require "log"
require "json"
{% if flag?(:enable_cuda) %}
  require "../cuda"
  require "../cudnn"
{% else %}
  require "../cuda_stub"
{% end %}
require "../math/simple_matrix"
require "../math/cuda_matrix"

module SHAInet
  class Network
    # ------------
    # This is a matrix-based neural network implementation. All operations
    # are performed on matrices rather than individual neurons/synapses.
    # This approach provides better performance and GPU acceleration capabilities.

    # This file contains all the methods for running and training the network,
    # for methods regarding creating and maintaining go to network_setup.cr
    # ------------

    @batch_in_ws : CudaMatrix?
    @batch_out_ws : CudaMatrix?
    @batch_grad_ws : CudaMatrix?

    # Run an input through the network to get an output (weights & biases do not change)
    # Simple wrapper that converts array input to matrix and calls the core matrix method
    def run(input : Array(GenNum), stealth : Bool = false) : Array(Float64)
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{expected_size}.") unless input.size == expected_size

      # Convert to matrix and use core matrix method
      processed = @mixed_precision ? input.map(&.to_f32.to_f64) : input.map(&.to_f64)
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([processed]))
      result_matrix = run(matrix, stealth: stealth)

      # Efficient array extraction - only sync once if needed
      output = if result_matrix.is_a?(CudaMatrix)
                 result_matrix.sync_from_device!("array_output") if result_matrix.device_dirty?
                 result_matrix.to_flat_array
               else
                 result_matrix.to_a.first
               end

      unless stealth
        Log.info { "Input => #{input}, network output => #{output}" }
      end
      output
    rescue ex : Exception
      raise NeuralNetRunError.new("Error running on layers: #{ex} #{ex.inspect_with_backtrace}")
    end

    # Overload allowing retrieval of the raw matrix
    def run(input : Array(GenNum), *, return_matrix : Bool, stealth : Bool = false) : Array(Float64) | CudaMatrix | SimpleMatrix
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      raise NeuralNetRunError.new(
        "Error input data size: #{input.size} doesn't fit input layer size: #{expected_size}.") unless input.size == expected_size

      processed = @mixed_precision ? input.map(&.to_f32.to_f64) : input.map(&.to_f64)
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([processed]))
      result_matrix = run(matrix, stealth: stealth)

      if return_matrix
        result_matrix
      else
        output = if result_matrix.is_a?(CudaMatrix)
                   result_matrix.sync_from_device!("array_output") if result_matrix.device_dirty?
                   result_matrix.to_flat_array
                 else
                   result_matrix.to_a.first
                 end

        Log.info { "Input => #{input}, network output => #{output}" } unless stealth
        output
      end
    rescue ex : Exception
      raise NeuralNetRunError.new("Error running on layers: #{ex} #{ex.inspect_with_backtrace}")
    end

    # GPU path - all CudaMatrix operations
    def run(input : CudaMatrix, stealth : Bool = false) : CudaMatrix
      verify_net_before_train

      matrix = input

      if !@transformer_layers.empty?
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            # Ensure embedding layer is on GPU for GPU path
            l.as(EmbeddingLayer).to_gpu!
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = extract_tokens_gpu(matrix)
            matrix = l.as(EmbeddingLayer).embed(tokens)
          when TransformerLayer
            # Ensure transformer layer is on GPU for GPU path
            l.as(TransformerLayer).to_gpu!
            matrix = l.as(TransformerLayer).forward(matrix)
          when LlamaLayer
            l.as(LlamaLayer).to_gpu!
            matrix = l.as(LlamaLayer).forward(matrix)
          else
            # Handle MatrixLayer and other layer types
            l.to_gpu! if l.responds_to?(:to_gpu!)
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        # Ensure output layer is on GPU for GPU path
        out_layer.to_gpu!
        w = out_layer.weights.as(CudaMatrix)
        b = out_layer.biases.as(CudaMatrix)
        if fn = @final_norm
          fn.to_gpu!
          matrix = fn.forward(matrix.as(CudaMatrix))
        end
        matrix = safe_output_transform(matrix.as(CudaMatrix), w)
        matrix.add_bias!(b)

        # Apply activation function - use GPU kernels when available
        unless out_layer.activation_function == SHAInet.identity
          # Try to use GPU kernels for common activation functions
          if try_gpu_activation(matrix, out_layer.activation_function)
            # GPU activation succeeded, update internal state matrices if needed
            if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
              # For now, keep last row handling simple - this is a rare case
              last_row_vals = matrix.rows == 1 ? matrix : slice_rows_helper(matrix, matrix.rows - 1, 1)
              out_layer.activations = last_row_vals.clone
              out_layer.sigma_primes = CudaMatrix.ones(1, matrix.cols)
            end
          else
            # Fallback to CPU for unsupported activation functions - minimize sync operations
            matrix.sync_from_device!("activation_fallback")
            # Use unsafe_get/unsafe_set for better performance
            matrix.rows.times do |i|
              matrix.cols.times do |j|
                val = matrix.unsafe_get(i, j)
                act, sig = out_layer.activation_function.call(val)
                matrix.unsafe_set(i, j, act)
                if i == matrix.rows - 1
                  # Update internal state matrices for output layer
                  if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
                    out_layer.activations.unsafe_set(0, j, act) if out_layer.activations.is_a?(CudaMatrix)
                    out_layer.sigma_primes.unsafe_set(0, j, sig) if out_layer.sigma_primes.is_a?(CudaMatrix)
                  end
                end
              end
            end
            matrix.mark_host_modified!
            matrix.sync_to_device!("activation_fallback")
          end
        end
        matrix.as(CudaMatrix)
      else
        # Standard matrix processing for non-transformer networks
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            # Ensure embedding layer is on GPU for GPU path
            l.as(EmbeddingLayer).to_gpu!
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = extract_tokens_gpu(matrix)
            matrix = l.as(EmbeddingLayer).embed(tokens)
          else
            # Handle MatrixLayer and other layer types
            l.to_gpu! if l.responds_to?(:to_gpu!)
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        # Ensure output layer is on GPU for GPU path
        out_layer.to_gpu!
        matrix = out_layer.forward(matrix)
        matrix.as(CudaMatrix)
      end
    rescue ex : Exception
      raise NeuralNetRunError.new("Error running on layers: #{ex} #{ex.inspect_with_backtrace}")
    end

    # CPU path - all SimpleMatrix operations
    # Scratch buffer for entering the device block chain, allocated once.
    # Keyed by shape: a prefill and a decode step have different row counts, and the
    # chain entry buffer is reused across turns. Bounded by MAX_CHAIN_SHAPES so a varying
    # prompt length cannot accumulate one buffer per length -- the same unbounded-cache
    # mistake that leaked several GB in the MoE prefill path.
    MAX_CHAIN_SHAPES = 3
    @dev_chain_bufs = Hash(Tuple(Int32, Int32), CudaMatrix).new

    private def chain_buf(rows : Int32, cols : Int32) : CudaMatrix
      key = {rows, cols}
      if existing = @dev_chain_bufs[key]?
        @dev_chain_bufs.delete(key)
        @dev_chain_bufs[key] = existing
        return existing
      end
      while @dev_chain_bufs.size >= MAX_CHAIN_SHAPES
        oldest = @dev_chain_bufs.first_key
        @dev_chain_bufs[oldest].free!
        @dev_chain_bufs.delete(oldest)
      end
      @dev_chain_bufs[key] = CudaMatrix.new(rows, cols)
    end

    # The single readback at the end of the device block chain. One per token for the
    # whole stack, rather than one per layer.
    private def device_row_to_host(src : CudaMatrix) : SimpleMatrix
      Profile.measure("net.dev_readback") do
        src.sync_from_device!("block_chain_out") if src.device_dirty?
        out = SimpleMatrix.new(src.rows, src.cols)
        out.data.to_unsafe.copy_from(src.raw_data.to_unsafe, src.rows * src.cols)
        out
      end
    end

    # Activation trace file, or nil. Set SHAINET_LAYER_TRACE to a path to enable.
    @@layer_trace_path : String? = begin
      v = ENV["SHAINET_LAYER_TRACE"]?
      v && !v.empty? ? v : nil
    end

    # Write one line per transformer block describing the activation leaving it.
    #
    # This exists for a specific failure mode: a stack where every weight passes its own numerical
    # check and the assembled model still does not predict. Component tests cannot find that, because
    # the fault is then in how the pieces are wired rather than in any one piece. Tracing the
    # activation through two builds of the SAME checkpoint and comparing block by block localizes the
    # divergence to one layer index, which maps to a layer type and a set of tensor types.
    #
    # It reads the activation back without clearing dev_act, so the device chain continues and a
    # traced run follows the same path as an untraced one.
    private def trace_layer_activation(layer_idx : Int32, l, matrix, dev_act : CudaMatrix?)
      path = @@layer_trace_path
      return unless path
      act = (da = dev_act) ? device_row_to_host(da) : matrix.as?(SimpleMatrix)
      return unless act
      row = act.rows - 1
      norm = 0.0
      act.cols.times { |c| norm += act[row, c] ** 2 }
      head = (0...Math.min(8, act.cols)).map { |c| act[row, c].round(5).to_s }.join(",")
      File.open(path, "a") do |f|
        f.puts "#{layer_idx}\t#{l.class.name.split("::").last}\t#{Math.sqrt(norm).round(5)}\t#{dev_act ? "dev" : "host"}\t#{head}"
      end

      # Dump the whole activation leaving this layer, so it can be fed into the SAME layer of a
      # different build. Feeding one build's activation through another's weights is what separates
      # "this layer's weights are wrong" from "this layer merely amplifies error accumulated
      # upstream" -- two explanations that look identical in a norm trace.
      if (after = ENV["SHAINET_DUMP_AFTER"]?) && after.to_i? == layer_idx && (dp = ENV["SHAINET_DUMP_PATH"]?)
        File.open(dp, "wb") do |f|
          act.cols.times { |c| f.write_bytes(act[row, c].to_f32, IO::ByteFormat::LittleEndian) }
        end
      end
    end

    # Rows per prefill pass through the stack. Tunable with SHAINET_PREFILL_ROWS; 0 disables.
    #
    # Without this, every activation buffer in the stack is sized to the WHOLE prompt, so peak VRAM
    # scales with context and a long conversation dies mid-turn. Measured on a 16 GB card with a
    # 27B model: a 9543-token prefill asked resident_buf for [9543, 5120] (186 MB) and the attention
    # workspace for 384 MB, and OOM'd -- while the same model prefills 2051 tokens with ~780 MB spare.
    # Chunking holds those buffers to [PREFILL_ROWS, *] no matter how long the prompt is.
    #
    # This is safe because the KV cache already makes a sliced prefill equivalent to a whole one --
    # it is the same mechanism a chat turn uses when it reuses cached tokens and prefills only the
    # new ones -- and because the head reduces to the last position anyway (gpu_lm_head_q slices the
    # final row), so the chunks before the last contribute cache state rather than logits.
    #
    # 512 keeps the matmuls comfortably in their batched regime while holding the widest activation
    # ([rows, 17408] appears three times in the SwiGLU trio) to about 100 MB.
    DEFAULT_PREFILL_ROWS = 512

    @@prefill_rows : Int32?

    def self.prefill_rows : Int32
      rows = @@prefill_rows
      return rows if rows
      rows = (ENV["SHAINET_PREFILL_ROWS"]?.try(&.to_i?) || DEFAULT_PREFILL_ROWS)
      @@prefill_rows = rows
      rows
    end

    def self.prefill_rows=(rows : Int32)
      @@prefill_rows = rows
    end

    def run(input : SimpleMatrix, stealth : Bool = false) : SimpleMatrix
      limit = Network.prefill_rows
      if limit > 0 && input.rows > limit && use_kv_cache? && !@transformer_layers.empty?
        result : SimpleMatrix? = nil
        offset = 0
        cols = input.cols
        slices = (input.rows + limit - 1) // limit

        # Report progress across the WHOLE prefill, not per slice.
        #
        # prefill_progress is called from the per-slice layer loop, so with the stack run once per
        # slice a caller's progress bar ran 1..N_layers, jumped back to 1, and climbed again -- which
        # reads as repeated work rather than as one prefill split into parts. Offsetting the layer
        # index by the slices already finished, and scaling the total, keeps it monotonic. The
        # callback's signature is unchanged, so callers need no edit.
        outer = @prefill_progress
        done_slices = 0
        if outer
          @prefill_progress = ->(layer_idx : Int32, total_layers : Int32) do
            outer.call(done_slices * total_layers + layer_idx, slices * total_layers)
          end
        end

        begin
          while offset < input.rows
            n = Math.min(limit, input.rows - offset)
            slice = SimpleMatrix.new(n, cols)
            # Contiguous row-major copy: row `offset + i` of the input is row `i` of the slice.
            slice.data.to_unsafe.copy_from(input.data.to_unsafe + offset.to_i64 * cols, n.to_i64 * cols)
            result = run_unchunked(slice, stealth)
            offset += n
            done_slices += 1
          end
        ensure
          @prefill_progress = outer
        end
        # The last slice carries the last position, which is the only one the head keeps.
        return result.not_nil!
      end

      run_unchunked(input, stealth)
    end

    private def run_unchunked(input : SimpleMatrix, stealth : Bool = false) : SimpleMatrix
      verify_net_before_train

      matrix = input

      if !@transformer_layers.empty?
        # Holds the activation when it is living on the device across llama blocks, so
        # a stack of blocks costs ONE upload and ONE readback instead of a host round
        # trip per layer. nil means the activation is currently the host `matrix`.
        dev_act : CudaMatrix? = nil

        # Prefill boost: uncap the hot-weight cache budget during multi-token passes
        # so that weights promoted layer-by-layer fill the available VRAM instead of
        # being evicted by the normal decode-safe budget. The device_has_room? check
        # is the real guard; the budget is just a soft ceiling that causes unnecessary
        # PCIe streaming during prefill.
        prefill_boosted = false
        if input.rows > 1
          Q4HostMatrix.prefill_boost!
          prefill_boosted = true
        end

        @hidden_layers.each_with_index do |l, layer_idx|
          if input.rows > 1 && @prefill_progress
            @prefill_progress.try &.call(layer_idx, @hidden_layers.size)
          end
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            matrix = Profile.measure("net.embed") { l.as(EmbeddingLayer).embed_cpu(tokens) }
          when TransformerLayer
            matrix = l.as(TransformerLayer).forward(matrix)
          when LlamaLayer
            block = l.as(LlamaLayer)
            # Per-block-type totals, which is what separates "the recurrence is slow" from "the
            # matmuls under it are slow". These NEST: stack.gdn and stack.llama each contain the
            # ffn.* and attn.* phases, so Profile's TOTAL over-counts and the shares to read are
            # against wall time, not against that total.
            #
            # This is how the chunked-prefill plan got cancelled. A 2048-token prefill measured
            # stack.gdn at 79% of wall, which looks like the sequential delta-rule scan -- but a
            # microbenchmark of that kernel at real shapes (48 heads, dk=dv=128) puts it at 340 ms
            # per 512-token chunk, so about 9% of stack.gdn and 6.5% of prefill. The other ~91% is
            # the ordinary projections and FFN, i.e. the quantized matmul path.
            Profile.measure("stack.llama") do
              if use_kv_cache?
                sm = matrix.as(SimpleMatrix)
                # Any row count now, not just a single token. A prefill used to be excluded
                # here, so every layer paid an upload and a readback around its attention and
                # its FFN even though both were device-resident internally.
                if (dev_act || sm.rows >= 1) && block.block_device_capable?
                  if dev_act.nil?
                    buf = chain_buf(sm.rows, sm.cols)
                    Profile.measure("net.dev_upload") do
                      buf.raw_data.to_unsafe.copy_from(sm.data.to_unsafe, sm.rows * sm.cols)
                      buf.mark_host_modified!
                      buf.sync_to_device!("block_chain_in")
                    end
                    dev_act = buf
                  end
                  nxt = if dev_act.not_nil!.rows > 1
                          block.forward_cached_device_multi(dev_act.not_nil!)
                        else
                          block.forward_cached_device(dev_act.not_nil!)
                        end
                  if nxt
                    dev_act = nxt
                  else
                    # The multi-token chain declined: finish this block on the host.
                    matrix = device_row_to_host(dev_act.not_nil!)
                    dev_act = nil
                    matrix = block.forward_cached(matrix.as(SimpleMatrix))
                  end
                else
                  # A block that cannot take the device path ends the chain: bring the
                  # activation home first so the host path sees the real activation
                  # rather than a stale copy.
                  if da = dev_act
                    matrix = device_row_to_host(da)
                    dev_act = nil
                  end
                  matrix = block.forward_cached(matrix.as(SimpleMatrix))
                end
              else
                matrix = block.forward(matrix)
              end
            end
          when GatedDeltaNetBlock
            # The linear-attention block type in a qwen3_5 hybrid stack.
            #
            # It now offers a device-in/device-out path, so it no longer ends the chain. That
            # mattered: in a hybrid stack 48 of 64 layers are this type, so breaking the chain here
            # brought the activation home and re-uploaded it about 96 times per token.
            #
            # Without this branch at all the case fell through and the block was SILENTLY SKIPPED:
            # a Qwen3.5-9B ran as an 8-layer attention-only model, produced one repeated token, and
            # gave byte-identical logits across changes that demonstrably altered the blocks --
            # which is the only reason it was noticed.
            gdn = l.as(GatedDeltaNetBlock)
            Profile.measure("stack.gdn") do
              nxt = nil
              if dev_act.nil? && (sm0 = matrix.as?(SimpleMatrix)) && gdn.device_chain_capable?
                buf = chain_buf(sm0.rows, sm0.cols)
                Profile.measure("net.dev_upload") do
                  buf.raw_data.to_unsafe.copy_from(sm0.data.to_unsafe, sm0.rows * sm0.cols)
                  buf.mark_host_modified!
                  buf.sync_to_device!("block_chain_in")
                end
                dev_act = buf
              end
              if da0 = dev_act
                nxt = gdn.forward_cached_device(da0)
              end
              if nxt
                dev_act = nxt
              else
                if da = dev_act
                  matrix = device_row_to_host(da)
                  dev_act = nil
                end
                sm = matrix.as(SimpleMatrix)
                matrix = use_kv_cache? && sm.rows == 1 ? gdn.forward_cached(sm) : gdn.forward(sm)
              end
            end
          end

          trace_layer_activation(layer_idx, l, matrix, dev_act)
        end

        if da = dev_act
          matrix = device_row_to_host(da)
        end

        Q4HostMatrix.prefill_restore! if prefill_boosted

        out_layer = @output_layers.last
        w = out_layer.weights
        b = out_layer.biases.as(SimpleMatrix)
        if fn = @final_norm
          matrix = Profile.measure("net.final_norm") { fn.forward(matrix.as(SimpleMatrix)) }
        end
        # When the quantized head applies the bias on the device, the host loop below
        # is skipped: it walked the whole vocab row per token.
        bias_on_device = false
        matrix = Profile.measure("net.lm_head") do
          if lq = @lm_head_q
            row_bias = matrix.as(SimpleMatrix).rows >= 1 ? b : nil
            bias_on_device = !row_bias.nil?
            gpu_lm_head_q(matrix.as(SimpleMatrix), lq, row_bias)
          elsif w.is_a?(CudaMatrix)
            gpu_lm_head(matrix.as(SimpleMatrix), w.as(CudaMatrix))
          else
            safe_output_transform(matrix.as(SimpleMatrix), w.as(SimpleMatrix))
          end
        end

        # CPU bias addition, unless the device head already added it.
        unless bias_on_device
          Profile.measure("net.out_bias") do
            matrix.rows.times do |i|
              matrix.cols.times do |j|
                matrix[i, j] += b[0, j]
              end
            end
          end
        end

        # Apply activation function - for identity, no operation needed
        unless out_layer.activation_function == SHAInet.identity
          matrix.rows.times do |i|
            matrix.cols.times do |j|
              val = matrix[i, j]
              act, sig = out_layer.activation_function.call(val)
              matrix[i, j] = act
              if i == matrix.rows - 1
                # Update internal state matrices for output layer
                if out_layer.responds_to?(:activations) && out_layer.responds_to?(:sigma_primes)
                  out_layer.activations[0, j] = act
                  out_layer.sigma_primes[0, j] = sig
                end
              end
            end
          end
        end
        matrix.as(SimpleMatrix)
      else
        # Standard matrix processing for non-transformer networks
        @hidden_layers.each do |l|
          case l
          when EmbeddingLayer
            raise NeuralNetRunError.new("Embedding input mismatch") unless matrix.cols == 1
            tokens = (0...matrix.rows).map { |r| matrix[r, 0].to_i }
            matrix = l.as(EmbeddingLayer).embed(tokens)
          else
            matrix = l.forward(matrix)
          end
        end
        out_layer = @output_layers.last
        matrix = out_layer.forward(matrix)
        matrix.as(SimpleMatrix)
      end
    rescue ex : Exception
      raise NeuralNetRunError.new("Error running on layers: #{ex} #{ex.inspect_with_backtrace}")
    end

    # Run a batch of sequences by calling `run` for each sequence
    # This is a convenience wrapper that can be consolidated with more direct matrix operations
    def run(input : Array(Array(Array(GenNum))), stealth : Bool = false) : Array(Array(Array(Float64)))
      input.map { |seq| run(seq, stealth: stealth) }
    end

    def run(input : Array(Array(Array(GenNum))), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Array(Float64))) | Array(CudaMatrix | SimpleMatrix)
      if return_matrix
        input.map { |seq| run(seq, stealth: stealth, return_matrix: true) }
      else
        input.map { |seq| run(seq, stealth: stealth) }
      end
    end

    # Accept a sequence of integer tokens for embedding layers
    # This is a convenience wrapper around the standard run method
    def run(input : Array(Array(Int32)), stealth : Bool = false) : Array(Array(Float64))
      seq = input.map(&.map(&.to_f64))
      run(seq, stealth: stealth)
    end

    def run(input : Array(Array(Int32)), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Float64)) | CudaMatrix | SimpleMatrix
      seq = input.map(&.map(&.to_f64))
      run(seq, stealth: stealth, return_matrix: return_matrix)
    end

    # Accept integer input for embedding layers
    # Creates a column vector [N, 1] so the embedding layer can look up token IDs.
    def run(input : Array(Int32), stealth : Bool = false) : Array(Float64)
      if @hidden_layers.any?(EmbeddingLayer)
        matrix = SimpleMatrix.new(input.size, 1)
        input.each_with_index { |id, i| matrix[i, 0] = id.to_f64 }
        result = run(matrix, stealth: stealth)
        result.to_a.last
      else
        run(input.map(&.to_f64), stealth: stealth)
      end
    end

    def run(input : Array(Int32), *, return_matrix : Bool, stealth : Bool = false) : Array(Float64) | CudaMatrix | SimpleMatrix
      if @hidden_layers.any?(EmbeddingLayer)
        matrix = SimpleMatrix.new(input.size, 1)
        input.each_with_index { |id, i| matrix[i, 0] = id.to_f64 }
        result = run(matrix, stealth: stealth)
        return_matrix ? result : result.to_a.last
      else
        float_in = input.map(&.to_f64)
        run(float_in, stealth: stealth, return_matrix: return_matrix)
      end
    end

    # Accept sequence input - converts to matrix and calls core matrix method
    def run(input : Array(Array(GenNum)), stealth : Bool = false) : Array(Array(Float64))
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      input.each do |step|
        raise NeuralNetRunError.new("Error input data size: #{step.size} doesn't fit input layer size: #{expected_size}.") unless step.size == expected_size
      end

      # Convert to matrix and use core matrix method
      processed = input.map do |x|
        @mixed_precision ? x.map(&.to_f32.to_f64) : x.map(&.to_f64)
      end
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(processed))
      result_matrix = run(matrix, stealth: stealth)

      # Efficient array extraction - sync only once if needed
      if result_matrix.is_a?(CudaMatrix)
        result_matrix.sync_from_device!("run_output") if result_matrix.device_dirty?
        result_matrix.to_a
      else
        result_matrix.to_a
      end
    rescue ex : Exception
      raise NeuralNetRunError.new("Error running on layers: #{ex} #{ex.inspect_with_backtrace}")
    end

    def run(input : Array(Array(GenNum)), *, return_matrix : Bool, stealth : Bool = false) : Array(Array(Float64)) | CudaMatrix | SimpleMatrix
      verify_net_before_train
      expected_size = @input_layers.reduce(0) { |acc, l| acc + l.size }
      input.each do |step|
        raise NeuralNetRunError.new("Error input data size: #{step.size} doesn't fit input layer size: #{expected_size}.") unless step.size == expected_size
      end

      processed = input.map do |x|
        @mixed_precision ? x.map(&.to_f32.to_f64) : x.map(&.to_f64)
      end
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(processed))
      result_matrix = run(matrix, stealth: stealth)

      if return_matrix
        result_matrix
      else
        if result_matrix.is_a?(CudaMatrix)
          result_matrix.sync_from_device!("run_output") if result_matrix.device_dirty?
          result_matrix.to_a
        else
          result_matrix.to_a
        end
      end
    rescue ex : Exception
      raise NeuralNetRunError.new("Error running on layers: #{ex} #{ex.inspect_with_backtrace}")
    end

    # Quantifies how good the network performed for a single input compared to the expected output
    # This function returns the actual output and updates the error gradient for the output layer
    def evaluate(input_data : Array(GenNum),
                 expected_output : Array(GenNum),
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      processed = @mixed_precision ? input_data.map(&.to_f32.to_f64) : input_data.map(&.to_f64)
      actual_output = run(processed, stealth: true)

      # Test for NaNs & exploading gradients
      validate_values(actual_output, "actual_output")

      # Get the error signal for the final layer, based on the cost function
      @error_signal = [] of Float64 # Collect all the errors for current run

      actual_output.size.times do |i|
        cost = cost_function.call(expected_output[i], actual_output[i])
        @error_signal << cost[:value]

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "---"
      end

      # Test for NaNs & exploading gradients
      validate_values(@error_signal, "error_signal")
      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i }

      if !@transformer_layers.empty?
        # Create matrices efficiently using GPU when available
        exp_data = expected_output.map(&.to_f64)

        exp = if CUDA.fully_available?
                mat = CudaMatrix.new(1, exp_data.size)
                exp_data.each_with_index { |val, i| mat[0, i] = val }
                mat.sync_to_device!("evaluate_expected_matrix")
                mat
              else
                SimpleMatrix.from_a([exp_data])
              end

        act = if CUDA.fully_available?
                mat = CudaMatrix.new(1, actual_output.size)
                actual_output.each_with_index { |val, i| mat[0, i] = val }
                mat.sync_to_device!("evaluate_actual_matrix")
                mat
              else
                SimpleMatrix.from_a([actual_output])
              end

        diff = if act.is_a?(CudaMatrix) && exp.is_a?(CudaMatrix)
                 act - exp
               else
                 act_s = act.is_a?(CudaMatrix) ? act.to_simple : act
                 exp_s = exp.is_a?(CudaMatrix) ? exp.to_simple : exp
                 act_s - exp_s
               end
        out_w = @output_layers.last.weights
        w = out_w.is_a?(CudaMatrix) ? out_w : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix))
        trans = if diff.is_a?(CudaMatrix) && w.is_a?(CudaMatrix)
                  diff * w
                else
                  d = diff.is_a?(CudaMatrix) ? diff.to_simple : diff
                  ww = w.is_a?(CudaMatrix) ? w.to_simple : w
                  d * ww
                end
        @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
      end

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue ex : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{ex}")
    end

    # Evaluate using matrices already on the desired device
    # GPU-compatible version that preserves matrix type
    def evaluate(input_data : SimpleMatrix | CudaMatrix,
                 expected_output : SimpleMatrix | CudaMatrix,
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      actual_matrix = run(input_data, stealth: true)

      grad = GPUMemory.like(actual_matrix, actual_matrix.rows, actual_matrix.cols)

      loss_value = 0.0

      if actual_matrix.is_a?(CudaMatrix) && expected_output.is_a?(CudaMatrix) && CUDNN.available?
        begin
          CUDNN.softmax_cross_entropy_loss_and_gradient(
            actual_matrix.as(CudaMatrix),
            expected_output.as(CudaMatrix),
            pointerof(loss_value),
            grad.as(CudaMatrix)
          )
        rescue
          loss_value = compute_cost_and_gradient_cpu(actual_matrix, expected_output, grad, cost_function)
        end
      else
        loss_value = compute_cost_and_gradient_cpu(actual_matrix, expected_output, grad, cost_function)
      end

      @error_signal = [loss_value]
      @total_error = loss_value

      if !@transformer_layers.empty?
        diff = grad
        out_w = @output_layers.last.weights
        w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
        trans = if diff.is_a?(CudaMatrix) && w.is_a?(CudaMatrix)
                  diff * w
                else
                  d = diff.is_a?(CudaMatrix) ? diff.to_simple : diff
                  ww = w.is_a?(CudaMatrix) ? w.to_simple : w
                  d * ww
                end
        @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
      end
    rescue ex : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{ex}")
    end

    # Accept integer input for embeddings
    # This is a convenience wrapper around the standard evaluate method
    def evaluate(input_data : Array(Int32),
                 expected_output : Array(GenNum),
                 cost_function : CostFunction = SHAInet.quadratic_cost)
      evaluate(input_data.map(&.to_f64), expected_output, cost_function)
    end

    def evaluate_sequence(input_data : Array(Array(GenNum)),
                          expected_output : Array(GenNum),
                          cost_function : CostFunction = SHAInet.quadratic_cost)
      seq = input_data.map do |x|
        @mixed_precision ? x.map(&.to_f32(to_f64)) : x.map(&.to_f64)
      end
      outputs = run(seq, stealth: true)
      actual_output = outputs.last

      # Test for NaNs & exploading gradients
      validate_values(actual_output, "actual_output")

      # Get the error signal for the final layer, based on the cost function
      @error_signal = [] of Float64 # Collect all the errors for current run

      actual_output.size.times do |i|
        cost = cost_function.call(expected_output[i], actual_output[i])
        @error_signal << cost[:value]

        # puts "Actual output: #{actual_output}"
        # puts "Cost value: #{cost[:value]}"
        # puts "Cost derivative: #{cost[:derivative]}"
        # puts "---"
      end

      # Test for NaNs & exploading gradients
      validate_values(@error_signal, "error_signal")
      @total_error = @error_signal.reduce(0.0) { |acc, i| acc + i }

      if !@transformer_layers.empty?
        exp_row = GPUMemory.to_gpu(SimpleMatrix.from_a([expected_output.map(&.to_f64)]))
        act_row = GPUMemory.to_gpu(SimpleMatrix.from_a([actual_output]))
        diff = act_row - exp_row
        tmp = GPUMemory.zeros_like(diff, outputs.size, diff.cols)
        tmp = tmp.to_simple if tmp.is_a?(CudaMatrix)

        # Use efficient row copying instead of element-by-element access
        tmp.set_row!(outputs.size - 1, diff.is_a?(CudaMatrix) ? diff.to_simple : diff, 0)
        @transformer_error = tmp
      end

      # puts "@error_signal: #{@error_signal}"
      # puts "@total_error: #{@total_error}"


    rescue ex : Exception
      raise NeuralNetRunError.new("Error in evaluate: #{ex}")
    end

    # Convenience wrapper for integer inputs
    def evaluate_sequence_label(input_data : Array(Array(Int32)), label : Int32)
      seq = input_data.map(&.map(&.to_f64))
      evaluate_sequence_label(seq, label)
    end

    # Evaluate a single example using a class label and softmax cross entropy
    def evaluate_label(input_data : Array(GenNum), label : Int32)
      processed = input_data.map(&.to_f64)
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a([processed]))
      logits = run(matrix, stealth: true)

      if logits.is_a?(CudaMatrix)
        if label < 0 || label >= logits.cols
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{logits.cols}")
        end

        target = CudaMatrix.zeros(1, logits.cols)
        target[0, label] = 1.0
        target.sync_to_device!

        grad = CudaMatrix.new(1, logits.cols)
        loss_val = 0.0

        if CUDNN.available?
          CUDNN.softmax_cross_entropy_loss_and_gradient(logits.as(CudaMatrix), target, pointerof(loss_val), grad)
        else
          logits.as(CudaMatrix).softmax_rows!
          grad.copy_from!(logits.as(CudaMatrix))
          grad[0, label] = grad[0, label] - 1.0
          logits.as(CudaMatrix).sync_from_device!("eval_label")
          loss_val = -Math.log(logits.as(CudaMatrix).unsafe_get(0, label).clamp(1e-9, 1.0))
        end

        @error_signal = Array(Float64).new(logits.cols, 0.0)
        @error_signal[label] = loss_val
        @total_error = loss_val

        if !@transformer_layers.empty?
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
          trans = grad * w
          @transformer_error = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
        end
      else
        actual_output = logits.as(SimpleMatrix).to_a.first
        validate_values(actual_output, "actual_output")
        probs = SHAInet.softmax(actual_output)

        if label < 0 || label >= probs.size
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{probs.size}")
        end

        @error_signal = [] of Float64
        probs.size.times do |i|
          @error_signal << (i == label ? -Math.log(probs[i].clamp(1e-9, 1.0)) : 0.0)
        end

        validate_values(@error_signal, "error_signal")
        @total_error = -Math.log(probs[label].clamp(1e-9, 1.0))
      end
    end

    # Convenience wrapper for integer inputs
    def evaluate_label(input_data : Array(Int32), label : Int32)
      evaluate_label(input_data.map(&.to_f64), label)
    end

    # Convenience wrapper for matrix inputs
    def evaluate_label(input_data : SimpleMatrix, label : Int32)
      vec = input_data.to_a.first.map(&.to_f64)
      evaluate_label(vec, label)
    end

    # Evaluate a sequence example with a class label and softmax cross entropy
    def evaluate_sequence_label(input_data : Array(Array(GenNum)), label : Int32)
      seq = input_data.map(&.map(&.to_f64))
      matrix = GPUMemory.to_gpu(SimpleMatrix.from_a(seq))
      logits = run(matrix, stealth: true)

      if logits.is_a?(CudaMatrix)
        if label < 0 || label >= logits.cols
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{logits.cols}")
        end

        target = CudaMatrix.zeros(1, logits.cols)
        target[0, label] = 1.0
        target.sync_to_device!

        grad = CudaMatrix.new(1, logits.cols)
        loss_val = 0.0

        if CUDNN.available?
          CUDNN.softmax_cross_entropy_loss_and_gradient(logits.as(CudaMatrix), target, pointerof(loss_val), grad)
        else
          logits.as(CudaMatrix).softmax_rows!
          grad.copy_from!(logits.as(CudaMatrix))
          grad[0, label] = grad[0, label] - 1.0
          logits.as(CudaMatrix).sync_from_device!("eval_seq_label")
          loss_val = -Math.log(logits.as(CudaMatrix).unsafe_get(0, label).clamp(1e-9, 1.0))
        end

        @error_signal = Array(Float64).new(logits.cols, 0.0)
        @error_signal[label] = loss_val
        @total_error = loss_val

        if !@transformer_layers.empty?
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.as(CudaMatrix) : GPUMemory.keep_on_gpu(out_w.as(SimpleMatrix)).as(CudaMatrix)
          trans = grad * w
          tmp = GPUMemory.zeros_like(trans, matrix.rows, trans.cols)
          tmp = tmp.to_simple if tmp.is_a?(CudaMatrix)
          trans_s = trans.is_a?(CudaMatrix) ? trans.to_simple : trans
          trans.cols.times do |j|
            tmp[matrix.rows - 1, j] = trans_s[0, j]
          end
          @transformer_error = tmp
        end
      else
        outputs = logits.as(SimpleMatrix).to_a
        actual_output = outputs.last
        validate_values(actual_output, "actual_output")
        probs = SHAInet.softmax(actual_output)

        if label < 0 || label >= probs.size
          raise NeuralNetRunError.new("Label #{label} out of bounds for output size #{probs.size}")
        end

        @error_signal = [] of Float64
        @total_error = -Math.log(probs[label].clamp(1e-9, 1.0))

        if !@transformer_layers.empty?
          exp_row = SimpleMatrix.zeros(1, probs.size)
          exp_row[0, label] = 1.0
          act_row = SimpleMatrix.from_a([probs])
          diff = act_row - exp_row
          out_w = @output_layers.last.weights
          w = out_w.is_a?(CudaMatrix) ? out_w.to_simple : out_w.as(SimpleMatrix)
          trans = diff * w
          tmp = SimpleMatrix.zeros(matrix.rows, trans.cols)
          trans.cols.times do |j|
            tmp[matrix.rows - 1, j] = trans[0, j]
          end
          @transformer_error = tmp
        end
      end
    end

    # Calculate MSE from the error signal of the output layer
    def update_mse
      n = @output_layers.last.size
      if @error_signal.size == 1
        error_avg = 0.0
      else
        error_avg = @total_error / n
      end
      sqrd_dists = 0.0
      @error_signal.each { |e| sqrd_dists += (e - error_avg)**2 }
      @mse = sqrd_dists / n
    end

    # Clean matrix-based training method
    def train(data : Array(Array) | SHAInet::TrainingData | SHAInet::StreamingData,
              training_type : Symbol | String = :sgdm,
              cost_function : Symbol | String | CostFunction = :mse,
              epochs : Int32 = 1,
              error_threshold : Float64 = 0.00000001,
              mini_batch_size : Int32 = 1,
              log_each : Int32 = 1,
              show_slice : Bool = false,
              autosave : NamedTuple(freq: Int32, path: String)? = nil)
      verify_net_before_train

      stream = data.is_a?(SHAInet::StreamingData) ? data : nil
      if data.is_a?(SHAInet::TrainingData) && data.preload_gpu? && CUDA.fully_available?
        data.preload_gpu!
      end
      # Convert TrainingData to raw data array
      raw_data = if data.is_a?(SHAInet::TrainingData)
                   data.data
                 elsif data.is_a?(Array)
                   data.as(Array)
                 else
                   [] of Array(Array(Float64))
                 end

      # Validate and convert cost function
      if cost_function.is_a?(Symbol) || cost_function.is_a?(String)
        raise NeuralNetRunError.new("Must define correct cost function type (:mse, :c_ent, :c_ent_sm).") if COST_FUNCTIONS.any? { |x| x == cost_function.to_s } == false
        cost_proc = get_cost_proc(cost_function.to_s)
      else
        cost_proc = cost_function
      end

      if stream
        Log.info { "Training started with #{epochs} epochs (streaming)" }
      else
        Log.info { "Training started with #{epochs} epochs, #{raw_data.size} samples" }
      end
      start_time = Time.monotonic

      batch_size = stream ? mini_batch_size : mini_batch_size.clamp(1, raw_data.size)

      epochs.times do |epoch|
        # Reset sync counters at start of each epoch
        if CUDA.fully_available?
          SHAInet::CudaMatrix.reset_sync_stats
        end

        # Autosave if configured
        if autosave && epoch % autosave[:freq] == 0 && epoch > 0
          save_to_file("#{autosave[:path]}/autosave_epoch_#{epoch}.nn")
        end

        # Shuffle or rewind data for each epoch
        total_error = 0.0
        sample_count = 0

        if stream
          stream.rewind if epoch > 0
          while (batch = stream.next_batch(batch_size)).size > 0
            batch_error = process_batch(batch, cost_proc, training_type)
            total_error += batch_error
            sample_count += batch.size
          end
        else
          shuffled_data = raw_data.shuffle
          # Process data in mini-batches
          shuffled_data.each_slice(batch_size) do |batch|
            batch_error = process_batch(batch, cost_proc, training_type)
            total_error += batch_error
            sample_count += batch.size
          end
        end

        avg_error = total_error / sample_count
        @total_error = total_error
        @error_signal = [avg_error]
        update_mse

        if epoch % log_each == 0
          elapsed = Time.monotonic - start_time
          gpu_stats = SHAInet::CudaMatrix.gpu_memory_stats
          sync_stats = CUDA.fully_available? ? SHAInet::CudaMatrix.sync_stats : nil

          if sync_stats
            sync_to_mb = (sync_stats[:total_sync_bytes_to_device] / 1024.0 / 1024.0).round(2)
            sync_from_mb = (sync_stats[:total_sync_bytes_from_device] / 1024.0 / 1024.0).round(2)
            total_syncs = sync_stats[:sync_to_device_count] + sync_stats[:sync_from_device_count]
            Log.info { "Epoch: #{epoch}, Error: #{avg_error.round(6)}, MSE: #{@mse.round(6)}, Time: #{elapsed.total_seconds.round(2)}s" }
            Log.debug { "  GPU: #{gpu_stats[:active_matrices]} matrices, #{(gpu_stats[:total_allocated_bytes] / 1024.0 / 1024.0).round(2)} MB" }
            Log.debug { "  Syncs: #{total_syncs} total (#{sync_stats[:sync_to_device_count]} to GPU, #{sync_stats[:sync_from_device_count]} from GPU)" }
            Log.debug { "  Data: #{sync_to_mb} MB to GPU, #{sync_from_mb} MB from GPU (#{(sync_to_mb + sync_from_mb).round(2)} MB total)" }
            Log.debug { "  Matrix creations: #{sync_stats[:matrix_creation_count]} this epoch" }
            SHAInet::CudaMatrix.print_top_allocation_sites

            # Log top sync sources
            sources = SHAInet::CudaMatrix.sync_sources_stats
            if sources.size > 0
              Log.debug { "  Top sync sources:" }
              sources.to_a.sort_by { |_, v| v }.reverse![0, 5].each do |source, count|
                Log.debug { "    #{source}: #{count} times" }
              end
            end
          else
            Log.info { "Epoch: #{epoch}, Error: #{avg_error.round(6)}, MSE: #{@mse.round(6)}, Time: #{elapsed.total_seconds.round(2)}s, GPU: #{gpu_stats[:active_matrices]} matrices, #{gpu_stats[:total_allocated_bytes]} bytes" }
          end

          # Log matrix pool statistics
          if CUDA.fully_available?
            pool_stats = CudaMatrix.pool_stats
            Log.debug { "  Matrix pool: #{pool_stats[:total_pooled]} matrices pooled across #{pool_stats[:pools].size} sizes" }
            if pool_stats[:pools].size > 0
              top_pools = pool_stats[:pools].to_a.sort_by(&.[1]).reverse!.first(3)
              Log.debug { "    Top pool sizes: #{top_pools.map { |k, v| "#{k}(#{v})" }.join(", ")}" }
            end
          end
        end

        if avg_error < error_threshold
          Log.info { "Training stopped early. Error threshold reached: #{avg_error} < #{error_threshold}" }
          break
        end

        @time_step += 1
      end

      elapsed = Time.monotonic - start_time
      Log.info { "Training completed in #{elapsed.total_seconds.round(2)} seconds" }
    end

    private def process_batch(batch, cost_proc, training_type)
      batch_error = 0.0

      # Zero gradients for all matrix layers
      @hidden_layers.each do |layer|
        layer.zero_gradients if layer.is_a?(MatrixLayer)
      end
      @output_layers.each do |layer|
        layer.zero_gradients if layer.is_a?(MatrixLayer)
      end

      # Determine matrix dimensions from first sample for workspace allocation
      first_input = batch.first[0]
      first_output = batch.first[1]
      # If the first output is a single label, expand to one-hot only when GPU
      # optimized label cross-entropy is unavailable.
      if first_output.is_a?(Array) && first_output.as(Array).size == 1 &&
         !first_output.as(Array)[0].is_a?(Array) && @output_layers.last.is_a?(MatrixLayer)
        if !(CUDA.fully_available? && CUDNN.available? &&
           @output_layers.last.as(MatrixLayer).size > 1)
          label = first_output.as(Array).first.as(GenNum).to_i
          oh = Array(Float64).new(@output_layers.last.as(MatrixLayer).size, 0.0)
          oh[label] = 1.0 if label >= 0 && label < oh.size
          first_output = oh
        end
      end

      get_dims = ->(obj : SimpleMatrix | CudaMatrix | Array(Array(Float64)) | Array(Float64)) do
        case obj
        when SimpleMatrix
          {obj.rows, obj.cols}
        when CudaMatrix
          {obj.rows, obj.cols}
        else
          arr = obj.as(Array)
          if arr.size > 0 && arr[0].is_a?(Array)
            {arr.size, arr[0].as(Array).size}
          else
            {1, arr.size}
          end
        end
      end

      in_rows, in_cols = get_dims.call(first_input)
      out_rows, out_cols = get_dims.call(first_output)

      input_workspace : CudaMatrix?
      expected_workspace : CudaMatrix?
      output_grad : SimpleMatrix | CudaMatrix?

      if CUDA.fully_available?
        if !first_input.is_a?(CudaMatrix)
          if @batch_in_ws.nil? || @batch_in_ws.not_nil!.rows != in_rows || @batch_in_ws.not_nil!.cols != in_cols
            @batch_in_ws = CudaMatrix.new(in_rows, in_cols)
          end
          input_workspace = @batch_in_ws
        end
        if !first_output.is_a?(CudaMatrix)
          if @batch_out_ws.nil? || @batch_out_ws.not_nil!.rows != out_rows || @batch_out_ws.not_nil!.cols != out_cols
            @batch_out_ws = CudaMatrix.new(out_rows, out_cols)
          end
          expected_workspace = @batch_out_ws
        end
        if @batch_grad_ws.nil? || @batch_grad_ws.not_nil!.rows != out_rows || @batch_grad_ws.not_nil!.cols != out_cols
          @batch_grad_ws = CudaMatrix.new(out_rows, out_cols)
        end
        output_grad = @batch_grad_ws
      end

      batch.each do |sample|
        input_data = sample[0]
        expected_output = sample[1]
        # If expected output is a single label, expand to one-hot only when GPU
        # accelerated label cross-entropy cannot be used.
        # IMPORTANT: Only convert to one-hot if output layer has multiple neurons (classification).
        # For regression tasks with a single output neuron, keep the expected value as-is.
        output_layer_size = @output_layers.last.is_a?(MatrixLayer) ? @output_layers.last.as(MatrixLayer).size : 0
        if expected_output.is_a?(Array) && expected_output.as(Array).size == 1 &&
           !expected_output.as(Array)[0].is_a?(Array) && @output_layers.last.is_a?(MatrixLayer) &&
           output_layer_size > 1 # Only for classification with multiple outputs
          if !(CUDA.fully_available? && CUDNN.available?)
            label = expected_output.as(Array).first.as(GenNum).to_i
            oh = Array(Float64).new(output_layer_size, 0.0)
            oh[label] = 1.0 if label >= 0 && label < oh.size
            expected_output = oh
          end
        end

        # Prepare expected output matrix using workspace when on GPU
        expected_matrix = case expected_output
                          when SimpleMatrix
                            if expected_workspace
                              GPUMemory.to_gpu!(expected_output, expected_workspace)
                            else
                              expected_output
                            end
                          when CudaMatrix
                            expected_output
                          else
                            arr = expected_output.as(Array)
                            if expected_workspace
                              case arr
                              when Array(Array(GenNum)), Array(Array(Float64))
                                # 2D: arr is Array(Array(GenNum)) or Array(Array(Float64))
                                GPUMemory.to_gpu!(arr.map(&.map(&.to_f64)), expected_workspace)
                              when Array(Float64)
                                # 1D: arr is Array(Float64)
                                GPUMemory.to_gpu!([arr], expected_workspace)
                              when Array(GenNum)
                                # 1D: arr is Array(GenNum)
                                GPUMemory.to_gpu!([arr.map(&.to_f64)], expected_workspace)
                              else
                                # Scalar fallback (should not happen, but for safety)
                                GPUMemory.to_gpu!([[arr.to_f64]], expected_workspace)
                              end
                              expected_workspace
                            else
                              to_matrix(expected_output)
                            end
                          end

        # Prepare input matrix using preallocated workspace when on GPU
        input_matrix = case input_data
                       when SimpleMatrix
                         if input_workspace
                           GPUMemory.to_gpu!(input_data, input_workspace)
                         else
                           input_data
                         end
                       when CudaMatrix
                         input_data
                       else
                         arr = input_data.as(Array)
                         if input_workspace
                           if arr.size > 0 && arr[0].is_a?(Array)
                             GPUMemory.to_gpu!(arr.as(Array(Array(Float64))), input_workspace)
                           else
                             GPUMemory.to_gpu!(arr.as(Array(Float64)), input_workspace)
                           end
                           input_workspace
                         else
                           to_matrix(input_data)
                         end
                       end

        actual_matrix = run(input_matrix, stealth: true)

        # Optimize: Use GPU-accelerated cost and gradient computation when possible
        output_layer = @output_layers.last

        if output_layer.is_a?(MatrixLayer)
          # Determine dimensions for output_grad matrix
          use_label_gpu = actual_matrix.is_a?(CudaMatrix) && expected_matrix.is_a?(CudaMatrix) &&
                          CUDNN.available? && expected_matrix.as(CudaMatrix).cols == 1 &&
                          actual_matrix.as(CudaMatrix).cols > 1

          grad_rows, grad_cols = if use_label_gpu
                                   {actual_matrix.as(CudaMatrix).rows, actual_matrix.as(CudaMatrix).cols}
                                 elsif expected_output.is_a?(SimpleMatrix)
                                   exp_mat = expected_output.as(SimpleMatrix)
                                   {exp_mat.rows, exp_mat.cols}
                                 elsif expected_output.is_a?(CudaMatrix)
                                   exp_mat = expected_output.as(CudaMatrix)
                                   {exp_mat.rows, exp_mat.cols}
                                 elsif expected_output.is_a?(Array) && expected_output.as(Array).size > 0 && expected_output.as(Array)[0].is_a?(Array)
                                   {expected_output.as(Array).size, expected_output.as(Array)[0].as(Array).size}
                                 else
                                   arr = expected_output.as(Array)
                                   {1, arr.size}
                                 end

          # Reuse preallocated output_grad matrix when available
          if output_grad.nil? || output_grad.not_nil!.rows != grad_rows || output_grad.not_nil!.cols != grad_cols
            output_grad = GPUMemory.like(actual_matrix, grad_rows, grad_cols)
            @batch_grad_ws = output_grad if CUDA.fully_available? && output_grad.is_a?(CudaMatrix)
          end

          existing_grad = output_grad.not_nil!
          if existing_grad.is_a?(CudaMatrix)
            existing_grad.zero!
          else
            existing_grad.rows.times do |i|
              existing_grad.cols.times do |j|
                existing_grad[i, j] = 0.0
              end
            end
          end

          grad_matrix = output_grad.not_nil!

          # Try GPU-accelerated cross-entropy when possible
          # Only use CUDNN softmax+CE when the output layer has identity activation
          # (i.e., raw logits). CUDNN applies softmax internally, so if the output
          # layer already applied sigmoid/other activation, this produces wrong gradients.
          output_is_raw_logits = output_layer.activation_function == SHAInet.identity ||
                                 output_layer.activation_function == SHAInet.none
          if output_is_raw_logits && actual_matrix.is_a?(CudaMatrix) && expected_matrix.is_a?(CudaMatrix) && CUDNN.available?
            begin
              loss_value = 0.0
              if use_label_gpu
                CUDNN.softmax_cross_entropy_label_loss_and_gradient(
                  actual_matrix.as(CudaMatrix),
                  expected_matrix.as(CudaMatrix),
                  pointerof(loss_value),
                  grad_matrix.as(CudaMatrix)
                )
              else
                CUDNN.softmax_cross_entropy_loss_and_gradient(
                  actual_matrix.as(CudaMatrix),
                  expected_matrix.as(CudaMatrix),
                  pointerof(loss_value),
                  grad_matrix.as(CudaMatrix)
                )
              end
              sample_error = loss_value
            rescue ex : Exception
              Log.debug { "GPU cross-entropy failed: #{ex}, falling back to CPU computation" } unless use_label_gpu
              # Fall back to CPU computation below
              if use_label_gpu
                # Convert label indices to one-hot for CPU fallback
                one_hot = SimpleMatrix.zeros(expected_matrix.rows, actual_matrix.cols)
                expected_matrix.rows.times do |i|
                  label = expected_matrix.as(CudaMatrix).unsafe_get(i, 0).to_i
                  one_hot[i, label] = 1.0 if label >= 0 && label < actual_matrix.cols
                end
                sample_error = compute_cost_and_gradient_cpu(actual_matrix, one_hot, grad_matrix, cost_proc)
              else
                sample_error = compute_cost_and_gradient_cpu(actual_matrix, expected_matrix, grad_matrix, cost_proc)
              end
            end
          else
            # CPU fallback for non-CudaMatrix types or when GPU computation is unavailable
            if use_label_gpu
              one_hot = SimpleMatrix.zeros(expected_matrix.rows, actual_matrix.cols)
              expected_matrix.rows.times do |i|
                label = expected_matrix.as(CudaMatrix).unsafe_get(i, 0).to_i
                one_hot[i, label] = 1.0 if label >= 0 && label < actual_matrix.cols
              end
              sample_error = compute_cost_and_gradient_cpu(actual_matrix, one_hot, grad_matrix, cost_proc)
            else
              sample_error = compute_cost_and_gradient_cpu(actual_matrix, expected_matrix, grad_matrix, cost_proc)
            end
          end

          batch_error += sample_error

          # grad_matrix is already GPU-compatible and reused across samples
          grad = output_layer.backward(grad_matrix)

          # Handle transformer layers backward pass with proper gradient reshaping
          if @transformer_layers.present?
            # For transformers, we need to map gradients from output space back to transformer space
            # The gradient from output layer is (1 x vocab_size), but transformer expects (seq_len x d_model)

            # Get the transformer's d_model dimension from layer size
            d_model = @transformer_layers.first.size
            seq_len = 16 # hardcoded for now, should be dynamic

            if grad.rows == 1 && grad.cols != d_model
              # We have output gradients (1 x vocab_size), need to transform to (seq_len x d_model)
              # This requires going through the output layer weights transpose

              # Use appropriate matrix type based on whether grad is CudaMatrix or SimpleMatrix
              if grad.is_a?(CudaMatrix)
                output_weights = @output_layers.last.weights.as(CudaMatrix)
              else
                output_weights = @output_layers.last.weights.as(SimpleMatrix)
              end

              # Transform: (1 x vocab_size) * (vocab_size x d_model) -> (1 x d_model)
              if grad.cols == output_weights.rows
                if grad.is_a?(CudaMatrix)
                  transformed_grad = grad * output_weights.as(CudaMatrix).transpose
                else
                  transformed_grad = grad * output_weights.as(SimpleMatrix).transpose
                end
              else
                # Fallback: create zero gradients with correct dimensions
                mat_klass = grad.is_a?(CudaMatrix) ? CudaMatrix : SimpleMatrix
                transformed_grad = mat_klass.zeros(1, d_model)
              end

              # Check if we can reuse cached gradient matrix
              if @cached_expanded_grad.nil? || @cached_seq_len != seq_len || @cached_d_model != d_model ||
                 @cached_expanded_grad.not_nil!.class != transformed_grad.class
                # Create new cached matrix
                @cached_expanded_grad = if transformed_grad.is_a?(CudaMatrix)
                                          CudaMatrix.zeros(seq_len, d_model)
                                        else
                                          SimpleMatrix.zeros(seq_len, d_model)
                                        end
                @cached_seq_len = seq_len
                @cached_d_model = d_model
              end

              # Reuse cached matrix - zero only the last row that we'll update
              expanded_grad = @cached_expanded_grad.not_nil!
              # Zero out only the last row efficiently
              d_model.times do |j|
                expanded_grad[seq_len - 1, j] = transformed_grad[0, j]
              end

              grad = expanded_grad
            end

            @transformer_layers.reverse_each do |layer|
              grad = layer.backward(grad)
            end
          end

          @hidden_layers.reverse_each do |layer|
            if layer.is_a?(MatrixLayer)
              grad = layer.backward(grad)
            elsif layer.is_a?(EmbeddingLayer)
              layer.accumulate_gradient
            end
          end
        end

        # Explicit GPU memory cleanup after processing each sample
        # Force cleanup of temporary matrices created during forward/backward pass
        # Persistent GPU buffers handle workspace reuse

        # NOTE: @batch_in_ws / @batch_out_ws are owned by the network and reused
        # for every sample in the batch (and across batches). They must NOT be
        # returned to the shared workspace pool: get_workspace would hand the
        # same buffer out as a temporary (e.g. the output layer's forward
        # scratch, which has the same shape as the expected-output workspace),
        # zero it and overwrite it mid-sample, corrupting the input/target.
      end

      learning_rate = current_learning_rate

      @hidden_layers.each do |layer|
        if layer.is_a?(MatrixLayer)
          layer.update_weights(learning_rate, training_type,
            momentum: @momentum,
            beta1: @beta1, beta2: @beta2,
            epsilon: @epsilon, time_step: @time_step,
            alpha: @alpha, weight_decay: @weight_decay)
        elsif layer.is_a?(EmbeddingLayer)
          layer.apply_gradients(learning_rate)
        end
      end

      @output_layers.each do |layer|
        if layer.is_a?(MatrixLayer)
          layer.update_weights(learning_rate, training_type,
            momentum: @momentum,
            beta1: @beta1, beta2: @beta2,
            epsilon: @epsilon, time_step: @time_step,
            alpha: @alpha, weight_decay: @weight_decay)
        end
      end

      update_transformer_layers if @transformer_layers.present?

      batch_error
    end

    # Convert raw data (arrays) to matrix format
    # This method is only called when input is NOT already a matrix
    private def to_matrix(obj) : SimpleMatrix | CudaMatrix
      arr = obj.as(Array)

      # When CUDA is available, allocate CudaMatrix directly and copy the
      # contiguous data buffer instead of filling element by element.
      if CUDA.fully_available?
        if arr.size > 0 && arr[0].is_a?(Array)
          rows = arr.size
          cols = arr[0].as(Array).size
          mat = CudaMatrix.new(rows, cols)

          # Write F32 values to host buffer, then sync to device
          rows.times do |r|
            cols.times do |c|
              mat.raw_data[r * cols + c] = arr[r].as(Array)[c].as(GenNum).to_f32
            end
          end
          mat.mark_host_modified!
          mat.sync_to_device!("to_matrix")

          mat
        else
          cols = arr.size
          mat = CudaMatrix.new(1, cols)

          # Write F32 values to host buffer, then sync to device
          cols.times do |i|
            mat.raw_data[i] = arr[i].as(GenNum).to_f32
          end
          mat.mark_host_modified!
          mat.sync_to_device!("to_matrix")

          mat
        end
      else
        # CPU-only fallback uses SimpleMatrix as before
        mat = if arr.size > 0 && arr[0].is_a?(Array)
                rows = arr.size
                cols = arr[0].as(Array).size
                SimpleMatrix.new(rows, cols).tap do |m|
                  rows.times do |i|
                    cols.times do |j|
                      m[i, j] = arr[i].as(Array)[j].as(GenNum).to_f64
                    end
                  end
                end
              else
                SimpleMatrix.new(1, arr.size).tap do |m|
                  arr.size.times do |i|
                    m[0, i] = arr[i].as(GenNum).to_f64
                  end
                end
              end
        mat
      end
    end

    def current_learning_rate
      if @warmup_steps > 0 && @time_step < @warmup_steps
        @learning_rate * (@time_step.to_f64 / @warmup_steps)
      else
        @learning_rate
      end
    end

    def update_transformer_layers
      lr = current_learning_rate
      @transformer_layers.each do |layer|
        case layer
        when TransformerLayer then layer.apply_gradients(lr)
        when LlamaLayer       then layer.apply_gradients(lr)
        end
      end
    end

    def validate_values(array : Array(Float64), location : String)
      # Detect NaNs in output
      array.each do |ar|
        raise NeuralNetRunError.new(
          "Found a NaN value, run stopped.\n#{location}: #{array}") if ar.nan?
      end
    end

    def get_cost_proc(function_name : String) : CostFunction
      case function_name
      when "mse"
        SHAInet.quadratic_cost
      when "c_ent"
        # raise MathError.new("Cross entropy cost is not implemented fully yet, please use quadratic cost for now.")
        SHAInet.cross_entropy_cost
      when "c_ent_sm"
        SHAInet.cross_entropy_cost
      else
        raise NeuralNetInitalizationError.new("Must choose correct cost function or provide a correct Proc")
      end
    end

    # Evaluate the network performance on a test set
    def test(test_set)
      correct = 0
      incorrect = 0
      test_set.normalized_inputs.each_with_index do |input, index|
        output_array = run(input: input, stealth: true)
        if test_set.label_for_array(output_array) == test_set.label_for_array(test_set.normalized_outputs[index])
          correct += 1
        else
          incorrect += 1
        end
      end
      Log.info { "Predicted #{correct} out of #{correct + incorrect} (#{(correct.to_f/(correct + incorrect).to_f)*100}% accuracy)" }
      correct.to_f/(correct + incorrect).to_f
    end

    # GPU path - all CudaMatrix operations
    private def safe_output_transform(matrix : CudaMatrix, weights : CudaMatrix) : CudaMatrix
      # For transformer architectures, use only the last token's representation
      if !@transformer_layers.empty?
        # Extract last token (row) from transformer output for language modeling using GPU kernel
        last_token = if CUDA.fully_available? && (mptr = matrix.device_ptr) && (wptr = weights.device_ptr) && !mptr.null? && !wptr.null?
                       begin
                         # Use more efficient GPU slice operation for last row
                         result = CudaMatrix.new(1, matrix.cols)
                         # Copy last row using GPU memory copy
                         last_row_offset = (matrix.rows - 1) * matrix.cols
                         CUDA.copy_device_to_device(
                           result.device_ptr.not_nil!,
                           mptr + last_row_offset,
                           (matrix.cols * 4).to_u64
                         )
                         result.mark_device_dirty!
                         result
                       rescue
                         # Fallback to elementwise copy if GPU operation fails
                         last_token_fallback = CudaMatrix.new(1, matrix.cols)
                         matrix.cols.times do |j|
                           last_token_fallback[0, j] = matrix[matrix.rows - 1, j]
                         end
                         last_token_fallback.sync_to_device!
                         last_token_fallback
                       end
                     else
                       # CPU fallback
                       last_token_cpu = CudaMatrix.new(1, matrix.cols)
                       matrix.cols.times do |j|
                         last_token_cpu[0, j] = matrix[matrix.rows - 1, j]
                       end
                       last_token_cpu.sync_to_device!
                       last_token_cpu
                     end

        # Now multiply: last_token (1 x d_model) * weights (d_model x vocab_size)
        if last_token.cols != weights.rows
          raise ArgumentError.new("Transformer output dimension mismatch: d_model (#{last_token.cols}) doesn't match weights input size (#{weights.rows})")
        end

        return last_token * weights
      end

      # For matrix * weights, we need matrix.cols == weights.rows
      if matrix.cols != weights.rows
        raise ArgumentError.new("Matrix dimension mismatch: input features (#{matrix.cols}) doesn't match weights input size (#{weights.rows})")
      end

      matrix * weights
    rescue ex : Exception
      # Check for dimension issues and try to reshape if possible
      if ex.message.to_s.includes?("size mismatch") || ex.message.to_s.includes?("dimension mismatch")
        # Try reshaping for a single token/sequence case
        if matrix.rows == 1 && matrix.cols > 0 && weights.rows > 0 && weights.cols > 0
          Log.info { "Reshaping matrix for single-token transformer operation" }
          reshaped = CudaMatrix.new(1, weights.cols)
          weights.cols.times do |j|
            sum = 0.0
            matrix.cols.times do |k|
              sum += matrix[0, k] * weights[j, k]
            end
            reshaped[0, j] = sum
          end
          reshaped.sync_to_device!
          return reshaped
        end
      end

      raise ex
    end

    # CPU path - all SimpleMatrix operations
    private def safe_output_transform(matrix : SimpleMatrix, weights : SimpleMatrix) : SimpleMatrix
      # For transformer architectures, use only the last token's representation
      if !@transformer_layers.empty?
        # Extract last token (row) from transformer output for language modeling
        last_token = SimpleMatrix.new(1, matrix.cols)
        matrix.cols.times do |j|
          last_token[0, j] = matrix[matrix.rows - 1, j]
        end

        # Now multiply: last_token (1 x d_model) * weights (d_model x vocab_size)
        if last_token.cols != weights.rows
          raise ArgumentError.new("Transformer output dimension mismatch: d_model (#{last_token.cols}) doesn't match weights input size (#{weights.rows})")
        end

        return last_token * weights
      end

      # For matrix * weights, we need matrix.cols == weights.rows
      if matrix.cols != weights.rows
        raise ArgumentError.new("Matrix dimension mismatch: input features (#{matrix.cols}) doesn't match weights input size (#{weights.rows})")
      end

      matrix * weights
    rescue ex : Exception
      # Check for dimension issues and try to reshape if possible
      if ex.message.to_s.includes?("size mismatch") || ex.message.to_s.includes?("dimension mismatch")
        # Try reshaping for a single token/sequence case
        if matrix.rows == 1 && matrix.cols > 0 && weights.rows > 0 && weights.cols > 0
          Log.info { "Reshaping matrix for single-token transformer operation" }
          reshaped = SimpleMatrix.new(1, weights.cols)
          weights.cols.times do |j|
            sum = 0.0
            matrix.cols.times do |k|
              sum += matrix[0, k] * weights[j, k]
            end
            reshaped[0, j] = sum
          end
          return reshaped
        end
      end

      raise ex
    end

    # GPU-accelerated lm_head projection: [1, d_model] × [d_model, vocab_size]
    # Uses direct cuBLAS GEMM to avoid the 2s CPU matmul for 128K vocab.
    private def gpu_lm_head(matrix : SimpleMatrix, weights : CudaMatrix) : SimpleMatrix
      # Extract last token for transformer architectures
      last = if !@transformer_layers.empty? && matrix.rows > 1
               sm = SimpleMatrix.new(1, matrix.cols)
               matrix.cols.times { |j| sm[0, j] = matrix[matrix.rows - 1, j] }
               sm
             else
               matrix.rows > 1 ? (sm = SimpleMatrix.new(1, matrix.cols); matrix.cols.times { |j| sm[0, j] = matrix[matrix.rows - 1, j] }; sm) : matrix
             end
      # Upload input, run GEMM, download result
      x_gpu = CudaMatrix.new(last.rows, last.cols)
      x_gpu.raw_data.to_unsafe.copy_from(last.data.to_unsafe, last.rows * last.cols)
      x_gpu.sync_to_device!("lm_head_in")
      r_gpu = CudaMatrix.new(last.rows, weights.cols)
      h = CUDA.create_handle
      CUDA.gemm(h, weights.device_ptr.not_nil!, x_gpu.device_ptr.not_nil!, r_gpu.device_ptr.not_nil!,
        weights.cols, last.rows, weights.rows, weights.cols, last.cols, weights.cols)
      CUDA.destroy_handle(h)
      CUDA.device_synchronize
      r_gpu.mark_device_dirty!
      r_gpu.sync_from_device!("lm_head_out")
      result = SimpleMatrix.new(r_gpu.rows, r_gpu.cols)
      result.data.to_unsafe.copy_from(r_gpu.raw_data.to_unsafe, r_gpu.rows * r_gpu.cols)
      result
    end

    # Quantized lm_head projection: [1, d_model] × dequant(W[d_model, vocab]).
    # Uses the dequant-in-kernel GEMV of the given quantized weight (Q8 or Q4).
    # Returns [1, vocab] logits.
    @lm_head_b : CudaMatrix? = nil

    # `bias` is added ON THE DEVICE before the readback when supplied, so the caller
    # can skip its host loop. That loop walked the full vocab row (151936 wide on
    # Qwen3) once per token, which the profiler showed at 0.25 ms/step.
    private def gpu_lm_head_q(matrix : SimpleMatrix, weights : QuantizedWeight,
                              bias : SimpleMatrix? = nil) : SimpleMatrix
      # Extract last token for transformer architectures (language modeling).
      last = if matrix.rows > 1
               sm = SimpleMatrix.new(1, matrix.cols)
               matrix.cols.times { |j| sm[0, j] = matrix[matrix.rows - 1, j] }
               sm
             else
               matrix
             end

      x_gpu = (@lm_head_x ||= CudaMatrix.new(last.rows, last.cols))
      x_gpu.raw_data.to_unsafe.copy_from(last.data.to_unsafe, last.rows * last.cols)
      x_gpu.mark_host_modified!
      x_gpu.sync_to_device!("lm_head_q_in")

      r_gpu = (@lm_head_r ||= CudaMatrix.new(last.rows, weights.cols))
      weights.gemv_into(x_gpu, r_gpu)

      if b = bias
        bd = @lm_head_b
        unless bd
          bd = CudaMatrix.new(1, b.cols)
          bd.raw_data.to_unsafe.copy_from(b.data.to_unsafe, b.cols)
          bd.mark_host_modified!
          bd.sync_to_device!("lm_head_bias")
          @lm_head_b = bd
        end
        CUDA.add_inplace(r_gpu.device_ptr.not_nil!, bd.device_ptr.not_nil!, r_gpu.cols)
        r_gpu.mark_device_dirty!
      end

      r_gpu.sync_from_device!("lm_head_q_out") if r_gpu.device_dirty?

      result = SimpleMatrix.new(r_gpu.rows, r_gpu.cols)
      result.data.to_unsafe.copy_from(r_gpu.raw_data.to_unsafe, r_gpu.rows * r_gpu.cols)
      result
    end

    # Optimized helper to extract tokens from GPU matrix without elementwise access

    # Uses GPU-to-CPU batch transfer instead of per-element sync
    private def extract_tokens_gpu(matrix : CudaMatrix) : Array(Int32)
      # Sync entire matrix from GPU in one operation instead of elementwise access
      matrix.sync_from_device!("extract_tokens") if matrix.device_dirty?
      # Extract tokens as a batch operation from column 0
      Array.new(matrix.rows) { |r| matrix.unsafe_get(r, 0).to_i }
    end

    # Optimized matrix creation from arrays using batch operations
    private def create_matrix_from_arrays(data : Array(Array(Float64)), use_gpu : Bool = true) : SimpleMatrix | CudaMatrix
      return SimpleMatrix.from_a(data) unless use_gpu && CUDA.fully_available?

      # Create GPU matrix directly from array data in batch
      rows = data.size
      cols = data[0].size
      result = CudaMatrix.new(rows, cols)

      # Copy data in batch instead of elementwise
      flat_data = data.flatten
      result.raw_data.copy_from(flat_data.to_unsafe, flat_data.size)
      result.sync_to_device!
      result
    end

    # Optimized matrix population from single array using batch operations
    private def populate_matrix_batch(matrix : CudaMatrix | SimpleMatrix, data : Array(Float64), row : Int32)
      if matrix.is_a?(CudaMatrix)
        # Use raw data access for batch copy instead of elementwise assignment
        start_idx = row * matrix.cols
        matrix.raw_data.to_slice[start_idx, data.size].copy_from(data.to_unsafe, data.size)
        matrix.sync_to_device!
      else
        # For SimpleMatrix, still do elementwise but at least batch the operation
        data.each_with_index { |val, col| matrix[row, col] = val }
      end
    end

    # Try to apply activation function using GPU kernels
    private def try_gpu_activation(matrix : CudaMatrix, activation_function : ActivationFunction) : Bool
      return false unless CUDA.fully_available?

      case activation_function
      when SHAInet.sigmoid
        # Use in-place GPU sigmoid operation
        begin
          matrix.sigmoid!
          return true
        rescue ex
          Log.debug { "GPU sigmoid failed: #{ex}, falling back to CPU" }
        end
      when SHAInet.relu
        # Use in-place ReLU operation
        begin
          matrix.relu!
          return true
        rescue ex
          Log.debug { "GPU ReLU failed: #{ex}, falling back to CPU" }
        end
      end

      false
    end

    # Helper method for matrix slicing (missing method)
    private def slice_rows_helper(matrix : CudaMatrix, start_row : Int32, num_rows : Int32) : CudaMatrix
      result = CudaMatrix.new(num_rows, matrix.cols)
      num_rows.times do |i|
        matrix.cols.times do |j|
          result[i, j] = matrix[start_row + i, j]
        end
      end
      result.sync_to_device! if CUDA.fully_available?
      result
    end

    # CPU fallback for cost and gradient computation when GPU acceleration fails
    private def compute_cost_and_gradient_cpu(actual_matrix, expected_output, grad_matrix, cost_proc)
      sample_error = 0.0

      if actual_matrix.is_a?(CudaMatrix)
        actual_matrix.as(CudaMatrix).sync_from_device!("cost_grad_cpu")
      end

      if expected_output.is_a?(CudaMatrix)
        expected_output.as(CudaMatrix).sync_from_device!("cost_grad_cpu")
      end

      if expected_output.is_a?(SimpleMatrix)
        exp_mat = expected_output.as(SimpleMatrix)
        exp_mat.rows.times do |i|
          exp_mat.cols.times do |j|
            expected = exp_mat[i, j]
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      elsif expected_output.is_a?(CudaMatrix)
        exp_mat = expected_output.as(CudaMatrix)
        exp_mat.rows.times do |i|
          exp_mat.cols.times do |j|
            expected = exp_mat.unsafe_get(i, j)
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      elsif expected_output.is_a?(Array) && expected_output.as(Array).size > 0 && expected_output.as(Array)[0].is_a?(Array)
        rows = expected_output.as(Array).size
        cols = expected_output.as(Array)[0].as(Array).size
        rows.times do |i|
          cols.times do |j|
            expected = expected_output.as(Array)[i].as(Array)[j].as(GenNum).to_f64
            actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(i, j) : actual_matrix.as(SimpleMatrix)[i, j]
            cost_result = cost_proc.call(expected, actual)
            sample_error += cost_result[:value]
            if grad_matrix.is_a?(CudaMatrix)
              grad_matrix.as(CudaMatrix).unsafe_set(i, j, cost_result[:derivative])
            else
              grad_matrix.as(SimpleMatrix)[i, j] = cost_result[:derivative]
            end
          end
        end
      else
        arr = expected_output.as(Array)
        arr.size.times do |i|
          expected = arr[i].as(GenNum).to_f64
          actual = actual_matrix.is_a?(CudaMatrix) ? actual_matrix.as(CudaMatrix).unsafe_get(0, i) : actual_matrix.as(SimpleMatrix)[0, i]
          cost_result = cost_proc.call(expected, actual)
          sample_error += cost_result[:value]
          if grad_matrix.is_a?(CudaMatrix)
            grad_matrix.as(CudaMatrix).unsafe_set(0, i, cost_result[:derivative])
          else
            grad_matrix.as(SimpleMatrix)[0, i] = cost_result[:derivative]
          end
        end
      end

      if grad_matrix.is_a?(CudaMatrix)
        grad_matrix.as(CudaMatrix).mark_host_modified!
        grad_matrix.as(CudaMatrix).sync_to_device!("cost_grad_cpu")
      end

      sample_error
    end
  end
end
