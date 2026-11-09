require "log"
require "json"
require "../math/simple_matrix"
require "./matrix_layer"

module SHAInet
  class Network
    # Notes:

    #
    # This file contains all the methods for creating and maintaining
    # the network, for methods regarding running and training go to network_run.cr
    # ------------

    Log = ::Log.for(self)

    LAYER_TYPES      = ["input", "hidden", "recurrent", "output"]
    CONNECTION_TYPES = ["full", "ind_to_ind", "random"]
    COST_FUNCTIONS   = ["mse", "c_ent", "c_ent_sm"] # , "exp", "hel_d", "kld", "gkld", "ita_sai_d"]

    # General network parameters
    getter :input_layers, :output_layers, :hidden_layers, :recurrent_layers, :lstm_layers
    getter :transformer_layers
    getter transformer_error : SimpleMatrix
    property final_norm : RMSNorm?
    getter error_signal : Array(Float64), total_error : Float64, :mse, w_gradient : Array(Float64), b_gradient : Array(Float64)

    # Parameters for SGD + Momentum
    property learning_rate : Float64, momentum : Float64

    # KV cache mode for autoregressive LLM inference
    property? use_kv_cache : Bool = false

    # Per-layer progress callback for prefill: called with (layer_idx, total_layers)
    property prefill_progress : Proc(Int32, Int32, Nil)?

    # Status flag set to true once quantize! has converted weights to Q8.
    # Informational only — the quantized lm_head dispatch in Network#run keys
    # off @lm_head_q (and block/FFN weights off their own QuantizedCudaMatrix
    # type), not off this flag.
    property? quantize_weights : Bool = false

    # Quantized lm_head weight (populated by quantize! when CUDA is available).
    # Readable so callers and specs can inspect its residency (a Q4HostMatrix
    # reports device_bytes 0; a Q4CudaMatrix reports its real VRAM footprint).
    property lm_head_q : QuantizedWeight?
    # Persistent decode buffers for the quantized lm_head GEMV (reused per token).
    @lm_head_x : CudaMatrix?
    @lm_head_r : CudaMatrix?

    # Parameters for Rprop
    property etah_plus : Float64, etah_minus : Float64, delta_max : Float64, delta_min : Float64
    getter prev_mse : Float64

    # Parameters for Adam
    property alpha : Float64
    getter beta1 : Float64, beta2 : Float64, epsilon : Float64, time_step : Int32
    property clip_threshold : Float64
    property warmup_steps : Int32
    property weight_decay : Float64
    property accumulation_steps : Int32
    property? mixed_precision : Bool

    @cached_expanded_grad : SimpleMatrix | CudaMatrix?

    # First creates an empty shell of the entire network
    def initialize
      @input_layers = Array(MatrixLayer).new
      @output_layers = Array(MatrixLayer).new
      @hidden_layers = Array(MatrixLayer).new
      @transformer_layers = Array(TransformerLayer | LlamaLayer).new
      @all_layers = Array(MatrixLayer).new
      @error_signal = Array(Float64).new # Array of errors for each element in the output layers
      @total_error = 1_f64               # Sum of errors from output layer, based on a specific input
      @mse = 1_f64                       # MSE of network, based on all errors of output layer for a specific input or batch
      @w_gradient = Array(Float64).new   # Needed for batch train
      @b_gradient = Array(Float64).new   # Needed for batch train

      @learning_rate = 0.005_f64 # Standard parameter for GD
      @momentum = 0.05_f64       # Improved GD

      @etah_plus = 1.2_f64  # For iRprop+ , how to increase step size
      @etah_minus = 0.5_f64 # For iRprop+ , how to decrease step size
      @delta_max = 50_f64   # For iRprop+ , max step size
      @delta_min = 0.1_f64  # For iRprop+ , min step size
      @prev_mse = 1_f64     # For iRprop+ , needed for backtracking

      @alpha = 0.001_f64   # For Adam , step size (recomeneded: only change this hyper parameter when fine-tuning)
      @beta1 = 0.9_f64     # For Adam , exponential decay rate (not recommended to change value)
      @beta2 = 0.999_f64   # For Adam , exponential decay rate (not recommended to change value)
      @epsilon = 10e-8_f64 # For Adam , prevents exploding gradients (not recommended to change value)
      @time_step = 0_i32   # For Adam
      @transformer_error = SimpleMatrix.zeros(1, 1)
      @clip_threshold = Float64::INFINITY
      @warmup_steps = 0
      @weight_decay = 0.0
      @accumulation_steps = 1
      @mixed_precision = false

      # Gradient transformation caching for efficient transformer backward pass
      @cached_seq_len = 0
      @cached_d_model = 0
    end

    # Create and populate a layer
    # l_type is: :input, :hidden or :output
    # l_size = size of the layer
    # n_type = advanced option for layer types
    def add_layer(l_type : Symbol | String, l_size : Int32, activation_function : ActivationFunction = SHAInet.sigmoid, num_heads : Int32 = 1, ff_hidden : Int32 = l_size*4, drop_percent : Int32 = 0, blocks : Int32 = 1, *, vocab_size : Int32 = 0, num_kv_heads : Int32 = num_heads, eps : Float64 = 1e-5, head_dim : Int32? = nil, moe_experts : Int32? = nil, moe_top_k : Int32 = 8, moe_norm_topk : Bool = true, moe_ff_hidden : Int32? = nil, moe_offload : Bool = false, linear_conv_kernel : Int32 = 4, allocate : Bool = true)
      if l_type.to_s == "transformer" && blocks > 1
        blocks.times do
          add_layer(l_type, l_size, activation_function, num_heads, ff_hidden, drop_percent, 1)
        end
        return
      end
      layer = case l_type.to_s
              when "embedding"
                raise NeuralNetRunError.new("vocab_size required for embedding layer") if vocab_size <= 0
                EmbeddingLayer.new(vocab_size, l_size, activation_function)
              when "transformer"
                TransformerLayer.new(l_size, num_heads, ff_hidden, drop_percent)
              when "llama"
                LlamaBlock.new(l_size, num_heads, ff_hidden, eps, 10000.0, num_kv_heads, head_dim,
                  moe_experts, moe_top_k, moe_norm_topk, moe_ff_hidden, moe_offload, allocate: allocate)
              when "gated_deltanet"
                # The linear-attention layer type of a hybrid qwen3_5 stack. Interchangeable
                # with "llama" in a stack, so a caller walking a layer_types list just picks
                # one name or the other per layer.
                #
                # num_heads carries the VALUE head count and num_kv_heads the KEY head count,
                # mirroring how attention uses them, so the existing keyword arguments describe
                # a linear layer without inventing parallel ones. head_dim, when given, sets
                # both key and value head dimensions -- Qwen3.5 uses 128 for each.
                hd = head_dim || 128
                GatedDeltaNetBlock.new(l_size, ff_hidden, num_heads, num_kv_heads, hd, hd,
                  linear_conv_kernel, eps, allocate: allocate)
              else
                MatrixLayer.new(l_size, activation_function)
              end

      # Add layer to appropriate collections
      case l_type.to_s
      when "input"
        @input_layers << layer
      when "hidden"
        @hidden_layers << layer
      when "embedding"
        @hidden_layers << layer
      when "transformer"
        @hidden_layers << layer
        @transformer_layers << layer.as(TransformerLayer)
      when "llama"
        @hidden_layers << layer
        @transformer_layers << layer.as(LlamaLayer)
      when "gated_deltanet"
        # NOT added to @transformer_layers, deliberately.
        #
        # That collection is typed Array(TransformerLayer | LlamaLayer) and is read from 24
        # sites, several on the hot inference path, so widening the union is a change that
        # deserves its own audit rather than a drive-by.
        #
        # Network#clear_cache! therefore reaches these blocks through @hidden_layers instead, which
        # it MUST: an earlier version of this comment argued the gap was harmless because "the loader
        # still refuses to build one of these from a checkpoint". GGUF support for Qwen3.5/3.8 made
        # that false -- every one of those models is a hybrid stack -- and the gap turned into a
        # second prompt in the same process producing garbage.
        @hidden_layers << layer
      when "output"
        if @output_layers.empty?
          @output_layers << layer
        else
          @output_layers.delete(@output_layers.first)
          @output_layers << layer
          connect_ltl(@hidden_layers.last, @output_layers.first, :full)
        end
      else
        raise NeuralNetRunError.new("Must define correct layer type (:input, :hidden, :embedding, :transformer, :output).")
      end

      # Add to all_layers collection
      @all_layers << layer
    end

    # Clear KV caches on all LLaMA blocks (call between conversations)
    def clear_cache!
      @transformer_layers.each do |l|
        l.as(LlamaBlock).clear_cache! if l.is_a?(LlamaBlock)
      end
      # Gated DeltaNet blocks are reached through @hidden_layers, because they are deliberately not
      # in @transformer_layers (see add_layer for why the union is not widened here).
      #
      # They MUST be cleared. A GDN block carries recurrent state rather than a KV cache, so a stale
      # state is not merely a stale prefix -- it feeds the first token of the next sequence and
      # corrupts everything after it. In a hybrid stack that is 48 of 64 layers.
      #
      # This was a live bug, not a theoretical one. add_layer's comment argued the gap was harmless
      # because "the loader still refuses to build one of these from a checkpoint", which stopped
      # being true once GGUF loading gained Qwen3.5/3.8 support: every one of those models is a hybrid
      # stack. Observed effect -- ask the model to write an HTML page, then ask it to review a file in
      # the same process with clear_cache! between: the first answer is correct and the second decodes
      # an immediate end-of-turn followed by empty im_start/im_end pairs forever. The same review
      # prompt in a fresh process answers correctly, which is what identified leftover state rather
      # than the prompt.
      @hidden_layers.each do |l|
        l.as(GatedDeltaNetBlock).clear_cache! if l.is_a?(GatedDeltaNetBlock)
      end
    end

    # Position the cache was saved at by save_cache!, or nil when there is no saved point.
    @saved_cache_positions : Int32? = nil

    # Save a point the cache can be rewound to: `positions` tokens, which must be every token the
    # cache holds right now (call it straight after a prefill, before generating).
    #
    # Attention blocks need nothing saved, since their KV cache can be cut back to a length. Gated
    # DeltaNet blocks fold every token into their recurrent state, so each one keeps a copy (in host
    # memory). One saved point at a time; saving again replaces it.
    def save_cache!(positions : Int32)
      llama_cache_blocks.each do |b|
        unless b.cached_length == positions
          drop_saved_cache!
          raise ArgumentError.new("save_cache!(#{positions}) but an attention block holds #{b.cached_length} positions")
        end
      end
      CUDA.device_synchronize if CUDA.fully_available?
      gdn_cache_blocks.each(&.save_state!)
      @saved_cache_positions = positions
    end

    # Rewind the cache to the saved point and return its position, or return nil (changing nothing)
    # when there is no saved point or an attention block no longer reaches it -- the caller then
    # clears the cache and prefills from scratch.
    def restore_cache! : Int32?
      pos = @saved_cache_positions
      return unless pos
      blocks = llama_cache_blocks
      return if blocks.any? { |b| b.cached_length < pos }
      blocks.each(&.truncate_cache!(pos))
      gdn_cache_blocks.each(&.restore_saved_state!)
      pos
    end

    def drop_saved_cache!
      @saved_cache_positions = nil
      gdn_cache_blocks.each(&.drop_saved_state!)
    end

    private def llama_cache_blocks : Array(LlamaBlock)
      @transformer_layers.compact_map { |l| l.as(LlamaBlock) if l.is_a?(LlamaBlock) }
    end

    private def gdn_cache_blocks : Array(GatedDeltaNetBlock)
      @hidden_layers.compact_map { |l| l.as(GatedDeltaNetBlock) if l.is_a?(GatedDeltaNetBlock) }
    end

    # Quantize all LLaMA block weights and the lm_head on the GPU to the given
    # bit width: bits == 8 -> Q8_0 (int8), bits == 4 -> Q4_0 (4-bit), both with
    # per-32-block fp32 scales. Opt-in; the fp32 path is untouched. Requires CUDA
    # + custom kernels. After this, run with use_kv_cache = true.
    #
    # When offload is true the dense weights (attention projections, dense FFN,
    # lm_head) are kept as packed Q4 in host RAM and streamed to the GPU on
    # demand, the same mechanism MoE experts already use. This is a CAPACITY
    # trade: it takes the dense weights off the VRAM budget so a much larger
    # dense model fits, at the cost of PCIe traffic on every token (dense weights
    # have no expert sparsity to amortize the transfer).
    def quantize!(bits : Int32 = 8, offload : Bool = false)
      raise ArgumentError.new("unsupported quantization bits: #{bits} (expected 8 or 4)") unless bits == 8 || bits == 4
      raise ArgumentError.new("dense offload currently supports 4-bit only (got #{bits}-bit)") if offload && bits != 4
      unless CUDA.fully_available?
        Log.warn { "quantize!: CUDA kernels unavailable, leaving weights in fp32" }
        return self
      end

      @quantize_weights = true
      @transformer_layers.each do |l|
        l.as(LlamaBlock).to_gpu!(quantize: true, bits: bits, offload: offload) if l.is_a?(LlamaBlock)
      end
      # gated_deltanet blocks are deliberately NOT in @transformer_layers (see add_layer), so
      # walking only that array would leave a hybrid stack's linear-attention layers in fp32 --
      # 24 of 32 on Qwen3.5-9B, which is most of the model.
      @hidden_layers.each do |l|
        l.as(GatedDeltaNetBlock).to_gpu!(quantize: true, bits: bits, offload: offload) if l.is_a?(GatedDeltaNetBlock)
      end

      # Quantize the output projection (lm_head). Stored separately so the
      # generic MatrixLayer weight union stays fp32/CudaMatrix only. On a large
      # vocabulary this is one of the biggest single dense weights (Qwen3's
      # 151936-wide head is ~1.2 GB in fp32), so it is the highest-value single
      # tensor to move off the device.
      out_layer = @output_layers.last?
      if out_layer
        w = out_layer.weights
        sm = w.is_a?(SimpleMatrix) ? w : (w.is_a?(CudaMatrix) ? w.to_simple : nil)
        if sm
          @lm_head_q = if offload
                         Q4HostMatrix.from_simple(sm)
                       else
                         bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm)
                       end
        end
      end

      self
    end

    # Connect all the layers in order (input and output don't connect between themselves): input, hidden, output
    def fully_connect
      if @hidden_layers.empty?
        # Connect all input layers to all output layers
        @output_layers.each do |out_layer|
          @input_layers.each do |in_layer|
            connect_ltl(in_layer, out_layer, :full)
          end
        end
      else
        # Connect all input layers to the first hidden layer
        @input_layers.each do |in_layer|
          connect_ltl(in_layer, @hidden_layers.first, :full)
        end

        # Connect all hidden layer between each other hierarchically
        (@hidden_layers.size).times do |l|
          next if (l + 1) == @hidden_layers.size
          connect_ltl(@hidden_layers[l], @hidden_layers[l + 1], :full)
        end

        # Connect last hidden layer to all output layers
        @output_layers.each do |out_layer|
          connect_ltl(@hidden_layers.last, out_layer, :full)
        end
      end
    rescue ex : Exception
      raise NeuralNetRunError.new("Error fully connecting network: #{ex}")
    end

    # Connect two specific layers
    def connect_ltl(src_layer : MatrixLayer, dest_layer : MatrixLayer, connection_type : Symbol | String)
      raise NeuralNetInitalizationError.new("Error initilizing network, must choose correct connection type.") if CONNECTION_TYPES.any? { |x| x == connection_type.to_s } == false
      case connection_type.to_s
      # Connect source layer to destination layer with full connections
      when "full"
        # Matrix-based layers handle weight initialization internally
        # Use SimpleMatrix for now to avoid CUDA type mismatches during debugging
        mat_klass = SimpleMatrix
        if src_layer.is_a?(TransformerLayer)
          # For transformer output, weights need to be (d_model, vocab_size) for correct matrix multiplication
          # (batch_size x d_model) * (d_model x vocab_size) = (batch_size x vocab_size)
          dest_layer.weights = mat_klass.new(src_layer.size, dest_layer.size).random_fill!
          dest_layer.biases = mat_klass.new(1, dest_layer.size).random_fill!
          # Gradients are NOT allocated here. They are materialized off the weights
          # by the first backward pass, so inference never pays for them: on the 30B
          # this projection is 2048x151936, whose g_w alone is 1.245 GB of host RAM.
        elsif dest_layer.is_a?(MatrixLayer)
          # For MatrixLayer, reinitialize with correct dimensions
          dest_layer.weights = mat_klass.new(src_layer.size, dest_layer.size).random_fill!
          dest_layer.biases = mat_klass.new(1, dest_layer.size).random_fill!
        else
          # Initialize weights randomly for all layer types
          dest_layer.weights = mat_klass.new(dest_layer.size, src_layer.size).random_fill!
          dest_layer.biases = mat_klass.new(dest_layer.size, 1).random_fill!
        end
      end
    rescue ex : Exception
      raise NeuralNetRunError.new("Error in connect_ltl: #{ex}")
    end

    def log_summary(e)
      Log.info { "Epoch: #{e}, Total error: #{@total_error}, MSE: #{@mse}" }
    end

    def reset_recurrent_state
      @recurrent_layers.each(&.reset_state)
      @lstm_layers.each(&.reset_state)
    end

    def clean_dead_neurons
      # Matrix-based layers don't require cleanup
      # This method is kept for API compatibility
      Log.info { "Matrix-based layers don't require cleanup" }
    end

    def verify_net_before_train
      if @input_layers.empty?
        raise NeuralNetRunError.new("No input layers defined")
        # elsif @hidden_layers.empty?
        #   raise NeuralNetRunError.new("Need atleast one hidden layer")
      elsif @output_layers.empty?
        raise NeuralNetRunError.new("No output layers defined")
      end
    end

    def randomize_all_weights
      # Matrix-based layers handle weight initialization during layer creation
      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        if layer.weights.is_a?(SimpleMatrix)
          layer.weights.random_fill!
        end
      end
    end

    def randomize_all_biases
      # Matrix-based layers handle bias initialization during layer creation
      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        if layer.biases.is_a?(SimpleMatrix)
          layer.biases.random_fill!
        end
      end
    end

    def save_to_file(file_path : String)
      Log.info { "Saving network to: #{file_path}" }
      File.open(file_path, "w") do |io|
        save_to_io(io)
      end
    end

    def save_as_string : String
      Log.info { "Saving network as string" }
      io = IO::Memory.new
      save_to_io(io)
      io.to_s
    end

    def save_to_io(io : IO)
      dump_network = [] of Hash(String, JSON::Any)

      [@input_layers, @hidden_layers, @output_layers].flatten.each do |layer|
        dump_layer = Hash(String, JSON::Any).new
        l_type = if @input_layers.includes?(layer)
                   "input"
                 elsif @hidden_layers.includes?(layer)
                   "hidden"
                 else
                   "output"
                 end

        dump_layer["l_type"] = JSON::Any.new(l_type)
        if layer.weights
          dump_layer["weights"] = JSON.parse(layer.weights.not_nil!.to_a.to_json)
          dump_layer["biases"] = JSON.parse(layer.biases.not_nil!.to_a.flatten.to_json)
        end
        dump_layer["activation_function"] = JSON::Any.new(layer.activation_function.to_s)
        dump_network << dump_layer
      end

      {"layers" => dump_network}.to_json(io)
      Log.info { "Network saved" }
    end

    def load_from_file(file_path : String)
      Log.info { "Loading network from: #{file_path}" }
      File.open(file_path) do |io|
        load_from_io(io)
      end
    end

    def load_from_string(data : String)
      Log.info { "Loading network from string (#{data.size} bytes)" }
      load_from_io(IO::Memory.new(data))
    end

    def load_from_io(io : IO)
      data = JSON.parse(io.gets_to_end)
      layers = data["layers"].as_a

      layers.each do |layer_data|
        l_type = layer_data["l_type"].as_s
        size = layer_data["biases"].as_a.size
        case l_type
        when "input"
          add_layer(:input, size)
        when "output"
          add_layer(:output, size)
        else
          add_layer(:hidden, size)
        end
      end

      fully_connect

      all_layers = @hidden_layers + @output_layers
      layers.each_with_index do |layer_data, idx|
        next if idx == 0 # input layer has no weights to set
        dest_layer = all_layers[idx - 1]
        w = layer_data["weights"].as_a.map(&.as_a.map(&.as_f))
        b = layer_data["biases"].as_a.map(&.as_f)
        dest_layer.weights = SimpleMatrix.from_a(w)
        dest_layer.biases = SimpleMatrix.from_a([b])
      end
      Log.info { "Network loaded" }
    end

    # Load a model from a HuggingFace SafeTensors directory.
    # The directory must contain `model.safetensors` and `config.json`.
    def self.load_from_hf(model_dir : String) : Network
      HFLoader.load(model_dir)
    end

    def inspect
      Log.info { @input_layers }
      Log.info { "--------------------------------" }
      Log.info { @hidden_layers }
      Log.info { "--------------------------------" }
      Log.info { @output_layers }
      Log.info { "--------------------------------" }
    end

    # Dummy layers property for compatibility with the matrix-based Network class
    getter layers : Array(MatrixLayer) { [] of MatrixLayer }
  end
end
