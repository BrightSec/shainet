module SHAInet
  # SwiGLU Feed-Forward Network as used in LLaMA/Mistral.
  # Formula: output = down_proj(silu(gate_proj(x)) * up_proj(x))
  class SwiGLUFF
    property gate_proj : SimpleMatrix | CudaMatrix | QuantizedWeight
    property up_proj : SimpleMatrix | CudaMatrix | QuantizedWeight
    property down_proj : SimpleMatrix | CudaMatrix | QuantizedWeight

    # allocate: when false, weights start as empty 0x0 placeholders instead of
    # full [d_model, ff_hidden] matrices. Used for MoE experts, which are always
    # overwritten by the loader — avoids allocating hundreds of fp32 expert
    # matrices up front (a 48-layer/128-expert model would otherwise need ~100GB
    # before streaming quantization ever runs).
    def initialize(d_model : Int32, ff_hidden : Int32, allocate : Bool = true)
      if allocate
        @gate_proj = SimpleMatrix.new(d_model, ff_hidden)
        @up_proj = SimpleMatrix.new(d_model, ff_hidden)
        @down_proj = SimpleMatrix.new(ff_hidden, d_model)
      else
        @gate_proj = SimpleMatrix.new(0, 0)
        @up_proj = SimpleMatrix.new(0, 0)
        @down_proj = SimpleMatrix.new(0, 0)
      end
    end

    # Persistent single-row GEMV workspaces for decode (M=1), keyed by width.
    @q8_in_bufs = Hash(Int32, CudaMatrix).new
    @q8_out_bufs = Hash(Int32, CudaMatrix).new

    # Device-resident workspaces for the decode FFN chain, keyed by width.
    #
    # CLASS level, not instance level, on purpose: a MoE layer holds up to 128
    # experts and a model up to 48 layers, and only one expert is ever mid-flight,
    # so per-instance buffers would burn hundreds of MB of VRAM to hold identical
    # shapes. gate and up are live at the same time as hidden, so they need
    # separate hashes rather than one keyed by width.
    #
    # This inherits the same single-inference-at-a-time assumption the attention
    # path's shared staging buffers already make.
    @@dev_gate_bufs = Hash(Int32, CudaMatrix).new
    @@dev_up_bufs = Hash(Int32, CudaMatrix).new
    @@dev_hidden_bufs = Hash(Int32, CudaMatrix).new

    # Batched workspaces, keyed by {rows, hidden_cols}: prefill tiles to a fixed row
    # count so this stays a handful of buffers rather than one per distinct expert
    # load.
    # One slot each, not one per shape: see the note in forward_device_batch.
    @@dev_gate_batch = Hash(Symbol, CudaMatrix).new
    @@dev_up_batch = Hash(Symbol, CudaMatrix).new
    @@dev_hidden_batch = Hash(Symbol, CudaMatrix).new

    protected def self.batch_buf(store : Hash(Symbol, CudaMatrix), rows : Int32, cols : Int32) : CudaMatrix
      if existing = store[:buf]?
        return existing if existing.rows == rows && existing.cols == cols
        existing.free!
      end
      store[:buf] = CudaMatrix.new(rows, cols)
    end

    # True when this expert can run the fully device-resident path: all three
    # projections quantized onto the device (or host-resident Q4, which streams
    # itself) and the fused SwiGLU kernel present in the loaded .so.
    def device_resident_capable? : Bool
      @gate_proj.is_a?(QuantizedWeight) &&
        @up_proj.is_a?(QuantizedWeight) &&
        @down_proj.is_a?(QuantizedWeight) &&
        CUDA.fully_available? && CUDA.swiglu_kernel_available?
    end

    # Decode forward that takes a device-resident activation row and returns one,
    # with no host round trip anywhere in between.
    #
    # The whole point: the host path spends three device-to-host readbacks per
    # expert (gate, up, down) plus a host SwiGLU loop, and at 8 experts x 48 layers
    # that readback latency dominated the decode step. Here gate and up are
    # produced into device buffers, combined by a device kernel, and fed straight
    # into the down projection. Nothing crosses PCIe.
    #
    # `xb` must already be device-resident. The returned CudaMatrix is a SHARED
    # workspace owned by the caller's buffer table, so consume it before the next
    # call to this method.
    def forward_device(xb : CudaMatrix, out_buf : CudaMatrix) : CudaMatrix
      gate_w = @gate_proj.as(QuantizedWeight)
      up_w = @up_proj.as(QuantizedWeight)
      down_w = @down_proj.as(QuantizedWeight)
      hidden_cols = gate_w.cols

      gate_buf = (@@dev_gate_bufs[hidden_cols] ||= CudaMatrix.new(1, hidden_cols))
      up_buf = (@@dev_up_bufs[hidden_cols] ||= CudaMatrix.new(1, hidden_cols))
      hidden_buf = (@@dev_hidden_bufs[hidden_cols] ||= CudaMatrix.new(1, hidden_cols))

      Profile.measure("ffn.dev_gate_up") do
        gate_w.gemv_into(xb, gate_buf)
        up_w.gemv_into(xb, up_buf)
      end
      Profile.measure("ffn.dev_swiglu") do
        CUDA.swiglu_forward(hidden_buf.device_ptr.not_nil!,
          gate_buf.device_ptr.not_nil!, up_buf.device_ptr.not_nil!, hidden_cols)
        hidden_buf.mark_device_dirty!
      end
      Profile.measure("ffn.dev_down") { down_w.gemv_into(hidden_buf, out_buf) }
      out_buf
    end

    # Batched variant: many token rows through ONE expert in three GEMMs instead of
    # three GEMVs per token. The quantized weights already take M rows (gemv_into
    # passes x.rows straight to gemm_q4_f32), so what this adds is the multi-row
    # workspaces and a SwiGLU over the whole contiguous batch rather than one row.
    #
    # This is what makes the expert's weights pay off: a Q4 weight read once serves
    # xb.rows tokens instead of being re-read per token, and prefill's cost after the
    # device-resident change is dominated by exactly those weight reads.
    def forward_device_batch(xb : CudaMatrix, out_buf : CudaMatrix) : CudaMatrix
      gate_w = @gate_proj.as(QuantizedWeight)
      up_w = @up_proj.as(QuantizedWeight)
      down_w = @down_proj.as(QuantizedWeight)
      hidden_cols = gate_w.cols
      n = xb.rows

      # One buffer per role, reallocated when the row count changes.
      #
      # These were keyed by {rows, hidden_cols} and kept a set per distinct sequence length. At
      # [1400, 17408] each is 97.5 MB, so every new prompt length cost another ~292 MB of VRAM and a
      # varying-length caller eventually OOM'd -- a 1400-token prefill failed even with a 2048 MB
      # reserve after a shorter one had run. A generation does one prefill shape then a steady
      # rows=1 shape, so eviction costs two reallocations and then nothing.
      gate_buf = SwiGLUFF.batch_buf(@@dev_gate_batch, n, hidden_cols)
      up_buf = SwiGLUFF.batch_buf(@@dev_up_batch, n, hidden_cols)
      hidden_buf = SwiGLUFF.batch_buf(@@dev_hidden_batch, n, hidden_cols)

      Profile.measure("ffn.batch_gate_up") do
        GGUFMatrix.with_shared_input(xb) do
          gate_w.gemv_into(xb, gate_buf)
          up_w.gemv_into(xb, up_buf)
        end
      end
      # The int8 path computes silu(gate) * up inside the down projection's input quantization, so
      # the [n, hidden] activation is never written out and read back.
      if down_w.is_a?(GGUFMatrix) && Profile.measure("ffn.batch_down") { down_w.gemm_into_swiglu(gate_buf, up_buf, out_buf) }
        return out_buf
      end
      Profile.measure("ffn.batch_swiglu") do
        # Elementwise over the whole batch: the buffers are contiguous [n, hidden].
        CUDA.swiglu_forward(hidden_buf.device_ptr.not_nil!,
          gate_buf.device_ptr.not_nil!, up_buf.device_ptr.not_nil!, n * hidden_cols)
        hidden_buf.mark_device_dirty!
      end
      Profile.measure("ffn.batch_down") { down_w.gemv_into(hidden_buf, out_buf) }
      out_buf
    end

    def to_gpu!(quantize : Bool = false, bits : Int32 = 8, offload : Bool = false)
      return unless CUDA.fully_available?
      # Catch placeholder experts (allocate: false) that were never loaded — promoting
      # or quantizing a 0x0 matrix would otherwise surface as a cryptic CUDA malloc(0)
      # failure.
      if (g = @gate_proj).is_a?(SimpleMatrix) && (g.rows == 0 || g.cols == 0)
        raise "SwiGLUFF weights are uninitialized (0x0); load weights before calling to_gpu!"
      end
      if quantize
        @gate_proj = to_quant(@gate_proj, bits, offload)
        @up_proj = to_quant(@up_proj, bits, offload)
        @down_proj = to_quant(@down_proj, bits, offload)
      else
        # Only promote host weights; leave existing CudaMatrix/QuantizedWeight as-is.
        @gate_proj = @gate_proj.as(SimpleMatrix).to_cuda if @gate_proj.is_a?(SimpleMatrix)
        @up_proj = @up_proj.as(SimpleMatrix).to_cuda if @up_proj.is_a?(SimpleMatrix)
        @down_proj = @down_proj.as(SimpleMatrix).to_cuda if @down_proj.is_a?(SimpleMatrix)
      end
    end

    # Quantize a weight to the requested bit width: bits == 4 -> Q4, bits == 8 ->
    # Q8. When offload is true the (Q4-only) weights are kept in host RAM as a
    # Q4HostMatrix and streamed to the GPU on demand. Already-quantized weights
    # are returned unchanged.
    private def to_quant(w : SimpleMatrix | CudaMatrix | QuantizedWeight, bits : Int32, offload : Bool = false) : QuantizedWeight
      raise ArgumentError.new("unsupported quantization bits: #{bits} (expected 8 or 4)") unless bits == 8 || bits == 4
      raise ArgumentError.new("expert offload currently supports 4-bit only (got #{bits}-bit)") if offload && bits != 4
      case w
      when QuantizedWeight then w
      when CudaMatrix
        sm = w.to_simple
        if offload
          Q4HostMatrix.from_simple(sm)
        else
          bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm)
        end
      else
        sm = w.as(SimpleMatrix)
        if offload
          Q4HostMatrix.from_simple(sm)
        else
          bits == 4 ? Q4CudaMatrix.from_simple(sm) : QuantizedCudaMatrix.from_simple(sm)
        end
      end
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      gate = matmul(x, @gate_proj)
      up = matmul(x, @up_proj)

      rows = gate.rows
      cols = gate.cols
      hidden = SimpleMatrix.new(rows, cols)
      Profile.measure("ffn.swiglu_act") do
        rows.times do |i|
          cols.times do |j|
            g = gate[i, j]
            hidden[i, j] = (g / (1.0 + Math.exp(-g))) * up[i, j]
          end
        end
      end

      matmul(hidden, @down_proj)
    end

    def forward(x : CudaMatrix) : CudaMatrix
      gate = x * @gate_proj.as(CudaMatrix) # cuBLAS GEMM
      up = x * @up_proj.as(CudaMatrix)     # cuBLAS GEMM

      # SiLU element-wise on CPU (no custom kernel yet)
      gate.sync_from_device!("swiglu") if gate.device_dirty?
      up.sync_from_device!("swiglu") if up.device_dirty?

      rows = gate.rows
      cols = gate.cols
      hidden = CudaMatrix.new(rows, cols)
      rows.times do |i|
        cols.times do |j|
          g = gate[i, j]
          hidden[i, j] = (g / (1.0 + Math.exp(-g))) * up[i, j]
        end
      end
      hidden.sync_to_device!("swiglu_done")

      hidden * @down_proj.as(CudaMatrix) # cuBLAS GEMM
    end

    private def matmul(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix | QuantizedWeight) : SimpleMatrix
      if w.is_a?(QuantizedWeight)
        if x.rows == 1
          xb = (@q8_in_bufs[x.cols] ||= CudaMatrix.new(1, x.cols))
          Profile.measure("gemm.in_h2d") do
            xb.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.cols)
            xb.mark_host_modified!
            xb.sync_to_device!("q8_ffn_in")
          end
          ob = (@q8_out_bufs[w.cols] ||= CudaMatrix.new(1, w.cols))
          Profile.measure("gemm.kernel") { w.gemv_into(xb, ob) }
          Profile.measure("gemm.out_d2h") { ob.sync_from_device!("q8_ffn_out") if ob.device_dirty? }
          result = SimpleMatrix.new(1, w.cols)
          Profile.measure("gemm.result_copy") do
            result.data.to_unsafe.copy_from(ob.raw_data.to_unsafe, w.cols)
          end
          result
        else
          x_gpu = CudaMatrix.new(x.rows, x.cols)
          x_gpu.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.rows * x.cols)
          x_gpu.sync_to_device!("q8_ffn_in")
          result_gpu = w.gemv(x_gpu)
          result_gpu.sync_from_device!("q8_ffn_out") if result_gpu.device_dirty?
          result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
          result.data.to_unsafe.copy_from(result_gpu.raw_data.to_unsafe, result_gpu.rows * result_gpu.cols)
          result
        end
      elsif w.is_a?(CudaMatrix)
        x_gpu = CudaMatrix.new(x.rows, x.cols)
        x.rows.times { |r| x.cols.times { |c| x_gpu[r, c] = x[r, c] } }
        x_gpu.sync_to_device!("ffn_in")
        result_gpu = x_gpu * w
        result_gpu.sync_from_device!("ffn_out") if result_gpu.device_dirty?
        result = SimpleMatrix.new(result_gpu.rows, result_gpu.cols)
        result_gpu.rows.times { |r| result_gpu.cols.times { |c| result[r, c] = result_gpu[r, c].to_f32 } }
        result
      else
        x * w
      end
    end
  end
end
