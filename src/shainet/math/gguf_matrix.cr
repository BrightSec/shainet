require "./cuda_matrix"           # CudaMatrix used in the GEMV interface
require "./quantized_cuda_matrix" # defines the QuantizedWeight module (included below)
require "../gguf"                 # GGUF::GGMLType
{% if flag?(:enable_cuda) %}
  require "../cuda"
{% else %}
  require "../cuda_stub"
{% end %}

module SHAInet
  # GGUF k-quant weight matrix (Q4_K / Q6_K) for GPU inference.
  #
  # Holds the raw GGUF k-quant bytes on the device untouched and dequantizes
  # inside the GEMV kernel. Logical shape matches the fp32 weight it replaces:
  # GGUF stores tensors as [in_features, out_features] row-major, so
  #   rows = K (in_features), cols = N (out_features)
  # and it is used as `y[M,N] = x[M,K] * W[N,K]`.
  class GGUFMatrix
    include QuantizedWeight

    getter rows : Int32               # K (in_features)
    getter cols : Int32               # N (out_features)
    getter ggml_type : GGUF::GGMLType # Q4_K or Q6_K
    getter dev_ptr : Pointer(UInt8)   # device pointer to raw k-quant bytes

    @byte_size : UInt64

    # Allocate device memory, copy the raw k-quant bytes from host, and keep the
    # device pointer. `host_data` points at `byte_size` bytes of GGUF k-quant
    # data laid out as N rows of blocks, each row covering K values.
    def initialize(@rows : Int32, @cols : Int32, @ggml_type : GGUF::GGMLType,
                   host_data : Pointer(UInt8), byte_size)
      @byte_size = byte_size.to_u64
      unless GGUFMatrix.device_type_supported?(@ggml_type)
        raise ArgumentError.new("GGUFMatrix: no device kernel for #{@ggml_type}")
      end

      dp = Pointer(UInt8).null
      CUDA.malloc(pointerof(dp).as(Pointer(Pointer(Void))), @byte_size)
      @dev_ptr = dp
      CUDA.memcpy(@dev_ptr.as(Pointer(Void)), host_data.as(Pointer(Void)),
        @byte_size, CUDA::MemcpyKind::HostToDevice)
    end

    # Create a GGUFMatrix that points into a pre-allocated device buffer.
    # No CUDA.malloc or memcpy -- the data is already on the device in the bulk pool.
    # The pool owns the memory; this matrix must NOT free it.
    def self.from_pool(rows : Int32, cols : Int32, ggml_type : GGUF::GGMLType,
                       dev_ptr : Pointer(UInt8), byte_size : UInt64) : GGUFMatrix
      m = GGUFMatrix.allocate
      m.init_from_pool(rows, cols, ggml_type, dev_ptr, byte_size)
      m
    end

    protected def init_from_pool(@rows : Int32, @cols : Int32, @ggml_type : GGUF::GGMLType,
                                 @dev_ptr : Pointer(UInt8), @byte_size : UInt64)
      @pool_owned = true
    end

    @pool_owned : Bool = false

    # Debug tracing flag, read ONCE at class init.
    #
    # This was ENV["SHAINET_DEBUG"]? evaluated inline on every gemv_into call, TWICE. A Crystal ENV
    # lookup calls getenv -- a linear scan of the environment -- and allocates a String for the
    # result. A generated token issues on the order of five hundred GEMV calls across the resident
    # layers, so the untaken branch of a debug print cost ~1000 getenv calls and ~1000 short-lived
    # allocations per token, for nothing.
    @@debug : Bool = ENV["SHAINET_DEBUG"]? == "1"

    # Allocate device memory only (no upload). Used by GGUFHostMatrix for its
    # shared scratch buffer -- the data is uploaded per-GEMV via cudaMemcpy.
    def self.new_empty(rows : Int32, cols : Int32, ggml_type : GGUF::GGMLType, byte_size : UInt64) : GGUFMatrix
      m = GGUFMatrix.allocate
      m.initialize_empty(rows, cols, ggml_type, byte_size)
      m
    end

    protected def initialize_empty(@rows : Int32, @cols : Int32, @ggml_type : GGUF::GGMLType, byte_size : UInt64)
      @byte_size = byte_size
      unless GGUFMatrix.device_type_supported?(@ggml_type)
        raise ArgumentError.new("GGUFMatrix: no device kernel for #{@ggml_type}")
      end
      dp = Pointer(UInt8).null
      CUDA.malloc(pointerof(dp).as(Pointer(Pointer(Void))), @byte_size)
      @dev_ptr = dp
    end

    def finalize
      free!
    end

    # y[M,N] = x[M,K] * dequant(W[N,K]) into a caller-provided result buffer.
    # result must be [x.rows, cols].
    def gemv_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      raise ArgumentError.new("dimension mismatch: x.cols=#{x.cols} vs K=#{@rows}") unless x.cols == @rows
      raise ArgumentError.new("result shape mismatch") unless result.rows == x.rows && result.cols == @cols
      raise RuntimeError.new("GGUF gemv requires a valid device pointer") if @dev_ptr.null?

      if @@debug
        STDERR.puts "  [gguf gemv] #{@ggml_type} M=#{x.rows} N=#{@cols} K=#{@rows} bytes=#{@byte_size} pool=#{@pool_owned}"
        STDERR.flush
      end

      # Ensure activation is resident on device (cheap no-op when already synced).
      x.sync_to_device!("gguf_gemv_in") unless x.device_dirty?

      # M == 1 (decode) is already optimal as a GEMV: the weight is streamed exactly once. For
      # M > 1 the GEMV kernel gives each (output, token) pair its own block, so it re-dequantizes
      # the whole weight per token and reuses nothing -- measured flat at 0.54 ms/token from M=1 to
      # M=256 on a 47.8 MB Q4_K weight, i.e. 54 GB of reads for a 1200-token prefill of ONE matmul.
      # Dequantizing once into a scratch tile and letting cuBLAS do the GEMM reads the weight once.
      if x.rows > 1 && gemm_capable?
        return gemm_into(x, result)
      end

      case @ggml_type
      when .iq4_xs?
        CUDA.gemv_iq4xs(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq3_s?
        CUDA.gemv_iq3s(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq3_xxs?
        CUDA.gemv_iq3xxs(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .q2_k?
        CUDA.gemv_q2k_lb(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq2_xxs?
        CUDA.gemv_iq2xxs(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq2_xs?
        CUDA.gemv_iq2xs(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq2_s?
        CUDA.gemv_iq2s(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq1_m?
        CUDA.gemv_iq1m(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .iq1_s?
        CUDA.gemv_iq1s(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .q4_k?
        CUDA.gemv_q4k(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .q6_k?
        CUDA.gemv_q6k(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      when .q5_k?
        CUDA.gemv_q5k(x.device_ptr.not_nil!, @dev_ptr, result.device_ptr.not_nil!,
          x.rows, @cols, @rows)
      else
        raise ArgumentError.new("unsupported GGUF type for gemv: #{@ggml_type}")
      end

      result.mark_device_dirty!
      result
    end

    # Same as gemv_into but allocates the result CudaMatrix [x.rows, cols].
    def gemv(x : CudaMatrix) : CudaMatrix
      result = CudaMatrix.new(x.rows, @cols)
      gemv_into(x, result)
    end

    # Scratch for the dequantized weight tile, shared across every GGUFMatrix.
    #
    # One buffer rather than one per weight: prefill runs the matmuls sequentially, so a single
    # tile is live at a time, and a per-weight buffer would cost hundreds of MB of VRAM on a card
    # that is already nearly full with the model.
    @@scratch : Pointer(Float32) = Pointer(Float32).null
    @@scratch_floats : Int32 = 0
    @@gemm_supported : Bool? = nil

    # Rows dequantized per cuBLAS call. 1024 x K=5120 fp32 is 20 MB, which is small next to the
    # model yet large enough that the GEMM is not launch-bound.
    SCRATCH_ROWS = 1024

    def self.release_scratch!
      unless @@scratch.null?
        CUDA.free(@@scratch.as(Pointer(Void)))
        @@scratch = Pointer(Float32).null
        @@scratch_floats = 0
      end
    end

    # Which quantization types have a device GEMV.
    #
    # Kept as one predicate because three places need the same answer: both constructors, and
    # stage_host when it decides whether a host weight can be borrowed onto the card. A type missing
    # here must be transcoded at load rather than reaching a kernel that cannot read it.
    def self.device_type_supported?(t : GGUF::GGMLType) : Bool
      return t.q4_k? || t.q6_k? if @@force_transcode_iq
      t.q4_k? || t.q5_k? || t.q6_k? || t.iq4_xs? || t.iq3_s? || t.iq3_xxs? ||
        t.q2_k? || t.iq2_xxs? || t.iq2_xs? || t.iq2_s? || t.iq1_m? || t.iq1_s?
    end

    # Diagnostic switch: treat the i-quant types as unsupported so every one of them is transcoded
    # through the scalar reference instead of reaching a kernel. The references are independently
    # verified against a differently-quantized build of the same model, so this separates "a kernel is
    # wrong" from "the loader wires something wrong" -- the two explanations that survive when every
    # component passes its own test but the model still does not predict.
    @@force_transcode_iq : Bool = ENV.fetch("SHAINET_IQ_FORCE_TRANSCODE", "0") == "1"

    # ── One reusable device staging slot for weights that live on the HOST ──
    #
    # A host-resident layer is the right choice when DECODING: at one row it streams its weights at
    # ~69 GB/s of DDR5 and the arithmetic is memory-bound, so the CPU keeps up. PREFILL is a
    # different problem. At 256 rows the same projection is COMPUTE-bound -- ffn_gate plus ffn_up is
    # 46 GFLOP per chunk -- and the CPU is about two orders of magnitude off the GPU there. Measured
    # on a 301-token prefill: the 24 host-layer ffn_gate_up calls took 5520 ms of the 5590 ms spent
    # in that phase, while the 104 device-layer calls took 73 ms between them.
    #
    # Above a batch threshold it is therefore cheaper to SEND the quantized weight to the card and
    # use the dequant+cuBLAS path, because the upload amortizes over the rows: ffn_gate is 47.8 MB,
    # roughly 4 ms over PCIe, against ~230 ms of CPU GEMM. One slot suffices because a staged weight
    # is consumed before the next is staged, and the slot is grown rather than reallocated per call.
    @@stage_ptr = Pointer(UInt8).null
    @@stage_bytes = 0_u64

    # Upload host-resident k-quant bytes into the staging slot and return a NON-OWNING view of them so
    # the caller can use the device GEMM path. Returns nil when the weight cannot be staged, which
    # leaves the host path as the fallback rather than turning it into a failure.
    def self.stage_host(rows : Int32, cols : Int32, ggml_type : GGUF::GGMLType,
                        host_data : Pointer(UInt8), byte_size : UInt64) : GGUFMatrix?
      return unless CUDA.fully_available?
      return unless GGUFMatrix.device_type_supported?(ggml_type)
      if @@stage_bytes < byte_size
        unless @@stage_ptr.null?
          CUDA.free(@@stage_ptr.as(Pointer(Void)))
          @@stage_ptr = Pointer(UInt8).null
          @@stage_bytes = 0_u64
        end
        p = Pointer(Void).null
        CUDA.malloc(pointerof(p), byte_size)
        return if p.null?
        @@stage_ptr = p.as(Pointer(UInt8))
        @@stage_bytes = byte_size
      end
      CUDA.memcpy(@@stage_ptr.as(Pointer(Void)), host_data.as(Pointer(Void)),
        byte_size, CUDA::MemcpyKind::HostToDevice)
      view = allocate
      view.init_view(rows, cols, ggml_type, @@stage_ptr, byte_size)
      view
    end

    # Adopt an existing device pointer. @pool_owned suppresses the free, so the staging slot outlives
    # every view taken of it.
    protected def init_view(@rows : Int32, @cols : Int32, @ggml_type : GGUF::GGMLType,
                            @dev_ptr : Pointer(UInt8), @byte_size : UInt64)
      @pool_owned = true
    end

    protected def self.scratch(floats : Int32) : Pointer(Float32)
      if @@scratch_floats < floats
        release_scratch!
        p = Pointer(Void).null
        CUDA.malloc(pointerof(p), (floats.to_u64 * 4))
        @@scratch = p.as(Pointer(Float32))
        @@scratch_floats = floats
      end
      @@scratch
    end

    private def gemm_capable? : Bool
      sup = @@gemm_supported
      return sup unless sup.nil?
      @@gemm_supported = sup = CUDA.fully_available? && CUDA.dequant_k_rows_available?
      sup
    end

    # Batched matmul for M > 1: dequantize the weight in row chunks and let cuBLAS multiply.
    #
    # result[M, N] = x[M, K] * dequant(W[N, K])^T. cuBLAS is column-major, so the row-major
    # result [M, N] is addressed as a column-major [N, M]: with the dequantized chunk as the
    # transposed operand and ldc set to the FULL N, each chunk writes straight into its own
    # columns of the finished result.
    def gemm_into(x : CudaMatrix, result : CudaMatrix) : CudaMatrix
      return result if gemm_into_mmq(x, result)
      return result if gemm_into_f16(x, result)

      k = @rows
      chunk = SCRATCH_ROWS > @cols ? @cols : SCRATCH_ROWS
      buf = GGUFMatrix.scratch(chunk * k)
      # TF32 here, true fp32 everywhere else.
      #
      # This path runs ONLY for M > 1, which is prefill. Decode goes through the hand-written GEMV
      # kernels and touches cuBLAS not at all, so the determinism that pedantic fp32 protects --
      # identical logits run to run -- is unaffected by this.
      #
      # Worth 2.1x on the dominant cost: the GEMM is 84% of this function (12.72 ms against 2.41 ms
      # of dequant on a real 5120x17408 IQ3_S weight), and TF32 measured 30.3 against 14.1 TFLOP/s at
      # exactly this shape. The weights are already 2-4 bit quantized, so TF32's 10-bit mantissa is
      # far finer than the values it is multiplying.
      handle = CUDA.create_handle(tf32: true)
      begin
        xp = x.device_ptr.not_nil!
        rp = result.device_ptr.not_nil!
        row0 = 0
        while row0 < @cols
          rows = @cols - row0
          rows = chunk if rows > chunk
          case @ggml_type
          when .iq4_xs?  then CUDA.dequant_iq4xs_rows(@dev_ptr, buf, row0, rows, k)
          when .iq3_s?   then CUDA.dequant_iq3s_rows(@dev_ptr, buf, row0, rows, k)
          when .iq3_xxs? then CUDA.dequant_iq3xxs_rows(@dev_ptr, buf, row0, rows, k)
          when .q2_k?    then CUDA.dequant_q2k_lb_rows(@dev_ptr, buf, row0, rows, k)
          when .iq2_xxs? then CUDA.dequant_iq2xxs_rows(@dev_ptr, buf, row0, rows, k)
          when .iq2_xs?  then CUDA.dequant_iq2xs_rows(@dev_ptr, buf, row0, rows, k)
          when .iq2_s?   then CUDA.dequant_iq2s_rows(@dev_ptr, buf, row0, rows, k)
          when .iq1_m?   then CUDA.dequant_iq1m_rows(@dev_ptr, buf, row0, rows, k)
          when .iq1_s?   then CUDA.dequant_iq1s_rows(@dev_ptr, buf, row0, rows, k)
          when .q4_k?    then CUDA.dequant_q4k_rows(@dev_ptr, buf, row0, rows, k)
          when .q6_k?    then CUDA.dequant_q6k_rows(@dev_ptr, buf, row0, rows, k)
          when .q5_k?    then CUDA.dequant_q5k_rows(@dev_ptr, buf, row0, rows, k)
          else                raise ArgumentError.new("unsupported GGUF type for gemm: #{@ggml_type}")
          end
          CUDA.gemm_tn(handle, buf, xp, rp + row0,
            rows, x.rows, k, k, k, @cols)
          row0 += rows
        end
      ensure
        CUDA.destroy_handle(handle)
      end
      result.mark_device_dirty!
      result
    end

    # Free the device allocation if not already freed.
    def free!
      unless @dev_ptr.null? || @pool_owned
        CUDA.free(@dev_ptr.as(Pointer(Void)))
        @dev_ptr = Pointer(UInt8).null
      end
    end

    # ── int8 MMQ prefill GEMM ──
    #
    # For the types it covers (Q4_K, IQ4_XS, IQ3_S, IQ3_XXS -- most of the bytes of the i-quant
    # builds), the weight slice is decoded once to int8 with a scale per 32 values, the activations
    # are quantized the same way (llama.cpp's q8 scheme), and an int8 tensor-core GEMM applies the
    # scales per 32-wide k step. One byte per weight through the scratch instead of the fp16 path's
    # two, and int8 MMA instead of fp16.
    #
    # Activations lose precision to int8 (per-32 absmax), which is the trade llama.cpp makes for its
    # prefill -- greedy output does change. Opt-in with SHAINET_MMQ=1: measured on the 27B GSQ IQ3_S
    # it is +11% prefill at 1024-token chunks and about even at 512, for ~80 MB more scratch, and the
    # fp16 path at the same chunk keeps fp16-class precision.
    @@mmq : Bool? = nil
    @@mq_w : Pointer(Int8) = Pointer(Int8).null
    @@mq_w_elems : Int64 = 0_i64
    @@mq_ws : Pointer(Float32) = Pointer(Float32).null
    @@mq_ws_elems : Int64 = 0_i64
    @@mq_wo : Pointer(Float32) = Pointer(Float32).null
    @@mq_wo_elems : Int64 = 0_i64
    @@mq_x : Pointer(Int8) = Pointer(Int8).null
    @@mq_x_elems : Int64 = 0_i64
    @@mq_xs : Pointer(Float32) = Pointer(Float32).null
    @@mq_xs_elems : Int64 = 0_i64
    @@mq_xsum : Pointer(Float32) = Pointer(Float32).null
    @@mq_xsum_elems : Int64 = 0_i64

    # Rows decoded per GEMM launch. Not small: a 2048-row slice at a 256-token chunk is a 32-block
    # grid on a 76-SM card, measured 3x slower than at 8192. 4096 keeps the int8 scratch near 80 MB
    # at K=17408 while the grid stays full at the 512+ token chunks the path is worth using at.
    MMQ_SLICE_ROWS = 4096

    MMQ_TYPES = {GGUF::GGMLType::Q4_K, GGUF::GGMLType::IQ4_XS, GGUF::GGMLType::IQ3_S, GGUF::GGMLType::IQ3_XXS}

    def self.mmq? : Bool
      flag = @@mmq
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_MMQ", "0") == "1" && CUDA.fully_available? && CUDA.mmq_available?
      @@mmq = flag
      flag
    end

    # Force the choice (nil re-reads SHAINET_MMQ on next use). For specs and A/B runs.
    def self.mmq=(value : Bool?)
      @@mmq = value
    end

    private def gemm_into_mmq(x : CudaMatrix, result : CudaMatrix) : Bool
      return false unless MMQ_TYPES.includes?(@ggml_type) && GGUFMatrix.mmq?
      k = @rows
      m = x.rows
      nb = (k // 32).to_i64
      slice = MMQ_SLICE_ROWS > @cols ? @cols : MMQ_SLICE_ROWS
      @@mq_w, @@mq_w_elems = GGUFMatrix.grow(@@mq_w, @@mq_w_elems, slice.to_i64 * k)
      @@mq_ws, @@mq_ws_elems = GGUFMatrix.grow(@@mq_ws, @@mq_ws_elems, slice.to_i64 * nb)
      @@mq_wo, @@mq_wo_elems = GGUFMatrix.grow(@@mq_wo, @@mq_wo_elems, slice.to_i64 * nb)
      @@mq_x, @@mq_x_elems = GGUFMatrix.grow(@@mq_x, @@mq_x_elems, m.to_i64 * k)
      @@mq_xs, @@mq_xs_elems = GGUFMatrix.grow(@@mq_xs, @@mq_xs_elems, m.to_i64 * nb)
      @@mq_xsum, @@mq_xsum_elems = GGUFMatrix.grow(@@mq_xsum, @@mq_xsum_elems, m.to_i64 * nb)
      return false if @@mq_w.null? || @@mq_ws.null? || @@mq_wo.null? || @@mq_x.null? || @@mq_xs.null? || @@mq_xsum.null?

      CUDA.mmq_quantize_x(x.device_ptr.not_nil!, @@mq_x, @@mq_xs, @@mq_xsum, m, k)
      rp = result.device_ptr.not_nil!
      offset = @ggml_type.q4_k?
      row0 = 0
      while row0 < @cols
        rows = @cols - row0
        rows = slice if rows > slice
        return false unless CUDA.mmq_decode_rows(@ggml_type.value.to_i32, @dev_ptr, @@mq_w, @@mq_ws, @@mq_wo, row0, rows, k)
        return false unless CUDA.mmq_gemm(offset, @@mq_x, @@mq_xs, @@mq_xsum, @@mq_w, @@mq_ws, @@mq_wo,
                              rp + row0, m, rows, k, @cols)
        row0 += rows
      end
      result.mark_device_dirty!
      true
    end

    # ── fp16 tensor-core prefill GEMM ──
    #
    # The TF32 path above was ~45% of a prefill: cuBLAS SGEMM at ~25 TFLOP/s on 256-token chunks,
    # fed by an fp32 dequant of every weight. Here the weight slice is dequantized straight to half
    # and the activations are converted to half once per call, and cublasGemmEx runs fp16 x fp16 with
    # fp32 accumulation -- twice TF32's tensor rate on Ada, from half the scratch bytes.
    #
    # Precision is TF32's: both keep a 10-bit mantissa. The difference is fp16's 5-bit exponent, which
    # the activations sidestep with a per-row power-of-two scale (exact, undone on the output by
    # scale_rows). The dequantized weights are products of an fp16 block scale and small integers, so
    # they are in range by construction.
    #
    # SHAINET_FP16_PREFILL_GEMM=0 keeps the TF32 path.
    @@f16_gemm : Bool? = nil
    @@scratch_h : Pointer(UInt16) = Pointer(UInt16).null
    @@scratch_h_elems : Int64 = 0_i64
    @@xh : Pointer(UInt16) = Pointer(UInt16).null
    @@xh_elems : Int64 = 0_i64
    @@inv_scale : Pointer(Float32) = Pointer(Float32).null
    @@inv_scale_elems : Int32 = 0

    F16_DEQUANT_NAMES = {
      GGUF::GGMLType::IQ4_XS  => "iq4xs",
      GGUF::GGMLType::IQ3_S   => "iq3s",
      GGUF::GGMLType::IQ3_XXS => "iq3xxs",
      GGUF::GGMLType::Q2_K    => "q2k_lb",
      GGUF::GGMLType::IQ2_XXS => "iq2xxs",
      GGUF::GGMLType::IQ2_XS  => "iq2xs",
      GGUF::GGMLType::IQ2_S   => "iq2s",
      GGUF::GGMLType::IQ1_M   => "iq1m",
      GGUF::GGMLType::IQ1_S   => "iq1s",
      GGUF::GGMLType::Q4_K    => "q4k",
      GGUF::GGMLType::Q5_K    => "q5k",
      GGUF::GGMLType::Q6_K    => "q6k",
    }

    def self.f16_gemm? : Bool
      flag = @@f16_gemm
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_FP16_PREFILL_GEMM", "1") != "0" && CUDA.fully_available? &&
             CUDA.f16_gemm_helpers_available?
      @@f16_gemm = flag
      flag
    end

    # Grow-only device buffer of `elems` elements of `T`.
    def self.grow(ptr : Pointer(T), have : Int64, want : Int64) : {Pointer(T), Int64} forall T
      return {ptr, have} if have >= want
      CUDA.free(ptr.as(Pointer(Void))) unless ptr.null?
      p = Pointer(Void).null
      CUDA.malloc(pointerof(p), (want * sizeof(T)).to_u64)
      {p.as(Pointer(T)), want}
    end

    # Returns false when this weight or device cannot take the fp16 path, leaving gemm_into's TF32
    # path to run.
    private def gemm_into_f16(x : CudaMatrix, result : CudaMatrix) : Bool
      return false unless GGUFMatrix.f16_gemm?
      name = F16_DEQUANT_NAMES[@ggml_type]?
      return false unless name

      k = @rows
      m = x.rows
      chunk = SCRATCH_ROWS > @cols ? @cols : SCRATCH_ROWS
      @@scratch_h, @@scratch_h_elems = GGUFMatrix.grow(@@scratch_h, @@scratch_h_elems, chunk.to_i64 * k)
      @@xh, @@xh_elems = GGUFMatrix.grow(@@xh, @@xh_elems, m.to_i64 * k)
      ip, ie = GGUFMatrix.grow(@@inv_scale, @@inv_scale_elems.to_i64, m.to_i64)
      @@inv_scale, @@inv_scale_elems = ip, ie.to_i32
      return false if @@scratch_h.null? || @@xh.null? || @@inv_scale.null?

      CUDA.rows_to_f16_scaled(x.device_ptr.not_nil!, @@xh, @@inv_scale, m, k)
      rp = result.device_ptr.not_nil!
      handle = CUDA.create_handle(tf32: true)
      begin
        row0 = 0
        while row0 < @cols
          rows = @cols - row0
          rows = chunk if rows > chunk
          return false unless CUDA.dequant_rows_f16(name, @dev_ptr, @@scratch_h, row0, rows, k)
          status = CUDA.gemm_tn_f16(handle, @@scratch_h, @@xh, rp + row0, rows, m, k, k, k, @cols)
          raise "cublasGemmEx failed (status #{status})" unless status == 0
          row0 += rows
        end
      ensure
        CUDA.destroy_handle(handle)
      end
      CUDA.scale_rows(rp, @@inv_scale, m, @cols)
      result.mark_device_dirty!
      true
    end

    # Device memory footprint in bytes.
    def device_bytes : UInt64
      @byte_size
    end
  end
end
