require "log"

module SHAInet
  module CUDA
    Log = ::Log.for(self)
    extend self

    # :nodoc:
    @[Link("cudart")]
    lib LibCUDARuntime
      fun cudaRuntimeGetVersion(version : Pointer(Int32)) : Int32
      fun cudaMalloc(ptr : Pointer(Pointer(Void)), size : LibC::SizeT) : Int32
      fun cudaFree(ptr : Pointer(Void)) : Int32
      fun cudaMemcpy(dst : Pointer(Void), src : Pointer(Void), count : LibC::SizeT, kind : Int32) : Int32
      fun cudaMallocHost(ptr : Pointer(Pointer(Void)), size : LibC::SizeT) : Int32
      fun cudaFreeHost(ptr : Pointer(Void)) : Int32
      fun cudaMemGetInfo(free : Pointer(LibC::SizeT), total : Pointer(LibC::SizeT)) : Int32
      fun cudaDeviceSynchronize : Int32
    end

    @[Link("cublas")]
    lib LibCUBLAS
      type Handle = Void*

      fun cublasCreate_v2(handle : Pointer(Handle)) : Int32
      fun cublasDestroy_v2(handle : Handle) : Int32
      fun cublasSetMathMode(handle : Handle, mode : Int32) : Int32
      fun cublasSgemm_v2(handle : Handle, transa : Int32, transb : Int32,
                         m : Int32, n : Int32, k : Int32,
                         alpha : Pointer(Float32), a : Pointer(Float32), lda : Int32,
                         b : Pointer(Float32), ldb : Int32,
                         beta : Pointer(Float32), c : Pointer(Float32), ldc : Int32) : Int32
      fun cublasSgeam(handle : Handle,
                      transa : Int32, transb : Int32,
                      m : Int32, n : Int32,
                      alpha : Pointer(Float32), a : Pointer(Float32), lda : Int32,
                      beta : Pointer(Float32), b : Pointer(Float32), ldb : Int32,
                      c : Pointer(Float32), ldc : Int32) : Int32
      fun cublasSscal_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float32), x : Pointer(Float32), incx : Int32) : Int32
      fun cublasSaxpy_v2(handle : Handle, n : Int32,
                         alpha : Pointer(Float32),
                         x : Pointer(Float32), incx : Int32,
                         y : Pointer(Float32), incy : Int32) : Int32
    end

    enum MemcpyKind
      HostToHost     = 0
      HostToDevice   = 1
      DeviceToHost   = 2
      DeviceToDevice = 3
    end

    enum Operation
      N = 0
      T = 1
    end

    # Check if CUDA runtime and cuBLAS libraries can be opened.
    @@checked = false
    @@available = false

    def available?
      return false if ENV["SHAINET_DISABLE_CUDA"]?
      return @@available if @@checked
      @@checked = true

      rt = LibC.dlopen("libcudart.so", LibC::RTLD_LAZY)
      blas = LibC.dlopen("libcublas.so", LibC::RTLD_LAZY)

      if rt.null? || blas.null?
        @@available = false
      else
        LibC.dlclose(rt)
        LibC.dlclose(blas)
        @@available = true
      end

      @@available
    rescue ex
      Log.error { "CUDA availability check raised: #{ex}" }
      @@available = false
    end

    # Returns the CUDA runtime version or nil if CUDA is unavailable.
    def version
      return unless available?
      out = 0
      if LibCUDARuntime.cudaRuntimeGetVersion(pointerof(out)) == 0
        out
      end
    rescue
      nil
    end

    # Returns true when the cuDNN library can be loaded.
    def cudnn_available?
      handle = LibC.dlopen("libcudnn.so", LibC::RTLD_LAZY)
      if handle.null?
        false
      else
        LibC.dlclose(handle)
        true
      end
    rescue ex
      Log.error { "cuDNN availability check raised: #{ex}" }
      false
    end

    # Check if optional CUDA kernels are available via libshainet_cuda_kernels.so
    def kernels_available?
      handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
      if handle.null?
        false
      else
        LibC.dlclose(handle)
        true
      end
    rescue ex
      Log.error { "kernel availability check raised: #{ex}" }
      false
    end

    def malloc(ptr : Pointer(Pointer(Void)), size : LibC::SizeT)
      rslt = LibCUDARuntime.cudaMalloc(ptr, size)
      unless rslt.zero?
        # Likely out of memory. Dead GPU matrices may not be reclaimed yet: their
        # Crystal wrappers are tiny, so GC rarely runs during GPU-heavy loops (a
        # prefill can create hundreds of large device buffers without enough host
        # pressure to trigger a collection). Force GC to run their finalizers
        # (which cudaFree), then retry before giving up.
        GC.collect
        rslt = LibCUDARuntime.cudaMalloc(ptr, size)
      end
      unless rslt.zero?
        Log.error { "CUDA.malloc: cudaMalloc failed with result #{rslt} for size #{size}" }
        raise "CUDA memory allocation failed"
      end
      rslt
    end

    def free(ptr : Pointer(Void))
      LibCUDARuntime.cudaFree(ptr)
    end

    def memcpy(dst : Pointer(Void), src : Pointer(Void), bytes : LibC::SizeT, kind : MemcpyKind)
      LibCUDARuntime.cudaMemcpy(dst, src, bytes, kind.value)
    end

    def device_synchronize
      LibCUDARuntime.cudaDeviceSynchronize
    end

    def copy_device_to_device(dst : Pointer(Float32), src : Pointer(Float32), bytes : LibC::SizeT)
      memcpy(dst.as(Pointer(Void)), src.as(Pointer(Void)), bytes, MemcpyKind::DeviceToDevice)
    end

    def malloc_host(ptr : Pointer(Pointer(Void)), size : LibC::SizeT)
      result = LibCUDARuntime.cudaMallocHost(ptr, size)
      unless result.zero?
        Log.error { "CUDA.malloc_host: cudaMallocHost failed with result #{result} for size #{size}" }
        raise "CUDA host memory allocation failed"
      end
      result
    end

    def free_host(ptr : Pointer(Void))
      LibCUDARuntime.cudaFreeHost(ptr)
    end

    # Returns a hash with free and total memory in bytes for the active CUDA device.
    def memory_info
      return unless fully_available?
      free = 0_u64
      total = 0_u64
      res = LibCUDARuntime.cudaMemGetInfo(pointerof(free), pointerof(total))
      if res.zero?
        {free: free, total: total}
      else
        Log.error { "CUDA.memory_info: cudaMemGetInfo failed with result #{res}" }
        nil
      end
    rescue ex
      Log.error { "CUDA.memory_info raised: #{ex}" }
      nil
    end

    # Convenience method returning the total memory in bytes or nil when unavailable.
    def total_memory
      if info = memory_info
        info[:total]
      end
    end

    # Handle pool to avoid creating/destroying handles frequently
    @@handle_pool = [] of LibCUBLAS::Handle
    @@handle_pool_mutex = Mutex.new
    @@max_pool_size = 4 # Limit pool size to avoid resource exhaustion

    # cuBLAS math modes, by their documented values. Naming them matters: the previous code treated
    # 16 as "allow TF32", but 16 alone is DEFAULT_MATH | DISALLOW_REDUCED, which enables no tensor
    # path at all. The escape hatch therefore did nothing, and the measurement taken through it
    # ("TF32 buys nothing here") was measuring pedantic fp32 against pedantic fp32.
    CUBLAS_DEFAULT_MATH         =  0
    CUBLAS_PEDANTIC_MATH        =  2
    CUBLAS_TF32_TENSOR_OP_MATH  =  3
    CUBLAS_DISALLOW_REDUCED_RED = 16

    # True fp32, deterministic: what decode and every default path gets.
    MATH_MODE_PEDANTIC = CUBLAS_PEDANTIC_MATH | CUBLAS_DISALLOW_REDUCED_RED # 18
    # TF32 tensor cores with deterministic accumulation: 2.1x on SGEMM, measured at the shape the
    # quantized prefill GEMM actually uses (M=1024 N=512 K=5120: 14.1 -> 30.3 TFLOP/s).
    MATH_MODE_TF32 = CUBLAS_TF32_TENSOR_OP_MATH | CUBLAS_DISALLOW_REDUCED_RED # 19

    # Acquire a cuBLAS handle. `tf32: true` asks for the TF32 tensor-core path.
    #
    # The math mode is set on every acquire, not only on creation, because handles are POOLED: a
    # handle returned by a TF32 caller would otherwise hand its mode to the next caller that wanted
    # pedantic fp32, which is a silent precision change depending on call order.
    def create_handle(tf32 : Bool = false)
      mode = if mm = ENV["SHAINET_CUBLAS_MATHMODE"]?
               mm.to_i
             elsif tf32 || ENV.fetch("SHAINET_CUBLAS_TF32", "0") == "1"
               MATH_MODE_TF32
             else
               MATH_MODE_PEDANTIC
             end

      pooled = nil.as(LibCUBLAS::Handle?)
      @@handle_pool_mutex.synchronize do
        pooled = @@handle_pool.pop unless @@handle_pool.empty?
      end
      if h = pooled
        apply_math_mode(h, mode)
        return h
      end

      handle = Pointer(LibCUBLAS::Handle).malloc(1)
      raise "cublasCreate failed" unless LibCUBLAS.cublasCreate_v2(handle) == 0
      apply_math_mode(handle.value, mode)
      handle.value
    end

    # Ask for true fp32 by default, so callers do not have to remember NVIDIA_TF32_OVERRIDE=0.
    #
    # On Ampere and Ada, cuBLAS can run SGEMM on TF32 tensor cores, cutting the mantissa from 23 bits
    # to 10 -- enough to make token generation vary between runs. The documented workaround is the
    # NVIDIA_TF32_OVERRIDE=0 environment variable, but that has to be set before the process starts,
    # so it is easy to forget and invisible when missing. Setting the mode per handle fixes that
    # in-process.
    #
    # Older drivers may reject the combined value; fall back to the reduction flag alone rather than
    # leaving the handle unconfigured.
    private def apply_math_mode(handle : LibCUBLAS::Handle, mode : Int32)
      result = LibCUBLAS.cublasSetMathMode(handle, mode)
      if result != 0 && mode != CUBLAS_DISALLOW_REDUCED_RED
        result = LibCUBLAS.cublasSetMathMode(handle, CUBLAS_DISALLOW_REDUCED_RED)
      end
      Log.warn { "cublasSetMathMode failed (code #{result})" } unless result == 0
    end

    def destroy_handle(handle : LibCUBLAS::Handle)
      @@handle_pool_mutex.synchronize do
        if @@handle_pool.size < @@max_pool_size
          @@handle_pool << handle
          return
        end
      end

      LibCUBLAS.cublasDestroy_v2(handle)
    end

    # Cleanup all pooled handles
    def cleanup_handles
      @@handle_pool_mutex.synchronize do
        @@handle_pool.each do |handle|
          LibCUBLAS.cublasDestroy_v2(handle)
        end
        @@handle_pool.clear
      end
    end

    def gemm(handle : LibCUBLAS::Handle, a : Pointer(Float32), b : Pointer(Float32), c : Pointer(Float32),
             m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32)
      alpha = 1.0_f32
      beta = 0.0_f32
      LibCUBLAS.cublasSgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, lda,
        b, ldb,
        pointerof(beta), c, ldc)
    end

    # C = A^T * B with A and B column-major, which is what a row-major
    # C[M, N] = X[M, K] * W[N, K]^T needs: pass the dequantized weight as A (lda = K), the
    # activation as B (ldb = K), m = N_chunk, n = M, ldc = the FULL N so a chunk of output columns
    # lands in place inside the complete result.
    def gemm_tn(handle : LibCUBLAS::Handle, a : Pointer(Float32), b : Pointer(Float32), c : Pointer(Float32),
                m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32)
      alpha = 1.0_f32
      beta = 0.0_f32
      LibCUBLAS.cublasSgemm_v2(handle,
        Operation::T.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, lda,
        b, ldb,
        pointerof(beta), c, ldc)
    end

    def gemm_accumulate(handle : LibCUBLAS::Handle, a : Pointer(Float32), b : Pointer(Float32), c : Pointer(Float32),
                        m : Int32, n : Int32, k : Int32, lda : Int32, ldb : Int32, ldc : Int32, alpha : Float32, beta : Float32)
      LibCUBLAS.cublasSgemm_v2(handle,
        Operation::N.value, Operation::N.value,
        m, n, k,
        pointerof(alpha), a, lda,
        b, ldb,
        pointerof(beta), c, ldc)
    end

    def geam(handle : LibCUBLAS::Handle, a : Pointer(Float32), b : Pointer(Float32), c : Pointer(Float32),
             m : Int32, n : Int32, alpha : Float32, beta : Float32)
      LibCUBLAS.cublasSgeam(handle,
        Operation::N.value, Operation::N.value,
        m, n,
        pointerof(alpha), a, m,
        pointerof(beta), b, m,
        c, m)
    end

    def scal(handle : LibCUBLAS::Handle, x : Pointer(Float32), n : Int32, alpha : Float64)
      a32 = alpha.to_f32
      LibCUBLAS.cublasSscal_v2(handle, n, pointerof(a32), x, 1)
    end

    def axpy(handle : LibCUBLAS::Handle, alpha : Float64, x : Pointer(Float32), y : Pointer(Float32), n : Int32)
      a32 = alpha.to_f32
      LibCUBLAS.cublasSaxpy_v2(handle, n, pointerof(a32), x, 1, y, 1)
    end

    # Optional kernels implemented in src/shainet/native/cuda_kernels.cu
    # These methods dynamically load from libshainet_cuda_kernels.so when available
    @@kernels_handle : Pointer(Void) = Pointer(Void).null
    @@softmax_rows_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@dropout_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, UInt64, Void)?
    @@gather_rows_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Int32), Int32, Int32, Void)?
    @@slice_cols_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Void)?
    @@set_cols_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Void)?
    @@row_mean_var_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@layer_norm_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, Void)?
    @@layer_norm_backward_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, Void)?
    @@sum_cols_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@mul_row_vector_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@transpose_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@sigmoid_forward_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void)?
    @@apply_gradient_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void)?
    @@accumulate_bias_grad_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@row_sum_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@zero_matrix_proc : Proc(Pointer(Float32), Int32, Void)?
    @@fill_matrix_proc : Proc(Pointer(Float32), Float64, Int32, Void)?
    @@element_div_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void)?
    @@count_pairs_proc : Proc(Pointer(Int32), Pointer(Int32), Pointer(Int32), Int32, Int32, Pointer(Int32), Void)?
    @@relu_backward_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void)?
    @@softmax_backward_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@element_log_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Void)?
    @@cross_entropy_loss_grad_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@softmax_cross_entropy_label_proc : Proc(Pointer(Float32), Pointer(Int32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@gemm_q8_f32_proc : Proc(Pointer(Float32), Pointer(Int8), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemm_q4_f32_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gather_rows_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Int32), Int32, Int32, Void)?
    @@scatter_add_rows_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Int32), Pointer(Float32), Int32, Int32, Void)?
    @@gather_available : Bool? = nil
    @@mul_sigmoid_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Void)?
    @@gated_delta_rule_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Int32, Void)?
    @@rope_forward_rows_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Void)?
    @@head_rmsnorm_rows_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Float32, Void)?
    # GGUF k-quant GEMV
    @@gemv_iq3s_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_iq3xxs_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_iq4xs_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_iq4xs_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_iq3s_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_iq3xxs_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_q2k_lb_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_q2k_lb_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_iq2xxs_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_iq2xxs_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_iq2xs_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_iq2xs_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_iq2s_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_iq2s_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_iq1m_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_iq1m_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_iq1s_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_iq1s_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_q4k_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_q6k_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gemv_q5k_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_q5k_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@dequant_q4k_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@gdn_gates_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@short_conv_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@short_conv_silu3_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Void)?
    @@dequant_q6k_rows_proc : Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@add_bias_rows_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void)?
    @@pack_kv_heads_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Void)?
    @@prefill_attn_available : Bool? = nil
    @@kv_cache_append_f32_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Void)?
    @@kv_cache_append_f16_proc : Proc(Pointer(Float32), Pointer(UInt16), Pointer(UInt16), Int32, Int32, Int32, Int32, Int32, Void)?
    @@kv_f16_available : Bool? = nil
    @@swiglu_forward_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void)?
    @@swiglu_available : Bool? = nil
    @@rms_norm_forward_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Float32, Void)?
    @@add_inplace_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Void)?
    @@block_device_available : Bool? = nil
    @@rope_forward_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Void)?
    @@head_rmsnorm_proc : Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Float32, Void)?
    @@attn_device_available : Bool? = nil
    @@attention_kv_f32_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void)?
    @@attention_kv_f16_proc : Proc(Pointer(Float32), Pointer(UInt16), Pointer(UInt16), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void)?
    @@attention_split_kv_f32_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void)?
    @@attention_split_kv_f16_proc : Proc(Pointer(Float32), Pointer(UInt16), Pointer(UInt16), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void)?
    @@kv_cache_append_fp8_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(UInt8), Int32, Int32, Int32, Int32, Int32, Void)?
    @@attention_kv_fp8_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(UInt8), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void)?
    @@attention_split_kv_fp8_proc : Proc(Pointer(Float32), Pointer(UInt8), Pointer(UInt8), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void)?
    @@kv_fp8_available : Bool? = nil
    @@attention_split_ws_floats_proc : Proc(Int32, Int32, Int32, Int32, Int32)?

    def softmax_rows(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32, cols : Int32)
      # Validate inputs
      if dst.null? || src.null? || rows <= 0 || cols <= 0
        Log.error { "CUDA softmax_rows: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, src: #{src.null? ? "null" : "valid"}, rows: #{rows}, cols: #{cols}" }
        return
      end

      unless fn = @@softmax_rows_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_rows")
          unless sym.null?
            @@softmax_rows_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_rows_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, src, rows, cols)
      rescue ex
        Log.error { "CUDA Error in softmax_rows: #{ex}" }
        raise ex
      end
    end

    def dropout(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32, cols : Int32, drop_p : Float64, seed : UInt64)
      unless fn = @@dropout_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "dropout")
          unless sym.null?
            @@dropout_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, UInt64, Void).new(sym, Pointer(Void).null)
            fn = @@dropout_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols, drop_p, seed)
    end

    def gather_rows(dst : Pointer(Float32), src : Pointer(Float32), ids : Pointer(Int32), rows : Int32, cols : Int32)
      unless fn = @@gather_rows_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gather_rows")
          unless sym.null?
            @@gather_rows_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Int32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gather_rows_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, ids, rows, cols)
    end

    def slice_cols(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32, src_cols : Int32, start_col : Int32, len : Int32)
      # Validate inputs
      if dst.null? || src.null? || rows <= 0 || src_cols <= 0 || len <= 0 || start_col < 0 || (start_col + len) > src_cols
        Log.error { "CUDA slice_cols: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, src: #{src.null? ? "null" : "valid"}, rows: #{rows}, src_cols: #{src_cols}, start_col: #{start_col}, len: #{len}" }
        return
      end

      unless fn = @@slice_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "slice_cols")
          unless sym.null?
            @@slice_cols_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@slice_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, src, rows, src_cols, start_col, len)
      rescue ex
        Log.error { "CUDA Error in slice_cols: #{ex}" }
        raise ex
      end
    end

    def set_cols(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32, dst_cols : Int32, start_col : Int32, len : Int32)
      # Validate inputs
      if dst.null? || src.null? || rows <= 0 || dst_cols <= 0 || len <= 0 || start_col < 0 || (start_col + len) > dst_cols
        Log.error { "CUDA set_cols: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, src: #{src.null? ? "null" : "valid"}, rows: #{rows}, dst_cols: #{dst_cols}, start_col: #{start_col}, len: #{len}" }
        return
      end

      unless fn = @@set_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "set_cols")
          unless sym.null?
            @@set_cols_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@set_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, src, rows, dst_cols, start_col, len)
      rescue ex
        Log.error { "CUDA Error in set_cols: #{ex}" }
        raise ex
      end
    end

    # hidden = silu(gate) * up, computed on the device so the FFN's gate and up
    # projections do not have to be read back to the host to be combined.
    def swiglu_forward(hidden : Pointer(Float32), gate : Pointer(Float32), up : Pointer(Float32), size : Int32)
      unless fn = @@swiglu_forward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "swiglu_forward")
          unless sym.null?
            @@swiglu_forward_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@swiglu_forward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(hidden, gate, up, size)
    end

    # Gather the rows named by idx into a contiguous batch, on the device. One launch
    # replaces a per-row cudaMemcpy, which is what made batching by expert pay.
    def gather_rows(dst : Pointer(Float32), src : Pointer(Float32), idx : Pointer(Int32),
                    n : Int32, cols : Int32)
      unless fn = @@gather_rows_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gather_rows")
          unless sym.null?
            @@gather_rows_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Int32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gather_rows_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, idx, n, cols)
    end

    # dst[idx[r]] += w[r] * src[r] for the whole batch in one launch.
    def scatter_add_rows(dst : Pointer(Float32), src : Pointer(Float32), idx : Pointer(Int32),
                         w : Pointer(Float32), n : Int32, cols : Int32)
      unless fn = @@scatter_add_rows_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "scatter_add_rows")
          unless sym.null?
            @@scatter_add_rows_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Int32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@scatter_add_rows_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, idx, w, n, cols)
    end

    # Multi-row RoPE for prefill: `rows` tokens at consecutive positions from base_pos.
    # rot_dim is how many leading dimensions of each head are rotated; it defaults to head_dim,
    # which is ordinary full rotary. Qwen3.5 rotates only a quarter of a 256-wide head.
    def rope_forward_rows(x : Pointer(Float32), inv_freq : Pointer(Float32), base_pos : Int32,
                          rows : Int32, heads : Int32, head_dim : Int32, rot_dim : Int32 = 0)
      unless fn = @@rope_forward_rows_proc
        @@rope_forward_rows_proc = fn = load_kernel_proc("rope_forward_rows",
          Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, inv_freq, base_pos, rows, heads, head_dim, rot_dim)
    end

    # out[i] *= sigmoid(gate[i]), for Qwen3.5's attention output gate.
    def mul_sigmoid(out_ptr : Pointer(Float32), gate : Pointer(Float32), size : Int32)
      unless fn = @@mul_sigmoid_proc
        @@mul_sigmoid_proc = fn = load_kernel_proc("mul_sigmoid",
          Proc(Pointer(Float32), Pointer(Float32), Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(out_ptr, gate, size)
    end

    def mul_sigmoid_available? : Bool
      return false unless fully_available?
      if @@kernels_handle.null?
        @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
      end
      return false if @@kernels_handle.null?
      !LibC.dlsym(@@kernels_handle, "mul_sigmoid").null?
    end

    # Gated delta rule (Qwen3.5 linear attention) over a whole sequence in one launch.
    #
    # `state` is [nv, dk, dv] and is read AND written: pass the carried state for a continued
    # sequence, or a zeroed buffer for a fresh one. q and k are indexed by KEY head, so nk may be
    # smaller than nv (grouped-query sharing). L2 normalization of q/k and the q_scale are applied
    # inside the kernel.
    #
    # `k_head_tiled` selects which key head a value head reads, which is a property of the weight
    # layout: false is grouped (h // heads_per_k, the SafeTensors layout), true is tiled
    # (h % nk, the GGUF layout, because llama.cpp widens q/k with ggml_repeat and that tiles).
    def gated_delta_rule(q : Pointer(Float32), k : Pointer(Float32), v : Pointer(Float32),
                         alpha : Pointer(Float32), beta : Pointer(Float32),
                         state : Pointer(Float32), out_ptr : Pointer(Float32),
                         seq : Int32, nv : Int32, nk : Int32, dk : Int32, dv : Int32,
                         heads_per_k : Int32, q_scale : Float32, k_head_tiled : Bool = false)
      unless fn = @@gated_delta_rule_proc
        @@gated_delta_rule_proc = fn = load_kernel_proc("gated_delta_rule",
          Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
               Pointer(Float32), Pointer(Float32), Pointer(Float32),
               Int32, Int32, Int32, Int32, Int32, Int32, Float32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(q, k, v, alpha, beta, state, out_ptr, seq, nv, nk, dk, dv, heads_per_k, q_scale,
        k_head_tiled ? 1 : 0)
    end

    def gated_delta_rule_available? : Bool
      return false unless fully_available?
      if @@kernels_handle.null?
        @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
      end
      return false if @@kernels_handle.null?
      !LibC.dlsym(@@kernels_handle, "gated_delta_rule").null?
    end

    # Multi-row Qwen3 QK-norm for prefill.
    def head_rmsnorm_rows(x : Pointer(Float32), gamma : Pointer(Float32), rows : Int32,
                          heads : Int32, head_dim : Int32, eps : Float32)
      unless fn = @@head_rmsnorm_rows_proc
        @@head_rmsnorm_rows_proc = fn = load_kernel_proc("head_rmsnorm_rows",
          Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Float32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, gamma, rows, heads, head_dim, eps)
    end

    # x[r, c] += bias[c], broadcast over rows.
    def add_bias_rows(x : Pointer(Float32), bias : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@add_bias_rows_proc
        @@add_bias_rows_proc = fn = load_kernel_proc("add_bias_rows",
          Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, bias, rows, cols)
    end

    # GGUF k-quant GEMV: y[M, N] = x[M, K] * dequant(W[N, K]).
    # W is in raw Q4_K format (144-byte blocks, K must be a multiple of 256).
    def gemv_iq3s(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                  m : Int32, n : Int32, k : Int32)
      unless fn = @@gemv_iq3s_proc
        @@gemv_iq3s_proc = fn = load_kernel_proc("gemv_iq3s",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k)
    end

    def gemv_iq3xxs(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                    m : Int32, n : Int32, k : Int32)
      unless fn = @@gemv_iq3xxs_proc
        @@gemv_iq3xxs_proc = fn = load_kernel_proc("gemv_iq3xxs",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k)
    end

    def gemv_iq4xs(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                   m : Int32, n : Int32, k : Int32)
      unless fn = @@gemv_iq4xs_proc
        @@gemv_iq4xs_proc = fn = load_kernel_proc("gemv_iq4xs",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k)
    end

    def gemv_q4k(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                 m : Int32, n : Int32, k : Int32)
      unless fn = @@gemv_q4k_proc
        @@gemv_q4k_proc = fn = load_kernel_proc("gemv_q4k",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k)
    end

    # GGUF k-quant GEMV for Q5_K format (176-byte blocks, K must be a multiple of 256).
    def gemv_q5k(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                 m : Int32, n : Int32, k : Int32)
      unless fn = @@gemv_q5k_proc
        @@gemv_q5k_proc = fn = load_kernel_proc("gemv_q5k",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k)
    end

    def dequant_q5k_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                         row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_q5k_rows_proc
        @@dequant_q5k_rows_proc = fn = load_kernel_proc("dequant_q5k_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    # GGUF k-quant GEMV for Q6_K format (210-byte blocks, K must be a multiple of 256).
    def gemv_q6k(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                 m : Int32, n : Int32, k : Int32)
      unless fn = @@gemv_q6k_proc
        @@gemv_q6k_proc = fn = load_kernel_proc("gemv_q6k",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k)
    end

    # Dequantize rows [row0, row0 + n_rows) of a k-quant weight into an fp32 [n_rows, K]
    # row-major buffer, so a batched prefill can run one cuBLAS GEMM per chunk instead of a GEMV
    # per token. The GEMV kernels reuse nothing across tokens; this reads the weight once.
    def dequant_iq4xs_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                           row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_iq4xs_rows_proc
        @@dequant_iq4xs_rows_proc = fn = load_kernel_proc("dequant_iq4xs_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def dequant_iq3s_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                          row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_iq3s_rows_proc
        @@dequant_iq3s_rows_proc = fn = load_kernel_proc("dequant_iq3s_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def dequant_iq3xxs_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                            row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_iq3xxs_rows_proc
        @@dequant_iq3xxs_rows_proc = fn = load_kernel_proc("dequant_iq3xxs_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def gemv_q2k_lb(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                    m : Int32, n : Int32, k_dim : Int32)
      unless fn = @@gemv_q2k_lb_proc
        @@gemv_q2k_lb_proc = fn = load_kernel_proc("gemv_q2k_lb",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k_dim)
    end

    def dequant_q2k_lb_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                            row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_q2k_lb_rows_proc
        @@dequant_q2k_lb_rows_proc = fn = load_kernel_proc("dequant_q2k_lb_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def gemv_iq2xxs(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                    m : Int32, n : Int32, k_dim : Int32)
      unless fn = @@gemv_iq2xxs_proc
        @@gemv_iq2xxs_proc = fn = load_kernel_proc("gemv_iq2xxs",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k_dim)
    end

    def dequant_iq2xxs_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                            row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_iq2xxs_rows_proc
        @@dequant_iq2xxs_rows_proc = fn = load_kernel_proc("dequant_iq2xxs_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def gemv_iq2xs(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                   m : Int32, n : Int32, k_dim : Int32)
      unless fn = @@gemv_iq2xs_proc
        @@gemv_iq2xs_proc = fn = load_kernel_proc("gemv_iq2xs",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k_dim)
    end

    def dequant_iq2xs_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                           row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_iq2xs_rows_proc
        @@dequant_iq2xs_rows_proc = fn = load_kernel_proc("dequant_iq2xs_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def gemv_iq2s(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                  m : Int32, n : Int32, k_dim : Int32)
      unless fn = @@gemv_iq2s_proc
        @@gemv_iq2s_proc = fn = load_kernel_proc("gemv_iq2s",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k_dim)
    end

    def dequant_iq2s_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                          row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_iq2s_rows_proc
        @@dequant_iq2s_rows_proc = fn = load_kernel_proc("dequant_iq2s_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def gemv_iq1m(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                  m : Int32, n : Int32, k_dim : Int32)
      unless fn = @@gemv_iq1m_proc
        @@gemv_iq1m_proc = fn = load_kernel_proc("gemv_iq1m",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k_dim)
    end

    def dequant_iq1m_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                          row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_iq1m_rows_proc
        @@dequant_iq1m_rows_proc = fn = load_kernel_proc("dequant_iq1m_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def gemv_iq1s(x : Pointer(Float32), w : Pointer(UInt8), y : Pointer(Float32),
                  m : Int32, n : Int32, k_dim : Int32)
      unless fn = @@gemv_iq1s_proc
        @@gemv_iq1s_proc = fn = load_kernel_proc("gemv_iq1s",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, w, y, m, n, k_dim)
    end

    def dequant_iq1s_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                          row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_iq1s_rows_proc
        @@dequant_iq1s_rows_proc = fn = load_kernel_proc("dequant_iq1s_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def dequant_q4k_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                         row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_q4k_rows_proc
        @@dequant_q4k_rows_proc = fn = load_kernel_proc("dequant_q4k_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def dequant_q6k_rows(w : Pointer(UInt8), dst : Pointer(Float32),
                         row0 : Int32, n_rows : Int32, k : Int32)
      unless fn = @@dequant_q6k_rows_proc
        @@dequant_q6k_rows_proc = fn = load_kernel_proc("dequant_q6k_rows",
          Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(w, dst, row0, n_rows, k)
    end

    def dequant_k_rows_available? : Bool
      return false unless kernels_available?
      !load_kernel_proc("dequant_q4k_rows",
        Proc(Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void)).nil?
    end

    # Gated DeltaNet per-head gates:
    #   alpha[t,h] = exp(-exp(a_log[h]) * softplus(a_proj[t,h] + dt_bias[h]))
    #   beta [t,h] = sigmoid(b_proj[t,h])
    def gdn_gates(alpha : Pointer(Float32), beta : Pointer(Float32),
                  a_proj : Pointer(Float32), b_proj : Pointer(Float32),
                  a_log : Pointer(Float32), dt_bias : Pointer(Float32),
                  seq : Int32, heads : Int32)
      unless fn = @@gdn_gates_proc
        @@gdn_gates_proc = fn = load_kernel_proc("gdn_gates",
          Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
               Pointer(Float32), Pointer(Float32), Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(alpha, beta, a_proj, b_proj, a_log, dt_bias, seq, heads)
    end

    @@gdn_ab_proj_proc : Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32)?
    @@gdn_ab_proj_checked = false

    # Single-token alpha/beta projections of a Gated DeltaNet layer in one pass (see the kernel).
    # Returns false when the kernel is missing from the loaded library or could not run, so the
    # caller falls back to cuBLAS.
    def gdn_ab_proj(x : Pointer(Float32), wa : Pointer(Float32), wb : Pointer(Float32),
                    ap : Pointer(Float32), bp : Pointer(Float32), d_model : Int32, heads : Int32) : Bool
      unless @@gdn_ab_proj_checked
        @@gdn_ab_proj_checked = true
        @@gdn_ab_proj_proc = load_kernel_proc("gdn_ab_proj",
          Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32))
      end
      fn = @@gdn_ab_proj_proc
      return false unless fn
      fn.call(x, wa, wb, ap, bp, d_model, heads) != 0
    end

    # Short causal depthwise convolution, weight[c, 0] being the current position. `state` carries
    # kernel-1 past positions and is updated in place, so a decode step continues the sequence.
    def short_conv(dst : Pointer(Float32), src : Pointer(Float32), state : Pointer(Float32),
                   w : Pointer(Float32), seq : Int32, channels : Int32, kernel : Int32)
      unless fn = @@short_conv_proc
        @@short_conv_proc = fn = load_kernel_proc("short_conv",
          Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
               Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, state, w, seq, channels, kernel)
    end

    # Causal conv (+ optional SiLU) for q, k and v in ONE launch pair instead of nine kernels.
    #
    # The unfused stage cost nine launches per GDN layer: three short_conv (each itself a conv plus a
    # state-roll launch) and three mul_sigmoid. Across 48 GDN layers that was roughly a third of the
    # ~1280 launches a generated token issues, and batch=1 decode is latency-bound, so those launches
    # are the cost rather than the arithmetic. Returns false when the kernel is unavailable so the
    # caller keeps the unfused path.
    def short_conv_silu3(d0 : Pointer(Float32), s0 : Pointer(Float32), st0 : Pointer(Float32), w0 : Pointer(Float32),
                         d1 : Pointer(Float32), s1 : Pointer(Float32), st1 : Pointer(Float32), w1 : Pointer(Float32),
                         d2 : Pointer(Float32), s2 : Pointer(Float32), st2 : Pointer(Float32), w2 : Pointer(Float32),
                         seq : Int32, ch0 : Int32, ch1 : Int32, ch2 : Int32,
                         kernel : Int32, apply_silu : Bool) : Bool
      unless fn = @@short_conv_silu3_proc
        @@short_conv_silu3_proc = fn = load_kernel_proc("short_conv_silu3",
          Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
               Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
               Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
               Int32, Int32, Int32, Int32, Int32, Int32, Void))
      end
      return false unless fn
      fn.call(d0, s0, st0, w0, d1, s1, st1, w1, d2, s2, st2, w2,
        seq, ch0, ch1, ch2, kernel, apply_silu ? 1 : 0)
      true
    end

    def short_conv_silu3_available? : Bool
      return false unless kernels_available?
      !load_kernel_proc("short_conv_silu3",
        Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
             Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
             Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
             Int32, Int32, Int32, Int32, Int32, Int32, Void)).nil?
    end

    def gdn_mixer_kernels_available? : Bool
      return false unless kernels_available?
      return false if load_kernel_proc("gdn_gates",
                        Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
                             Pointer(Float32), Pointer(Float32), Int32, Int32, Void)).nil?
      !load_kernel_proc("short_conv",
        Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
             Int32, Int32, Int32, Void)).nil?
    end

    def pack_kv_heads(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32,
                      kv_heads : Int32, head_dim : Int32)
      unless fn = @@pack_kv_heads_proc
        @@pack_kv_heads_proc = fn = load_kernel_proc("pack_kv_heads",
          Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, kv_heads, head_dim)
    end

    # dlopen/dlsym once per symbol. The four multi-row prefill kernels share this rather
    # than repeating the same twelve lines each.
    private def load_kernel_proc(name : String, type : T.class) : T? forall T
      if @@kernels_handle.null?
        @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
      end
      return if @@kernels_handle.null?
      sym = LibC.dlsym(@@kernels_handle, name)
      return if sym.null?
      T.new(sym, Pointer(Void).null)
    end

    # False when the loaded .so predates the multi-row prefill kernels, so the caller
    # falls back to the host path instead of raising.
    def prefill_attn_kernels_available? : Bool
      avail = @@prefill_attn_available
      return avail unless avail.nil?
      result = begin
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        if @@kernels_handle.null?
          false
        else
          ["rope_forward_rows", "head_rmsnorm_rows", "add_bias_rows", "pack_kv_heads"].all? do |n|
            !LibC.dlsym(@@kernels_handle, n).null?
          end
        end
      end
      @@prefill_attn_available = result
      result
    end

    # False when the loaded .so predates the gather/scatter kernels, so the batched
    # prefill path can decline instead of raising.
    def gather_kernels_available? : Bool
      avail = @@gather_available
      return avail unless avail.nil?
      result = begin
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        if @@kernels_handle.null?
          false
        else
          !LibC.dlsym(@@kernels_handle, "gather_rows").null? &&
          !LibC.dlsym(@@kernels_handle, "scatter_add_rows").null?
        end
      end
      @@gather_available = result
      result
    end

    # False when the loaded .so predates the fused SwiGLU kernel, so callers can
    # fall back to the host path instead of raising.
    def swiglu_kernel_available? : Bool
      avail = @@swiglu_available
      return avail unless avail.nil?
      result = begin
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        @@kernels_handle.null? ? false : !LibC.dlsym(@@kernels_handle, "swiglu_forward").null?
      rescue
        false
      end
      @@swiglu_available = result
      result
    end

    # out = x / sqrt(mean(x^2) + eps) * gamma, computed on the device so a norm no
    # longer breaks a device-resident chain.
    def rms_norm_forward(dst : Pointer(Float32), src : Pointer(Float32), gamma : Pointer(Float32),
                         rows : Int32, cols : Int32, eps : Float32)
      unless fn = @@rms_norm_forward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "rms_norm_forward")
          unless sym.null?
            @@rms_norm_forward_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Float32, Void).new(sym, Pointer(Void).null)
            fn = @@rms_norm_forward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, gamma, rows, cols, eps)
    end

    # dst += src elementwise, for the block's residual adds.
    def add_inplace(dst : Pointer(Float32), src : Pointer(Float32), size : Int32)
      unless fn = @@add_inplace_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "add_inplace")
          unless sym.null?
            @@add_inplace_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@add_inplace_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, size)
    end

    # False when the loaded .so predates the block-residency kernels, so callers fall
    # back to the host path instead of raising.
    def block_device_kernels_available? : Bool
      avail = @@block_device_available
      return avail unless avail.nil?
      result = begin
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        if @@kernels_handle.null?
          false
        else
          !LibC.dlsym(@@kernels_handle, "rms_norm_forward").null? &&
          !LibC.dlsym(@@kernels_handle, "add_inplace").null?
        end
      rescue
        false
      end
      @@block_device_available = result
      result
    end

    # Rotary position embedding in place on one token's row, HF half-split.
    def rope_forward(x : Pointer(Float32), inv_freq : Pointer(Float32), pos : Int32,
                     heads : Int32, head_dim : Int32, rot_dim : Int32 = 0)
      unless fn = @@rope_forward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "rope_forward")
          unless sym.null?
            @@rope_forward_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@rope_forward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, inv_freq, pos, heads, head_dim, rot_dim)
    end

    # Qwen3 QK-norm: RMSNorm per head slice, in place.
    def head_rmsnorm(x : Pointer(Float32), gamma : Pointer(Float32), heads : Int32,
                     head_dim : Int32, eps : Float32)
      unless fn = @@head_rmsnorm_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "head_rmsnorm")
          unless sym.null?
            @@head_rmsnorm_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Float32, Void).new(sym, Pointer(Void).null)
            fn = @@head_rmsnorm_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(x, gamma, heads, head_dim, eps)
    end

    # False when the loaded .so predates the device-attention kernels.
    def attention_device_kernels_available? : Bool
      avail = @@attn_device_available
      return avail unless avail.nil?
      result = begin
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        if @@kernels_handle.null?
          false
        else
          !LibC.dlsym(@@kernels_handle, "rope_forward").null? &&
          !LibC.dlsym(@@kernels_handle, "head_rmsnorm").null?
        end
      rescue
        false
      end
      @@attn_device_available = result
      result
    end

    def row_mean_var(src : Pointer(Float32), mean : Pointer(Float32), var : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@row_mean_var_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "row_mean_var")
          unless sym.null?
            @@row_mean_var_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@row_mean_var_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(src, mean, var, rows, cols)
    end

    def layer_norm(dst : Pointer(Float32), src : Pointer(Float32), mean : Pointer(Float32), var : Pointer(Float32), rows : Int32, cols : Int32, eps : Float64)
      unless fn = @@layer_norm_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "apply_layer_norm") # Note: the actual function name is apply_layer_norm
          unless sym.null?
            @@layer_norm_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, Void).new(sym, Pointer(Void).null)
            fn = @@layer_norm_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, mean, var, rows, cols, eps)
    end

    def layer_norm_backward(d_x : Pointer(Float32), d_gamma : Pointer(Float32), d_beta : Pointer(Float32),
                            d_out : Pointer(Float32), x : Pointer(Float32), gamma : Pointer(Float32),
                            mean : Pointer(Float32), var : Pointer(Float32), norm : Pointer(Float32),
                            rows : Int32, cols : Int32, eps : Float64)
      unless fn = @@layer_norm_backward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "layer_norm_backward")
          unless sym.null?
            @@layer_norm_backward_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, Void).new(sym, Pointer(Void).null)
            fn = @@layer_norm_backward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(d_x, d_gamma, d_beta, d_out, x, gamma, mean, var, norm, rows, cols, eps)
    end

    def sum_cols(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@sum_cols_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "sum_cols")
          unless sym.null?
            @@sum_cols_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@sum_cols_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(dst, src, rows, cols)
    end

    def mul_row_vector(matrix : Pointer(Float32), vec : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@mul_row_vector_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "mul_row_vector")
          unless sym.null?
            @@mul_row_vector_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@mul_row_vector_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(matrix, vec, rows, cols)
    end

    def transpose(output : Pointer(Float32), input : Pointer(Float32), rows : Int32, cols : Int32)
      # Validate inputs
      if output.null? || input.null? || rows <= 0 || cols <= 0
        Log.error { "CUDA transpose: invalid parameters - output: #{output.null? ? "null" : "valid"}, input: #{input.null? ? "null" : "valid"}, rows: #{rows}, cols: #{cols}" }
        return
      end

      unless fn = @@transpose_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "transpose")
          unless sym.null?
            @@transpose_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@transpose_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(output, input, rows, cols)
      rescue ex
        Log.error { "CUDA Error in transpose: #{ex}, output=#{output.address}, input=#{input.address}, rows=#{rows}, cols=#{cols}" }
        Log.warn { "Falling back to CPU transpose due to GPU error" }
        # GPU operation failed - let the caller handle the fallback
        raise ex
      end
    end

    def sigmoid_forward(activations : Pointer(Float32), derivatives : Pointer(Float32), linear : Pointer(Float32), size : Int32)
      unless fn = @@sigmoid_forward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "sigmoid_forward")
          unless sym.null?
            @@sigmoid_forward_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@sigmoid_forward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(activations, derivatives, linear, size)
    end

    def apply_gradient(local_grad : Pointer(Float32), grad : Pointer(Float32), derivatives : Pointer(Float32), size : Int32)
      unless fn = @@apply_gradient_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "apply_gradient")
          unless sym.null?
            @@apply_gradient_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@apply_gradient_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(local_grad, grad, derivatives, size)
    end

    def accumulate_bias_grad(bias_grad : Pointer(Float32), local_grad : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@accumulate_bias_grad_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "accumulate_bias_grad")
          unless sym.null?
            @@accumulate_bias_grad_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@accumulate_bias_grad_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(bias_grad, local_grad, rows, cols)
    end

    def zero_matrix(matrix : Pointer(Float32), size : Int32)
      # Validate inputs
      if matrix.null? || size <= 0
        Log.error { "CUDA zero_matrix: invalid parameters - matrix: #{matrix.null? ? "null" : "valid"}, size: #{size}" }
        return
      end

      # Add detailed logging for zero_matrix operations

      unless fn = @@zero_matrix_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "zero_matrix")
          unless sym.null?
            @@zero_matrix_proc = Proc(Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@zero_matrix_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(matrix, size)
      rescue ex
        Log.error { "CUDA Error in zero_matrix: #{ex}, matrix=#{matrix.address}, size=#{size}" }
        raise ex
      end
    end

    def fill_matrix(matrix : Pointer(Float32), value : Float64, size : Int32)
      if matrix.null? || size <= 0
        Log.error { "CUDA fill_matrix: invalid parameters - matrix: #{matrix.null? ? "null" : "valid"}, size: #{size}, value: #{value}" }
        return
      end

      unless fn = @@fill_matrix_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "fill_matrix")
          unless sym.null?
            @@fill_matrix_proc = Proc(Pointer(Float32), Float64, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@fill_matrix_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(matrix, value, size)
      rescue ex
        Log.error { "CUDA Error in fill_matrix: #{ex}, matrix=#{matrix.address}, size=#{size}, value=#{value}" }
        raise ex
      end
    end

    def element_div(dst : Pointer(Float32), a : Pointer(Float32), b : Pointer(Float32), size : Int32)
      if dst.null? || a.null? || b.null? || size <= 0
        Log.error { "CUDA element_div: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, a: #{a.null? ? "null" : "valid"}, b: #{b.null? ? "null" : "valid"}, size: #{size}" }
        return
      end

      unless fn = @@element_div_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "element_div")
          unless sym.null?
            @@element_div_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@element_div_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, a, b, size)
      rescue ex
        Log.error { "CUDA Error in element_div: #{ex}" }
        raise ex
      end
    end

    # In-place element-wise ReLU on GPU memory. This fallback implementation
    # copies the data to the host, applies ReLU and writes the result back. It
    # avoids additional synchronization logic in the caller while still keeping
    # the computation on the GPU when proper kernels are available.
    def relu(ptr : Pointer(Float32), len : Int32)
      host = Array(Float32).new(len, 0.0_f32)
      bytes = (len * 4).to_u64
      memcpy(host.to_unsafe.as(Pointer(Void)), ptr.as(Pointer(Void)), bytes, MemcpyKind::DeviceToHost)
      len.times do |i|
        v = host[i]
        host[i] = v > 0 ? v : 0.0_f32
      end
      memcpy(ptr.as(Pointer(Void)), host.to_unsafe.as(Pointer(Void)), bytes, MemcpyKind::HostToDevice)
    end

    # Add a bias row vector to each row of a matrix in GPU memory.
    # Uses multiple AXPY operations instead of DGER to avoid row-major/column-major issues.
    def add_bias(mat : Pointer(Float32), bias : Pointer(Float32), rows : Int32, cols : Int32)
      handle = create_handle

      # Add bias to each row using AXPY: row_i += 1.0 * bias
      rows.times do |i|
        row_start = mat + (i * cols) # Pointer to start of row i
        axpy(handle, 1.0, bias, row_start, cols)
      end

      destroy_handle(handle)
    end

    # Accumulate the sum over rows of a matrix into an existing row vector.
    # Performs: dst += ones^T * src
    #
    # cuBLAS assumes column-major layout which doesn't match the row-major
    # storage used by `CudaMatrix`.  The previous implementation tried to use a
    # GEMM with an implicit ones vector but produced incorrect results because
    # of the layout mismatch.  Instead, use repeated AXPY operations on each
    # row which works regardless of the underlying memory layout and avoids
    # creating temporary matrices.
    def row_sum(dst : Pointer(Float32), src : Pointer(Float32), rows : Int32, cols : Int32)
      unless fn = @@row_sum_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "row_sum")
          unless sym.null?
            @@row_sum_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@row_sum_proc
          end
        end
      end

      if fn
        begin
          fn.call(dst, src, rows, cols)
          return
        rescue ex
          Log.error { "CUDA Error in row_sum: #{ex}" }
        end
      end

      handle = create_handle
      rows.times do |i|
        row_start = src + (i * cols)
        axpy(handle, 1.0, row_start, dst, cols)
      end
      destroy_handle(handle)
    end

    # Count token pairs using a custom CUDA kernel when available.
    # C signature: count_token_pairs(a, b, freq, pair_count, vocab_size, counts)
    def count_token_pairs(counts : Pointer(Int32), a : Pointer(Int32), b : Pointer(Int32), freqs : Pointer(Int32), pair_count : Int32, vocab : Int32)
      unless fn = @@count_pairs_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "count_token_pairs")
          unless sym.null?
            @@count_pairs_proc = Proc(Pointer(Int32), Pointer(Int32), Pointer(Int32), Int32, Int32, Pointer(Int32), Void).new(sym, Pointer(Void).null)
            fn = @@count_pairs_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn
      fn.call(a, b, freqs, pair_count, vocab, counts)
    end

    # Check if both CUDA runtime and custom kernels are available
    def fully_available?
      available? && kernels_available?
    end

    # Q8_0-style quantized matmul: y[M,N] = x[M,K] * dequant(W).
    # q: int8 weights laid out [N, K]; scales: fp32 [N, ceil(K/32)].
    def gemm_q8_f32(x : Pointer(Float32), q : Pointer(Int8), scales : Pointer(Float32),
                    y : Pointer(Float32), m : Int32, n : Int32, k : Int32)
      unless fn = @@gemm_q8_f32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gemm_q8_f32")
          unless sym.null?
            @@gemm_q8_f32_proc = Proc(Pointer(Float32), Pointer(Int8), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gemm_q8_f32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(x, q, scales, y, m, n, k)
      rescue ex
        Log.error { "CUDA Error in gemm_q8_f32: #{ex}" }
        raise ex
      end
    end

    # Q4_0-style 4-bit GEMM. q holds two nibbles per byte along K, laid out
    # [N, ceil(K/2)]; scales are fp32 [N, ceil(K/32)] (one per 32-element block).
    def gemm_q4_f32(x : Pointer(Float32), q : Pointer(UInt8), d : Pointer(Float32),
                    sub : Pointer(UInt8), y : Pointer(Float32), m : Int32, n : Int32, k : Int32)
      unless fn = @@gemm_q4_f32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "gemm_q4_f32")
          unless sym.null?
            @@gemm_q4_f32_proc = Proc(Pointer(Float32), Pointer(UInt8), Pointer(Float32), Pointer(UInt8), Pointer(Float32), Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@gemm_q4_f32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(x, q, d, sub, y, m, n, k)
      rescue ex
        Log.error { "CUDA Error in gemm_q4_f32: #{ex}" }
        raise ex
      end
    end

    # Scatter newly appended KV rows from a staging buffer into the device
    # KV cache laid out [num_kv_heads, capacity, head_dim]. The staging buffer
    # holds K chunks (kv_head-major, each chunk new_tokens*head_dim floats)
    # followed by V chunks in the same layout.
    def kv_cache_append_f32(staging : Pointer(Float32), kc : Pointer(Float32), vc : Pointer(Float32),
                            new_tokens : Int32, start_pos : Int32, num_kv_heads : Int32,
                            head_dim : Int32, capacity : Int32)
      unless fn = @@kv_cache_append_f32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "kv_cache_append_f32")
          unless sym.null?
            @@kv_cache_append_f32_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@kv_cache_append_f32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(staging, kc, vc, new_tokens, start_pos, num_kv_heads, head_dim, capacity)
      rescue ex
        Log.error { "CUDA Error in kv_cache_append_f32: #{ex}" }
        raise ex
      end
    end

    # Causal attention over the device KV cache (one block per head/token).
    # q/out: [new_tokens, num_heads*head_dim] row-major, q already RoPE'd.
    # ws: scratch of at least num_heads*new_tokens*(start_pos+new_tokens) floats.
    def attention_kv_f32(q : Pointer(Float32), kc : Pointer(Float32), vc : Pointer(Float32),
                         out_ptr : Pointer(Float32), ws : Pointer(Float32),
                         new_tokens : Int32, start_pos : Int32, num_heads : Int32,
                         heads_per_kv : Int32, head_dim : Int32, capacity : Int32,
                         scale : Float32)
      unless fn = @@attention_kv_f32_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "attention_kv_f32")
          unless sym.null?
            @@attention_kv_f32_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void).new(sym, Pointer(Void).null)
            fn = @@attention_kv_f32_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(q, kc, vc, out_ptr, ws, new_tokens, start_pos, num_heads, heads_per_kv, head_dim, capacity, scale)
      rescue ex
        Log.error { "CUDA Error in attention_kv_f32: #{ex}" }
        raise ex
      end
    end

    # fp16 KV cache variants. The cache buffers are __half on the device; Crystal
    # has no native half type, so they travel as UInt16 pointers. Staging, q, out
    # and ws all stay fp32 — only the stored cache precision changes.
    def kv_cache_append_f16(staging : Pointer(Float32), kc : Pointer(UInt16), vc : Pointer(UInt16),
                            new_tokens : Int32, start_pos : Int32, num_kv_heads : Int32,
                            head_dim : Int32, capacity : Int32)
      unless fn = @@kv_cache_append_f16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "kv_cache_append_f16")
          unless sym.null?
            @@kv_cache_append_f16_proc = Proc(Pointer(Float32), Pointer(UInt16), Pointer(UInt16), Int32, Int32, Int32, Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@kv_cache_append_f16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(staging, kc, vc, new_tokens, start_pos, num_kv_heads, head_dim, capacity)
      rescue ex
        Log.error { "CUDA Error in kv_cache_append_f16: #{ex}" }
        raise ex
      end
    end

    # Split-KV ("flash-decoding") attention. Same contract as attention_kv_* but the grid also spans
    # the KV length, so decode occupies the whole card instead of num_heads blocks, and the scores
    # stay in shared memory rather than a global workspace. Exact, not approximate: the per-split
    # partials are merged with the standard max-rescale.
    #
    # `ws` must hold attention_split_ws_floats(...) floats. Returns false when the kernels are
    # missing so the caller keeps the single-block path.
    def attention_split_kv_f32(q : Pointer(Float32), kc : Pointer(Float32), vc : Pointer(Float32),
                               out_ptr : Pointer(Float32), ws : Pointer(Float32),
                               new_tokens : Int32, start_pos : Int32, num_heads : Int32,
                               heads_per_kv : Int32, head_dim : Int32, capacity : Int32,
                               scale : Float32) : Bool
      unless fn = @@attention_split_kv_f32_proc
        @@attention_split_kv_f32_proc = fn = load_kernel_proc("attention_split_kv_f32",
          Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32),
               Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void))
      end
      return false unless fn
      fn.call(q, kc, vc, out_ptr, ws, new_tokens, start_pos, num_heads,
        heads_per_kv, head_dim, capacity, scale)
      true
    end

    def attention_split_kv_f16(q : Pointer(Float32), kc : Pointer(UInt16), vc : Pointer(UInt16),
                               out_ptr : Pointer(Float32), ws : Pointer(Float32),
                               new_tokens : Int32, start_pos : Int32, num_heads : Int32,
                               heads_per_kv : Int32, head_dim : Int32, capacity : Int32,
                               scale : Float32) : Bool
      unless fn = @@attention_split_kv_f16_proc
        @@attention_split_kv_f16_proc = fn = load_kernel_proc("attention_split_kv_f16",
          Proc(Pointer(Float32), Pointer(UInt16), Pointer(UInt16), Pointer(Float32),
               Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void))
      end
      return false unless fn
      fn.call(q, kc, vc, out_ptr, ws, new_tokens, start_pos, num_heads,
        heads_per_kv, head_dim, capacity, scale)
      true
    end

    # Floats of partial-buffer space the split path needs. Kept on the kernel side so the split size
    # is defined in exactly one place.
    def attention_split_ws_floats(new_tokens : Int32, num_heads : Int32, head_dim : Int32,
                                  total_len : Int32) : Int32
      unless fn = @@attention_split_ws_floats_proc
        @@attention_split_ws_floats_proc = fn = load_kernel_proc("attention_split_ws_floats",
          Proc(Int32, Int32, Int32, Int32, Int32))
      end
      return 0 unless fn
      fn.call(new_tokens, num_heads, head_dim, total_len)
    end

    def attention_split_kv_available? : Bool
      return false unless kernels_available?
      !load_kernel_proc("attention_split_ws_floats", Proc(Int32, Int32, Int32, Int32, Int32)).nil?
    end

    def attention_kv_f16(q : Pointer(Float32), kc : Pointer(UInt16), vc : Pointer(UInt16),
                         out_ptr : Pointer(Float32), ws : Pointer(Float32),
                         new_tokens : Int32, start_pos : Int32, num_heads : Int32,
                         heads_per_kv : Int32, head_dim : Int32, capacity : Int32,
                         scale : Float32)
      unless fn = @@attention_kv_f16_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "attention_kv_f16")
          unless sym.null?
            @@attention_kv_f16_proc = Proc(Pointer(Float32), Pointer(UInt16), Pointer(UInt16), Pointer(Float32), Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void).new(sym, Pointer(Void).null)
            fn = @@attention_kv_f16_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(q, kc, vc, out_ptr, ws, new_tokens, start_pos, num_heads, heads_per_kv, head_dim, capacity, scale)
      rescue ex
        Log.error { "CUDA Error in attention_kv_f16: #{ex}" }
        raise ex
      end
    end

    # Reports whether the fp16 KV kernels are present in the loaded kernel
    # library. Older prebuilt .so files predate them, so the KV cache falls back
    # to fp32 rather than failing when they are missing.
    def kv_f16_kernels_available? : Bool
      avail = @@kv_f16_available
      return avail unless avail.nil?
      result = begin
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        if @@kernels_handle.null?
          false
        else
          !LibC.dlsym(@@kernels_handle, "kv_cache_append_f16").null? &&
          !LibC.dlsym(@@kernels_handle, "attention_kv_f16").null?
        end
      rescue
        false
      end
      @@kv_f16_available = result
      result
    end

    # fp8 (e4m3) KV cache variants: one byte per cached element, half of fp16. Same contracts as the
    # f16 entry points; the buffers travel as byte pointers.
    def kv_cache_append_fp8(staging : Pointer(Float32), kc : Pointer(UInt8), vc : Pointer(UInt8),
                            new_tokens : Int32, start_pos : Int32, num_kv_heads : Int32,
                            head_dim : Int32, capacity : Int32)
      unless fn = @@kv_cache_append_fp8_proc
        @@kv_cache_append_fp8_proc = fn = load_kernel_proc("kv_cache_append_fp8",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(UInt8), Int32, Int32, Int32, Int32, Int32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(staging, kc, vc, new_tokens, start_pos, num_kv_heads, head_dim, capacity)
    end

    def attention_kv_fp8(q : Pointer(Float32), kc : Pointer(UInt8), vc : Pointer(UInt8),
                         out_ptr : Pointer(Float32), ws : Pointer(Float32),
                         new_tokens : Int32, start_pos : Int32, num_heads : Int32,
                         heads_per_kv : Int32, head_dim : Int32, capacity : Int32,
                         scale : Float32)
      unless fn = @@attention_kv_fp8_proc
        @@attention_kv_fp8_proc = fn = load_kernel_proc("attention_kv_fp8",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(UInt8), Pointer(Float32),
               Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void))
      end
      raise "CUDA kernels not available" unless fn
      fn.call(q, kc, vc, out_ptr, ws, new_tokens, start_pos, num_heads, heads_per_kv, head_dim, capacity, scale)
    end

    def attention_split_kv_fp8(q : Pointer(Float32), kc : Pointer(UInt8), vc : Pointer(UInt8),
                               out_ptr : Pointer(Float32), ws : Pointer(Float32),
                               new_tokens : Int32, start_pos : Int32, num_heads : Int32,
                               heads_per_kv : Int32, head_dim : Int32, capacity : Int32,
                               scale : Float32) : Bool
      unless fn = @@attention_split_kv_fp8_proc
        @@attention_split_kv_fp8_proc = fn = load_kernel_proc("attention_split_kv_fp8",
          Proc(Pointer(Float32), Pointer(UInt8), Pointer(UInt8), Pointer(Float32),
               Pointer(Float32), Int32, Int32, Int32, Int32, Int32, Int32, Float32, Void))
      end
      return false unless fn
      fn.call(q, kc, vc, out_ptr, ws, new_tokens, start_pos, num_heads,
        heads_per_kv, head_dim, capacity, scale)
      true
    end

    # Whether the loaded kernel library has the fp8 KV kernels. A prebuilt .so from before them
    # makes an fp8 request fall back to fp16 instead of failing.
    def kv_fp8_kernels_available? : Bool
      avail = @@kv_fp8_available
      return avail unless avail.nil?
      result = !load_kernel_proc("kv_cache_append_fp8", Proc(Void)).nil? &&
               !load_kernel_proc("attention_kv_fp8", Proc(Void)).nil? &&
               !load_kernel_proc("attention_split_kv_fp8", Proc(Void)).nil?
      @@kv_fp8_available = result
      result
    end

    # Cross-entropy loss and gradient computation kernel
    def cross_entropy_loss_gradient(predicted : Pointer(Float32), target : Pointer(Float32),
                                    grad_output : Pointer(Float32), loss_output : Pointer(Float32),
                                    rows : Int32, cols : Int32) : Int32
      unless fn = @@cross_entropy_loss_grad_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "cross_entropy_loss_gradient")
          unless sym.null?
            @@cross_entropy_loss_grad_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@cross_entropy_loss_grad_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        loss_device = Pointer(Float32).null
        # Allocate device memory for the scalar loss
        CUDA.malloc(pointerof(loss_device).as(Pointer(Pointer(Void))), 4)
        fn.call(predicted, target, grad_output, loss_device, rows, cols)
        # Copy loss back to host
        CUDA.memcpy(loss_output.as(Pointer(Void)), loss_device.as(Pointer(Void)), 4_u64, MemcpyKind::DeviceToHost)
        CUDA.free(loss_device.as(Pointer(Void)))
        0
      rescue ex
        Log.error { "CUDA Error in cross_entropy_loss_gradient: #{ex}" }
        1
      end
    end

    def softmax_cross_entropy_label(predicted : Pointer(Float32), labels : Pointer(Int32),
                                    grad_out : Pointer(Float32), loss_out : Pointer(Float32),
                                    rows : Int32, cols : Int32) : Int32
      unless fn = @@softmax_cross_entropy_label_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_cross_entropy_label")
          unless sym.null?
            @@softmax_cross_entropy_label_proc = Proc(Pointer(Float32), Pointer(Int32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_cross_entropy_label_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        loss_device = Pointer(Float32).null
        CUDA.malloc(pointerof(loss_device).as(Pointer(Pointer(Void))), 4)
        fn.call(predicted, labels, grad_out, loss_device, rows, cols)
        CUDA.memcpy(loss_out.as(Pointer(Void)), loss_device.as(Pointer(Void)), 4_u64, MemcpyKind::DeviceToHost)
        CUDA.free(loss_device.as(Pointer(Void)))
        0
      rescue ex
        Log.error { "CUDA Error in softmax_cross_entropy_label: #{ex}" }
        1
      end
    end

    # Dropout kernel using cuRAND/cuDNN. Applies dropout in-place on a contiguous
    # buffer of `size` Float64 values. Returns 0 on success and 1 on failure.
    def dropout(data : Pointer(Float32), size : Int32, dropout_prob : Float32, seed : UInt64) : Int32
      return 1 if data.null? || size <= 0

      begin
        unless fn = @@dropout_proc
          if @@kernels_handle.null?
            @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
          end
          unless @@kernels_handle.null?
            sym = LibC.dlsym(@@kernels_handle, "dropout")
            unless sym.null?
              @@dropout_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Int32, Float64, UInt64, Void).new(sym, Pointer(Void).null)
              fn = @@dropout_proc
            end
          end
        end

        if fn
          fn.call(data, data, size, 1, dropout_prob.to_f64, seed)
          return 0
        end
      rescue ex
        Log.error { "CUDA dropout kernel failed: #{ex}" }
      end

      1
    end

    # ReLU backward kernel
    def relu_backward(dst : Pointer(Float32), input : Pointer(Float32), grad : Pointer(Float32), size : Int32)
      if dst.null? || input.null? || grad.null? || size <= 0
        Log.error { "CUDA relu_backward: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, input: #{input.null? ? "null" : "valid"}, grad: #{grad.null? ? "null" : "valid"}, size: #{size}" }
        return
      end

      unless fn = @@relu_backward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "relu_backward")
          unless sym.null?
            @@relu_backward_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@relu_backward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, input, grad, size)
      rescue ex
        Log.error { "CUDA Error in relu_backward: #{ex}" }
        raise ex
      end
    end

    # Softmax backward kernel
    def softmax_backward(dst : Pointer(Float32), grad : Pointer(Float32), softmax_out : Pointer(Float32), rows : Int32, cols : Int32)
      if dst.null? || grad.null? || softmax_out.null? || rows <= 0 || cols <= 0
        Log.error { "CUDA softmax_backward: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, grad: #{grad.null? ? "null" : "valid"}, softmax_out: #{softmax_out.null? ? "null" : "valid"}, rows: #{rows}, cols: #{cols}" }
        return
      end

      unless fn = @@softmax_backward_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "softmax_backward")
          unless sym.null?
            @@softmax_backward_proc = Proc(Pointer(Float32), Pointer(Float32), Pointer(Float32), Int32, Int32, Void).new(sym, Pointer(Void).null)
            fn = @@softmax_backward_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, grad, softmax_out, rows, cols)
      rescue ex
        Log.error { "CUDA Error in softmax_backward: #{ex}" }
        raise ex
      end
    end

    def element_log(dst : Pointer(Float32), src : Pointer(Float32), size : Int32)
      if dst.null? || src.null? || size <= 0
        Log.error { "CUDA element_log: invalid parameters - dst: #{dst.null? ? "null" : "valid"}, src: #{src.null? ? "null" : "valid"}, size: #{size}" }
        return
      end

      unless fn = @@element_log_proc
        if @@kernels_handle.null?
          @@kernels_handle = LibC.dlopen("libshainet_cuda_kernels.so", LibC::RTLD_LAZY)
        end
        unless @@kernels_handle.null?
          sym = LibC.dlsym(@@kernels_handle, "element_log")
          unless sym.null?
            @@element_log_proc = Proc(Pointer(Float32), Pointer(Float32), Int32, Void).new(sym, Pointer(Void).null)
            fn = @@element_log_proc
          end
        end
      end
      raise "CUDA kernels not available" unless fn

      begin
        fn.call(dst, src, size)
      rescue ex
        Log.error { "CUDA Error in element_log: #{ex}" }
        raise ex
      end
    end

    # GPU kernel for mean squared error cost and gradient computation
    def mse_cost_gradient(actual_ptr : Pointer(Float32), expected_ptr : Pointer(Float32),
                          cost_ptr : Pointer(Float32), grad_ptr : Pointer(Float32), size : Int32)
      # This would be a custom CUDA kernel implementation
      # For now, fallback is handled in the calling code
      raise RuntimeError.new("GPU MSE kernel not yet implemented")
    end
  end
end
