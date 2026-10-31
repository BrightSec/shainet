require "./spec_helper"

# Prefill runs the k-quant matmuls with M > 1, and both backends take a DIFFERENT code path there
# than at M = 1:
#
#   device: M = 1 stays a GEMV (the weight is streamed once anyway); M > 1 dequantizes row chunks
#           into a scratch tile and calls cuBLAS, because the GEMV kernel gives each
#           (output, token) pair its own block and so re-dequantizes the whole weight per token --
#           measured flat at 0.54 ms/token from M=1 to M=256 on a 47.8 MB Q4_K weight.
#   host:   M = 1 keeps the fused dequant+dot; M > 1 unpacks each row once and dots it against
#           four tokens at a time.
#
# A batched path that disagrees with the per-token one would make prefill and decode compute
# different things from the same weights, which is exactly the class of bug that leaves output
# fluent but wrong. So assert they agree rather than assuming it.

# Little-endian fp16 for the constants below. 1.0 = 0x3C00, 0.5 = 0x3800.
private def f16(bits : UInt16, into : Bytes, at : Int32)
  into[at] = (bits & 0xFF).to_u8
  into[at + 1] = (bits >> 8).to_u8
end

# One Q4_K block: d = 1.0, dmin = 0.5, all group scales and mins = 1, qs bytes vary by index so a
# transposed or mis-strided read cannot pass by symmetry.
private def build_q4k_block : Bytes
  block = Bytes.new(144, 0_u8)
  f16(0x3C00_u16, block, 0)
  f16(0x3800_u16, block, 2)
  6.times { |i| block[4 + i] = 0x41_u8 }  # scales: 6-bit 1s
  6.times { |i| block[10 + i] = 0x41_u8 } # mins:   6-bit 1s
  128.times { |i| block[16 + i] = ((i % 13) | (((i + 5) % 11) << 4)).to_u8 }
  block
end

private def build_q6k_block : Bytes
  block = Bytes.new(210, 0_u8)
  128.times { |i| block[i] = ((i * 7) % 251).to_u8 }                           # ql
  64.times { |i| block[128 + i] = ((i * 11) % 253).to_u8 }                     # qh
  16.times { |i| block[192 + i] = (i.odd? ? -(i + 1) : (i + 1)).to_i8.to_u8! } # scales, signed
  f16(0x3C00_u16, block, 208)
  block
end

describe "batched k-quant matmul" do
  {"Q4_K" => 144, "Q6_K" => 210}.each do |kind, bs|
    it "host M>1 agrees with M=1 for #{kind}" do
      pending! "CPU kernels not loaded" unless SHAInet::CPUKernels.available?

      k = 256
      n = 8
      m = 6 # not a multiple of 4, so the blocked loop and its remainder tail both run
      block = kind == "Q4_K" ? build_q4k_block : build_q6k_block

      w = Pointer(UInt8).malloc(n * bs)
      n.times do |r|
        block.each_with_index do |byte, i|
          # Perturb one byte per row so no two rows are identical.
          w[r * bs + i] = i == 0 ? (byte + r).to_u8! : byte
        end
      end

      x = Pointer(Float32).malloc(m * k)
      (m * k).times { |i| x[i] = (0.03 * Math.sin(i * 0.017) + 0.001 * i).to_f32 }

      batched = Pointer(Float32).malloc(m * n)
      per_token = Pointer(Float32).malloc(m * n)

      ok = if kind == "Q4_K"
             SHAInet::CPUKernels.gemv_q4k(x, w, batched, m, n, k)
           else
             SHAInet::CPUKernels.gemv_q6k(x, w, batched, m, n, k)
           end
      pending! "kernel unavailable" unless ok

      m.times do |r|
        if kind == "Q4_K"
          SHAInet::CPUKernels.gemv_q4k(x + r * k, w, per_token + r * n, 1, n, k)
        else
          SHAInet::CPUKernels.gemv_q6k(x + r * k, w, per_token + r * n, 1, n, k)
        end
      end

      scale = (0...(m * n)).max_of { |i| per_token[i].abs }
      scale = 1.0_f32 if scale < 1e-6
      (m * n).times do |i|
        ((batched[i] - per_token[i]).abs / scale).should be < 1e-5
      end
    end
  end

  it "device M>1 GEMM agrees with the per-token GEMV", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no dequant kernels" unless SHAInet::CUDA.dequant_k_rows_available?

    k = 256
    n = 8
    m = 5
    block = build_q4k_block
    host = Pointer(UInt8).malloc(n * 144)
    n.times { |r| block.each_with_index { |byte, i| host[r * 144 + i] = i == 0 ? (byte + r).to_u8! : byte } }

    w = SHAInet::GGUFMatrix.new(k, n, SHAInet::GGUF::GGMLType::Q4_K, host, (n * 144).to_u64)
    x = SHAInet::CudaMatrix.new(m, k)
    (m * k).times { |i| x.raw_data[i] = (0.02 * Math.sin(i * 0.031)).to_f32 }
    x.mark_host_modified!
    x.sync_to_device!("spec_in")

    gemm = SHAInet::CudaMatrix.new(m, n)
    gemv = SHAInet::CudaMatrix.new(m, n)
    # The dequant + cuBLAS path specifically; the int8 MMQ path has its own example below.
    SHAInet::GGUFMatrix.mmq = false
    begin
      w.gemm_into(x, gemm)
    ensure
      SHAInet::GGUFMatrix.mmq = nil
    end
    SHAInet::CUDA.gemv_q4k(x.device_ptr.not_nil!, w.dev_ptr, gemv.device_ptr.not_nil!, m, n, k)
    gemv.mark_device_dirty!
    gemm.sync_from_device!("spec_gemm")
    gemv.sync_from_device!("spec_gemv")

    scale = (0...(m * n)).max_of { |i| gemv.raw_data[i].abs }
    scale = 1.0_f32 if scale < 1e-6
    (m * n).times do |i|
      ((gemm.raw_data[i] - gemv.raw_data[i]).abs / scale).should be < 1e-4
    end

    x.free!
    gemm.free!
    gemv.free!
    w.free!
    SHAInet::GGUFMatrix.release_scratch!
  end

  it "int8 MMQ GEMM agrees with the per-token GEMV", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no int8 MMQ kernels or device below sm_80" unless SHAInet::CUDA.mmq_available?

    # Shapes past one tile in every direction, with partial tiles: 300 tokens over 128-token tiles,
    # 200 rows over 128-row tiles, two 256-value blocks along K.
    k = 512
    n = 200
    m = 300
    block = build_q4k_block
    blocks_per_row = k // 256
    host = Pointer(UInt8).malloc(n * blocks_per_row * 144)
    (n * blocks_per_row).times do |b|
      block.each_with_index { |byte, i| host[b * 144 + i] = i == 0 ? (byte + b % 7).to_u8! : (i >= 16 ? (byte ^ (b % 5)).to_u8 : byte) }
    end

    w = SHAInet::GGUFMatrix.new(k, n, SHAInet::GGUF::GGMLType::Q4_K, host, (n * blocks_per_row * 144).to_u64)
    x = SHAInet::CudaMatrix.new(m, k)
    (m * k).times { |i| x.raw_data[i] = (0.5 * Math.sin(i * 0.013) + 0.1 * Math.cos(i * 0.71)).to_f32 }
    x.mark_host_modified!
    x.sync_to_device!("spec_in")

    mmq = SHAInet::CudaMatrix.new(m, n)
    gemv = SHAInet::CudaMatrix.new(m, n)
    SHAInet::GGUFMatrix.mmq = true
    begin
      w.gemm_into(x, mmq)
    ensure
      SHAInet::GGUFMatrix.mmq = nil
    end
    SHAInet::CUDA.gemv_q4k(x.device_ptr.not_nil!, w.dev_ptr, gemv.device_ptr.not_nil!, m, n, k)
    gemv.mark_device_dirty!
    mmq.sync_from_device!("spec_mmq")
    gemv.sync_from_device!("spec_gemv")

    # int8 activations (per-32 absmax) cost about 1e-3 of the output scale; a wrong tile index,
    # scale or offset costs order 1.
    scale = (0...(m * n)).max_of { |i| gemv.raw_data[i].abs }
    (m * n).times do |i|
      ((mmq.raw_data[i] - gemv.raw_data[i]).abs / scale).should be < 1e-2
    end

    x.free!
    mmq.free!
    gemv.free!
    w.free!
  end

  it "fused SwiGLU int8 GEMM matches swiglu_forward then gemm_into", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no int8 MMQ kernels or device below sm_80" unless SHAInet::CUDA.mmq_available?

    k = 512
    n = 160
    m = 70
    block = build_q4k_block
    blocks_per_row = k // 256
    host = Pointer(UInt8).malloc(n * blocks_per_row * 144)
    (n * blocks_per_row).times do |b|
      block.each_with_index { |byte, i| host[b * 144 + i] = i >= 16 ? (byte ^ (b % 9)).to_u8 : byte }
    end
    w = SHAInet::GGUFMatrix.new(k, n, SHAInet::GGUF::GGMLType::Q4_K, host, (n * blocks_per_row * 144).to_u64)

    gate = SHAInet::CudaMatrix.new(m, k)
    up = SHAInet::CudaMatrix.new(m, k)
    (m * k).times do |i|
      gate.raw_data[i] = (1.5 * Math.sin(i * 0.017)).to_f32
      up.raw_data[i] = (0.8 * Math.cos(i * 0.029)).to_f32
    end
    gate.mark_host_modified!
    up.mark_host_modified!
    gate.sync_to_device!("spec_gate")
    up.sync_to_device!("spec_up")

    fused = SHAInet::CudaMatrix.new(m, n)
    hidden = SHAInet::CudaMatrix.new(m, k)
    plain = SHAInet::CudaMatrix.new(m, n)
    SHAInet::GGUFMatrix.mmq = true
    begin
      w.gemm_into_swiglu(gate, up, fused).should be_true
      SHAInet::CUDA.swiglu_forward(hidden.device_ptr.not_nil!, gate.device_ptr.not_nil!, up.device_ptr.not_nil!, m * k)
      hidden.mark_device_dirty!
      w.gemm_into(hidden, plain)
    ensure
      SHAInet::GGUFMatrix.mmq = nil
    end
    fused.sync_from_device!("spec_fused")
    plain.sync_from_device!("spec_plain")

    # Same activation values reach the same quantizer, so the two agree to rounding.
    scale = (0...(m * n)).max_of { |i| plain.raw_data[i].abs }
    (m * n).times do |i|
      ((fused.raw_data[i] - plain.raw_data[i]).abs / scale).should be < 1e-5
    end

    {gate, up, fused, hidden, plain}.each(&.free!)
    w.free!
  end
end
