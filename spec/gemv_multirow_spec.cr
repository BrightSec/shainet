require "./spec_helper"

# The multi-row GEMV (gemv_multirow.cuh) serves 2-8 rows -- a speculative verify pass -- by reading
# each weight block once for all rows. Each GGUF type has its own decoder there, separate from the
# single-row GEMV's, so every type is checked against the single-row path on the same weights.
# Activations go through int8 (one scale per 32 values), so agreement is to that precision rather
# than bit for bit.

private GEMV_MR_TYPES = {
  # type => {block bytes, fp16 scale offsets}
  SHAInet::GGUF::GGMLType::Q2_K    => {84, [80, 82]},
  SHAInet::GGUF::GGMLType::Q4_K    => {144, [0, 2]},
  SHAInet::GGUF::GGMLType::Q5_K    => {176, [0, 2]},
  SHAInet::GGUF::GGMLType::Q6_K    => {210, [208]},
  SHAInet::GGUF::GGMLType::IQ2_XXS => {66, [0]},
  SHAInet::GGUF::GGMLType::IQ2_XS  => {74, [0]},
  SHAInet::GGUF::GGMLType::IQ2_S   => {82, [0]},
  SHAInet::GGUF::GGMLType::IQ3_XXS => {98, [0]},
  SHAInet::GGUF::GGMLType::IQ3_S   => {110, [0]},
  SHAInet::GGUF::GGMLType::IQ4_XS  => {136, [0]},
  SHAInet::GGUF::GGMLType::IQ1_S   => {50, [0]},
  SHAInet::GGUF::GGMLType::IQ1_M   => {56, [] of Int32},
}

# Pseudo-random block bytes with every fp16 scale set to a small finite value, so any bit pattern
# in the quant fields decodes to finite weights.
private def gemv_mr_weight(type : SHAInet::GGUF::GGMLType, bs : Int32, scales : Array(Int32), n : Int32, k : Int32) : Bytes
  blocks = n * (k // 256)
  rng = Random.new(42 + type.value)
  w = Bytes.new(blocks * bs) { rng.rand(256).to_u8 }
  blocks.times do |b|
    base = b * bs
    scales.each do |off|
      v = 0x2000_u16 | rng.rand(0x0400).to_u16 # fp16 in [2^-7, 2^-6)
      w[base + off] = (v & 0xFF).to_u8
      w[base + off + 1] = (v >> 8).to_u8
    end
    if type.iq1_m?
      # IQ1_M assembles its fp16 scale from the top nibble of each of its four u16 scale words.
      {0x0, 0x0, 0xC, 0x3}.each_with_index do |nib, i|
        w[base + 48 + 2 * i + 1] = ((w[base + 48 + 2 * i + 1] & 0x0F) | (nib << 4)).to_u8
      end
    end
  end
  w
end

describe "multi-row GEMV" do
  GEMV_MR_TYPES.each do |type, (bs, scales)|
    it "matches the single-row GEMV for #{type}", tags: "cuda" do
      pending! "no CUDA" unless SHAInet::CUDA.fully_available?

      k = 512
      n = 96
      m = 5
      w = gemv_mr_weight(type, bs, scales, n, k)
      mat = SHAInet::GGUFMatrix.new(k, n, type, w.to_unsafe, w.size)

      x = SHAInet::CudaMatrix.new(m, k)
      (m * k).times { |i| x.raw_data[i] = (0.5 * Math.sin(i * 0.013 + 0.3) + 0.05 * Math.cos(i * 0.71)).to_f32 }
      x.mark_host_modified!
      x.sync_to_device!

      # The multi-row kernel must take this type: without it gemv_into falls back to the batched GEMM.
      y = SHAInet::CudaMatrix.new(m, n)
      SHAInet::CUDA.gemv_mr(type.value.to_i32, x.device_ptr.not_nil!, mat.dev_ptr, y.device_ptr.not_nil!, m, n, k).should be_true
      y.mark_device_dirty!
      y.sync_from_device!

      worst = 0.0
      scale = 0.0
      m.times do |r|
        xr = SHAInet::CudaMatrix.new(1, k)
        k.times { |i| xr.raw_data[i] = x.raw_data[r * k + i] }
        xr.mark_host_modified!
        xr.sync_to_device!
        yr = SHAInet::CudaMatrix.new(1, n)
        mat.gemv_into(xr, yr)
        yr.sync_from_device!
        n.times do |j|
          scale = Math.max(scale, yr.raw_data[j].abs.to_f64)
          worst = Math.max(worst, (yr.raw_data[j] - y.raw_data[r * n + j]).abs.to_f64)
        end
        xr.free!
        yr.free!
      end
      scale.should be > 0.0
      (worst / scale).should be < 2e-2

      # gemv_into routes 2-8 rows here.
      z = SHAInet::CudaMatrix.new(m, n)
      mat.gemv_into(x, z)
      z.sync_from_device!
      (m * n).times { |i| z.raw_data[i].should eq(y.raw_data[i]) }
      x.free!
      y.free!
      z.free!
    end
  end
end
