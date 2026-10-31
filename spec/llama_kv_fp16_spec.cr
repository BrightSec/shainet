require "./spec_helper"

# The device KV cache is the dominant VRAM consumer at long context, so it is
# stored in fp16 by default while all arithmetic stays fp32 (the kernels convert
# on load). These specs pin three things: the fp16 result tracks the fp32 one,
# the footprint actually halves, and the growth path — which re-uploads the fp32
# host mirror through the append kernel — stays correct.
private def fill_random!(w, rng : Random)
  m = w.as(SHAInet::SimpleMatrix)
  m.rows.times { |r| m.cols.times { |c| m[r, c] = (rng.rand * 0.2 - 0.1) } }
end

private def build_block(rng : Random, d_model : Int32) : SHAInet::LlamaBlock
  blk = SHAInet::LlamaBlock.new(d_model, 8, 128, num_kv_heads: 4)
  {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, rng) }
  ffn = blk.ffn.as(SHAInet::SwiGLUFF)
  fill_random!(ffn.gate_proj, rng)
  fill_random!(ffn.up_proj, rng)
  fill_random!(ffn.down_proj, rng)
  blk
end

private def max_abs_diff(a : SHAInet::SimpleMatrix, b : SHAInet::SimpleMatrix) : Float64
  worst = 0.0
  a.rows.times do |r|
    a.cols.times do |c|
      d = (a[r, c] - b[r, c]).abs
      worst = d if d > worst
    end
  end
  worst
end

private def with_kv_fp16(value : String, &)
  prev = ENV["SHAINET_KV_FP16"]?
  ENV["SHAINET_KV_FP16"] = value
  begin
    yield
  ensure
    if prev
      ENV["SHAINET_KV_FP16"] = prev
    else
      ENV.delete("SHAINET_KV_FP16")
    end
  end
end

private def with_kv_dtype(value : String, &)
  prev = ENV["SHAINET_KV_DTYPE"]?
  ENV["SHAINET_KV_DTYPE"] = value
  begin
    yield
  ensure
    if prev
      ENV["SHAINET_KV_DTYPE"] = prev
    else
      ENV.delete("SHAINET_KV_DTYPE")
    end
  end
end

# Prefill then a few decode steps, returning every output.
private def drive(blk : SHAInet::LlamaBlock, prompt : SHAInet::SimpleMatrix,
                  steps : Array(SHAInet::SimpleMatrix)) : Array(SHAInet::SimpleMatrix)
  blk.clear_cache!
  outs = [blk.forward_cached(prompt)]
  steps.each { |t| outs << blk.forward_cached(t) }
  outs
end

describe "LlamaBlock fp16 KV cache" do
  it "exposes the fp16 kernels from the built kernel library" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    SHAInet::CUDA.kv_f16_kernels_available?.should be_true
  end

  it "tracks the fp32 cache result and halves the footprint" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 64
    seq = 24
    prng = Random.new(9)
    prompt = SHAInet::SimpleMatrix.new(seq, d_model)
    seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }
    steps = Array.new(3) do
      tok = SHAInet::SimpleMatrix.new(1, d_model)
      d_model.times { |c| tok[0, c] = (prng.rand * 2.0 - 1.0) }
      tok
    end

    fp32_blk = build_block(Random.new(42), d_model)
    fp32_outs = with_kv_fp16("0") { drive(fp32_blk, prompt, steps) }
    fp32_blk.kv_cache_fp16?.should be_false

    fp16_blk = build_block(Random.new(42), d_model)
    fp16_outs = with_kv_fp16("1") { drive(fp16_blk, prompt, steps) }
    fp16_blk.kv_cache_fp16?.should be_true

    fp32_outs.each_with_index do |expected, i|
      max_abs_diff(expected, fp16_outs[i]).should be < 5e-3
    end

    fp32_blk.kv_cache_bytes.should be > 0
    fp16_blk.kv_cache_bytes.should eq(fp32_blk.kv_cache_bytes // 2)
  end

  it "fp8 KV tracks the fp32 result at a quarter of the footprint" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?
    pending! "fp8 KV kernels not in the loaded library" unless SHAInet::CUDA.kv_fp8_kernels_available?

    d_model = 64
    seq = 24
    prng = Random.new(9)
    prompt = SHAInet::SimpleMatrix.new(seq, d_model)
    seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }
    steps = Array.new(3) do
      tok = SHAInet::SimpleMatrix.new(1, d_model)
      d_model.times { |c| tok[0, c] = (prng.rand * 2.0 - 1.0) }
      tok
    end

    fp32_blk = build_block(Random.new(42), d_model)
    fp32_outs = with_kv_fp16("0") { drive(fp32_blk, prompt, steps) }

    fp8_blk = build_block(Random.new(42), d_model)
    fp8_outs = with_kv_dtype("fp8") { drive(fp8_blk, prompt, steps) }
    fp8_blk.kv_cache_fp8?.should be_true

    # e4m3 keeps 3 mantissa bits, so the bound is looser than fp16's, but still far below the
    # activations' scale: a broken store or load would miss it by orders of magnitude.
    fp32_outs.each_with_index do |expected, i|
      max_abs_diff(expected, fp8_outs[i]).should be < 5e-2
    end

    fp8_blk.kv_cache_bytes.should eq(fp32_blk.kv_cache_bytes // 4)
  end

  it "tensor-core prefill attention tracks the CPU path (head_dim 128, GQA)" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    # head_dim 128 is one of the two shapes the WMMA prefill kernel serves; the other specs here
    # use head dims it declines, so without this it would only be exercised by a real model.
    d_model = 512
    seq = 80
    prng = Random.new(11)
    prompt = SHAInet::SimpleMatrix.new(seq, d_model)
    seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }
    steps = Array.new(2) do
      tok = SHAInet::SimpleMatrix.new(1, d_model)
      d_model.times { |c| tok[0, c] = (prng.rand * 2.0 - 1.0) }
      tok
    end

    cpu_blk = SHAInet::LlamaBlock.new(d_model, 4, 256, num_kv_heads: 2)
    {cpu_blk.w_q, cpu_blk.w_k, cpu_blk.w_v, cpu_blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }
    cpu_blk.force_cpu_attention = true
    cpu_outs = drive(cpu_blk, prompt, steps)

    gpu_blk = SHAInet::LlamaBlock.new(d_model, 4, 256, num_kv_heads: 2)
    {gpu_blk.w_q, gpu_blk.w_k, gpu_blk.w_v, gpu_blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }
    gpu_outs = with_kv_fp16("1") { drive(gpu_blk, prompt, steps) }

    # Q and P are rounded to half inside the kernel, so the bound is looser than the fp16-cache
    # specs above, and still far below the activations' scale.
    cpu_outs.each_with_index do |expected, i|
      max_abs_diff(expected, gpu_outs[i]).should be < 2e-2
    end
  end

  it "stays correct through a device cache growth" do
    pending! "CUDA kernels not available" unless SHAInet::CUDA.fully_available?

    d_model = 32
    # Initial capacity is 256, so 290 positions forces a grow plus the fp32
    # host-mirror re-upload through the fp16 append kernel.
    seq = 290
    prng = Random.new(4)
    prompt = SHAInet::SimpleMatrix.new(seq, d_model)
    seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }
    steps = Array.new(5) do
      tok = SHAInet::SimpleMatrix.new(1, d_model)
      d_model.times { |c| tok[0, c] = (prng.rand * 2.0 - 1.0) }
      tok
    end

    cpu_blk = SHAInet::LlamaBlock.new(d_model, 4, 64, num_kv_heads: 2)
    {cpu_blk.w_q, cpu_blk.w_k, cpu_blk.w_v, cpu_blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }
    cpu_blk.force_cpu_attention = true
    cpu_outs = drive(cpu_blk, prompt, steps)

    gpu_blk = SHAInet::LlamaBlock.new(d_model, 4, 64, num_kv_heads: 2)
    {gpu_blk.w_q, gpu_blk.w_k, gpu_blk.w_v, gpu_blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }
    gpu_outs = with_kv_fp16("1") { drive(gpu_blk, prompt, steps) }
    gpu_blk.kv_cache_fp16?.should be_true

    cpu_outs.each_with_index do |expected, i|
      max_abs_diff(expected, gpu_outs[i]).should be < 5e-3
    end
  end
end
