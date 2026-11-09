require "./spec_helper"

# The q4 KV cache: q4_0 rows (32-value groups, fp16 scales) with keys stored after a fixed Hadamard
# rotation and queries rotated to match. These pin that it tracks the fp32 cache about as closely as
# fp8 does, at 144/256 of fp8's bytes (head_dim 256), through prefill, decode and a cache growth.
private def fill_random!(w, rng : Random)
  m = w.as(SHAInet::SimpleMatrix)
  m.rows.times { |r| m.cols.times { |c| m[r, c] = (rng.rand * 0.2 - 0.1) } }
end

private def block(d_model : Int32, heads : Int32, kv_heads : Int32) : SHAInet::LlamaBlock
  blk = SHAInet::LlamaBlock.new(d_model, heads, 128, num_kv_heads: kv_heads)
  {blk.w_q, blk.w_k, blk.w_v, blk.w_o}.each { |w| fill_random!(w, Random.new(42)) }
  blk
end

private def with_kv_dtype(value : String, &)
  prev = ENV["SHAINET_KV_DTYPE"]?
  ENV["SHAINET_KV_DTYPE"] = value
  begin
    yield
  ensure
    prev ? (ENV["SHAINET_KV_DTYPE"] = prev) : ENV.delete("SHAINET_KV_DTYPE")
  end
end

private def inputs(seq : Int32, d_model : Int32, steps : Int32, seed : Int32)
  prng = Random.new(seed)
  prompt = SHAInet::SimpleMatrix.new(seq, d_model)
  seq.times { |r| d_model.times { |c| prompt[r, c] = (prng.rand * 2.0 - 1.0) } }
  toks = Array.new(steps) do
    t = SHAInet::SimpleMatrix.new(1, d_model)
    d_model.times { |c| t[0, c] = (prng.rand * 2.0 - 1.0) }
    t
  end
  {prompt, toks}
end

private def drive(blk : SHAInet::LlamaBlock, prompt, steps) : Array(SHAInet::SimpleMatrix)
  blk.clear_cache!
  outs = [blk.forward_cached(prompt)]
  steps.each { |t| outs << blk.forward_cached(t) }
  outs
end

# Worst absolute deviation from the reference over every output, and the outputs' own scale.
private def worst(ref : Array(SHAInet::SimpleMatrix), got : Array(SHAInet::SimpleMatrix)) : Float64
  w = 0.0
  ref.each_with_index do |a, i|
    b = got[i]
    a.rows.times { |r| a.cols.times { |c| d = (a[r, c] - b[r, c]).abs; w = d if d > w } }
  end
  w
end

private def q4_ready? : Bool
  SHAInet::CUDA.fully_available? && SHAInet::CUDA.kv_q4_kernels_available? &&
    SHAInet::CUDA.kv_fp8_kernels_available?
end

describe "LlamaBlock q4 KV cache" do
  {128, 256}.each do |head_dim|
    it "tracks fp32 about as closely as fp8 at head_dim #{head_dim}" do
      pending! "q4 KV kernels not in the loaded library" unless q4_ready?

      heads = 4
      d_model = heads * head_dim
      prompt, steps = inputs(80, d_model, 4, 11)
      ref = with_kv_dtype("f32") { drive(block(d_model, heads, 2), prompt, steps) }

      fp8_blk = block(d_model, heads, 2)
      fp8 = with_kv_dtype("fp8") { drive(fp8_blk, prompt, steps) }
      q4_blk = block(d_model, heads, 2)
      q4 = with_kv_dtype("q4") { drive(q4_blk, prompt, steps) }
      q4_blk.kv_cache_q4?.should be_true

      e8 = worst(ref, fp8)
      e4 = worst(ref, q4)
      scale = ref.max_of { |m| (0...m.rows * m.cols).max_of { |i| m.data[i].abs.to_f64 } }
      STDERR.puts "  head_dim #{head_dim}: output max |x| #{scale.round(3)}  fp8 err #{e8.round(5)}  q4 err #{e4.round(5)}" if ENV["Q4_SPEC_VERBOSE"]?
      # 4 bits per value against fp8's 3 mantissa bits: a few times fp8's error, not orders of
      # magnitude. A wrong rotation (q and k transformed differently) or a mis-addressed row gives
      # errors on the scale of the outputs themselves.
      # Measured: 2.6x fp8 at head_dim 128, 3.1x at 256 (random weights, no QK-norm, so the outputs
      # reach 6-14 and both caches' absolute errors are large in these units).
      e4.should be > 0.0
      e4.should be < 5 * e8
      e4.should be < 0.25 * scale

      expected = head_dim == 256 ? 144 : 80
      q4_blk.kv_cache_bytes.should eq(fp8_blk.kv_cache_bytes * expected // head_dim)
    end
  end

  it "decodes the same with and without the GQA decode kernel" do
    pending! "q4 KV kernels not in the loaded library" unless q4_ready?

    head_dim = 256
    d_model = 4 * head_dim
    prompt, steps = inputs(40, d_model, 4, 5)
    grouped = with_kv_dtype("q4") { drive(block(d_model, 4, 2), prompt, steps) }
    ENV["SHAINET_DECODE_GQA_ATTN"] = "0"
    begin
      plain = with_kv_dtype("q4") { drive(block(d_model, 4, 2), prompt, steps) }
    ensure
      ENV.delete("SHAINET_DECODE_GQA_ATTN")
    end
    # The kernel caches its enable flag at first use, so this compares the two only when the flag was
    # still unset; either way both runs must agree with each other to fp32 reassociation.
    worst(grouped, plain).should be < 1e-4
  end

  it "stays correct through a device cache growth" do
    pending! "q4 KV kernels not in the loaded library" unless q4_ready?

    head_dim = 128
    d_model = 2 * head_dim
    # Initial capacity is 256 positions, so 290 forces a grow, which copies q4 rows to the new buffer.
    prompt, steps = inputs(290, d_model, 5, 4)
    ref = with_kv_dtype("f32") { drive(block(d_model, 2, 1), prompt, steps) }
    fp8 = with_kv_dtype("fp8") { drive(block(d_model, 2, 1), prompt, steps) }
    q4 = with_kv_dtype("q4") { drive(block(d_model, 2, 1), prompt, steps) }
    # A row copied to the wrong offset on growth would make the decode steps' errors output-sized.
    worst(ref, q4).should be < 5 * worst(ref, fp8)
  end
end
