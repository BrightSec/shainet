require "./spec_helper"

# Prefix cache reuse in the OpenAI server rewinds the cache to the end of the previous prompt and
# prefills only the new tail. That is correct only if "rewind, then feed C" produces exactly what
# "fresh, feed A, then feed C" produces, for both block kinds in a hybrid stack:
#   - attention (LlamaBlock): the KV cache is cut back to a length;
#   - Gated DeltaNet: the recurrent state cannot be cut back, so a saved copy is restored.
# Each example also checks that skipping the rewind gives a DIFFERENT result, so the equality cannot
# pass on a block whose output ignores its cache.

private def seeded(rows, cols, seed)
  m = SHAInet::SimpleMatrix.new(rows, cols)
  rows.times { |i| cols.times { |j| m[i, j] = Math.sin(seed + i * 0.31 + j * 0.17) * 0.5 } }
  m
end

private def max_diff(a : SHAInet::SimpleMatrix, b : SHAInet::SimpleMatrix) : Float64
  d = 0.0
  a.rows.times { |i| a.cols.times { |j| d = Math.max(d, (a[i, j] - b[i, j]).abs.to_f64) } }
  d
end

private def llama_block(d_model = 32, heads = 4, kv_heads = 2, head_dim = 8)
  blk = SHAInet::LlamaBlock.new(d_model, heads, d_model * 2, num_kv_heads: kv_heads, head_dim: head_dim)
  blk.w_q = seeded(d_model, heads * head_dim, 1.0)
  blk.w_k = seeded(d_model, kv_heads * head_dim, 2.0)
  blk.w_v = seeded(d_model, kv_heads * head_dim, 3.0)
  blk.w_o = seeded(heads * head_dim, d_model, 4.0)
  blk
end

# Real weights, as in hybrid_stack_spec: with zero weights nothing reaches the recurrent state and
# every comparison would pass vacuously.
private def gdn_block : SHAInet::GatedDeltaNetBlock
  net = SHAInet::Network.new
  net.add_layer("gated_deltanet", 16, num_heads: 4, ff_hidden: 32, num_kv_heads: 2, head_dim: 4)
  block = net.hidden_layers.find!(&.is_a?(SHAInet::GatedDeltaNetBlock)).as(SHAInet::GatedDeltaNetBlock)
  [block.w_q.as(SHAInet::SimpleMatrix), block.w_k.as(SHAInet::SimpleMatrix),
   block.w_v.as(SHAInet::SimpleMatrix), block.w_o.as(SHAInet::SimpleMatrix),
   block.w_gate.as(SHAInet::SimpleMatrix), block.w_alpha, block.w_beta,
   block.conv_q.weight, block.conv_k.weight, block.conv_v.weight].each_with_index do |m, idx|
    m.rows.times do |i|
      m.cols.times { |j| m[i, j] = 0.08 * Math.sin((i * 7 + j * 13 + idx * 3) * 0.37) }
    end
  end
  block
end

describe "LlamaBlock#truncate_cache!" do
  it "continues from the cut point exactly as a fresh cache would" do
    a, b, c = seeded(5, 32, 0.1), seeded(3, 32, 0.7), seeded(3, 32, 1.3)

    blk = llama_block
    blk.forward_cached(a)
    blk.forward_cached(b)
    blk.cached_length.should eq(8)
    blk.truncate_cache!(5)
    blk.cached_length.should eq(5)
    rewound = blk.forward_cached(c)

    fresh = llama_block
    fresh.forward_cached(a)
    expected = fresh.forward_cached(c)

    max_diff(rewound, expected).should be < 1e-5

    # Counter-check: without the cut, C attends over B too and the output changes.
    dirty = llama_block
    dirty.forward_cached(a)
    dirty.forward_cached(b)
    max_diff(dirty.forward_cached(c), expected).should be > 1e-4
  end

  it "refuses to extend the cache" do
    blk = llama_block
    blk.forward_cached(seeded(2, 32, 0.4))
    expect_raises(ArgumentError) { blk.truncate_cache!(3) }
  end
end

describe "GatedDeltaNetBlock saved state" do
  it "restores the recurrent state so the next tokens match a fresh run" do
    a, b, c = seeded(4, 16, 0.2), seeded(3, 16, 0.9), seeded(2, 16, 1.6)

    blk = gdn_block
    blk.forward(a)
    blk.save_state!
    blk.forward(b)
    blk.restore_saved_state!.should be_true
    rewound = blk.forward(c)

    blk.clear_cache!
    blk.forward(a)
    expected = blk.forward(c)

    max_diff(rewound, expected).should be < 1e-6

    blk.clear_cache!
    blk.forward(a)
    blk.forward(b)
    max_diff(blk.forward(c), expected).should be > 1e-6
  end

  it "can restore the same point twice and reports when nothing is saved" do
    blk = gdn_block
    blk.restore_saved_state!.should be_false

    a, b = seeded(3, 16, 0.3), seeded(2, 16, 1.1)
    blk.forward(a)
    blk.save_state!
    blk.forward(b)
    blk.restore_saved_state!
    first = blk.forward(b)
    blk.restore_saved_state!
    second = blk.forward(b)
    max_diff(first, second).should be < 1e-6

    blk.drop_saved_state!
    blk.restore_saved_state!.should be_false
  end
end

describe "Network#save_cache! / #restore_cache!" do
  it "rewinds every attention block to the saved length" do
    net = SHAInet::Network.new
    net.add_layer("llama", 32, num_heads: 4, ff_hidden: 64, num_kv_heads: 2, head_dim: 8)
    net.add_layer("gated_deltanet", 32, num_heads: 4, ff_hidden: 64, num_kv_heads: 2, head_dim: 8)
    llama = net.hidden_layers.find!(&.is_a?(SHAInet::LlamaBlock)).as(SHAInet::LlamaBlock)

    net.restore_cache!.should be_nil

    llama.forward_cached(seeded(4, 32, 0.5))
    net.save_cache!(4)
    llama.forward_cached(seeded(3, 32, 0.8))
    llama.cached_length.should eq(7)

    net.restore_cache!.should eq(4)
    llama.cached_length.should eq(4)

    # A cache cleared since the save no longer reaches the saved point: no rewind.
    net.clear_cache!
    net.restore_cache!.should be_nil

    net.drop_saved_cache!
    net.restore_cache!.should be_nil
  end

  it "refuses to save at a length the cache does not hold" do
    net = SHAInet::Network.new
    net.add_layer("llama", 32, num_heads: 4, ff_hidden: 64, num_kv_heads: 2, head_dim: 8)
    llama = net.hidden_layers.find!(&.is_a?(SHAInet::LlamaBlock)).as(SHAInet::LlamaBlock)
    llama.forward_cached(seeded(2, 32, 0.5))
    expect_raises(ArgumentError) { net.save_cache!(5) }
    net.restore_cache!.should be_nil
  end
end

# The server runs the DEVICE paths: the recurrent state lives in device buffers (@dev_state and the
# device conv windows) and the KV cache is device-resident with a stale host mirror. Same claims as
# above, on those paths.
private def device_gdn_block : SHAInet::GatedDeltaNetBlock
  d = 64
  v_heads = 6
  blk = SHAInet::GatedDeltaNetBlock.new(d, 2 * d, v_heads, 2, 8, 8, 4)
  fill = ->(m : SHAInet::SimpleMatrix, scale : Float64, phase : Float64) do
    m.rows.times { |i| m.cols.times { |j| m[i, j] = (scale * Math.sin(phase + (i * 7 + j * 13) * 0.37)).to_f32 } }
  end
  fill.call(blk.w_q.as(SHAInet::SimpleMatrix), 0.09, 0.1)
  fill.call(blk.w_k.as(SHAInet::SimpleMatrix), 0.08, 0.2)
  fill.call(blk.w_v.as(SHAInet::SimpleMatrix), 0.07, 0.3)
  fill.call(blk.w_o.as(SHAInet::SimpleMatrix), 0.06, 0.4)
  fill.call(blk.w_gate.as(SHAInet::SimpleMatrix), 0.05, 0.5)
  fill.call(blk.w_alpha.as(SHAInet::SimpleMatrix), 0.04, 0.6)
  fill.call(blk.w_beta.as(SHAInet::SimpleMatrix), 0.03, 0.7)
  fill.call(blk.conv_q.weight, 0.3, 0.8)
  fill.call(blk.conv_k.weight, 0.3, 0.9)
  fill.call(blk.conv_v.weight, 0.3, 1.0)
  v_heads.times { |h| blk.a_log[h] = -2.0 - 0.1 * h; blk.dt_bias[h] = 0.05 * h }
  fill.call(blk.out_norm.gamma.as(SHAInet::SimpleMatrix), 1.0, 0.2)
  blk.to_gpu!(quantize: true, bits: 8)
  blk
end

describe "cache rewind on the device paths" do
  it "restores device-resident Gated DeltaNet state", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no GDN kernels" unless SHAInet::CUDA.gated_delta_rule_available?
    pending! "no mixer kernels" unless SHAInet::CUDA.gdn_mixer_kernels_available?

    ENV["SHAINET_GDN_RESIDENT"] = "1"
    begin
      blk = device_gdn_block
      a, b, c = seeded(4, 64, 0.2), seeded(3, 64, 0.9), seeded(2, 64, 1.6)

      blk.clear_cache!
      blk.mix(a.clone)
      blk.save_state!
      blk.mix(b.clone)
      blk.restore_saved_state!.should be_true
      rewound = blk.mix(c.clone)

      blk.clear_cache!
      blk.mix(a.clone)
      expected = blk.mix(c.clone)
      err = max_diff(rewound, expected)

      blk.clear_cache!
      blk.mix(a.clone)
      blk.mix(b.clone)
      dirty = max_diff(blk.mix(c.clone), expected)
      STDERR.puts "  device GDN rewind: rewound diff #{err}, unrewound diff #{dirty}" if ENV["SPEC_VERBOSE_DIFF"]?
      # The state's influence on this small block's output is itself small (~1e-4), so the bar is
      # relative: the rewound run must be far closer to the fresh run than the unrewound one is.
      dirty.should be > 1e-6
      err.should be < dirty / 100
    ensure
      ENV.delete("SHAINET_GDN_RESIDENT")
    end
  end

  it "truncates the device KV cache", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no prefill kernels" unless SHAInet::CUDA.prefill_attn_kernels_available? &&
                                         SHAInet::CUDA.kv_f16_kernels_available?

    prev = SHAInet::LlamaBlock.prefill_attn_device_enabled?
    SHAInet::LlamaBlock.prefill_attn_device_enabled = true
    begin
      build = -> do
        blk = SHAInet::LlamaBlock.new(64, 4, 128, num_kv_heads: 2, head_dim: 16)
        blk.w_q = seeded(64, 64, 1.0)
        blk.w_k = seeded(64, 32, 2.0)
        blk.w_v = seeded(64, 32, 3.0)
        blk.w_o = seeded(64, 64, 4.0)
        blk.to_gpu!(quantize: true, bits: 4)
        blk.clear_cache!
        blk
      end
      a, b, c = seeded(6, 64, 0.1), seeded(4, 64, 0.7), seeded(3, 64, 1.3)

      blk = build.call
      blk.forward_cached(a)
      blk.forward_cached(b)
      blk.host_kv_stale?.should be_true
      blk.truncate_cache!(6)
      rewound = blk.forward_cached(c)

      fresh = build.call
      fresh.forward_cached(a)
      expected = fresh.forward_cached(c)
      max_diff(rewound, expected).should be < 1e-3

      dirty = build.call
      dirty.forward_cached(a)
      dirty.forward_cached(b)
      max_diff(dirty.forward_cached(c), expected).should be > 1e-3
    ensure
      SHAInet::LlamaBlock.prefill_attn_device_enabled = prev
    end
  end
end

# A snapshot parks one sequence while the cache is cleared and used for a DIFFERENT sequence, then
# resumes it. truncate_cache!/save_cache! cannot do that: the other sequence overwrites the rows.
describe "Network#snapshot_cache / #restore_cache_snapshot!" do
  it "resumes a parked sequence after another one used the cache" do
    net = SHAInet::Network.new
    net.add_layer("llama", 16, num_heads: 4, ff_hidden: 32, num_kv_heads: 2, head_dim: 4)
    net.add_layer("gated_deltanet", 16, num_heads: 4, ff_hidden: 32, num_kv_heads: 2, head_dim: 4)
    llama = net.hidden_layers.find!(&.is_a?(SHAInet::LlamaBlock)).as(SHAInet::LlamaBlock)
    llama.w_q = seeded(16, 16, 1.0)
    llama.w_k = seeded(16, 8, 2.0)
    llama.w_v = seeded(16, 8, 3.0)
    llama.w_o = seeded(16, 16, 4.0)
    gdn = net.hidden_layers.find!(&.is_a?(SHAInet::GatedDeltaNetBlock)).as(SHAInet::GatedDeltaNetBlock)
    [gdn.w_q.as(SHAInet::SimpleMatrix), gdn.w_k.as(SHAInet::SimpleMatrix),
     gdn.w_v.as(SHAInet::SimpleMatrix), gdn.w_o.as(SHAInet::SimpleMatrix),
     gdn.w_gate.as(SHAInet::SimpleMatrix), gdn.w_alpha, gdn.w_beta,
     gdn.conv_q.weight, gdn.conv_k.weight, gdn.conv_v.weight].each_with_index do |m, idx|
      m.rows.times { |i| m.cols.times { |j| m[i, j] = 0.08 * Math.sin((i * 7 + j * 13 + idx * 3) * 0.37) } }
    end

    a, other, c = seeded(5, 16, 0.1), seeded(7, 16, 0.6), seeded(3, 16, 1.3)
    step = ->(x : SHAInet::SimpleMatrix) { {llama.forward_cached(x), gdn.forward(x)} }

    net.clear_cache!
    step.call(a)
    snap = net.snapshot_cache
    snap.bytes.should be > 0

    # A different sequence takes over the cache.
    net.clear_cache!
    step.call(other)
    llama.cached_length.should eq(7)

    net.restore_cache_snapshot!(snap).should be_true
    llama.cached_length.should eq(5)
    resumed_attn, resumed_gdn = step.call(c)

    net.clear_cache!
    step.call(a)
    want_attn, want_gdn = step.call(c)
    max_diff(resumed_attn, want_attn).should be < 1e-5
    max_diff(resumed_gdn, want_gdn).should be < 1e-6

    # Counter-check: continuing from the other sequence instead gives a different result.
    net.clear_cache!
    step.call(other)
    dirty_attn, dirty_gdn = step.call(c)
    max_diff(dirty_attn, want_attn).should be > 1e-4
    max_diff(dirty_gdn, want_gdn).should be > 1e-6

    # The snapshot is not consumed: it can be restored again.
    net.restore_cache_snapshot!(snap).should be_true
    llama.cached_length.should eq(5)
  end

  it "parks a device-resident q4 KV cache", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no q4 kernels" unless SHAInet::CUDA.prefill_attn_kernels_available? &&
                                    SHAInet::CUDA.kv_q4_kernels_available?

    prev = SHAInet::LlamaBlock.prefill_attn_device_enabled?
    prev_dtype = ENV["SHAINET_KV_DTYPE"]?
    SHAInet::LlamaBlock.prefill_attn_device_enabled = true
    ENV["SHAINET_KV_DTYPE"] = "q4"
    begin
      build = -> do
        blk = SHAInet::LlamaBlock.new(64, 4, 128, num_kv_heads: 2, head_dim: 32)
        blk.w_q = seeded(64, 128, 1.0)
        blk.w_k = seeded(64, 64, 2.0)
        blk.w_v = seeded(64, 64, 3.0)
        blk.w_o = seeded(128, 64, 4.0)
        blk.to_gpu!(quantize: true, bits: 4)
        blk.kv_max_context = 64
        blk.clear_cache!
        blk
      end
      a, other, c = seeded(6, 64, 0.1), seeded(9, 64, 0.7), seeded(3, 64, 1.3)

      blk = build.call
      blk.kv_cache_q4?.should be_true
      blk.forward_cached(a)
      blk.host_kv_stale?.should be_true
      snap = blk.snapshot_kv
      blk.clear_cache!
      blk.forward_cached(other)
      blk.restore_kv!(snap).should be_true
      blk.cached_length.should eq(6)
      resumed = blk.forward_cached(c)

      fresh = build.call
      fresh.forward_cached(a)
      expected = fresh.forward_cached(c)
      max_diff(resumed, expected).should be < 1e-3

      dirty = build.call
      dirty.forward_cached(other)
      max_diff(dirty.forward_cached(c), expected).should be > 1e-3
    ensure
      SHAInet::LlamaBlock.prefill_attn_device_enabled = prev
      prev_dtype ? (ENV["SHAINET_KV_DTYPE"] = prev_dtype) : ENV.delete("SHAINET_KV_DTYPE")
    end
  end
end
