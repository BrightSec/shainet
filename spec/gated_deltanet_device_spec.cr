require "./spec_helper"

# The device recurrence must agree with the host one, which is the reference implementation and the
# thing already pinned against real transformers. A fast kernel that computes something else is
# worse than the slow loop it replaces, so equivalence is asserted before any timing is believed.
#
# SHAINET_GDN_DEVICE=0 forces the host path, which is how the A/B is taken and how the fallback
# stays exercised rather than rotting.

def gdev_fill!(m : SHAInet::SimpleMatrix, seed : Int32)
  r = Random.new(seed)
  m.rows.times { |i| m.cols.times { |j| m[i, j] = (r.next_float - 0.5) * 0.4 } }
  m
end

def gdev_block(nv = 4, nk = 2, hk = 8, hv = 8, d = 32, ff = 64)
  b = SHAInet::GatedDeltaNetBlock.new(d, ff, num_v_heads: nv, num_k_heads: nk,
    head_k: hk, head_v: hv, conv_kernel: 4)
  gdev_fill!(b.w_q.as(SHAInet::SimpleMatrix), 1)
  gdev_fill!(b.w_k.as(SHAInet::SimpleMatrix), 2)
  gdev_fill!(b.w_v.as(SHAInet::SimpleMatrix), 3)
  gdev_fill!(b.w_o.as(SHAInet::SimpleMatrix), 4)
  gdev_fill!(b.w_gate.as(SHAInet::SimpleMatrix), 5)
  gdev_fill!(b.w_alpha.as(SHAInet::SimpleMatrix), 6)
  gdev_fill!(b.w_beta.as(SHAInet::SimpleMatrix), 7)
  gdev_fill!(b.conv_q.weight, 8)
  gdev_fill!(b.conv_k.weight, 9)
  gdev_fill!(b.conv_v.weight, 10)
  nv.times { |h| b.a_log[h] = -0.4 + h * 0.1; b.dt_bias[h] = 0.05 }
  b
end

def gdev_worst(a : SHAInet::SimpleMatrix, b : SHAInet::SimpleMatrix) : Float64
  w = 0.0
  a.rows.times { |i| a.cols.times { |j| w = Math.max(w, (a[i, j].to_f64 - b[i, j].to_f64).abs) } }
  w
end

def gdev_scale(m : SHAInet::SimpleMatrix) : Float64
  s = 0.0
  m.rows.times { |i| m.cols.times { |j| s = Math.max(s, m[i, j].to_f64.abs) } }
  s
end

describe "GatedDeltaNetBlock device recurrence" do
  it "matches the host recurrence at a shape the column-parallel kernel serves" do
    pending! "requires the gated_delta_rule kernel" unless SHAInet::CUDA.gated_delta_rule_available?
    # head_v 32 (a multiple of the kernel's 32-column slice) and head_k 32. The default block above
    # uses head_v 8, which that kernel declines, so only this case covers it.
    x = gdev_fill!(SHAInet::SimpleMatrix.new(9, 32), 32)

    ENV["SHAINET_GDN_DEVICE"] = "0"
    host = gdev_block(nv: 4, nk: 2, hk: 32, hv: 32).forward(x)
    ENV.delete("SHAINET_GDN_DEVICE")
    dev = gdev_block(nv: 4, nk: 2, hk: 32, hv: 32).forward(x)

    (gdev_worst(dev, host) / gdev_scale(host)).should be < 1e-4
  end

  it "matches the host recurrence over a whole sequence" do
    pending! "requires the gated_delta_rule kernel" unless SHAInet::CUDA.gated_delta_rule_available?
    x = gdev_fill!(SHAInet::SimpleMatrix.new(9, 32), 30)

    ENV["SHAINET_GDN_DEVICE"] = "0"
    host = gdev_block.forward(x)
    ENV.delete("SHAINET_GDN_DEVICE")
    dev = gdev_block.forward(x)

    # Float32 with different summation orders, so relative rather than exact. The bound is far
    # tighter than any wiring error survives: a wrong head mapping or a missing decay moves this by
    # order 1, not by 1e-4.
    (gdev_worst(dev, host) / gdev_scale(host)).should be < 1e-4
  end

  it "carries state across calls the same way on both paths" do
    pending! "requires the gated_delta_rule kernel" unless SHAInet::CUDA.gated_delta_rule_available?
    # Prefill then a decode step, which is what generation does and where a state that lives on the
    # device could silently diverge from the host's.
    x = gdev_fill!(SHAInet::SimpleMatrix.new(6, 32), 31)
    head = SHAInet::SimpleMatrix.new(5, 32)
    5.times { |t| 32.times { |j| head[t, j] = x[t, j] } }
    last = SHAInet::SimpleMatrix.new(1, 32)
    32.times { |j| last[0, j] = x[5, j] }

    ENV["SHAINET_GDN_DEVICE"] = "0"
    hb = gdev_block
    hb.clear_cache!
    hb.forward(head)
    host = hb.forward_cached(last)
    ENV.delete("SHAINET_GDN_DEVICE")

    db = gdev_block
    db.clear_cache!
    db.forward(head)
    dev = db.forward_cached(last)

    (gdev_worst(dev, host) / gdev_scale(host)).should be < 1e-4
  end

  it "starts a fresh sequence after clear_cache!" do
    pending! "requires the gated_delta_rule kernel" unless SHAInet::CUDA.gated_delta_rule_available?
    # The device state is reused across calls, so clear_cache! must actually free it. Without this
    # a second sequence would continue from the first one's tail -- wrong output, no error.
    x = gdev_fill!(SHAInet::SimpleMatrix.new(5, 32), 32)
    b = gdev_block
    b.clear_cache!
    first = b.forward(x)
    b.clear_cache!
    again = b.forward(x)
    gdev_worst(first, again).should be < 1e-6
  end

  it "differs when the state is NOT cleared, so the check above is not vacuous" do
    pending! "requires the gated_delta_rule kernel" unless SHAInet::CUDA.gated_delta_rule_available?
    x = gdev_fill!(SHAInet::SimpleMatrix.new(5, 32), 32)
    b = gdev_block
    b.clear_cache!
    first = b.forward(x)
    second = b.forward(x) # no clear: state carries
    gdev_worst(first, second).should be > 1e-6
  end
end
