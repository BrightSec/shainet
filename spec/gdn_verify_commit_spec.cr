require "./spec_helper"

# A speculative verify pass runs k+1 rows through a Gated DeltaNet block without knowing yet how many
# will be kept. In verify mode the block leaves its recurrent state and conv windows untouched and
# keeps the rows' inputs; commit_verify!(n) then folds in the first n. The state after that must be
# the one n plain rows leave, which is what this checks: the row after the commit has to come out
# the same as after a plain run of those n rows.

private def gdn_verify_block : SHAInet::GatedDeltaNetBlock
  d = 64
  # dv and dk of 32 so the column kernel (the one the verify pass uses) covers the shape.
  blk = SHAInet::GatedDeltaNetBlock.new(d, 2 * d, 4, 2, 32, 32, 4)
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
  4.times { |h| blk.a_log[h] = -2.0 - 0.1 * h; blk.dt_bias[h] = 0.05 * h }
  fill.call(blk.out_norm.gamma.as(SHAInet::SimpleMatrix), 1.0, 0.2)
  ffn = blk.ffn.as(SHAInet::SwiGLUFF)
  fill.call(ffn.gate_proj.as(SHAInet::SimpleMatrix), 0.05, 1.1)
  fill.call(ffn.up_proj.as(SHAInet::SimpleMatrix), 0.05, 1.2)
  fill.call(ffn.down_proj.as(SHAInet::SimpleMatrix), 0.05, 1.3)
  blk.to_gpu!(quantize: true, bits: 8)
  blk
end

private def gdn_rows(x : SHAInet::SimpleMatrix, from : Int32, n : Int32) : SHAInet::CudaMatrix
  m = SHAInet::CudaMatrix.new(n, x.cols)
  n.times { |r| x.cols.times { |c| m.raw_data[r * x.cols + c] = x[from + r, c].to_f32 } }
  m.mark_host_modified!
  m.sync_to_device!
  m
end

private def gdn_run(blk : SHAInet::GatedDeltaNetBlock, x : SHAInet::SimpleMatrix, from : Int32, n : Int32) : Array(Float32)
  inp = gdn_rows(x, from, n)
  res = blk.forward_cached_device(inp) || raise "device chain declined"
  res.sync_from_device! if res.device_dirty?
  out_rows = Array(Float32).new(res.rows * res.cols) { |i| res.raw_data[i] }
  inp.free!
  out_rows
end

private def gdn_cos(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0
  na = 0.0
  nb = 0.0
  a.size.times do |i|
    dot += a[i].to_f64 * b[i]
    na += a[i].to_f64 ** 2
    nb += b[i].to_f64 ** 2
  end
  dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-30)
end

describe "Gated DeltaNet verify and commit" do
  it "commits exactly the kept rows", tags: "cuda" do
    pending! "no CUDA" unless SHAInet::CUDA.fully_available?
    pending! "no verify kernels" unless SHAInet::CUDA.gdn_verify_kernels_available?
    pending! "no mixer kernels" unless SHAInet::CUDA.gdn_mixer_kernels_available?

    blk = gdn_verify_block
    x = SHAInet::SimpleMatrix.new(8, 64)
    x.rows.times { |i| x.cols.times { |j| x[i, j] = (0.5 * Math.sin(1.5 + (i * 11 + j * 5) * 0.21)).to_f32 } }

    # Reference: 2 prefix rows, then 3 rows, then the probe row 7.
    blk.clear_cache!
    gdn_run(blk, x, 0, 2)
    plain = gdn_run(blk, x, 2, 3)
    ref_next = gdn_run(blk, x, 7, 1)

    # Verify 5 rows (2..6), keep 3.
    blk.clear_cache!
    gdn_run(blk, x, 0, 2)
    SHAInet::GatedDeltaNetBlock.verify = true
    verified = begin
      gdn_run(blk, x, 2, 5)
    ensure
      SHAInet::GatedDeltaNetBlock.verify = false
    end
    blk.verify_rows.should eq(5)
    # The verify outputs of the kept rows are those of a plain run.
    gdn_cos(verified[0, 3 * 64], plain).should be > 0.99999
    blk.commit_verify!(3)
    blk.verify_rows.should eq(0)
    got_next = gdn_run(blk, x, 7, 1)
    err_kept = gdn_max_diff(got_next, ref_next)

    # Control: committing a different row count leaves a different state, so the check is sensitive
    # to which rows were folded in.
    blk.clear_cache!
    gdn_run(blk, x, 0, 2)
    SHAInet::GatedDeltaNetBlock.verify = true
    begin
      gdn_run(blk, x, 2, 5)
    ensure
      SHAInet::GatedDeltaNetBlock.verify = false
    end
    blk.commit_verify!(5)
    wrong_next = gdn_run(blk, x, 7, 1)
    err_wrong = gdn_max_diff(wrong_next, ref_next)

    err_kept.should be < 1e-5
    err_wrong.should be > 100 * err_kept + 1e-4
  end
end

private def gdn_max_diff(a : Array(Float32), b : Array(Float32)) : Float64
  (0...a.size).max_of { |i| (a[i] - b[i]).abs.to_f64 }
end
