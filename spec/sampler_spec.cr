require "./spec_helper"

private def logits_row(values : Array(Float32)) : SHAInet::SimpleMatrix
  m = SHAInet::SimpleMatrix.new(1, values.size)
  values.each_with_index { |v, j| m.data[j] = v }
  m
end

describe SHAInet::Sampler do
  describe ".greedy" do
    it "picks the argmax" do
      m = logits_row([0.1_f32, 3.2_f32, -1.0_f32, 2.9_f32])
      SHAInet::Sampler.greedy(m).should eq(1)
    end

    it "never selects NaN/Inf" do
      m = logits_row([Float32::NAN, Float32::INFINITY, 0.5_f32, -Float32::INFINITY])
      SHAInet::Sampler.greedy(m).should eq(2)
    end

    it "samples from the requested row" do
      m = SHAInet::SimpleMatrix.new(2, 3)
      [1.0_f32, 0.0_f32, 0.0_f32].each_with_index { |v, j| m.data[j] = v }
      [0.0_f32, 0.0_f32, 5.0_f32].each_with_index { |v, j| m.data[3 + j] = v }
      SHAInet::Sampler.greedy(m, 0).should eq(0)
      SHAInet::Sampler.greedy(m, 1).should eq(2)
    end
  end

  describe "top_p" do
    it "samples only from the smallest set reaching top_p" do
      # Probabilities ~0.84, 0.11, 0.04, ...: top_p 0.9 keeps the first two, never the rest.
      m = logits_row([4.0_f32, 2.0_f32, 1.0_f32, 0.5_f32, 0.0_f32])
      s = SHAInet::Sampler.new(temperature: 1.0, top_k: 5, top_p: 0.9, rng: Random.new(5))
      picks = Array.new(500) { s.sample(m) }
      picks.all? { |id| id <= 1 }.should be_true
      picks.includes?(1).should be_true
    end

    it "always keeps the top token, however small top_p is" do
      m = logits_row([1.0_f32, 0.9_f32, 0.8_f32])
      s = SHAInet::Sampler.new(temperature: 1.0, top_k: 3, top_p: 0.01, rng: Random.new(2))
      50.times { s.sample(m).should eq(0) }
    end

    it "rejects a top_p outside (0, 1]" do
      expect_raises(ArgumentError) { SHAInet::Sampler.new(top_p: 0.0) }
      expect_raises(ArgumentError) { SHAInet::Sampler.new(top_p: 1.5) }
    end
  end

  describe "#apply_presence_frequency!" do
    it "subtracts presence once and frequency per occurrence" do
      m = logits_row([5.0_f32, 5.0_f32, 5.0_f32])
      s = SHAInet::Sampler.new
      s.apply_presence_frequency!(m, {0 => 1, 1 => 3}, presence: 0.5, frequency: 0.25)
      m.data[0].should eq(4.25_f32) # 5 - 0.5 - 0.25
      m.data[1].should eq(3.75_f32) # 5 - 0.5 - 0.75
      m.data[2].should eq(5.0_f32)  # never generated
    end

    it "is a no-op when both are zero" do
      m = logits_row([1.0_f32, 2.0_f32])
      SHAInet::Sampler.new.apply_presence_frequency!(m, {0 => 4}, 0.0, 0.0)
      m.data[0].should eq(1.0_f32)
    end
  end

  describe "#sample" do
    it "is deterministic with a seeded RNG" do
      m = logits_row([1.0_f32, 2.0_f32, 3.0_f32, 0.5_f32, -2.0_f32])
      a = SHAInet::Sampler.new(temperature: 0.8, top_k: 3, rng: Random.new(42))
      b = SHAInet::Sampler.new(temperature: 0.8, top_k: 3, rng: Random.new(42))
      picks_a = Array.new(20) { a.sample(m) }
      picks_b = Array.new(20) { b.sample(m) }
      picks_a.should eq(picks_b)
    end

    it "with top_k = 1 is equivalent to greedy" do
      m = logits_row([0.3_f32, 1.7_f32, 1.69_f32, -0.4_f32, 0.9_f32])
      s = SHAInet::Sampler.new(temperature: 0.7, top_k: 1, rng: Random.new(1))
      10.times { s.sample(m).should eq(SHAInet::Sampler.greedy(m)) }
    end

    it "never selects a NaN logit" do
      m = logits_row([Float32::NAN, 2.0_f32, Float32::NAN, 1.5_f32, Float32::NAN])
      s = SHAInet::Sampler.new(temperature: 1.0, top_k: 5, rng: Random.new(7))
      200.times do
        id = s.sample(m)
        (id == 1 || id == 3).should be_true
      end
    end

    it "only samples from the top-k set" do
      # One dominant + k-1 close values; everything else far lower.
      vals = Array(Float32).new(100) { |j| j < 5 ? (10.0_f32 - j) : -50.0_f32 }
      m = logits_row(vals)
      s = SHAInet::Sampler.new(temperature: 1.0, top_k: 5, rng: Random.new(3))
      300.times { s.sample(m).should be < 5 }
    end
  end

  describe "#apply_repetition_penalty!" do
    it "divides positive logits and multiplies negative ones" do
      m = logits_row([2.0_f32, -2.0_f32, 0.0_f32, 4.0_f32])
      s = SHAInet::Sampler.new(repetition_penalty: 2.0)
      s.apply_repetition_penalty!(m, [0, 1, 3])
      m.data[0].should eq(1.0_f32)  # 2.0 / 2
      m.data[1].should eq(-4.0_f32) # -2.0 * 2
      m.data[2].should eq(0.0_f32)  # untouched
      m.data[3].should eq(2.0_f32)  # 4.0 / 2
    end

    it "is a no-op when penalty == 1.0" do
      m = logits_row([2.0_f32, -2.0_f32])
      s = SHAInet::Sampler.new(repetition_penalty: 1.0)
      s.apply_repetition_penalty!(m, [0, 1])
      m.data[0].should eq(2.0_f32)
      m.data[1].should eq(-2.0_f32)
    end

    it "only penalizes the last `window` ids" do
      m = logits_row([4.0_f32, 4.0_f32, 4.0_f32])
      s = SHAInet::Sampler.new(repetition_penalty: 2.0)
      s.apply_repetition_penalty!(m, [0, 1, 2], window: 1)
      m.data[0].should eq(4.0_f32) # outside window
      m.data[1].should eq(4.0_f32) # outside window
      m.data[2].should eq(2.0_f32) # penalized
    end
  end
end
