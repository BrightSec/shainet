require "spec"
require "../examples/spec_accept"

# The server's speculative loop keeps drafts while the trunk's own sample agrees with them. The order
# matters: each kept draft must reach `accept` (streaming, stop strings, penalty counts) before the
# next row is sampled, and the first disagreeing sample becomes the next token.
describe "OpenAIServer.accept_drafts" do
  it "keeps the agreeing prefix and returns the first disagreeing sample" do
    samples = [11, 12, 99, 14]
    taken = [] of Int32
    order = [] of String
    sample_row = ->(r : Int32) { order << "sample#{r}"; samples[r] }
    accept = ->(t : Int32) { order << "accept#{t}"; taken << t; true }
    a, y, outcome = OpenAIServer.accept_drafts([11, 12, 13], sample_row, accept)
    a.should eq(2)
    y.should eq(99)
    outcome.should eq(OpenAIServer::DraftOutcome::Continue)
    taken.should eq([11, 12])
    order.should eq(%w[sample0 accept11 sample1 accept12 sample2])
  end

  it "takes the bonus row's sample when every draft is kept" do
    samples = [5, 6, 7]
    a, y, outcome = OpenAIServer.accept_drafts([5, 6], ->(r : Int32) { samples[r] }, ->(_t : Int32) { true })
    {a, y, outcome}.should eq({2, 7, OpenAIServer::DraftOutcome::Continue})
  end

  it "stops at the kept draft where accept says stop, without sampling further" do
    sampled = 0
    sample_row = ->(r : Int32) { sampled += 1; [1, 2, 3][r] }
    accept = ->(t : Int32) { t != 2 }
    a, y, outcome = OpenAIServer.accept_drafts([1, 2, 3], sample_row, accept)
    {a, y, outcome}.should eq({2, -1, OpenAIServer::DraftOutcome::Stopped})
    sampled.should eq(2)
  end

  it "ends on a non-finite row" do
    a, _, outcome = OpenAIServer.accept_drafts([4], ->(_r : Int32) { OpenAIServer::NONFINITE }, ->(_t : Int32) { true })
    {a, outcome}.should eq({0, OpenAIServer::DraftOutcome::Nonfinite})
  end
end
