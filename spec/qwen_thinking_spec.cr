require "spec"
require "../examples/qwen_thinking"

describe QwenThinking do
  it "defaults to xhigh, the chat template's default effort" do
    m = QwenThinking.mode(nil)
    m.enabled.should be_true
    m.instruction.should eq(QwenThinking::XHIGH)
  end

  it "maps OpenAI efforts onto Qwen3.8's three levels" do
    QwenThinking.mode("high").instruction.should eq(QwenThinking::XHIGH)
    QwenThinking.mode("xhigh").instruction.should eq(QwenThinking::XHIGH)
    QwenThinking.mode("medium").instruction.should be_nil
    QwenThinking.mode("medium").enabled.should be_true
    QwenThinking.mode("low").instruction.should eq(QwenThinking::LOW)
    QwenThinking.mode("minimal").instruction.should eq(QwenThinking::LOW)
  end

  it "turns thinking off for none and for enable_thinking=false" do
    QwenThinking.mode("none").enabled.should be_false
    QwenThinking.mode("high", enable_thinking: false).enabled.should be_false
    QwenThinking.mode(nil, enable_thinking: true).enabled.should be_true
  end

  it "rejects an unknown effort" do
    expect_raises(QwenThinking::EffortError) { QwenThinking.mode("extreme") }
  end

  it "puts the effort line first in the system message" do
    QwenThinking.with_instruction("tools", QwenThinking.mode(nil)).should eq("#{QwenThinking::XHIGH}\n\ntools")
    QwenThinking.with_instruction("", QwenThinking.mode("low")).should eq(QwenThinking::LOW)
    QwenThinking.with_instruction("tools", QwenThinking.mode("medium")).should eq("tools")
  end
end
