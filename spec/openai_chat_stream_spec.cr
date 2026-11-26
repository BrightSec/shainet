require "spec"
require "../examples/openai_chat_stream"

# The chat-completions stream used to send the raw model output as content: a client showed <think>
# and the <tool_call> XML to its user (seen in the fuzzer lab). ChatStream must split it the way the
# whole response is split, one small piece at a time, with tags that straddle pieces held back.
private def replay(raw : String, size : Int32) : {String, String}
  s = OpenAIServer::ChatStream.new
  reasoning = ""
  content = ""
  i = 0
  while i < raw.size
    r, c = s.feed(raw[i, size])
    reasoning += r
    content += c
    i += size
  end
  final_r, final_c = OpenAIServer.split_reasoning(raw)
  r, c = s.finish(final_r, final_c)
  {reasoning + r, content + c}
end

describe OpenAIServer::ChatStream do
  raw = "<think>\nLet me analyze this finding.\n\nFirst, the source.\n</think>\n\n" \
        "I'll read the file.\n<tool_call>\n<function=read_file>\n<parameter=path>\n" \
        "/app/userspace/libscap/scap_print_event.c\n</parameter>\n</function>\n</tool_call>\n" \
        "<tool_call>\n<function=run_command>\n<parameter=command>\nxxd /payload\n</parameter>\n" \
        "</function>\n</tool_call>"

  [1, 2, 3, 5, 7, 64].each do |size|
    it "streams reasoning and content without markup (#{size}-char pieces)" do
      reasoning, content = replay(raw, size)
      reasoning.should eq("Let me analyze this finding.\n\nFirst, the source.")
      content.should eq("I'll read the file.")
      {reasoning, content}.each do |text|
        %w[<think </think <tool_call </tool_call <function <parameter].each { |tag| text.includes?(tag).should be_false }
      end
    end
  end

  it "streams an unterminated thought as reasoning only" do
    reasoning, content = replay("<think>\nstill thinking about it", 3)
    reasoning.should eq("still thinking about it")
    content.should eq("")
  end

  it "passes a plain answer through as content" do
    reasoning, content = replay("Just an answer, no tags.", 4)
    {reasoning, content}.should eq({"", "Just an answer, no tags."})
  end
end
