require "spec"
require "../examples/openai_responses"

private def j(s : String) : JSON::Any
  JSON.parse(s)
end

# Same reasoning/tool-call stripping the server applies, reduced to what the stream filter needs.
private def answer_of(raw : String) : String
  a = raw.gsub(/<think>.*?<\/think>/m, "")
  a = a[0, a.index!("<think>")] if a.includes?("<think>")
  a = a.gsub(/<tool_call>.*?<\/tool_call>/m, "")
  a = a[0, a.index!("<tool_call>")] if a.includes?("<tool_call>")
  a.strip
end

describe OpenAIResponses do
  describe ".parse_input" do
    it "turns a string input and instructions into system + user messages" do
      msgs = OpenAIResponses.parse_input(j(%("hi")), "be brief")
      msgs.map(&.role).should eq(["system", "user"])
      msgs[0].content.should eq("be brief")
      msgs[1].content.should eq("hi")
    end

    it "reads message items with text parts and maps developer to system" do
      msgs = OpenAIResponses.parse_input(j(<<-JSON))
        [
          {"role": "developer", "content": "rules"},
          {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]}
        ]
        JSON
      msgs.map(&.role).should eq(["system", "user"])
      msgs[1].content.should eq("a\nb")
    end

    it "groups function_call items into one assistant turn and maps outputs to tool messages" do
      msgs = OpenAIResponses.parse_input(j(<<-JSON))
        [
          {"role": "user", "content": "go"},
          {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{\\"path\\":\\"a\\"}"},
          {"type": "function_call", "call_id": "c2", "name": "read", "arguments": {"path": "b"}},
          {"type": "function_call_output", "call_id": "c1", "output": "A"},
          {"type": "reasoning", "summary": []}
        ]
        JSON
      msgs.map(&.role).should eq(["user", "assistant", "tool"])
      msgs[1].content.should be_nil
      msgs[1].calls.map(&.call_id).should eq(["c1", "c2"])
      msgs[1].calls[1].arguments.should eq(%({"path":"b"}))
      msgs[2].tool_call_id.should eq("c1")
      msgs[2].content.should eq("A")
    end

    it "attaches an echoed reasoning item to the assistant turn after it" do
      msgs = OpenAIResponses.parse_input(j(<<-JSON))
        [
          {"role": "user", "content": "go"},
          {"type": "reasoning", "summary": [{"type": "summary_text", "text": "plan it"}]},
          {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
          {"type": "function_call_output", "call_id": "c1", "output": "A"},
          {"type": "reasoning", "summary": [{"type": "summary_text", "text": "done"}]},
          {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}
        ]
        JSON
      msgs.map(&.role).should eq(["user", "assistant", "tool", "assistant"])
      msgs[1].reasoning.should eq("plan it")
      msgs[3].reasoning.should eq("done")
      msgs[3].content.should eq("ok")
    end

    it "rejects a missing input and an input with no messages" do
      expect_raises(OpenAIResponses::RequestError) { OpenAIResponses.parse_input(nil) }
      expect_raises(OpenAIResponses::RequestError) { OpenAIResponses.parse_input(j("[]"), "sys") }
    end
  end

  describe ".tool_defs / .tool_choice" do
    it "accepts flat Responses tools and nested Chat tools, skipping built-ins" do
      defs = OpenAIResponses.tool_defs(j(<<-JSON))
        [
          {"type": "function", "name": "a", "description": "d", "parameters": {"type": "object"}},
          {"type": "function", "function": {"name": "b"}},
          {"type": "web_search"}
        ]
        JSON
      defs.map(&.name).should eq(["a", "b"])
      defs[0].description.should eq("d")
    end

    it "reads none and forced function choices" do
      OpenAIResponses.tool_choice(j(%("none"))).should eq({true, nil})
      OpenAIResponses.tool_choice(j(%("auto"))).should eq({false, nil})
      OpenAIResponses.tool_choice(j(%({"type": "function", "name": "a"}))).should eq({false, "a"})
    end
  end

  describe ".format_instruction / .with_system / .strip_code_fence" do
    it "asks for schema JSON and merges it into the leading system message" do
      ins = OpenAIResponses.format_instruction(j(%({"format": {"type": "json_schema", "schema": {"type": "object"}}})))
      ins.not_nil!.should contain(%({"type":"object"}))
      OpenAIResponses.format_instruction(j(%({"format": {"type": "text"}}))).should be_nil

      msgs = OpenAIResponses.parse_input(j(%("q")), "sys")
      merged = OpenAIResponses.with_system(msgs, "JSON")
      merged.size.should eq(2)
      merged[0].content.should eq("sys\n\nJSON")
      OpenAIResponses.with_system(OpenAIResponses.parse_input(j(%("q"))), "JSON").map(&.role).should eq(["system", "user"])
    end

    it "removes a code fence around JSON" do
      OpenAIResponses.strip_code_fence("```json\n{\"a\":1}\n```").should eq(%({"a":1}))
      OpenAIResponses.strip_code_fence(%({"a":1})).should eq(%({"a":1}))
    end
  end

  describe OpenAIResponses::AnswerStream do
    it "streams only answer text and never a partial tag" do
      s = OpenAIResponses::AnswerStream.new(->(raw : String) { answer_of(raw) })
      out = [] of String
      ["<think>plan", "</think>", "Hello", " wor", "ld <tool_", "call>\n<function=x>", "</function>\n</tool_call>"].each do |p|
        out << s.feed(p)
      end
      out.join.should eq("Hello world")
      out.none?(&.includes?("<")).should be_true
    end

    it "advance sends only the missing tail of the final text" do
      s = OpenAIResponses::AnswerStream.new(->(raw : String) { answer_of(raw) })
      s.feed("Hel")
      s.advance("Hello").should eq("lo")
      s.advance("different").should eq("")
    end
  end

  describe OpenAIResponses::Builder do
    it "builds a completed response with message, function_call and usage" do
      b = OpenAIResponses::Builder.new("m")
      call = OpenAIResponses::OutputCall.new("call_1", "read", %({"path":"a"}))
      r = JSON.parse(b.response_json(status: "completed", text: "hi", reasoning: "r",
        calls: [call], usage: {10, 5}))
      r["object"].should eq("response")
      r["status"].should eq("completed")
      r["output_text"].should eq("hi")
      types = r["output"].as_a.map(&.["type"].as_s)
      types.should eq(["message", "function_call", "reasoning"])
      r["output"][0]["content"][0]["text"].should eq("hi")
      r["output"][1]["call_id"].should eq("call_1")
      r["output"][1]["arguments"].should eq(%({"path":"a"}))
      r["usage"]["total_tokens"].should eq(15)
      r["incomplete_details"].raw.should be_nil
    end

    it "marks a length-limited response incomplete" do
      r = JSON.parse(OpenAIResponses::Builder.new("m").response_json(status: "incomplete",
        usage: {1, 1}, incomplete: true))
      r["incomplete_details"]["reason"].should eq("max_output_tokens")
      r["output"].as_a.should be_empty
    end
  end

  describe OpenAIResponses::EventWriter do
    it "writes typed SSE events with increasing sequence numbers" do
      buf = [] of String
      w = OpenAIResponses::EventWriter.new { |s| buf << s; nil }
      w.emit("response.created", &.field("x", 1))
      w.emit("response.completed", &.field("x", 2))
      buf[0].should start_with("event: response.created\ndata: ")
      JSON.parse(buf[1].split("data: ")[1])["sequence_number"].should eq(1)
    end
  end
end
