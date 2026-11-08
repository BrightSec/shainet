require "json"
require "random/secure"
require "./tool_protocol"

# OpenAI Responses API (POST /v1/responses) translation for openai_server.cr.
#
# The server generates from a chat-style message list. This module converts a Responses request into
# that list and builds Responses-shaped output (JSON and SSE events) from a finished generation. It has
# no dependency on the server so it can be specced without loading a model.
#
# Supported: string or item-array `input`, `instructions`, message items (system/developer/user/
# assistant, text parts), function_call and function_call_output items, function tools,
# tool_choice none/auto/forced, text.format json_schema/json_object, streaming.
# Not supported: previous_response_id (the server is stateless; send the whole conversation), stored
# items (item_reference), built-in tools (web_search etc.), image/file input parts.
module OpenAIResponses
  extend self

  record FunctionCall, call_id : String, name : String, arguments : String

  # One chat message, in the shape the server's prompt builder understands.
  record InputMessage, role : String, content : String?,
    calls : Array(FunctionCall) = [] of FunctionCall, tool_call_id : String? = nil,
    reasoning : String? = nil

  # A parsed tool call ready to be returned to the client.
  record OutputCall, call_id : String, name : String, arguments : String

  class RequestError < Exception
  end

  # Convert `input` (+ `instructions`) into chat messages.
  def parse_input(input : JSON::Any?, instructions : String? = nil) : Array(InputMessage)
    raise RequestError.new("'input' is required.") if input.nil?
    msgs = [] of InputMessage
    msgs << InputMessage.new("system", instructions) if instructions && !instructions.empty?

    if s = input.as_s?
      msgs << InputMessage.new("user", s)
      return msgs
    end
    items = input.as_a? || raise RequestError.new("'input' must be a string or an array of items.")

    # A reasoning item precedes the assistant output it belongs to; it is attached to that turn.
    pending_reasoning : String? = nil
    items.each do |item|
      h = item.as_h? || raise RequestError.new("each 'input' item must be an object.")
      type = h["type"]?.try(&.as_s?)
      case type
      when "reasoning"
        text = [content_text(h["summary"]?), content_text(h["content"]?)].reject(&.empty?).join("\n")
        pending_reasoning = [pending_reasoning, text].compact.reject(&.empty?).join("\n")
      when "function_call"
        call = FunctionCall.new(
          str(h["call_id"]? || h["id"]?) || raise(RequestError.new("function_call item needs 'call_id'.")),
          str(h["name"]?) || raise(RequestError.new("function_call item needs 'name'.")),
          json_string(h["arguments"]?))
        # Consecutive calls belong to one assistant turn, attached after any prose it wrote.
        if (last = msgs.last?) && last.role == "assistant" && pending_reasoning.nil?
          msgs[-1] = last.copy_with(calls: last.calls + [call])
        else
          msgs << InputMessage.new("assistant", nil, [call], reasoning: pending_reasoning)
          pending_reasoning = nil
        end
      when "function_call_output"
        id = str(h["call_id"]?) || raise(RequestError.new("function_call_output item needs 'call_id'."))
        msgs << InputMessage.new("tool", content_text(h["output"]?), tool_call_id: id)
      when "message", nil
        role = str(h["role"]?) || raise(RequestError.new("message item needs 'role'."))
        role = "system" if role == "developer"
        if role == "assistant"
          msgs << InputMessage.new(role, content_text(h["content"]?), reasoning: pending_reasoning)
          pending_reasoning = nil
        else
          msgs << InputMessage.new(role, content_text(h["content"]?))
        end
      else
        # item_reference, built-in tool calls: nothing the prompt can carry.
      end
    end
    raise RequestError.new("'input' contains no messages.") if msgs.none? { |m| m.role != "system" }
    msgs
  end

  # Function tools, in either the Responses (flat) or Chat (nested `function`) shape.
  def tool_defs(tools : JSON::Any?) : Array(ToolProtocol::FunctionDef)
    defs = [] of ToolProtocol::FunctionDef
    (tools.try(&.as_a?) || [] of JSON::Any).each do |t|
      next unless t.as_h? && t["type"]?.try(&.as_s?) == "function"
      f = t["function"]?.try { |x| x.as_h? ? x : nil } || t
      name = str(f["name"]?)
      next unless name
      defs << ToolProtocol::FunctionDef.new(name, str(f["description"]?) || "", f["parameters"]?)
    end
    defs
  end

  # {tools disabled?, forced function name}
  def tool_choice(tc : JSON::Any?) : Tuple(Bool, String?)
    return {false, nil} unless tc
    return {tc.as_s == "none", nil} if tc.as_s?
    return {false, nil} unless tc.as_h?
    name = str(tc["name"]?) || tc["function"]?.try { |f| f.as_h? ? str(f["name"]?) : nil }
    {false, name}
  end

  # Prompt instruction for `text.format`, or nil when the client asked for plain text.
  def format_instruction(text : JSON::Any?) : String?
    fmt = text.try { |t| t.as_h? ? t["format"]? : nil }
    fmt = nil unless fmt.try(&.as_h?)
    case fmt.try(&.["type"]?).try(&.as_s?)
    when "json_schema"
      schema = fmt.not_nil!["schema"]?
      "Respond with ONLY a JSON value matching this JSON Schema. No prose, no code fences.\n" \
      "#{schema ? schema.to_json : "{}"}"
    when "json_object"
      "Respond with ONLY a valid JSON object. No prose, no code fences."
    end
  end

  # Add `extra` to the leading system message, creating one when there is none.
  def with_system(msgs : Array(InputMessage), extra : String?) : Array(InputMessage)
    return msgs unless extra
    if (first = msgs.first?) && first.role == "system"
      body = first.content.to_s
      msgs = msgs.dup
      msgs[0] = first.copy_with(content: body.empty? ? extra : "#{body}\n\n#{extra}")
      msgs
    else
      [InputMessage.new("system", extra)] + msgs
    end
  end

  # Remove a ```json ... ``` fence around a JSON answer.
  def strip_code_fence(text : String) : String
    if m = text.strip.match(/\A```[a-zA-Z]*\n(.*?)\n?```\z/m)
      m[1].strip
    else
      text
    end
  end

  private def str(v : JSON::Any?) : String?
    v.try(&.as_s?)
  end

  # Arguments arrive as a JSON string; accept an object too and serialize it.
  private def json_string(v : JSON::Any?) : String
    return "{}" unless v
    v.as_s? || v.to_json
  end

  # Text of a content value: a string, or an array of parts whose text is concatenated.
  private def content_text(v : JSON::Any?) : String
    return "" unless v
    return v.as_s if v.as_s?
    parts = v.as_a? || return v.to_json
    parts.compact_map { |p| p.as_s? || (p.as_h? ? p["text"]?.try(&.as_s?) : nil) }.join("\n")
  end

  # Visible answer text while streaming: the raw stream minus <think> and <tool_call> markup.
  #
  # `feed` returns only the new part. A tag split across tokens is held back until it resolves, so a
  # half-written "<tool_" never reaches the client.
  class AnswerStream
    TAGS = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
    getter sent = ""

    def initialize(@split : String -> String)
      @raw = ""
    end

    def feed(piece : String) : String
      @raw += piece
      advance(visible(@raw))
    end

    # Bring the sent text up to `final`. Returns the remaining delta, or "" when `final` is not an
    # extension of what was already sent.
    def advance(target : String) : String
      return "" unless target.starts_with?(@sent) && target.size > @sent.size
      delta = target[@sent.size..]
      @sent = target
      delta
    end

    private def visible(raw : String) : String
      answer = @split.call(raw)
      TAGS.each do |tag|
        (1...tag.size).reverse_each do |k|
          if answer.ends_with?(tag[0, k])
            return answer[0, answer.size - k].rstrip
          end
        end
      end
      answer
    end
  end

  # Ids and output items for one response, shared by the JSON body and the SSE events.
  class Builder
    getter id : String
    getter msg_id : String
    getter created : Int64
    getter model : String

    def initialize(@model : String)
      @id = "resp_#{Random::Secure.hex(12)}"
      @msg_id = "msg_#{Random::Secure.hex(12)}"
      @rs_id = "rs_#{Random::Secure.hex(12)}"
      @created = Time.utc.to_unix
      @fc_ids = {} of String => String
    end

    def fc_id(c : OutputCall) : String
      @fc_ids[c.call_id] ||= "fc_#{Random::Secure.hex(12)}"
    end

    def text_part(j : JSON::Builder, text : String)
      j.object do
        j.field "type", "output_text"
        j.field "text", text
        j.field("annotations") { j.array { } }
      end
    end

    def message_item(j : JSON::Builder, text : String, status : String)
      j.object do
        j.field "type", "message"
        j.field "id", @msg_id
        j.field "status", status
        j.field "role", "assistant"
        j.field "content" do
          j.array { text_part(j, text) unless status == "in_progress" && text.empty? }
        end
      end
    end

    def call_item(j : JSON::Builder, c : OutputCall, status : String, arguments : String = c.arguments)
      j.object do
        j.field "type", "function_call"
        j.field "id", fc_id(c)
        j.field "call_id", c.call_id
        j.field "name", c.name
        j.field "arguments", arguments
        j.field "status", status
      end
    end

    def reasoning_item(j : JSON::Builder, text : String)
      j.object do
        j.field "type", "reasoning"
        j.field "id", @rs_id
        j.field "summary" do
          j.array do
            j.object do
              j.field "type", "summary_text"
              j.field "text", text
            end
          end
        end
      end
    end

    # The full response object. `usage` is nil while the response is still in progress.
    def response(j : JSON::Builder, status : String, text : String = "", reasoning : String = "",
                 calls : Array(OutputCall) = [] of OutputCall, usage : Tuple(Int32, Int32)? = nil,
                 incomplete : Bool = false)
      j.object do
        j.field "id", @id
        j.field "object", "response"
        j.field "created_at", @created
        j.field "status", status
        j.field "model", @model
        j.field "output" do
          j.array do
            message_item(j, text, "completed") unless text.empty?
            calls.each { |c| call_item(j, c, "completed") }
            reasoning_item(j, reasoning) unless reasoning.empty?
          end
        end
        j.field "output_text", text
        if incomplete
          j.field("incomplete_details") { j.object { j.field "reason", "max_output_tokens" } }
        else
          j.field "incomplete_details", nil
        end
        j.field "error", nil
        if u = usage
          j.field "usage" do
            j.object do
              j.field "input_tokens", u[0]
              j.field "output_tokens", u[1]
              j.field "total_tokens", u[0] + u[1]
            end
          end
        else
          j.field "usage", nil
        end
      end
    end

    def response_json(**args) : String
      JSON.build { |j| response(j, **args) }
    end
  end

  # SSE event writer: `event:` line plus a JSON payload carrying the same type and a sequence number.
  class EventWriter
    @seq = 0

    def initialize(&@out : String -> Nil)
    end

    def emit(type : String, &block : JSON::Builder ->)
      payload = JSON.build do |j|
        j.object do
          j.field "type", type
          j.field "sequence_number", @seq
          block.call(j)
        end
      end
      @seq += 1
      @out.call("event: #{type}\ndata: #{payload}\n\n")
    end
  end
end
