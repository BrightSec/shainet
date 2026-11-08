require "json"

# Translation between the OpenAI tool-calling wire format and the XML dialect Qwen3-style models
# actually emit.
#
# Kept separate from both examples/agent.cr and examples/openai_server.cr because both need it and the
# parsing half has already cost real debugging: a tool call that is syntactically complete but
# TRUNCATED looks exactly like a turn with no call at all, and the difference decides whether the loop
# runs a tool or prints a preamble as the final answer. One copy, one set of fixes.
module ToolProtocol
  # A function the model may call, as an OpenAI client declares it.
  #
  # `parameters` is kept as raw JSON rather than parsed into a schema type: clients send arbitrary
  # JSON Schema, only the property names and descriptions are needed to build the prompt, and
  # re-serializing a partially-understood schema would silently drop whatever this code does not model.
  record FunctionDef, name : String, description : String, parameters : JSON::Any?

  # A call the model emitted.
  record Call, id : String, name : String, args : Hash(String, String)

  # Render the tool list in the format the model was trained on.
  #
  # Qwen3-Coder models are trained on this XML dialect, not on JSON function blocks, so a client's
  # JSON Schema is translated rather than passed through. Only names, types and descriptions survive
  # the translation, which is what the model acts on; constraints like enum or minimum are advisory in
  # the prompt at best and are left in the description where the client put them.
  def self.render_tools(tools : Array(FunctionDef)) : String
    String.build do |s|
      s << "# Tools\n\nYou have access to the following functions:\n\n<tools>"
      tools.each do |t|
        s << "\n<function>\n<name>" << t.name << "</name>"
        s << "\n<description>" << t.description << "</description>\n<parameters>"
        each_param(t) do |pname, ptype, pdesc, required|
          s << "\n<parameter>\n<name>" << pname << "</name>\n<type>" << ptype << "</type>"
          s << "\n<description>" << pdesc
          s << " (required)" if required
          s << "</description>\n</parameter>"
        end
        s << "\n</parameters>\n</function>"
      end
      s << "\n</tools>"
      s << "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
      s << "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
      s << "</parameter>\n</function>\n</tool_call>\n\n"
      s << "<IMPORTANT>\n- Function calls MUST be wrapped in <tool_call></tool_call> with an inner"
      s << " <function=...></function> block.\n- Provide any reasoning BEFORE the call, never after.\n"
      s << "- If no function is needed, just answer normally.\n</IMPORTANT>"
    end
  end

  # Walk a JSON Schema's properties. Tolerates a missing or malformed schema by yielding nothing, since
  # a client is free to declare a function that takes no arguments.
  private def self.each_param(t : FunctionDef, &)
    params = t.parameters
    return unless params
    props = params["properties"]?
    return unless props
    required = (params["required"]?.try(&.as_a?) || [] of JSON::Any).compact_map(&.as_s?).to_set
    props.as_h?.try &.each do |name, spec|
      h = spec.as_h?
      next unless h
      yield name,
        (h["type"]?.try(&.as_s?) || "string"),
        (h["description"]?.try(&.as_s?) || ""),
        required.includes?(name)
    end
  end

  # Instruct the model to call one specific function, for a client that sets
  # tool_choice: {type: "function", function: {name: ...}}.
  #
  # The model has no notion of a forced call, so this is a prompt-level instruction rather than a
  # decoding constraint -- it is strong but not a guarantee. It matters that it exists at all: a client
  # may use a forced call as its way of extracting a final structured answer, and silently treating
  # that as "auto" turns a required result into an optional one.
  def self.render_forced_choice(name : String) : String
    "\n\n<REQUIRED>\nYou MUST call the function `#{name}` now, with no prose before or after it.\n" \
    "</REQUIRED>"
  end

  # Parse the model's XML calls. Returns them in emission order.
  def self.parse_calls(text : String) : Array(Call)
    text = text.scrub # never run a regex over invalid UTF-8 from a broken decode
    calls = [] of Call
    i = 0
    text.scan(/<tool_call>(.*?)<\/tool_call>/m) do |m|
      body = m[1]
      fmatch = body.match(/<function=([^>\s]+)>(.*)/m)
      next unless fmatch
      name = fmatch[1].strip
      next if name.empty? || name == "none" # placeholder the model emits when it means "no call"
      args = {} of String => String
      fmatch[2].scan(/<parameter=([^>\s]+)>\n?(.*?)\n?<\/parameter>/m) do |pm|
        args[pm[1].strip] = pm[2]
      end
      i += 1
      calls << Call.new("call_#{i}_#{name}", name, args)
    end
    calls
  end

  # True when a call was opened and never closed.
  #
  # Counting tags rather than guessing from prose. An earlier attempt inferred truncation from phrases
  # like "let's" or "I'll", passed sixteen hand-written cases, and then failed on the real input --
  # the model had put its preamble inside <think>, so after stripping that the visible text BEGAN with
  # <tool_call> and no prose heuristic could see anything wrong.
  def self.truncated?(text : String) : Bool
    text.split("<tool_call>").size > text.split("</tool_call>").size
  end

  # Turn parsed arguments into the JSON object string an OpenAI client expects in
  # `tool_calls[].function.arguments`, which it will JSON.parse.
  #
  # The XML carries every value as text, so a JSON Schema declaring a number or a boolean would
  # otherwise receive a quoted string and fail the client's own validation. Values are coerced to the
  # type the schema asks for, and left as strings when the schema says nothing or the text does not
  # parse as that type -- a wrong-but-honest string beats a malformed number.
  def self.arguments_json(call : Call, definition : FunctionDef?) : String
    types = {} of String => String
    if d = definition
      each_param(d) { |name, type, _desc, _req| types[name] = type }
    end
    JSON.build do |j|
      j.object do
        call.args.each do |k, v|
          j.field(k) { emit_typed(j, v, types[k]?) }
        end
      end
    end
  end

  private def self.emit_typed(j : JSON::Builder, raw : String, type : String?)
    text = raw.strip
    case type
    when "integer"
      if n = text.to_i64?
        j.number(n)
      else
        j.string(raw)
      end
    when "number"
      if f = text.to_f64?
        j.number(f)
      else
        j.string(raw)
      end
    when "boolean"
      case text.downcase
      when "true"  then j.bool(true)
      when "false" then j.bool(false)
      else              j.string(raw)
      end
    when "array", "object"
      # The model is asked for JSON directly for these; pass it through when it parses, and fall back
      # to a string so a malformed value is visible to the client rather than breaking the envelope.
      begin
        JSON.parse(text).to_json(j)
      rescue
        j.string(raw)
      end
    else
      j.string(raw)
    end
  end
end
