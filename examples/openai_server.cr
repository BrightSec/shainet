require "../src/shainet"
require "./tool_protocol"
require "./openai_responses"
require "./qwen_thinking"
require "http/server"
require "json"
require "random/secure"

# OpenAI-compatible HTTP API server backed by SHAInet inference (Network#run).
#
# Exposes a subset of the OpenAI REST API so existing OpenAI client libraries
# (python `openai`, `curl`, LangChain, etc.) can talk to a locally-loaded model:
#
#   GET  /v1/models               — list the loaded model
#   POST /v1/chat/completions      — chat completion (streaming + non-streaming)
#   POST /v1/responses             — Responses API, stateless (streaming + non-streaming)
#   GET  /health                   — liveness probe
#
# Usage:
#   crystal run examples/openai_server.cr -Denable_cuda -- /path/to/model-dir [port] [host]
#   crystal run examples/openai_server.cr -Denable_cuda -- /models/Qwen3-Coder-30B-A3B-Instruct
#
# Load-time defaults match examples/agent.cr: Q4 weights, MoE offload on, q4 KV cache at a 128K
# context, expert cache sized to leave room for the context. Opt out with SHAINET_Q8=1,
# SHAINET_MOE_OFFLOAD=0, SHAINET_KV_DTYPE=fp8|f16, SHAINET_EXPERT_CACHE_MB=<mb>, or SHAINET_FP32=1.
#
# SECURITY: this is an example server with NO authentication by default and it
# binds to 127.0.0.1 (localhost only). Do not expose it to an untrusted network.
# Set SHAINET_API_KEY to require an `Authorization: Bearer <key>` header, and
# pass an explicit host (e.g. 0.0.0.0) only when you understand the exposure.
module OpenAIServer
  # Logged under its own source so server lines are distinguishable from the library's `sha_inet:` ones
  # at a glance, and can be filtered separately.
  Log = ::Log.for("api")
  extend self

  # ---- OpenAI request schema (the fields we honor; unknown fields ignored) ----

  # One tool call, in the shape a client sends back and we emit.
  struct ToolCallRef
    include JSON::Serializable
    getter id : String
    getter type : String = "function"
    getter function : FunctionRef

    def initialize(@id : String, @function : FunctionRef)
    end
  end

  struct FunctionRef
    include JSON::Serializable
    getter name : String
    getter arguments : String = "{}"

    def initialize(@name : String, @arguments : String = "{}")
    end
  end

  struct ChatMessage
    include JSON::Serializable
    getter role : String
    # Nilable because an assistant message that ONLY calls tools has content: null, and clients echo
    # that message straight back into the next request. A non-nilable field here rejected the whole
    # conversation with a 400 on the second turn of every tool-using session.
    getter content : String?
    getter tool_calls : Array(ToolCallRef)?
    getter tool_call_id : String?
    # An assistant turn's reasoning, when the client echoes it back (both names are in use; see
    # chat_completion_json). Qwen3.8 keeps reasoning in history by default (preserve_thinking).
    getter reasoning_content : String?
    getter reasoning : String?

    def initialize(@role : String, @content : String? = nil,
                   @tool_calls : Array(ToolCallRef)? = nil, @tool_call_id : String? = nil,
                   @reasoning_content : String? = nil)
    end

    def reasoning_text : String
      (@reasoning_content || @reasoning || "").strip
    end
  end

  # Sampling settings for one request.
  record Sampling, temperature : Float64, top_p : Float64, top_k : Int32,
    presence : Float64, frequency : Float64

  # Chat messages for a /v1/responses request (see openai_responses.cr).
  def chat_messages(msgs : Array(OpenAIResponses::InputMessage)) : Array(ChatMessage)
    msgs.map do |m|
      calls = m.calls.empty? ? nil : m.calls.map { |c| ToolCallRef.new(c.call_id, FunctionRef.new(c.name, c.arguments)) }
      ChatMessage.new(m.role, m.content, calls, m.tool_call_id, m.reasoning)
    end
  end

  struct ToolDef
    include JSON::Serializable
    getter type : String = "function"
    getter function : ToolFunctionDef
  end

  struct ToolFunctionDef
    include JSON::Serializable
    getter name : String
    getter description : String = ""
    getter parameters : JSON::Any?
  end

  struct ChatCompletionRequest
    include JSON::Serializable
    getter model : String?
    getter messages : Array(ChatMessage)
    getter? stream : Bool = false
    getter temperature : Float64?
    getter top_p : Float64?
    # OpenAI deprecated max_tokens for max_completion_tokens; both are accepted, the new name first.
    getter max_tokens : Int32?
    getter max_completion_tokens : Int32?
    getter seed : Int64?
    getter frequency_penalty : Float64?
    getter presence_penalty : Float64?
    # Not in OpenAI's schema, but vLLM and SGLang accept it and Qwen's own examples send it.
    getter top_k : Int32?
    getter reasoning_effort : String?
    # {"enable_thinking": false} turns thinking off, as with vLLM and SGLang.
    getter chat_template_kwargs : JSON::Any?
    # `stop` may be a single string or an array of strings in the OpenAI API.
    getter stop : JSON::Any?
    getter tools : Array(ToolDef)?
    # "auto", "none", "required", or {type: "function", function: {name: ...}}.
    getter tool_choice : JSON::Any?

    # The function name a client is forcing, if any.
    def forced_tool : String?
      tc = @tool_choice
      return unless tc
      tc["function"]?.try(&.["name"]?).try(&.as_s?)
    end

    def tool_defs : Array(ToolProtocol::FunctionDef)
      (@tools || [] of ToolDef).map do |t|
        ToolProtocol::FunctionDef.new(t.function.name, t.function.description, t.function.parameters)
      end
    end

    def enable_thinking : Bool?
      @chat_template_kwargs.try { |k| k.as_h? ? k["enable_thinking"]?.try(&.as_bool?) : nil }
    end
  end

  # Result of one generation pass.
  record Generation, text : String, prompt_tokens : Int32,
    completion_tokens : Int32, finish_reason : String,
    calls : Array(ToolProtocol::Call) = [] of ToolProtocol::Call,
    tool_defs : Array(ToolProtocol::FunctionDef) = [] of ToolProtocol::FunctionDef,
    reasoning : String = ""

  # Wraps a loaded model + tokenizer and serializes generation. The Network's
  # KV cache is shared mutable state, so only one request may generate at a
  # time — concurrent requests are queued behind @mutex.
  # Raised when a request cannot fit the model's context. Carried as its own type so the handler can
  # answer 400 with a message a client can act on, rather than letting an allocation failure take the
  # process down.
  class ContextLengthError < Exception
  end

  class Engine
    getter model_name : String

    @mutex = Mutex.new
    @nl : Array(Int32)
    @nl2 : Array(Int32)
    @bos : Int32?
    @start_hdr : Int32?
    @end_hdr : Int32?
    @eot : Int32?
    @im_start : Int32?
    @im_end : Int32?
    @stop_ids : Array(Int32)

    def initialize(@net : SHAInet::Network, @tok : SHAInet::BPETokenizer, @model_name : String,
                   @max_context : Int32 = 65536)
      @net.use_kv_cache = true
      # Special tokens used by the supported chat templates (nil when absent).
      @bos = sp("<|begin_of_text|>")
      @start_hdr = sp("<|start_header_id|>")
      @end_hdr = sp("<|end_header_id|>")
      @eot = sp("<|eot_id|>")
      @im_start = sp("<|im_start|>")
      @im_end = sp("<|im_end|>")
      @nl = @tok.encode("\n")
      @nl2 = @tok.encode("\n\n")
      # End-of-turn / end-of-text ids that always terminate generation.
      @stop_ids = [] of Int32
      ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"].each do |name|
        if id = sp(name)
          @stop_ids << id unless @stop_ids.includes?(id)
        end
      end
    end

    private def sp(name : String) : Int32?
      @tok.vocab[name]?
    end

    # Flatten the OpenAI message list into {role, content} pairs the chat template can carry.
    #
    # Three shapes need translating, and the model only understands the last of them:
    #
    #   * the tool DECLARATIONS, which OpenAI passes as a separate `tools` array and this model expects
    #     as XML inside the system message. When the client sent no system message one is created, or
    #     the block would have nowhere to live.
    #   * an ASSISTANT message carrying tool_calls and usually content: null -- the client echoes its
    #     own previous turn back. Re-rendered as the XML the model originally emitted, so the
    #     conversation it sees is the one it wrote rather than a gap where its call used to be.
    #   * a TOOL result, which OpenAI sends as its own role with a tool_call_id. This model was trained
    #     on <tool_response> inside a USER turn, so that is what it gets.
    #
    # Each entry is {role, head, body}. `head` is tokenized on its own, ahead of `body`: an assistant
    # turn's "<think>\n" is encoded exactly as the generation prompt encoded it, so the previous
    # request's prompt stays a token prefix of this one and the cached prefill can be reused.
    #
    # Thinking follows Qwen3.8's chat template: the reasoning-effort line opens the system message,
    # and assistant turns keep their reasoning in a <think> block (preserve_thinking, the template
    # default; an empty block when the client did not echo the reasoning back).
    private def render_messages(messages : Array(ChatMessage),
                                tools : Array(ToolProtocol::FunctionDef),
                                forced_tool : String?,
                                mode : QwenThinking::Mode) : Array(Tuple(String, String, String))
      result = [] of Tuple(String, String, String)
      tool_block = tools.empty? ? "" : ToolProtocol.render_tools(tools)
      tool_block += ToolProtocol.render_forced_choice(forced_tool) if forced_tool && !tools.empty?

      # One system message: effort line, tool declarations, then the client's own system text.
      client_system = messages.select { |m| m.role == "system" }.map { |m| m.content || "" }
      system = [tool_block, client_system.join("\n\n")].reject(&.empty?).join("\n\n")
      system = QwenThinking.with_instruction(system, mode)
      result << {"system", "", system} unless system.empty?

      messages.each do |m|
        case m.role
        when "system"
          next
        when "tool"
          result << {"user", "", "<tool_response>\n#{m.content || ""}\n</tool_response>"}
        when "assistant"
          body = m.content || ""
          if calls = m.tool_calls
            calls.each do |c|
              body += "\n" unless body.empty?
              body += render_call_xml(c)
            end
          end
          if mode.enabled
            result << {"assistant", "<think>\n", "#{m.reasoning_text}\n</think>\n\n#{body}"}
          else
            result << {"assistant", "", body}
          end
        else
          result << {m.role, "", m.content || ""}
        end
      end
      result
    end

    # Re-render a client-supplied tool call as the XML the model emits, so its own prior turn reads
    # back to it in the format it produced. Arguments arrive as a JSON string and are unpacked to one
    # <parameter=> each; a value that is not a plain scalar is written back as JSON, which is what the
    # model was asked to produce for an array or object parameter.
    private def render_call_xml(c : ToolCallRef) : String
      params = String.build do |s|
        JSON.parse(c.function.arguments).as_h.each do |k, v|
          s << "<parameter=" << k << ">\n" << (v.as_s? || v.to_json) << "\n</parameter>\n"
        end
      rescue
        # Unparseable arguments: keep them verbatim rather than silently emitting an empty call.
        s << "<parameter=arguments>\n" << c.function.arguments << "\n</parameter>\n"
      end
      "<tool_call>\n<function=#{c.function.name}>\n#{params}</function>\n</tool_call>"
    end

    # Render an OpenAI message list into prompt token ids using whichever chat
    # template the tokenizer advertises (ChatML, then LLaMA 3, else a plain
    # concatenation), ending with the assistant generation prompt.
    def build_prompt(messages : Array(ChatMessage),
                     tools : Array(ToolProtocol::FunctionDef) = [] of ToolProtocol::FunctionDef,
                     forced_tool : String? = nil,
                     mode : QwenThinking::Mode = QwenThinking.mode(nil)) : Array(Int32)
      ids = [] of Int32
      im_start, im_end = @im_start, @im_end
      bos, start_hdr, end_hdr, eot = @bos, @start_hdr, @end_hdr, @eot
      rendered = render_messages(messages, tools, forced_tool, mode)

      if im_start && im_end
        rendered.each do |role, head, content|
          ids << im_start
          ids.concat(@tok.encode(role))
          ids.concat(@nl)
          ids.concat(@tok.encode(head)) unless head.empty?
          ids.concat(@tok.encode(content))
          ids << im_end
          ids.concat(@nl)
        end
        ids << im_start
        ids.concat(@tok.encode("assistant"))
        ids.concat(@nl)
        # The template's generation prompt: open the thought, or close an empty one to skip it.
        ids.concat(@tok.encode(mode.enabled ? "<think>\n" : QwenThinking::OFF_PRIMER))
      elsif bos && start_hdr && end_hdr && eot
        ids << bos
        rendered.each do |role, head, content|
          ids << start_hdr
          ids.concat(@tok.encode(role))
          ids << end_hdr
          ids.concat(@nl2)
          ids.concat(@tok.encode(head + content))
          ids << eot
        end
        ids << start_hdr
        ids.concat(@tok.encode("assistant"))
        ids << end_hdr
        ids.concat(@nl2)
      else
        ids << (bos || 0)
        messages.each do |m|
          ids.concat(@tok.encode("#{m.role}: #{m.content}"))
          ids.concat(@nl)
        end
      end
      ids
    end

    # Run one generation. Each newly-decoded text piece is yielded to the block
    # (used for SSE streaming); the full text is also returned. Generation is
    # serialized across requests via @mutex.
    # Pre-flight the context budget, returning an error message or nil.
    #
    # Called BEFORE the handler decides between streaming and not, because a streaming response has
    # already sent its 200 and its SSE headers by the time generation starts -- at that point a 400 is
    # no longer possible and the client would get a stream that simply dies. Re-tokenizing the prompt
    # to answer this costs a fraction of one token's generation.
    def context_error(messages : Array(ChatMessage), max_tokens : Int32?,
                      mode : QwenThinking::Mode = QwenThinking.mode(nil)) : String?
      OpenAIServer.context_message(@max_context, build_prompt(messages, mode: mode).size, max_tokens)
    end

    # `max_tokens` is the OpenAI limit (max_completion_tokens / max_output_tokens): an upper bound on
    # EVERY generated token, reasoning included. nil means the client set no limit, and generation may
    # use whatever room the context window has left.
    def generate(messages : Array(ChatMessage), max_tokens : Int32?,
                 sampling : Sampling, mode : QwenThinking::Mode,
                 stop_strings : Array(String), seed : Int64?,
                 tools : Array(ToolProtocol::FunctionDef) = [] of ToolProtocol::FunctionDef,
                 forced_tool : String? = nil, &block : String ->) : Generation
      @mutex.synchronize do
        prompt_ids = build_prompt(messages, tools, forced_tool, mode)
        # Refuse a prompt that cannot fit rather than dying on it.
        #
        # The agent controls its own prompt and compacts when it grows; a server is handed whatever a
        # client sends. Without this, one oversized request exhausts VRAM and takes the process down,
        # losing every other client's session with it. OpenAI's own API answers this case with an
        # error, so a client already knows how to handle it.
        if msg = OpenAIServer.context_message(@max_context, prompt_ids.size, max_tokens)
          raise ContextLengthError.new(msg)
        end
        # Never decode past the KV budget, whatever the limit.
        room = @max_context - prompt_ids.size
        limit = Math.min(max_tokens || Int32::MAX, room)
        if (asked = max_tokens) && asked > room
          Log.info do
            "token limit #{asked} capped to #{room}: #{prompt_ids.size} prompt tokens leave " \
            "#{room} of the #{@max_context}-token context"
          end
        end
        rng = seed ? Random.new(seed) : Random.new
        sampler = SHAInet::Sampler.new(temperature: sampling.temperature, top_k: sampling.top_k,
          top_p: sampling.top_p, rng: rng)
        counts = {} of Int32 => Int32

        # Prefill is timed and logged separately from decode because they fail differently and are
        # bound by different things. A slow prefill means the prompt is large or the cache missed; slow
        # decode means layers did not fit on the card. One combined duration cannot tell an operator
        # which of those is happening, and on this model a host-resident layer costs about a second per
        # TOKEN, so the difference is the whole diagnosis.
        pf = Time.instant
        reused = 0
        begin
          reused = rewind_to_prefix(prompt_ids)
          tail = prompt_ids[reused..]
          logits = @net.run(tail, stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
          # Saved BEFORE generating, at exactly the prompt: the next request in this conversation
          # starts with this prompt and can rewind to here.
          @net.save_cache!(prompt_ids.size)
          @saved_prompt = prompt_ids
        rescue ex
          forget_cache!
          raise ex
        end
        pf_secs = (Time.instant - pf).total_seconds
        filled = prompt_ids.size - reused
        Log.info do
          "prefill #{filled} tok#{reused > 0 ? " (reused #{reused} cached)" : ""} in #{pf_secs.round(2)}s " \
          "(#{(filled / (pf_secs > 0 ? pf_secs : 1.0)).round(0).to_i} tok/s)"
        end
        dec = Time.instant
        generated = [] of Int32
        # The generation prompt already opened the thought, so the text starts inside it.
        text = mode.enabled && @im_start ? "<think>\n" : ""
        emitted = 0
        finish = "length"

        # The limit counts reasoning, as OpenAI's max_completion_tokens and max_output_tokens do. A
        # reply whose reasoning used the whole limit ends with finish_reason "length" (Responses:
        # status "incomplete"), which is what OpenAI returns and what clients already handle.
        while generated.size < limit
          # Let other fibers run.
          #
          # Crystal's HTTP server is fiber-based, and this loop is CPU/GPU bound with no yield point of
          # its own. Without this the WHOLE server is frozen for the duration of a generation -- not
          # just this request: the accept loop stops, health checks time out, and a streaming response
          # cannot flush. At roughly 85 ms per token a 500-token answer is 42 seconds of a server that
          # looks dead to everything else. A streaming request gets away with it by accident, because
          # writing each chunk is I/O and yields; a non-streaming one does no I/O per token at all.
          #
          # Same root cause as the agent's Ctrl-C handler never firing: a tight Crystal loop starves
          # every other fiber, including the ones the runtime needs.
          Fiber.yield
          last = logits.rows - 1
          sampler.apply_presence_frequency!(logits, counts, sampling.presence, sampling.frequency, row: last)
          id = sampler.sample(logits, last)
          break if id < 0 || @stop_ids.includes?(id) ? (finish = "stop"; true) : false
          break unless logits[last, id].finite?

          generated << id
          counts[id] = (counts[id]? || 0) + 1
          text += @tok.decode([id])

          # Offsets here are BYTES. A character split across byte-fallback tokens (an emoji, CJK) is
          # invalid UTF-8 until its last byte arrives, and joining the halves makes the CHARACTER count
          # shrink -- which made a char offset point past the end and raised IndexError mid-reply.
          # An incomplete character is held back until it completes.

          # Honor user-supplied stop sequences: cut at the first occurrence.
          if si = first_stop_index(text, stop_strings)
            piece = text.byte_slice(emitted, si - emitted).scrub
            block.call(piece) unless piece.empty?
            text = text.byte_slice(0, si)
            finish = "stop"
            break
          end

          piece = text.byte_slice(emitted)
          if piece.valid_encoding?
            block.call(piece) unless piece.empty?
            emitted = text.bytesize
          end

          logits = @net.run([id], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
        end
        # A character still incomplete when generation stopped: send what there is, scrubbed.
        if emitted < text.bytesize
          rest = text.byte_slice(emitted).scrub
          block.call(rest) unless rest.empty?
        end
        text = text.scrub

        # Extract any tool calls the model emitted, and report the finish reason OpenAI clients branch
        # on. A client decides whether to run tools by the presence of tool_calls, so a call the parser
        # misses is indistinguishable from a final answer -- which is exactly how a truncated call once
        # got printed as one.
        calls = tools.empty? ? [] of ToolProtocol::Call : ToolProtocol.parse_calls(text)
        finish = "tool_calls" unless calls.empty?
        reasoning, answer = OpenAIServer.split_reasoning(text)
        dec_secs = (Time.instant - dec).total_seconds
        ms_per_tok = generated.size > 0 ? (dec_secs * 1000 / generated.size).round(0).to_i : 0
        Log.info do
          "decode #{generated.size} tok in #{dec_secs.round(2)}s (#{ms_per_tok} ms/tok) " \
          "finish=#{finish} content=#{answer.size}ch" \
          "#{calls.empty? ? "" : " calls=#{calls.map(&.name).join(",")}"}" \
          "#{reasoning.empty? ? "" : " reasoning=#{reasoning.size}ch"}"
        end
        # The failure this turn CANNOT be seen from the numbers above alone, and it is a real one: the
        # model spent its whole token budget inside <think> and was cut off before writing either an
        # answer or a tool call. The reply is then structurally valid and useless -- empty content, no
        # calls -- so a client re-prompts, gets the same thing, and loops. Observed: three consecutive
        # rounds of 512 tokens and ~2100 characters of reasoning, 145 seconds each, with the caller's
        # message list growing assistant>user>assistant>user because it had nothing to act on.
        #
        # Named explicitly with the remedy, because the fix is the CALLER's max_tokens and nothing in
        # the response says so.
        if finish == "length" && answer.empty? && calls.empty? && !reasoning.empty?
          Log.warn do
            "reply is EMPTY: all #{generated.size} tokens went to reasoning (#{reasoning.size} chars) " \
            "and the turn hit the token limit before any answer or tool call. The caller sees " \
            "content=\"\" with finish_reason=length -- raise max_completion_tokens / " \
            "max_output_tokens above #{generated.size} for this prompt, or omit it."
          end
        elsif calls.empty? && ToolProtocol.truncated?(text)
          # A DIFFERENT failure with the same symptom. The model opened <tool_call> and never closed it,
          # so the parser correctly finds nothing and the client sees a reply with no calls -- which it
          # reads as a final answer. The remedy is the same knob, but the cause is not "it thought too
          # long", and conflating the two sends anyone debugging in the wrong direction.
          Log.warn do
            "TRUNCATED tool call: a <tool_call> was opened and never closed (finish=#{finish}), so no " \
            "call could be parsed and the caller will read this as a final answer. Raise the token limit."
          end
        end
        Generation.new(answer, prompt_ids.size, generated.size, finish, calls, tools, reasoning)
      end
    end

    # Prompt the cache was saved at (see Network#save_cache!), or nil.
    @saved_prompt : Array(Int32)? = nil

    # Rewind the cache to the previous request's prompt when this prompt extends it, and return how
    # many tokens are reused; otherwise clear it and return 0.
    #
    # Clients resend the whole conversation each turn, so turn N+1's prompt starts with turn N's
    # prompt. The cache holds more than that -- the reply as generated, <think> included -- while the
    # client echoes the reply back without its reasoning, so the cache itself is never a prefix of
    # the new prompt. The saved point at the end of the previous PROMPT is: rewinding to it keeps the
    # whole earlier history and prefills only the new tail (the echoed reply and the tool results).
    # A different conversation, an edited history, or an identical prompt falls back to a full
    # prefill, which is the old behaviour.
    private def rewind_to_prefix(prompt : Array(Int32)) : Int32
      if (saved = @saved_prompt) && saved.size < prompt.size && prompt[0, saved.size] == saved
        if (pos = @net.restore_cache!) && pos == saved.size
          return pos
        end
      end
      forget_cache!
      0
    end

    private def forget_cache!
      @saved_prompt = nil
      @net.drop_saved_cache!
      @net.clear_cache!
    end

    # Byte offset of the earliest stop sequence in `text`, or nil.
    private def first_stop_index(text : String, stops : Array(String)) : Int32?
      idx = nil
      stops.each do |s|
        next if s.empty?
        if i = text.byte_index(s)
          idx = idx ? Math.min(idx, i) : i
        end
      end
      idx
    end
  end

  # ---- response helpers -------------------------------------------------------

  # A request's sampling: the client's values where given, else Qwen3.8's recommended thinking-mode
  # values (temperature 1.0, top_p 0.95, top_k 20, presence/frequency penalty 0).
  def sampling(temperature : Float64?, top_p : Float64?, top_k : Int32?,
               presence : Float64?, frequency : Float64?) : Sampling
    t = temperature || QwenThinking::TEMPERATURE
    t = 0.01 if t <= 0.0 # the sampler requires > 0; 0.01 is effectively greedy
    p = top_p || QwenThinking::TOP_P
    p = 1.0 unless 0.0 < p <= 1.0
    k = top_k || QwenThinking::TOP_K
    k = QwenThinking::TOP_K if k < 1
    Sampling.new(t, p, k, presence || 0.0, frequency || 0.0)
  end

  # The thinking mode for a request, or the error message for an unsupported effort.
  def thinking_mode(effort : String?, enable_thinking : Bool?) : QwenThinking::Mode | String
    QwenThinking.mode(effort, enable_thinking)
  rescue ex : QwenThinking::EffortError
    ex.message.to_s
  end

  def describe(s : Sampling, mode : QwenThinking::Mode) : String
    effort = !mode.enabled ? "none" : (mode.instruction == QwenThinking::LOW ? "low" : (mode.instruction ? "xhigh" : "medium"))
    "temp=#{s.temperature.round(2)} top_p=#{s.top_p} top_k=#{s.top_k}" \
    "#{s.presence != 0 ? " presence=#{s.presence}" : ""}#{s.frequency != 0 ? " frequency=#{s.frequency}" : ""} " \
    "effort=#{effort}"
  end

  # Error text when the prompt cannot fit the context, else nil.
  #
  # Only the prompt has to fit. A token limit larger than the room left is capped to that room at
  # generation time rather than rejected, as llama.cpp's server and Ollama do. OpenAI rejects it, but
  # its windows are hundreds of thousands of tokens; on a 64K local window, clients sized for those
  # (bright-agent asks for 32768) would otherwise be refused once the history passed 32K, with room
  # still left to answer.
  def context_message(max_context : Int32, prompt : Int32, max_tokens : Int32?) : String?
    return if prompt < max_context
    "This model's maximum context length is #{max_context} tokens, but the prompt alone is " \
    "#{prompt} tokens. Shorten the messages."
  end

  def completion_id : String
    "chatcmpl-#{Random::Secure.hex(12)}"
  end

  # Separate the model's private reasoning from its answer. Returns {reasoning, answer}.
  #
  # Every OpenAI-compatible server that handles reasoning models agrees on the invariant: `content`
  # carries the conclusion only. They differ on the reasoning, and none of them throw it away -- DeepSeek
  # returns it as `reasoning_content` "at the same level as content", vLLM the same and now migrating to
  # `reasoning`, Anthropic as a separate thinking block, and OpenAI's own reasoning models withhold the
  # text but still bill it in usage. An earlier version of this code deleted it, which kept content
  # correct and made the server useless to a client that wants to display or log the chain of thought.
  #
  # An unterminated block -- generation stopped mid-thought -- yields all reasoning and no answer, which
  # is what actually happened: there is no conclusion to report, and inventing one from the scratchpad
  # would be worse than an empty answer.
  def split_reasoning(text : String) : Tuple(String, String)
    reasoning = [] of String
    text.scan(/<think>(.*?)<\/think>/m) { |m| reasoning << m[1].strip }
    answer = text.gsub(/<think>.*?<\/think>/m, "")
    if i = answer.index("<think>")
      reasoning << answer[(i + "<think>".size)..].strip
      answer = answer[0, i]
    end
    # Tool-call markup is not prose and must not travel as content.
    #
    # The calls are parsed from the raw text separately and returned as a structured tool_calls array, so
    # leaving the XML in here would send it TWICE -- once structured and once as text a client would
    # display to its user and echo back on the next turn. Same failure as the <think> leak, one layer
    # down. An unterminated opener is dropped along with everything after it, since a half-written call
    # is not something a caller can use either.
    answer = answer.gsub(/<tool_call>.*?<\/tool_call>/m, "")
    if i = answer.index("<tool_call>")
      answer = answer[0, i]
    end
    {reasoning.reject(&.empty?).join("\n\n"), answer.strip}
  end

  def chat_completion_json(id : String, model : String, gen : Generation) : String
    JSON.build do |j|
      j.object do
        j.field "id", id
        j.field "object", "chat.completion"
        j.field "created", Time.utc.to_unix
        j.field "model", model
        j.field "choices" do
          j.array do
            j.object do
              j.field "index", 0
              j.field "message" do
                j.object do
                  j.field "role", "assistant"
                  # Both names, because the ecosystem is mid-rename: DeepSeek and older vLLM read
                  # `reasoning_content`, current vLLM reads `reasoning` and warns that a client still
                  # reading the old name will silently see nothing. Emitting one would quietly fail for
                  # half of them, and an unknown field is ignored by every client, so emitting both is
                  # the only option with no silent-failure mode. Omitted entirely when there was no
                  # reasoning, so a non-thinking turn does not carry empty fields.
                  unless gen.reasoning.empty?
                    j.field "reasoning_content", gen.reasoning
                    j.field "reasoning", gen.reasoning
                  end
                  # content is null when the turn is only a tool call, which is what OpenAI does and
                  # what a client's own type expects. The XML is stripped either way: a client that
                  # received the raw <tool_call> markup as content would show it to its user and, worse,
                  # echo it back as text on the next turn alongside the structured call.
                  # content and tool_calls are not exclusive. OpenAI's schema allows both, and this model
                  # routinely writes a sentence of intent beside its calls -- measured at 561-703
                  # characters per turn on a real run, all of which was previously thrown away because
                  # this branch set content to null whenever calls were present. A client that shows
                  # intermediate progress to a user had nothing to show.
                  #
                  # null only when the prose is genuinely empty, which is what OpenAI returns and what a
                  # client's own type expects for a pure tool-call turn.
                  if gen.text.empty?
                    j.field "content", nil
                  else
                    j.field "content", gen.text
                  end
                  unless gen.calls.empty?
                    j.field "tool_calls" do
                      j.array do
                        gen.calls.each do |c|
                          definition = gen.tool_defs.find { |d| d.name == c.name }
                          j.object do
                            j.field "id", c.id
                            j.field "type", "function"
                            j.field "function" do
                              j.object do
                                j.field "name", c.name
                                # A JSON *string*, not an object: the client calls JSON.parse on it.
                                j.field "arguments", ToolProtocol.arguments_json(c, definition)
                              end
                            end
                          end
                        end
                      end
                    end
                  end
                end
              end
              j.field "finish_reason", gen.finish_reason
            end
          end
        end
        j.field "usage" do
          j.object do
            j.field "prompt_tokens", gen.prompt_tokens
            j.field "completion_tokens", gen.completion_tokens
            j.field "total_tokens", gen.prompt_tokens + gen.completion_tokens
          end
        end
      end
    end
  end

  # One SSE chat.completion.chunk. `delta_role`/`delta_content` are optional;
  # `finish_reason` is nil for content chunks and set on the final chunk.
  def chunk_json(id : String, model : String, created : Int64,
                 delta_role : String? = nil, delta_content : String? = nil,
                 finish_reason : String? = nil) : String
    JSON.build do |j|
      j.object do
        j.field "id", id
        j.field "object", "chat.completion.chunk"
        j.field "created", created
        j.field "model", model
        j.field "choices" do
          j.array do
            j.object do
              j.field "index", 0
              j.field "delta" do
                j.object do
                  j.field "role", delta_role if delta_role
                  j.field "content", delta_content if delta_content
                end
              end
              j.field "finish_reason", finish_reason
            end
          end
        end
      end
    end
  end

  def error_json(message : String, type : String = "invalid_request_error") : String
    JSON.build do |j|
      j.object do
        j.field "error" do
          j.object do
            j.field "message", message
            j.field "type", type
          end
        end
      end
    end
  end

  def normalize_stop(stop : JSON::Any?) : Array(String)
    return [] of String unless stop
    if s = stop.as_s?
      [s]
    elsif arr = stop.as_a?
      arr.compact_map(&.as_s?)
    else
      [] of String
    end
  end
end

# =============================================================================
# Entry point
# =============================================================================

model_dir = ARGV[0]?
port = (ARGV[1]? || "8080").to_i
host = ARGV[2]? || "127.0.0.1"

# Accept the same three model forms the agent does: a SafeTensors directory, a .gguf FILE, or an Ollama
# model name.
#
# The server only ever accepted a directory, so pointing it at the .gguf everything else in this repo
# runs was rejected by the usage check -- and passing the containing directory instead got further and
# then failed looking for config.json, which reads like a corrupt download rather than a server that
# does not support the format. Same resolution, same tokenizer split, so the two entry points cannot
# drift on which models they accept.
gguf_mode = false
# The name clients will ask for. For an Ollama model this must be the Ollama name, not the resolved
# blob path: the blob is named by its digest, and advertising "sha256-..." in /v1/models made clients
# that validate their configured model against that list (bright-agent) refuse to start.
ollama_name : String? = nil
if model_dir && SHAInet::OllamaResolve.ollama_name?(model_dir)
  resolved = SHAInet::OllamaResolve.resolve(model_dir)
  unless resolved
    STDERR.puts "Ollama model '#{model_dir}' not found. Is it pulled?"
    exit 1
  end
  STDERR.puts "Resolved Ollama '#{model_dir}' -> #{resolved}"
  ollama_name = model_dir
  model_dir = resolved
  gguf_mode = true
elsif model_dir && File.file?(model_dir)
  gguf_mode = true
end

unless model_dir && (Dir.exists?(model_dir) || File.file?(model_dir))
  STDERR.puts "Usage: openai_server <model-dir | gguf-file | ollama-name> [port] [host]"
  STDERR.puts "  openai_server ~/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf 8080"
  STDERR.puts "  openai_server qwen3.8:27b"
  STDERR.puts "  openai_server ~/models/Qwen3-0.6B        # SafeTensors directory"
  exit 1
end

# --- Load model (same knobs as llama_chat.cr) ---
STDERR.puts "Loading model from #{model_dir}..."
t = Time.instant
quantize = SHAInet::CUDA.fully_available? && !ENV["SHAINET_FP32"]?
# Same load-time defaults as the agent: Q4 and MoE offload are what a large model needs to fit on one
# consumer GPU, so they are on unless opted out (SHAINET_Q8=1, SHAINET_MOE_OFFLOAD=0). SHAINET_Q4 is
# still accepted but is now the default.
ENV["SHAINET_MOE_OFFLOAD"] = "1" unless ENV.has_key?("SHAINET_MOE_OFFLOAD")
bits = ENV.fetch("SHAINET_Q8", "0") == "1" ? 8 : 4
mode = ENV["SHAINET_FP32"]? ? "fp32" : "Q#{bits}"
offload = ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1"
# q4 KV cache, as in the agent: 4-bit rows with rotated keys, 144 bytes per head per position at
# head_dim 256 against fp8's 256, which is what lets the default context be 128K. SHAINET_KV_DTYPE=fp8
# or f16 opts out.
ENV["SHAINET_KV_DTYPE"] ||= "q4"
STDERR.puts "  Mode: #{mode}#{offload ? " (MoE offload)" : ""}"

# Decide the context BEFORE loading, and tell the loader.
#
# The loader sizes its KV reserve from SHAINET_MAX_CONTEXT at load time, so a context decided after
# the model is in memory is a context the loader never reserved for. The agent had exactly this bug:
# two independent defaults that happened to agree, and when they disagreed the symptom was an OOM
# part-way through a conversation rather than an error at startup. A server is worse off than the
# agent here, because it will accept whatever prompt a client sends.
#
# 131072 by default, with the q4 cache: on a 16 GB card the 27B still places all 64 layers, and a
# 127.8K-token prefill completed with 402 MB free. With SHAINET_KV_DTYPE=fp8 use 65536.
server_context = (ENV["SHAINET_SERVER_CONTEXT"]? || ENV["SHAINET_MAX_CONTEXT"]? || "131072").to_i
ENV["SHAINET_MAX_CONTEXT"] = server_context.to_s
STDERR.puts "  Context: #{server_context} tok (KV reserved for this at load)"

net = SHAInet::HFLoader.load(model_dir, quantize: quantize, bits: bits)
# A GGUF carries its own tokenizer; there is no tokenizer.json beside it. Reading one unconditionally
# is what turned "this format is not supported here" into a missing-file error about the wrong file.
tokenizer =
  if gguf_mode
    SHAInet::GGUF.extract_tokenizer(model_dir)
  else
    path = ENV["SHAINET_TOKENIZER_PATH"]? || File.join(model_dir, "tokenizer.json")
    unless File.exists?(path)
      STDERR.puts "Error: tokenizer.json not found at #{path} (set SHAINET_TOKENIZER_PATH to override)."
      exit 1
    end
    SHAInet::BPETokenizer.from_hf(path)
  end
STDERR.puts "Loaded in #{(Time.instant - t).total_seconds.round(1)}s (vocab #{tokenizer.vocab.size})"

# Size the MoE expert cache to leave room for the KV cache, as the agent does. Without this the cache
# defaults to 70% of free VRAM and competes with the context the loader just reserved for.
# SHAINET_EXPERT_CACHE_MB overrides (0 disables). Dense models have no experts, so nothing is reserved.
has_moe = net.hidden_layers.any? { |l| l.is_a?(SHAInet::LlamaBlock) && l.as(SHAInet::LlamaBlock).ffn.is_a?(SHAInet::MoEFF) }
if has_moe && offload && !ENV["SHAINET_EXPERT_CACHE_MB"]? && (info = SHAInet::CUDA.memory_info)
  kv_reserve = server_context.to_u64 * 128_u64 * 1024_u64
  reserve = 6_u64 * 1024 * 1024 * 1024 + kv_reserve
  free = info[:free]
  budget_mb = free > reserve ? ((free - reserve) // (1024_u64 * 1024_u64)) : 0_u64
  ENV["SHAINET_EXPERT_CACHE_MB"] = budget_mb.to_s
  STDERR.puts "  Expert cache budget: #{budget_mb} MB " \
              "(free #{free // (1024*1024)} MB − 6GB − #{kv_reserve // (1024*1024)} MB for #{server_context} tok context)"
end

# Clients display and echo this, so drop the .gguf extension rather than advertising a filename.
model_name = ollama_name || File.basename(model_dir.rstrip("/")).sub(/\.gguf$/i, "")
# Every id /v1/models lists for this one model. A namespaced Ollama name is also listed without its
# namespace ("incu6us/qwen3.8-gsq:27b-iq3_s" and "qwen3.8-gsq:27b-iq3_s"), since clients are
# configured with either form.
model_ids = [model_name]
if (n = ollama_name) && n.includes?("/")
  model_ids << n.split("/").last
end
engine = OpenAIServer::Engine.new(net, tokenizer, model_name, server_context)

api_key = ENV["SHAINET_API_KEY"]?
# Token limit for a request that sends none. Unset by default, as on OpenAI: generation may then use
# the room left in the context window. A fixed default cut reasoning models off mid-thought and
# returned empty replies to clients that never asked for a limit. SHAINET_MAX_TOKENS sets one.
default_max_tokens = ENV["SHAINET_MAX_TOKENS"]?.try(&.to_i?)

authorized = ->(req : HTTP::Request) : Bool {
  return true unless k = api_key
  req.headers["Authorization"]? == "Bearer #{k}"
}

server = HTTP::Server.new do |ctx|
  req = ctx.request
  res = ctx.response
  started = Time.instant
  peer = req.remote_address.try(&.to_s) || "-"

  begin
    unless authorized.call(req)
      res.status = HTTP::Status::UNAUTHORIZED
      res.content_type = "application/json"
      res.print OpenAIServer.error_json("Missing or invalid API key.", "invalid_request_error")
      next
    end

    # Accept routes with or without the `/v1` prefix: some OpenAI clients are
    # configured with a base URL that already includes `/v1`, others without.
    route = req.path.sub(/^\/v1/, "")

    case {req.method, route}
    when {"GET", "/health"}
      res.content_type = "application/json"
      res.print %({"status":"ok"})
    when {"GET", "/models"}
      res.content_type = "application/json"
      models = JSON.build do |j|
        j.object do
          j.field "object", "list"
          j.field "data" do
            j.array do
              model_ids.each do |mid|
                j.object do
                  j.field "id", mid
                  j.field "object", "model"
                  j.field "created", Time.utc.to_unix
                  j.field "owned_by", "shainet"
                end
              end
            end
          end
        end
      end
      res.print models
    when {"POST", "/chat/completions"}
      body = req.body.try(&.gets_to_end) || ""
      request =
        begin
          OpenAIServer::ChatCompletionRequest.from_json(body)
        rescue ex : JSON::ParseException | ArgumentError
          res.status = HTTP::Status::BAD_REQUEST
          res.content_type = "application/json"
          res.print OpenAIServer.error_json("Invalid request body: #{ex.message}")
          next
        end

      if request.messages.empty?
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json("'messages' must not be empty.")
        next
      end

      max_tokens = request.max_completion_tokens || request.max_tokens || default_max_tokens
      # Defaults are Qwen3.8's recommended thinking-mode sampling; a client's own values win.
      sampling = OpenAIServer.sampling(request.temperature, request.top_p, request.top_k,
        request.presence_penalty, request.frequency_penalty)
      effort_mode = OpenAIServer.thinking_mode(request.reasoning_effort, request.enable_thinking)
      if effort_mode.is_a?(String)
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json(effort_mode)
        next
      end
      think = effort_mode.as(QwenThinking::Mode)
      stops = OpenAIServer.normalize_stop(request.stop)
      model = request.model || model_name

      # What the caller actually asked for. Logged before generation rather than after, so a request that
      # hangs or dies mid-way still shows what it was -- which is the case where you most want to know.
      # Message ROLES rather than content: the shape is what explains a bad answer (a missing system
      # message, a tool result that never arrived), and prompts are the caller's data, not ours to spill
      # into a log file.
      OpenAIServer::Log.info do
        roles = request.messages.map(&.role).join(">")
        tool_names = request.tool_defs.map(&.name)
        "#{peer} chat model=#{model} msgs=#{request.messages.size} [#{roles}] " \
        "max_tokens=#{max_tokens || "none"} #{OpenAIServer.describe(sampling, think)}" \
        "#{request.stream? ? " stream" : ""}" \
        "#{tool_names.empty? ? "" : " tools=#{tool_names.join(",")}"}" \
        "#{request.forced_tool ? " forced=#{request.forced_tool}" : ""}" \
        "#{stops.empty? ? "" : " stops=#{stops.size}"}"
      end

      if msg = engine.context_error(request.messages, max_tokens, think)
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json(msg, "context_length_exceeded")
        next
      end

      if request.stream?
        res.content_type = "text/event-stream"
        res.headers["Cache-Control"] = "no-cache"
        res.headers["Connection"] = "keep-alive"
        id = OpenAIServer.completion_id
        created = Time.utc.to_unix

        send = ->(chunk : String) {
          res.print "data: #{chunk}\n\n"
          res.flush
        }
        send.call(OpenAIServer.chunk_json(id, model, created, delta_role: "assistant"))

        gen = engine.generate(request.messages, max_tokens, sampling, think, stops, request.seed, request.tool_defs, request.forced_tool) do |piece|
          send.call(OpenAIServer.chunk_json(id, model, created, delta_content: piece))
        end

        send.call(OpenAIServer.chunk_json(id, model, created, finish_reason: gen.finish_reason))
        res.print "data: [DONE]\n\n"
        res.flush
      else
        gen = engine.generate(request.messages, max_tokens, sampling, think, stops, request.seed, request.tool_defs, request.forced_tool) { }
        res.content_type = "application/json"
        res.print OpenAIServer.chat_completion_json(OpenAIServer.completion_id, model, gen)
      end
    when {"POST", "/responses"}
      # Responses API, translated onto the same generation path as chat completions (see
      # openai_responses.cr). Stateless: the client sends the whole conversation in `input`.
      body = req.body.try(&.gets_to_end) || ""
      rq = begin
        JSON.parse(body)
      rescue ex : JSON::ParseException
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json("Invalid request body: #{ex.message}")
        next
      end

      unless rq.as_h?
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json("Request body must be a JSON object.")
        next
      end

      if rq["previous_response_id"]?.try(&.as_s?)
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json("previous_response_id is not supported: this server stores no " \
                                          "responses. Send the full conversation in 'input'.")
        next
      end

      json_format = OpenAIResponses.format_instruction(rq["text"]?)
      tools_off, forced = OpenAIResponses.tool_choice(rq["tool_choice"]?)
      tool_defs = tools_off ? [] of ToolProtocol::FunctionDef : OpenAIResponses.tool_defs(rq["tools"]?)
      messages = begin
        msgs = OpenAIResponses.parse_input(rq["input"]?, rq["instructions"]?.try(&.as_s?))
        OpenAIServer.chat_messages(OpenAIResponses.with_system(msgs, json_format))
      rescue ex : OpenAIResponses::RequestError
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json(ex.message.to_s)
        next
      end

      max_tokens = rq["max_output_tokens"]?.try(&.as_i?) || default_max_tokens
      num = ->(key : String) { rq[key]?.try { |t| t.as_f? || t.as_i?.try(&.to_f) } }
      sampling = OpenAIServer.sampling(num.call("temperature"), num.call("top_p"),
        rq["top_k"]?.try(&.as_i?), nil, nil)
      effort = rq["reasoning"]?.try { |r| r.as_h? ? r["effort"]?.try(&.as_s?) : nil }
      effort_mode = OpenAIServer.thinking_mode(effort, nil)
      if effort_mode.is_a?(String)
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json(effort_mode)
        next
      end
      think = effort_mode.as(QwenThinking::Mode)
      seed = rq["seed"]?.try(&.as_i64?)
      model = rq["model"]?.try(&.as_s?) || model_name
      stream = rq["stream"]?.try(&.as_bool?) || false

      OpenAIServer::Log.info do
        "#{peer} responses model=#{model} msgs=#{messages.size} [#{messages.map(&.role).join(">")}] " \
        "max_tokens=#{max_tokens || "none"} #{OpenAIServer.describe(sampling, think)}#{stream ? " stream" : ""}" \
        "#{tool_defs.empty? ? "" : " tools=#{tool_defs.map(&.name).join(",")}"}" \
        "#{forced ? " forced=#{forced}" : ""}#{json_format ? " json" : ""}"
      end

      if msg = engine.context_error(messages, max_tokens, think)
        res.status = HTTP::Status::BAD_REQUEST
        res.content_type = "application/json"
        res.print OpenAIServer.error_json(msg, "context_length_exceeded")
        next
      end

      rb = OpenAIResponses::Builder.new(model)
      to_calls = ->(gen : OpenAIServer::Generation) do
        gen.calls.map do |c|
          OpenAIResponses::OutputCall.new(c.id, c.name,
            ToolProtocol.arguments_json(c, gen.tool_defs.find { |d| d.name == c.name }))
        end
      end
      final_text = ->(gen : OpenAIServer::Generation) do
        json_format && gen.calls.empty? ? OpenAIResponses.strip_code_fence(gen.text) : gen.text
      end

      if stream
        res.content_type = "text/event-stream"
        res.headers["Cache-Control"] = "no-cache"
        res.headers["Connection"] = "keep-alive"
        ev = OpenAIResponses::EventWriter.new { |s| res.print s; res.flush }
        ev.emit("response.created") { |j| j.field("response") { rb.response(j, "in_progress") } }
        ev.emit("response.in_progress") { |j| j.field("response") { rb.response(j, "in_progress") } }

        answer = OpenAIResponses::AnswerStream.new(->(raw : String) { OpenAIServer.split_reasoning(raw)[1] })
        started_msg = false
        send_delta = ->(delta : String) do
          unless delta.empty?
            # The message item opens on the first visible text, so a pure tool-call turn has none.
            unless started_msg
              started_msg = true
              ev.emit("response.output_item.added") do |j|
                j.field "output_index", 0
                j.field("item") { rb.message_item(j, "", "in_progress") }
              end
              ev.emit("response.content_part.added") do |j|
                j.field "item_id", rb.msg_id
                j.field "output_index", 0
                j.field "content_index", 0
                j.field("part") { rb.text_part(j, "") }
              end
            end
            ev.emit("response.output_text.delta") do |j|
              j.field "item_id", rb.msg_id
              j.field "output_index", 0
              j.field "content_index", 0
              j.field "delta", delta
            end
          end
          nil
        end

        gen = engine.generate(messages, max_tokens, sampling, think, [] of String, seed, tool_defs, forced) do |piece|
          # A JSON answer is sent whole at the end, after its code fence (if any) is removed.
          send_delta.call(answer.feed(piece)) unless json_format
        end

        text = final_text.call(gen)
        send_delta.call(answer.advance(text))
        index = 0
        if started_msg
          ev.emit("response.output_text.done") do |j|
            j.field "item_id", rb.msg_id
            j.field "output_index", 0
            j.field "content_index", 0
            j.field "text", text
          end
          ev.emit("response.content_part.done") do |j|
            j.field "item_id", rb.msg_id
            j.field "output_index", 0
            j.field "content_index", 0
            j.field("part") { rb.text_part(j, text) }
          end
          ev.emit("response.output_item.done") do |j|
            j.field "output_index", 0
            j.field("item") { rb.message_item(j, text, "completed") }
          end
          index = 1
        end
        calls = to_calls.call(gen)
        calls.each do |c|
          ev.emit("response.output_item.added") do |j|
            j.field "output_index", index
            j.field("item") { rb.call_item(j, c, "in_progress", "") }
          end
          ev.emit("response.function_call_arguments.delta") do |j|
            j.field "item_id", rb.fc_id(c)
            j.field "output_index", index
            j.field "delta", c.arguments
          end
          ev.emit("response.function_call_arguments.done") do |j|
            j.field "item_id", rb.fc_id(c)
            j.field "output_index", index
            j.field "arguments", c.arguments
          end
          ev.emit("response.output_item.done") do |j|
            j.field "output_index", index
            j.field("item") { rb.call_item(j, c, "completed") }
          end
          index += 1
        end
        unless gen.reasoning.empty?
          ev.emit("response.output_item.added") do |j|
            j.field "output_index", index
            j.field("item") { rb.reasoning_item(j, gen.reasoning) }
          end
          ev.emit("response.output_item.done") do |j|
            j.field "output_index", index
            j.field("item") { rb.reasoning_item(j, gen.reasoning) }
          end
        end
        incomplete = gen.finish_reason == "length"
        ev.emit(incomplete ? "response.incomplete" : "response.completed") do |j|
          j.field("response") do
            rb.response(j, incomplete ? "incomplete" : "completed", text, gen.reasoning, calls,
              {gen.prompt_tokens, gen.completion_tokens}, incomplete)
          end
        end
      else
        gen = engine.generate(messages, max_tokens, sampling, think, [] of String, seed, tool_defs, forced) { }
        incomplete = gen.finish_reason == "length"
        res.content_type = "application/json"
        res.print rb.response_json(status: incomplete ? "incomplete" : "completed", text: final_text.call(gen),
          reasoning: gen.reasoning, calls: to_calls.call(gen),
          usage: {gen.prompt_tokens, gen.completion_tokens}, incomplete: incomplete)
      end
    else
      res.status = HTTP::Status::NOT_FOUND
      res.content_type = "application/json"
      res.print OpenAIServer.error_json("Unknown route: #{req.method} #{req.path}", "not_found")
    end
  ensure
    # One line per request, on EVERY exit path -- which is why it is an ensure rather than a call at the
    # end of each branch. A client that gets a 404 on a route this server does not implement, or a 400
    # on a body it rejected, previously left no trace at all: the failure was visible only to the
    # client, and from here the server looked idle. That is the first thing anyone needs when a client
    # will not talk to it.
    OpenAIServer::Log.info do
      "#{peer} #{req.method} #{req.path} -> #{res.status.code} " \
      "in #{(Time.instant - started).total_milliseconds.round(0).to_i}ms"
    end
  end
end

address = server.bind_tcp(host, port)
STDERR.puts "OpenAI-compatible API on http://#{address} (model: #{model_name})"
STDERR.puts "  POST /v1/chat/completions   POST /v1/responses   GET /v1/models   GET /health"
STDERR.puts "  tools: OpenAI function calling (tools / tool_choice / role:\"tool\")"
STDERR.puts "  auth: #{api_key ? "Bearer token required (SHAINET_API_KEY)" : "none (localhost only)"}"
server.listen
