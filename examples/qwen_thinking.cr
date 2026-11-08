# Qwen3.8 thinking controls, as its own chat template implements them.
#
# The template (tokenizer.chat_template in the GGUF) does two things:
#   - reasoning_effort puts one instruction line at the start of the system message: a sentence for
#     xhigh (the default) and for low, nothing for medium;
#   - enable_thinking=false ends the generation prompt with an empty "<think>\n\n</think>\n\n", so
#     the model answers directly.
# Recommended thinking-mode sampling, from the Qwen3.8-27B model card: temperature 1.0, top_p 0.95,
# top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0.
module QwenThinking
  XHIGH = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key " \
          "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and " \
          "clarity in the final answer."
  LOW = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to " \
        "the conclusion without unnecessary elaboration."

  # The primer that disables thinking for one generation, appended after "<|im_start|>assistant\n".
  OFF_PRIMER = "<think>\n\n</think>\n\n"

  TEMPERATURE =  1.0
  TOP_P       = 0.95
  TOP_K       =   20

  # How one request should think. `instruction` is the system-message line (nil for none).
  record Mode, enabled : Bool, instruction : String?

  class EffortError < Exception
  end

  # Resolve an OpenAI reasoning effort (and an explicit enable_thinking) to the template's behaviour.
  # OpenAI's levels map onto Qwen's three: none disables thinking, minimal/low -> low, medium ->
  # medium, high/xhigh/absent -> xhigh (the template default).
  def self.mode(effort : String?, enable_thinking : Bool? = nil) : Mode
    return Mode.new(false, nil) if enable_thinking == false
    case effort.try(&.downcase)
    when nil, "", "high", "xhigh" then Mode.new(true, XHIGH)
    when "medium"                 then Mode.new(true, nil)
    when "low", "minimal"         then Mode.new(true, LOW)
    when "none"                   then Mode.new(false, nil)
    else
      raise EffortError.new("Unsupported reasoning effort '#{effort}'. Supported: none, minimal, low, " \
                            "medium, high, xhigh.")
    end
  end

  # A system message body with the effort line in front, as the template orders it.
  def self.with_instruction(body : String, mode : Mode) : String
    line = mode.instruction
    return body unless line
    body.empty? ? line : "#{line}\n\n#{body}"
  end
end
