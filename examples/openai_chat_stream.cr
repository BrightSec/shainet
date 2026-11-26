# Splitting the model's raw output into reasoning, answer and tool-call markup, for the chat
# completions endpoint (whole responses and SSE streams). Its own file so specs can load it:
# openai_server.cr starts the HTTP server at top level.
require "./openai_responses"

module OpenAIServer
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
  def self.split_reasoning(text : String) : Tuple(String, String)
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

  # The chat-completions SSE stream split the same way as a whole response: each raw piece becomes a
  # reasoning delta and a content delta (either may be ""), with <think> and <tool_call> markup removed
  # and a tag split across tokens held back until it resolves. Tool calls are not streamed; the caller
  # sends the parsed calls once generation ends.
  class ChatStream
    def initialize
      @content = OpenAIResponses::AnswerStream.new(->(raw : String) { OpenAIServer.split_reasoning(raw)[1] })
      @reasoning = OpenAIResponses::AnswerStream.new(->(raw : String) { OpenAIServer.split_reasoning(raw)[0] })
    end

    # {reasoning delta, content delta} for one raw piece.
    def feed(piece : String) : {String, String}
      {@reasoning.feed(piece), @content.feed(piece)}
    end

    # The rest of each, given the final reasoning and answer.
    def finish(reasoning : String, answer : String) : {String, String}
      {@reasoning.advance(reasoning), @content.advance(answer)}
    end
  end
end
