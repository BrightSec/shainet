# Sub-agents: run a delegated task in a fresh, empty context and hand back only its report.
#
# Same model, same process, one KV cache. The parent conversation is parked rather than thrown away:
# its messages stay in RAM and its whole sequence state (every attention KV cache and every Gated
# DeltaNet recurrent state) is copied to host memory with Network#snapshot_cache. The cache is then
# cleared and the sub-agent runs its own tool loop from just the system prompt and the task. When it
# finishes, its final reply is written to a file, the parent's snapshot is copied back, and the reply
# becomes the parent's tool result. Resuming costs a host-to-device copy instead of re-prefilling the
# parent's context, which at tens of thousands of tokens is minutes of prefill.
module AgentDemo
  SUBAGENT_TOOL = "subagent"

  # Reports up to this size are inlined in the parent's tool result; longer ones are previewed and
  # the parent reads the file. Either way the full report is on disk.
  SUBAGENT_INLINE_BYTES  = 8_000
  SUBAGENT_PREVIEW_BYTES = 2_000

  SUBAGENT_PREAMBLE = "You are a sub-agent. Another agent delegated the task below to you and is " \
                      "paused until you finish. You start with no memory of its conversation: " \
                      "everything you know is in this message and in what you look up with your " \
                      "tools. The delegating agent sees ONLY your final reply, not your tool calls or " \
                      "reasoning, so end with a complete, self-contained report of what you found or " \
                      "did, with the file paths, names and numbers it will need. Do not ask questions " \
                      "back; make reasonable assumptions and state them.\n\nTask:\n"

  # Declared like any tool so it appears in the system prompt; the agent loop runs it (it needs the
  # Agent's state), so this handler is never reached.
  def self.subagent_tool : Tool
    Tool.new(
      SUBAGENT_TOOL,
      "Delegate a self-contained task to a sub-agent that starts with an EMPTY context (same tools, " \
      "same workspace) and returns only its final report. Use it when the work's intermediate output " \
      "would flood your context and you need only the conclusion: exploring an unfamiliar part of " \
      "the codebase, reading many files, a long search, or researching web pages. The sub-agent " \
      "cannot see this conversation, so put everything it needs in `task`: the goal, relevant paths, " \
      "constraints, and what the report must contain. Your context is parked while it runs and " \
      "restored unchanged afterwards. A sub-agent cannot start another sub-agent.",
      [ToolParam.new("task", "string", "Complete, standalone instructions for the sub-agent.")]
    ) do |_|
      "Error: #{SUBAGENT_TOOL} is run by the agent loop"
    end
  end

  class Agent
    # Prompt label and nesting depth; a sub-agent runs at depth 1 under the label "Sub-agent".
    @label = "Agent"
    @depth = 0

    private def run_subagent(args : Hash(String, String)) : String
      task = (args["task"]? || "").strip
      return "Error: empty task" if task.empty?
      return "Error: a sub-agent cannot start another sub-agent; do this work yourself" if @depth > 0

      parent_messages = @messages
      parent_recent = @recent_calls
      parent_cache = @cache_ids
      parked_tokens = parent_cache.size

      t_park = Time.instant
      snap = parked_tokens > 0 ? @net.snapshot_cache : nil
      park_secs = (Time.instant - t_park).total_seconds
      preview = task.gsub(/\s+/, " ")
      preview = "#{preview[0, 100]}…" if preview.size > 100
      STDERR.puts ""
      STDERR.puts "  ╭─ sub-agent ─ #{preview}".colorize(:light_magenta)
      if s = snap
        STDERR.puts "  │ parked #{parked_tokens} tok (#{(s.bytes / 1_048_576.0).round(0).to_i} MB in host RAM, " \
                    "#{park_secs.round(2)}s)".colorize(:dark_gray)
      end

      @messages = [] of Message
      @recent_calls = [] of String
      @cache_ids = [] of Int32
      @net.clear_cache!
      @depth += 1
      @label = "Sub-agent"
      t_run = Time.instant
      begin
        chat("#{SUBAGENT_PREAMBLE}#{task}")
        sub_messages = @messages
        sub_tokens = @cache_ids.size
      ensure
        @depth -= 1
        @label = "Agent"
        @messages = parent_messages
        @recent_calls = parent_recent
        resumed = false
        if s = snap
          @net.clear_cache!
          resumed = @net.restore_cache_snapshot!(s)
        end
        if resumed
          @cache_ids = parent_cache
        else
          # Nothing parked, or the cache can no longer hold it: the next prefill rebuilds from scratch.
          reset_cache!
        end
      end

      report = subagent_report(sub_messages)
      interrupted = AgentDemo.cancelled?
      path = File.join(Dir.tempdir, "shainet-subagent-#{Time.utc.to_unix_ms}.md")
      File.write(path, report)
      steps = sub_messages.count { |m| m.role == "assistant" }
      secs = (Time.instant - t_run).total_seconds
      STDERR.puts "  ╰─ sub-agent #{interrupted ? "interrupted" : "done"} · #{steps} step(s) · " \
                  "#{sub_tokens || 0} tok · #{secs.round(0).to_i}s · report #{path}".colorize(:light_magenta)
      STDERR.puts "    resumed #{parked_tokens} tok #{resumed ? "from the snapshot (no re-prefill)" : "-- will re-prefill"}".colorize(:dark_gray) if parked_tokens > 0

      head = interrupted ? "[The user interrupted the sub-agent; this is its partial work.]\n\n" : ""
      if report.bytesize <= SUBAGENT_INLINE_BYTES
        "#{head}Sub-agent report (also saved to #{path}):\n\n#{report}"
      else
        "#{head}Sub-agent report is #{report.bytesize} bytes, saved to #{path}. Read it with read_file " \
        "in line ranges if the preview is not enough.\n\n--- preview ---\n" \
        "#{report.byte_slice(0, SUBAGENT_PREVIEW_BYTES).scrub}\n..."
      end
    end

    # The sub-agent's last reply with its reasoning and any tool-call markup removed.
    private def subagent_report(messages : Array(Message)) : String
      messages.reverse_each do |m|
        next unless m.role == "assistant"
        text = m.content.gsub(/<think>.*?<\/think>/m, "").gsub(/<tool_call>.*?<\/tool_call>/m, "").strip
        # An unclosed think block (cut off by an interrupt) is all reasoning, not a report.
        text = text.sub(/<think>.*\z/m, "").strip
        return text unless text.empty?
      end
      "(the sub-agent finished without a final reply)"
    end
  end
end
