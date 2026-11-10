require "../src/shainet"
require "json"
require "colorize"
require "readline"
require "./agent_workspace"
require "./agent_v4a"
require "./agent_markdown"
require "./qwen_thinking"
require "./agent_browser"

# Agentic chat demo: a multi-turn conversation with a growing context and a
# tool-calling loop, built entirely on Network#run. Think of it as a very light
# Claude/Copilot/Kiro-style CLI agent.
#
# The model (Qwen3-Coder-style: ChatML + XML tool calls) can call local tools;
# this harness parses the calls, runs the tools, feeds the results back, and
# lets the model continue until it produces a plain answer.
#
# Usage:
#   SHAINET_Q4=1 SHAINET_MOE_OFFLOAD=1 \
#     crystal run examples/agent.cr --release -Denable_cuda -- ~/models/Qwen3-Coder-30B-A3B-Instruct
#
# Built-in tools are intentionally read-only (list_directory, read_file). Add
# your own in build_tools — that is the extension point for an MCP bridge etc.

@[Link("c")]
lib LibWinsize
  struct Winsize
    ws_row : LibC::UShort
    ws_col : LibC::UShort
    ws_xpixel : LibC::UShort
    ws_ypixel : LibC::UShort
  end

  # One void-pointer signature serves both uses (TIOCGWINSZ wants a Winsize*, FIONREAD an Int32*);
  # Crystal refuses two `fun`s that differ only in a parameter type.
  fun ioctl(fd : LibC::Int, request : LibC::ULong, arg : Void*) : LibC::Int
end

@[Link("readline")]
lib LibRLHook
  $rl_startup_hook : -> LibC::Int
  fun rl_insert_text(text : LibC::Char*) : LibC::Int
end

module AgentDemo
  MAX_READ_BYTES = 64_000

  # Tool-calling iterations allowed per user turn, before the agent hands control back.
  #
  # A runaway guard, not a work budget: without it a model that keeps emitting tool calls
  # never returns to the prompt. The number was 8, chosen when a turn's prefill cost minutes
  # and a runaway was expensive. Prefill is now ~29 s at 4096 with KV prefix reuse making
  # turn 2+ about a second, so 8 stopped being a runaway guard and started truncating
  # ordinary work -- reading four files and grepping twice already exceeds it.
  #
  # 40 is high enough that hitting it means something is actually looping.
  # AGENT_MAX_TOOL_STEPS overrides it, clamped to at least 1: a 0 would break out before
  # generating anything and every turn would answer with silence.
  MAX_TOOL_STEPS = begin
    n = (ENV["AGENT_MAX_TOOL_STEPS"]? || "40").to_i
    n < 1 ? 1 : n
  end

  # How many times an IDENTICAL tool call is allowed before the loop intervenes, and how many recent
  # call signatures to remember. Small on purpose: two identical calls are a retry, three is a loop.
  REPEAT_CALL_LIMIT = 2

  # How many times one turn may be told to finish a dropped tool call before its reply is
  # accepted as-is. Two is enough for a formatting slip without letting a model that simply
  # wants to answer briefly be nudged in a loop.
  EMPTY_REPLY_NUDGE_LIMIT =  2
  RECENT_CALL_MEMORY      = 12

  # Turn a failed list_directory into an answer rather than a dead end.
  #
  # Three cases, in the order a caller actually hits them:
  #   - the argument is a shell command, not a path (a model reaching for `ls a b 2>/dev/null`
  #     because no tool offered "check several paths at once") -- name the right tool;
  #   - the path is a FILE -- say so and list its parent, which is what was wanted;
  #   - the path does not exist -- list the nearest existing ancestor and point out entries whose
  #     names are close, so a typo like "example/" for "examples/" is corrected in one round trip.
  def self.directory_miss_help(raw : String, resolved : String) : String
    if raw.matches?(/[;|&><$`]|\*\s|\s2>/)
      return "Error: '#{raw}' looks like a shell command, not a directory path. list_directory " \
             "takes ONE directory path. Use run_command to run shell, or call list_directory once " \
             "per path."
    end

    if File.exists?(resolved) && !Dir.exists?(resolved)
      parent = File.dirname(raw)
      parent = "." if parent.empty?
      listing = safe_children(File.dirname(resolved))
      return "Error: #{raw} is a FILE, not a directory (use read_file to read it).\n" \
             "Contents of #{parent}:\n#{listing}"
    end

    # Walk up to the nearest ancestor that exists, and list it.
    rel = raw
    anc = resolved
    3.times do
      break if Dir.exists?(anc)
      anc = File.dirname(anc)
      rel = File.dirname(rel)
    end
    rel = "." if rel.empty? || rel == "/"

    unless Dir.exists?(anc)
      return "Error: no such directory: #{raw}"
    end

    wanted = File.basename(raw)
    entries = begin
      Dir.children(anc)
    rescue
      [] of String
    end
    # "Close" deliberately means a shared prefix rather than an edit distance: the typos that
    # actually occur here are a missing or extra trailing character (example/ for examples/).
    near = entries.select do |e|
      next false if wanted.empty?
      e.starts_with?(wanted[0, [wanted.size, 3].min]) || wanted.starts_with?(e[0, [e.size, 3].min])
    end.sort!.first(8)

    out = ["Error: no such directory: #{raw}"]
    out << "Did you mean: #{near.join(", ")}" unless near.empty?
    out << "Contents of #{rel}:\n#{safe_children(anc)}"
    out.join("\n")
  end

  # Directory listing with directories marked, or a readable reason it could not be listed.
  def self.safe_children(dir : String) : String
    Dir.children(dir).sort.map { |e| Dir.exists?(File.join(dir, e)) ? "#{e}/" : e }.join("\n")
  rescue ex
    "(could not list: #{ex.message})"
  end

  record ToolParam, name : String, type : String, description : String, required : Bool = true
  # `ids` holds the EXACT tokens the model produced for an assistant turn, when they are
  # known. Re-encoding the decoded text does not reliably reproduce them -- a decode,
  # UTF-8 scrub and re-encode round trip can shift token boundaries -- and a single
  # mismatched token destroys KV prefix reuse for the whole conversation.
  record Message, role : String, content : String, ids : Array(Int32)? = nil
  record ToolCall, name : String, args : Hash(String, String)

  class Tool
    getter name : String
    getter description : String
    getter params : Array(ToolParam)
    getter handler : Hash(String, String) -> String

    def initialize(@name, @description, @params, &@handler : Hash(String, String) -> String)
    end

    def call(args : Hash(String, String)) : String
      handler.call(args)
    end
  end

  # Root that every file tool is confined to: the directory the agent was launched in, unless
  # AGENT_WORKSPACE says otherwise.
  #
  # Resolved once at startup rather than per call, so nothing can widen its own sandbox by
  # chdir-ing partway through a turn.
  @@workspace = File.realpath(ENV["AGENT_WORKSPACE"]? || Dir.current)

  def self.workspace : String
    @@workspace
  end

  # A line the user typed WHILE the model was generating.
  #
  # Last session the user watched the model loop and typed "You're looping, make smaller changes" --
  # and had to wait for the turn to finish before it could be delivered. Steering makes that land
  # immediately: the current generation stops, what it produced so far is kept, and the new instruction
  # goes in as the next user message, so the model sees both its own partial attempt and the
  # correction. Ctrl-C throws the turn away; steering redirects it.
  @@steer : String? = nil

  def self.take_steer : String?
    s = @@steer
    @@steer = nil
    s
  end

  def self.steer_pending? : Bool
    !@@steer.nil?
  end

  # ---- The steering input line ----------------------------------------------------------------------
  #
  # Steering used to be read in the terminal's normal line mode: keystrokes echoed wherever the output
  # cursor happened to be, so a correction typed mid-answer was smeared through the streaming text,
  # and nothing reached the agent until Enter. Now, while a turn runs, the terminal is put in
  # non-canonical, no-echo mode (signals stay on, so Ctrl-C still cancels), keystrokes are read as
  # they arrive and drawn on a row of their own above the status bar, and Enter delivers the line.
  #
  # Reads happen only when FIONREAD says bytes are waiting, never blocking: directly from the decode
  # loop and the prefill progress callback (the GPU phases, where no fiber runs), and from a small
  # fiber during tool execution, which waits on the event loop. The approval prompt and readline get
  # the normal mode back for as long as they read.
  @@steer_mode = false
  @@saved_termios = LibC::Termios.new
  @@typed = ""
  @@esc = ""
  @@utf8 = [] of UInt8
  @@in_paste = false
  @@prefill_line : String? = nil

  def self.steer_mode_on
    return if @@steer_mode || !STDIN.tty?
    t = LibC::Termios.new
    return if LibC.tcgetattr(STDIN.fd, pointerof(t)) != 0
    @@saved_termios = t
    t.c_lflag &= ~(LibC::ECHO | LibC::ICANON)
    t.c_cc[LibC::VMIN] = 0
    t.c_cc[5] = 0 # VTIME
    return if LibC.tcsetattr(STDIN.fd, LibC::TCSANOW, pointerof(t)) != 0
    @@steer_mode = true
    @@typed = ""
    @@esc = ""
    @@utf8.clear
    @@in_paste = false
    BAR.input = ""
    BAR.input_active = true
    spawn do
      while @@steer_mode
        pump_input
        sleep 40.milliseconds
      end
    end
  end

  # Back to the terminal's normal mode. Unsent text is kept for the next prompt.
  def self.steer_mode_off
    return unless @@steer_mode
    pump_input
    @@steer_mode = false
    t = @@saved_termios
    LibC.tcsetattr(STDIN.fd, LibC::TCSANOW, pointerof(t))
    @@prefill_line = @@typed unless @@typed.strip.empty?
    @@typed = ""
    BAR.input_active = false
  end

  # Run the block in normal terminal mode (an approval prompt reading a line), then resume.
  def self.with_cooked(&)
    was = @@steer_mode
    steer_mode_off if was
    begin
      yield
    ensure
      if was
        # Whatever was half-typed before the prompt goes back on the input line.
        keep = @@prefill_line
        @@prefill_line = nil
        steer_mode_on
        if keep
          @@typed = keep
          BAR.input = keep
        end
      end
    end
  end

  # Text typed during the last turn and never sent; readline offers it pre-filled.
  def self.take_prefill_line : String?
    l = @@prefill_line
    @@prefill_line = nil
    l
  end

  def self.pump_input
    return unless @@steer_mode
    pending = 0
    return if LibWinsize.ioctl(STDIN.fd, FIONREAD, pointerof(pending).as(Void*)) != 0
    return if pending <= 0
    buf = Bytes.new(pending)
    n = LibC.read(STDIN.fd, buf, pending)
    return if n <= 0
    before = @@typed
    buf[0, n].each { |byte| input_byte(byte) }
    BAR.input = @@typed if @@typed != before || steer_pending?
  rescue
  end

  private def self.input_byte(b : UInt8)
    unless @@esc.empty? && b != 0x1b
      @@esc += b.chr
      seq = @@esc
      done = if seq.size == 1
               false
             elsif seq[1] == '['
               seq.size > 2 && (0x40..0x7e).includes?(seq[-1].ord)
             elsif seq[1] == 'O'
               seq.size >= 3
             else
               true
             end
      if done
        @@in_paste = true if seq == PASTE_START
        @@in_paste = false if seq == PASTE_END
        @@esc = ""
      end
      return
    end
    if b >= 0x80
      @@utf8 << b
      lead = @@utf8[0]
      need = lead >= 0xF0 ? 4 : (lead >= 0xE0 ? 3 : (lead >= 0xC0 ? 2 : 1))
      if @@utf8.size >= need
        str = String.new(Slice.new(@@utf8.to_unsafe, @@utf8.size))
        @@typed += str if str.valid_encoding?
        @@utf8.clear
      end
      return
    end
    case b
    when 13, 10
      if @@in_paste
        @@typed += " "
      else
        text = @@typed.strip
        @@typed = ""
        @@steer = text unless text.empty?
      end
    when 127, 8
      @@typed = @@typed.rchop unless @@typed.empty?
    when 21 # Ctrl-U
      @@typed = ""
    when 23 # Ctrl-W
      @@typed = @@typed.rstrip.sub(/\S+\z/, "")
    when 9
      @@typed += " "
    else
      @@typed += b.chr if b >= 32
    end
  end

  # readline's startup hook: insert the carried-over line so it can be edited and sent.
  def self.readline_prefill : LibC::Int
    if l = take_prefill_line
      LibRLHook.rl_insert_text(l)
    end
    0
  end

  # Terminal size, via ioctl(TIOCGWINSZ).
  #
  # Crystal 1.21 has no public API for this -- no tty_size?, no console_size, and no TIOCGWINSZ
  # anywhere in the stdlib -- so it is bound here. The alternatives were worse: COLUMNS/LINES are not
  # exported by default, shelling out to `tput` costs a process per query, and the ANSI size report
  # ("\e[18t") would have to read its answer from stdin, which is where steering is listening.
  TIOCGWINSZ = {% if flag?(:darwin) %}0x40087468_u64{% else %}0x5413_u64{% end %}

  # Columns of the terminal on stderr, or 100 when it is not one.
  def self.term_cols : Int32
    ws = LibWinsize::Winsize.new
    return 100 if LibWinsize.ioctl(STDERR.fd, TIOCGWINSZ, pointerof(ws).as(Void*)) != 0 || ws.ws_col == 0
    ws.ws_col.to_i
  end

  # Bytes waiting on a descriptor, used to poll stdin without blocking. IO.select does not exist in
  # Crystal, and a background reader fiber would compete with the confirmation prompt for the same
  # line, so asking the kernel how much is buffered is both the simplest and the only safe option.
  FIONREAD = {% if flag?(:darwin) %}0x4004667f_u64{% else %}0x541b_u64{% end %}

  # A status line pinned to the bottom row of the terminal.
  #
  # Implemented with a SCROLL REGION (DECSTBM, "\e[1;{rows-1}r") rather than by tracking which row the
  # cursor is on. Setting the region tells the TERMINAL that ordinary output may only occupy rows 1 to
  # rows-1, so the bottom line is reserved by the terminal itself and no write can land on it.
  #
  # The alternative -- counting newlines and wrapping to maintain a row number, pushing a blank line
  # whenever content is about to reach the last row -- needs every single write to pass through one
  # wrapper, and two things here guarantee that is not true. The output is heavily colorized, so
  # `text.size` counts escape bytes that occupy no columns and the row count drifts on the first
  # coloured line. And the loader logs through SHAInet's Log, which writes to the same handle without
  # going through the agent at all. A row counter that is wrong once stays wrong, and the bar then
  # overwrites real output.
  #
  # With a region, the bar also does not need redrawing when output scrolls: scrolling happens inside
  # the region and leaves the last line alone, so a redraw is needed only when the TEXT changes.
  class StatusBar
    @text = ""
    @installed = false
    # Size the scroll region was set for; a different size on the next draw means a resize.
    @rows = 0
    @cols = 0
    # The steering input line, the row above the status line. Active while a turn runs.
    @input = ""
    @input_active = false

    # Rows kept out of the scroll region: the input line and the status line.
    RESERVED = 2

    def initialize(@io : IO::FileDescriptor)
      @tty = @io.tty?
    end

    # Reserve the bottom rows. Safe to call when not on a tty: it does nothing and every other method
    # degrades to a no-op, which is the same gate Colorize.enabled uses.
    def install
      return unless @tty && (size = term_size)
      rows, cols = size
      return if rows < RESERVED + 2 # too short to give lines away
      @io.print "\n" * RESERVED     # make room so nothing on screen is under the reserved rows
      @io.print "\e[1;#{rows - RESERVED}r"
      @io.print "\e[#{rows - RESERVED};1H"
      @io.flush
      @rows, @cols = rows, cols
      @installed = true
      paint(rows, cols)
    end

    # Re-fit the region after a terminal resize. Called from draw, the SIGWINCH trap and before each
    # prompt; a no-op while the size is unchanged.
    #
    # The region is set in absolute rows, so a resize left it describing the OLD terminal: on a taller
    # window the reserved rows sat in the middle of the output, on a shorter one the region ran past
    # the bottom (or the terminal reset it to full screen), and from then on every bar redraw wrote
    # over real text and scrolling no longer avoided the bar -- the screen turned to garbage.
    #
    # The cursor's row is unknown here (asking the terminal means reading stdin, where steering and
    # readline listen), so the fix works without it. With full-screen scrolling, RESERVED newlines
    # then as many cursor-ups is a no-op unless the cursor is within RESERVED rows of the bottom, in
    # which case the screen scrolls just enough to free those rows and the cursor keeps its column.
    # DECSTBM homes the cursor, hence the save/restore around it.
    def refresh
      return unless @installed && (size = term_size)
      rows, cols = size
      return if rows == @rows && cols == @cols
      @rows, @cols = rows, cols
      return if rows < RESERVED + 2
      @io.print "\e7\e[r\e8"
      @io.print "\n" * RESERVED + "\e[#{RESERVED}A"
      @io.print "\e7\e[1;#{rows - RESERVED}r\e8"
      paint(rows, cols)
    end

    # Release the rows and restore the terminal. Registered with at_exit, because a process that dies
    # with a scroll region set leaves the user's shell scrolling in a box.
    def uninstall
      return unless @installed
      @installed = false
      if size = term_size
        rows, _ = size
        @io.print "\e[r" # full-screen scrolling again
        RESERVED.times { |i| @io.print "\e[#{rows - i};1H\e[2K" }
        @io.print "\e[?25h" # cursor visible
        @io.print "\e[#{rows - RESERVED + 1};1H"
      end
      @io.flush
    end

    def text=(t : String)
      return if t == @text
      @text = t
      draw
    end

    def input=(t : String)
      @input = t
      draw
    end

    def input_active=(v : Bool)
      @input_active = v
      draw
    end

    def hide_cursor
      @io.print "\e[?25l" if @installed
    end

    def show_cursor
      @io.print "\e[?25h" if @installed
    end

    # Re-queries the size on every draw, re-fitting the scroll region first if it changed.
    private def draw
      return unless @installed && (size = term_size)
      rows, cols = size
      if rows != @rows || cols != @cols
        refresh # re-fits the region and paints
        return
      end
      paint(rows, cols)
    end

    private def paint(rows : Int32, cols : Int32)
      body = @text.size > cols ? @text[0, cols] : @text.ljust(cols)
      @io.print "\e7" # save cursor
      @io.print "\e[#{rows - 1};1H\e[2K"
      @io.print input_line(cols)
      @io.print "\e[#{rows};1H\e[2K"
      @io.print body.colorize(:dark_gray)
      @io.print "\e8" # restore cursor
      @io.flush
    end

    # "› text▏" while a turn runs (the tail of a long line, so the end being typed stays visible), a
    # hint when nothing is typed yet, and a plain rule between turns, when readline has the prompt.
    private def input_line(cols : Int32) : String
      unless @input_active
        return ("─" * cols).colorize(:dark_gray).to_s
      end
      prompt = "› "
      if @input.empty?
        return "#{prompt.colorize(:light_green)}#{"type to steer · Enter sends · Ctrl-C stops the turn".colorize(:dark_gray)}"
      end
      room = cols - prompt.size - 2
      shown = @input
      while AgentDemo::TermWidth.of(shown) > room && !shown.empty?
        shown = shown[1..]
      end
      shown = "…" + shown[1..] if shown.size < @input.size && !shown.empty?
      "#{prompt.colorize(:light_green)}#{shown}\e[7m \e[0m"
    end

    private def term_size
      return unless @tty
      ws = LibWinsize::Winsize.new
      return if LibWinsize.ioctl(@io.fd, AgentDemo::TIOCGWINSZ, pointerof(ws).as(Void*)) != 0
      return if ws.ws_row == 0 || ws.ws_col == 0
      {ws.ws_row.to_i, ws.ws_col.to_i}
    rescue
      nil
    end
  end

  # Which OS the tools will actually run on.
  #
  # The model guessed wrong in practice, reaching for a macOS path on a Linux host, and a wrong guess
  # here costs a whole tool call to a command that cannot work.
  def self.host_os : String
    {% if flag?(:linux) %}
      "Linux"
    {% elsif flag?(:darwin) %}
      "macOS"
    {% elsif flag?(:win32) %}
      "Windows"
    {% else %}
      "Unix"
    {% end %}
  end

  # Change the workspace root at runtime. Realpath'd the same way as at startup, so a symlinked
  # argument cannot point the sandbox somewhere its label does not say. Must be an existing
  # directory: a file would make every relative path escape the root.
  #
  # Named as a verb rather than as a `workspace=` setter on purpose: it validates and RAISES, and an
  # assignment that can reject its value reads as though it cannot.
  def self.switch_workspace!(path : String)
    resolved = File.realpath(path)
    raise ArgumentError.new("not a directory: #{path}") unless Dir.exists?(resolved)
    @@workspace = resolved
  end

  # Resolve a model-supplied path. Returns {path, ""} on success and {nil, error_text} on
  # refusal, so a tool body stays a single line and the model gets told WHY.
  #
  # Inside the workspace needs no approval. OUTSIDE it, the path is negotiable rather than
  # forbidden: the user is asked once per location and the answer is remembered, so working on
  # a file elsewhere costs one prompt instead of one per tool call. An approval is per PATH on
  # purpose -- approving a model config should not also hand over ~/.ssh/id_rsa. Answering "a"
  # still opens everything, matching the prompt's existing meaning.
  def self.safe_path(raw : String?) : {String?, String}
    {WorkspacePath.resolve(@@workspace, raw || ""), ""}
  rescue WorkspaceEscapeError
    outside_path(raw || "")
  rescue ex : WorkspacePathError
    {nil, "Error: #{ex.message}."}
  end

  private def self.outside_path(raw : String) : {String?, String}
    resolved = WorkspacePath.resolve_anywhere(@@workspace, raw)
    return {resolved, ""} if allow_all? || WorkspacePath.approved_outside?(resolved)
    unless confirm?("access #{resolved} — OUTSIDE the workspace (#{@@workspace})")
      return {nil, "Error: #{raw} is outside the workspace and the user declined access. " \
                   "Work within #{@@workspace} instead."}
    end
    # Remembered, so the next tool call on this file does not ask again.
    WorkspacePath.approve_outside(resolved)
    {resolved, ""}
  rescue ex : WorkspacePathError
    {nil, "Error: #{ex.message}."}
  end

  # A unified-ish diff of a proposed whole-file write, for the confirmation prompt.
  #
  # Approving "write 4182 bytes to x.cr" says nothing about what actually changes, which is
  # the real problem with a blanket approval: allow-all is only reasonable if each prompt
  # showed enough to judge. Capped on purpose -- a 3000-line rewrite scrolling past defeats
  # the point as thoroughly as showing nothing.
  def self.preview_diff(path : String, new_content : String, max_lines : Int32 = 40) : String
    old_lines = File.exists?(path) ? File.read(path).lines : [] of String
    new_lines = new_content.lines
    # Trim the common head and tail, so the window shown is the change and not the file.
    head = 0
    while head < old_lines.size && head < new_lines.size && old_lines[head] == new_lines[head]
      head += 1
    end
    tail = 0
    while tail < (old_lines.size - head) && tail < (new_lines.size - head) &&
          old_lines[old_lines.size - 1 - tail] == new_lines[new_lines.size - 1 - tail]
      tail += 1
    end
    removed = old_lines[head, old_lines.size - head - tail]
    added = new_lines[head, new_lines.size - head - tail]
    return "  (no textual change)" if removed.empty? && added.empty?

    half = max_lines // 2
    out = [] of String
    out << "  @@ line #{head + 1} @@"
    removed.first(half).each { |l| out << "  -#{l}" }
    out << "  ... #{removed.size - half} more removed" if removed.size > half
    added.first(half).each { |l| out << "  +#{l}" }
    out << "  ... #{added.size - half} more added" if added.size > half
    out.join("\n")
  end

  # Set by answering "a" at a confirmation prompt: allow every later mutating tool call for
  # the rest of the process, without re-asking.
  #
  # Deliberately in-memory and process-scoped, not persisted: a blanket approval that
  # outlived the session would be a much bigger promise than the one the user made at the
  # prompt. SHAINET_AGENT_YOLO=1 is the same thing chosen up front.
  @@allow_all = false

  def self.allow_all? : Bool
    @@allow_all
  end

  def self.allow_all=(value : Bool)
    @@allow_all = value
  end

  # Confirmation gate for mutating/executing tools.
  #
  #   y  allow this call
  #   n  deny this call (the default, so a bare Enter is safe)
  #   a  allow this and every later call this session
  #
  # Skipped entirely with SHAINET_AGENT_YOLO=1.
  def self.confirm?(desc : String) : Bool
    return true if ENV.fetch("SHAINET_AGENT_YOLO", "0") == "1"
    return true if @@allow_all
    ans = with_cooked do
      STDERR.print "  #{"⚠ allow".colorize(:yellow)} #{desc}? [y/N/a=all] "
      STDERR.flush
      gets
    end
    # EOF (piped input, closed stdin) must not read as approval.
    return false unless ans
    case ans.strip.downcase
    when "a", "all"
      @@allow_all = true
      STDERR.puts "  #{"⚠ approving all further tool calls this session".colorize(:yellow)}"
      true
    when .starts_with?("y")
      true
    else
      false
    end
  end

  # Basic code-agent tools. Read-only ones run freely; write/edit/run ask for
  # confirmation first (unless SHAINET_AGENT_YOLO=1).
  def self.build_tools : Array(Tool)
    [
      Tool.new(
        "list_directory",
        "List the files and subdirectories in a directory. Paths are relative to the workspace root; an absolute path outside it needs the user to approve that location once.",
        [ToolParam.new("path", "string", "Directory path to list (default: current directory).")]
      ) do |args|
        raw = (args["path"]? || "").strip
        raw = "." if raw.empty?
        path, err = AgentDemo.safe_path(raw)
        if path.nil?
          next err
        end
        if Dir.exists?(path)
          Dir.children(path).sort.map { |e| Dir.exists?(File.join(path, e)) ? "#{e}/" : e }.join("\n")
        else
          # A miss used to dead-end with just "not a directory", which is what sent a model into a
          # loop: its real question was "which of these paths exist?", and the error answered
          # nothing, so it reissued variants -- eventually smuggling a shell command into this
          # argument. Answer the underlying question instead.
          AgentDemo.directory_miss_help(raw, path)
        end
      end,
      Tool.new(
        "read_file",
        "Read a text file. Returns the whole file when it is small, or the lines from start_line to " \
        "end_line (1-based, inclusive) when given. A large file returns only its first lines plus its " \
        "total line count -- ask again with a range to see the rest. Prefer `search` to FIND the lines " \
        "you want and then read that range: reading a whole large file usually costs more context than " \
        "the answer is worth. Output is line-numbered. Binary files are refused. Paths are relative to " \
        "the workspace root; an absolute path outside it needs the user to approve that location once.",
        [ToolParam.new("path", "string", "File path to read."),
         ToolParam.new("start_line", "integer", "Optional first line to return (1-based).", false),
         ToolParam.new("end_line", "integer", "Optional last line to return (1-based, inclusive).", false)]
      ) do |args|
        path, err = AgentDemo.safe_path(args["path"]?)
        if path.nil?
          next err
        end
        if Dir.exists?(path) || !File.exists?(path)
          "Error: not a file: #{args["path"]?}"
        else
          c = File.read(path)
          # Refuse binaries outright. A model asked to inspect a directory will happily try
          # to read a compiled executable, and the bytes are useless to it while being
          # ruinously expensive: 64 KB of binary tokenized to 60534 tokens in one observed
          # run, against an 8192-token window. A NUL byte is the cheap, reliable signal.
          if c.byte_slice(0, Math.min(c.bytesize, 8192)).includes?('\u0000')
            "Error: #{path} looks like a binary file (contains NUL bytes); refusing to read it"
          elsif !c.valid_encoding?
            "Error: #{path} is not valid UTF-8; refusing to read it"
          else
            AgentDemo.render_file(args["path"]? || path, c,
              args["start_line"]?.try(&.to_i?), args["end_line"]?.try(&.to_i?))
          end
        end
      end,
      Tool.new(
        "search",
        "Search file CONTENTS and get back path:line: text for every hit. This is how you find the few " \
        "ranges worth reading instead of reading whole files — search first, then read_file with " \
        "start_line/end_line around a hit. Case-insensitive; literal text unless regex is true.",
        [ToolParam.new("pattern", "string", "Text to find, or a regular expression when regex is true."),
         ToolParam.new("path", "string", "Directory to search (default: workspace root).", false),
         ToolParam.new("glob", "string", "Only search files whose name matches this, e.g. '*.cr'.", false),
         ToolParam.new("context", "string", "Lines of context around each hit, 0-5 (default 0).", false),
         ToolParam.new("regex", "string", "'true' to treat pattern as a regular expression.", false)]
      ) do |args|
        pat = args["pattern"]? || ""
        next "Error: empty pattern" if pat.empty?
        root, err = AgentDemo.safe_path(args["path"]? || ".")
        if root.nil?
          next err
        end
        ctx = Math.min(5, Math.max(0, (args["context"]? || "0").to_i? || 0))
        rx = (args["regex"]? || "").downcase == "true"
        AgentDemo.search_contents(root, pat, args["glob"]?, ctx, rx)
      end,
      Tool.new(
        "glob",
        "List FILES BY NAME matching a glob, e.g. '**/*.cr' or 'src/**/gguf*.cr'. Use it to see what " \
        "exists before reading anything; it returns paths only, never file contents.",
        [ToolParam.new("pattern", "string", "Glob pattern, e.g. '**/*.cr'."),
         ToolParam.new("path", "string", "Directory to search from (default: workspace root).", false)]
      ) do |args|
        pat = args["pattern"]? || ""
        next "Error: empty pattern" if pat.empty?
        root, err = AgentDemo.safe_path(args["path"]? || ".")
        if root.nil?
          next err
        end
        AgentDemo.glob_files(root, pat)
      end,
      Tool.new(
        "write_file",
        "Create a NEW text file. An existing file is refused: use apply_patch or edit_file to change one. Paths are relative to the workspace root; an absolute path outside it needs the user to approve that location once.",
        [ToolParam.new("path", "string", "File path to write."),
         ToolParam.new("content", "string", "Full file content.")]
      ) do |args|
        path, err = AgentDemo.safe_path(args["path"]?)
        if path.nil?
          next err
        end
        # Refuse to clobber. A model that means to change three lines will cheerfully pass a
        # whole-file rewrite, and any part it did not remember is silently gone. Making it
        # patch or edit instead keeps the rest of the file out of the blast radius.
        if File.exists?(path)
          next "Error: #{args["path"]?} already exists. Use apply_patch or edit_file to modify it, " \
               "or delete it first with run_command if replacing it wholesale is really intended."
        end
        content = args["content"]? || ""
        rel = args["path"]?
        STDERR.puts AgentDemo.preview_diff(path, content) unless AgentDemo.allow_all?
        next "Declined by user." unless AgentDemo.confirm?("create #{rel} (#{content.bytesize} bytes)")
        # mkdir -p the parents: otherwise a new file in a new directory needs a shell call.
        Dir.mkdir_p(File.dirname(path))
        File.write(path, content)
        "Wrote #{content.bytesize} bytes to #{rel}"
      end,
      Tool.new(
        "current_dir",
        "Show the workspace root that every relative path resolves against, the host OS, and what is " \
        "directly inside it. Call this if you are unsure where you are — never search the filesystem " \
        "to find out.",
        [] of ToolParam
      ) do |_args|
        ws = AgentDemo.workspace
        entries = begin
          Dir.children(ws).reject(&.starts_with?('.')).sort!
        rescue
          [] of String
        end
        shown = entries.first(40)
        more = entries.size > shown.size ? " … and #{entries.size - shown.size} more" : ""
        "workspace: #{ws}\nhost: #{AgentDemo.host_os}\ncontains (#{entries.size}): #{shown.join(", ")}#{more}"
      end,
      Tool.new(
        "insert_lines",
        "INSERT new lines into a file after a given line number, without touching anything already " \
        "there. Use this for ADDING code — it is safer than replace_lines for an addition, because " \
        "nothing existing has to be retyped. Pass after_line 0 to insert at the very top of the file.",
        [ToolParam.new("path", "string", "File path to edit."),
         ToolParam.new("after_line", "string", "Insert after this line number; 0 means before line 1."),
         ToolParam.new("content", "string", "Lines to insert.")]
      ) do |args|
        path, err = AgentDemo.safe_path(args["path"]?)
        if path.nil?
          next err
        end
        next "Error: #{args["path"]?} does not exist" unless File.exists?(path)
        at = (args["after_line"]? || "").to_i?
        next "Error: after_line must be an integer" if at.nil?
        body = File.read(path)
        lines = body.split('\n')
        total = lines.size
        total -= 1 if total > 0 && lines[-1].empty? && body.ends_with?('\n')
        next "Error: after_line #{at} is outside #{args["path"]?} (#{total} lines)" if at < 0 || at > total
        added = (args["content"]? || "").split('\n')
        next "Declined by user." unless AgentDemo.confirm?("insert #{added.size} line(s) after line #{at} of #{args["path"]?}")
        File.write(path, (lines[0, at] + added + lines[at..]).join('\n'))
        "Inserted #{added.size} line(s) after line #{at} in #{args["path"]?}. " \
        "Lines after that point have shifted down by #{added.size}; the file is now #{total + added.size} lines."
      end,
      Tool.new(
        "replace_lines",
        "Replace a RANGE OF LINES in a file, addressed by number (1-based, inclusive) — you never have " \
        "to reproduce the existing text. Prefer this over edit_file and apply_patch when search or " \
        "read_file has already told you the line numbers. Pass the same number twice to replace one " \
        "line, and an empty content to delete the range.",
        [ToolParam.new("path", "string", "File path to edit."),
         ToolParam.new("start_line", "string", "First line to replace (1-based, inclusive)."),
         ToolParam.new("end_line", "string", "Last line to replace (1-based, inclusive)."),
         ToolParam.new("content", "string", "Replacement text for those lines (may be several lines; empty deletes them).")]
      ) do |args|
        path, err = AgentDemo.safe_path(args["path"]?)
        if path.nil?
          next err
        end
        next "Error: #{args["path"]?} does not exist" unless File.exists?(path)
        s = (args["start_line"]? || "").to_i?
        e = (args["end_line"]? || "").to_i?
        next "Error: start_line and end_line must be integers" if s.nil? || e.nil?
        body = File.read(path)
        lines = body.split('\n')
        # A trailing newline makes split produce a final empty element; it is not a line the user can
        # address, and counting it would put every range off by one at the end of the file.
        total = lines.size
        total -= 1 if total > 0 && lines[-1].empty? && body.ends_with?('\n')
        next "Error: start_line #{s} is outside #{args["path"]?} (#{total} lines)" if s < 1 || s > total
        next "Error: end_line #{e} is before start_line #{s}" if e < s
        e = total if e > total

        replacement = args["content"]? || ""
        before = lines[0, s - 1]
        after = lines[e..]
        middle = replacement.empty? ? [] of String : replacement.split('\n')
        preview = "#{s}-#{e} of #{args["path"]?} → #{middle.size} line(s)"
        next "Declined by user." unless AgentDemo.confirm?("replace lines #{preview}")
        File.write(path, (before + middle + after).join('\n'))
        # State the shift explicitly. Left to work it out, the model computed the drift after its first
        # edit twice and got a different answer each time, then re-ran search before every later edit to
        # recover the numbers. Line-addressed editing is only cheap if the tool keeps the accounting.
        delta = middle.size - (e - s + 1)
        shift = if delta == 0
                  "Line numbers are unchanged."
                elsif delta > 0
                  "Lines after #{e} have shifted DOWN by #{delta}."
                else
                  "Lines after #{e} have shifted UP by #{-delta}."
                end
        "Replaced lines #{s}-#{e} with #{middle.size} line(s) in #{args["path"]?}. " \
        "#{shift} The file is now #{total + delta} lines."
      end,
      Tool.new(
        "edit_file",
        "Replace an exact string in a file. 'find' must match EXACTLY ONE occurrence, including whitespace; include surrounding context to make it unique. Paths are relative to the workspace root; an absolute path outside it needs the user to approve that location once.",
        [ToolParam.new("path", "string", "File path to edit."),
         ToolParam.new("find", "string", "Exact text to find (must be unique in the file)."),
         ToolParam.new("replace", "string", "Replacement text.")]
      ) do |args|
        path, err = AgentDemo.safe_path(args["path"]?)
        if path.nil?
          next err
        end
        next "Error: not a file: #{args["path"]?}" if Dir.exists?(path) || !File.exists?(path)
        find = args["find"]? || ""
        next "Error: empty 'find'" if find.empty?
        replace = args["replace"]? || ""
        content = File.read(path)
        count = content.scan(find).size
        next "No occurrences of the given text in #{args["path"]?}" if count == 0
        # Refuse a non-unique match instead of replacing every one of them.
        #
        # This used to gsub. A model asked to change one `end` rewrote every `end` in the
        # file, and the only hint was a count buried in the confirmation line, which an
        # approved-all session never shows. Ambiguity is the model's to resolve, not ours to
        # guess at, so hand it back and say how to fix it.
        if count > 1
          next "Error: 'find' matches #{count} times in #{args["path"]?}. Include more " \
               "surrounding context so it matches exactly once, or use apply_patch for a " \
               "multi-hunk change."
        end
        updated = content.sub(find, replace)
        STDERR.puts AgentDemo.preview_diff(path, updated) unless AgentDemo.allow_all?
        next "Declined by user." unless AgentDemo.confirm?("edit #{args["path"]?}")
        File.write(path, updated)
        "Edited #{args["path"]?} (replaced #{find.size} chars with #{replace.size} chars)"
      end,
      Tool.new(
        "apply_patch",
        <<-DESC,
          Apply a V4A context-diff patch to one or more files. PREFERRED for all file edits: it supports multi-file, multi-hunk changes in a single call, and edits are located by surrounding context rather than line numbers, so they survive minor drift. Read the file first so your context lines match.

          Format:
          *** Begin Patch
          *** Update File: path/to/file
          @@ optional anchor (e.g. a function name)
           unchanged context line
          -removed line
          +added line
          *** Add File: path/to/new-file
          +line 1
          *** Delete File: path/to/old-file
          *** End Patch

          Rules: ' ' prefix = context, '-' = remove, '+' = add. Include ~3 context lines around each change. Use @@ anchors when the context is not unique. All patch paths must be INSIDE the workspace: unlike read_file and edit_file, a patch cannot reach an approved outside location, so use edit_file for those.
          DESC
        [ToolParam.new("patch", "string", "Full patch in V4A format, including the *** Begin Patch / *** End Patch markers.")]
      ) do |args|
        patch = args["patch"]? || ""
        next "Error: empty patch" if patch.empty?
        # Show which files the patch touches BEFORE asking: the patch body is the diff, so a
        # separate preview would just repeat it, but the file list is what the approval is
        # actually about.
        targets = patch.lines.compact_map do |l|
          m = l.match(/^\*\*\* (Update|Add|Delete) File: (.+)$/)
          "#{m[1].downcase} #{m[2].strip}" if m
        end
        next "Error: no *** Update/Add/Delete File: directives found in the patch" if targets.empty?
        next "Declined by user." unless AgentDemo.confirm?("apply patch to #{targets.size} file(s): #{targets.join(", ")}")
        begin
          result = AgentDemo::V4A.apply(AgentDemo.workspace, patch)
        rescue ex : AgentDemo::V4A::PatchParseError
          next "Error: patch does not parse: #{ex.message}"
        rescue ex : AgentDemo::WorkspacePathError
          next "Error: #{ex.message}. Paths must be relative to the workspace."
        rescue ex
          next "Error applying patch: #{ex.message}"
        end
        lines = [] of String
        lines << "Applied: #{result[:applied].join(", ")}" unless result[:applied].empty?
        # Report partial success honestly: some hunks landing and others not is the normal
        # failure mode of context matching, and the model must know which is which to retry.
        lines << "Failed: #{result[:errors].join("; ")}" unless result[:errors].empty?
        lines << "Patch matched nothing." if lines.empty?
        lines.join("\n")
      end,
      Tool.new(
        "run_command",
        "Run a shell command and return its combined stdout/stderr and exit code.",
        [ToolParam.new("command", "string", "The shell command to execute.")]
      ) do |args|
        cmd = args["command"]? || ""
        next "Error: empty command" if cmd.empty?
        next "Declined by user." unless AgentDemo.confirm?("run: #{cmd}")
        buf = IO::Memory.new
        # Run in the workspace, not the directory the process was launched in.
        #
        # Without chdir, /workspace moved where the FILE tools look but not where a shell command runs,
        # so list_directory and `ls` reported different directories and the model could not reconcile
        # them: it saw a file that "existed" via list_directory and did not exist via ls, and burned
        # several commands chasing the contradiction. The two views must be the same directory. Reads
        # the workspace per call so a /workspace switch mid-session takes effect here too.
        status = Process.run("/bin/sh", ["-c", cmd], output: buf, error: buf, chdir: AgentDemo.workspace)
        result = buf.to_s
        result = "#{result.byte_slice(0, MAX_READ_BYTES)}\n... [truncated]" if result.bytesize > MAX_READ_BYTES
        "exit=#{status.exit_code}\n#{result}"
      end,
    ] + browser_tools
  end

  # Streams assistant output token-by-token, coloring reasoning (<think>…
  # </think>) dim and the answer normal, while hiding tool-call XML. Buffers a
  # small tail so tags split across tokens are never shown half-rendered.
  class StreamRenderer
    TAGS = {"<think>" => :think_open, "</think>" => :think_close,
            "<tool_call>" => :tool_open, "</tool_call>" => :tool_close}
    SPINNER = %w[⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏]

    def initialize(@io : IO)
      # The answer is rendered as markdown; "Agent ❯ " is already on the line, hence col0 false.
      tty = @io.is_a?(IO::FileDescriptor) && @io.as(IO::FileDescriptor).tty?
      @md = AgentDemo::Markdown.new(@io, -> { AgentDemo.term_cols }, Colorize.enabled? && tty, col0: false)
      @mode = :normal
      @pending = ""
      @started = false
      @tool_chars = 0
      @spin = 0
      @status_shown = false
      @status_width = 0
    end

    def feed(text : String)
      @pending += text
      loop do
        idx = nil
        found = nil
        TAGS.each_key do |tag|
          i = @pending.index(tag)
          if i && (idx.nil? || i < idx.not_nil!)
            idx, found = i, tag
          end
        end
        if (i = idx) && (tag = found)
          emit(@pending[0...i])
          apply(TAGS[tag])
          @pending = @pending[(i + tag.size)..]
        else
          hold = partial_tag_suffix(@pending)
          emit(@pending[0, @pending.size - hold])
          @pending = @pending[(@pending.size - hold)..]
          break
        end
      end
    end

    def finish
      clear_status
      emit(@pending)
      @pending = ""
      @md.finish
    end

    private def emit(t : String)
      return if t.empty?
      # Hidden tool-call tokens (the model writing a whole file inline) would
      # otherwise be dead silence — show a live spinner + byte count instead.
      if @mode == :tool
        @tool_chars += t.size
        @spin = (@spin + 1) % SPINNER.size
        status = "  #{SPINNER[@spin]} writing tool call… (#{@tool_chars} chars)"
        @io.print "\r#{status.colorize(:dark_gray)}"
        @io.flush
        # Remember how wide this actually was. The erase used to be a hardcoded 36, which is exactly
        # the width at a five-digit count and one short from six digits on -- so a tool call over
        # 99,999 characters, which an apply_patch of a large file reaches, left residue on the line.
        # Tracking the real width also means the string above can be reworded without silently
        # breaking the erase.
        @status_width = status.size if status.size > @status_width
        @status_shown = true
        return
      end
      # Trim leading whitespace before the very first visible chars.
      unless @started
        t = t.lstrip
        return if t.empty?
        @started = true
      end
      case @mode
      when :think
        @io.print t.colorize(:dark_gray)
        @md.col0 = t.ends_with?('\n')
      when :normal then @md.feed(t)
      end
      @io.flush
    end

    # Erase the in-progress status line (so following output starts clean).
    private def clear_status
      return unless @status_shown
      @io.print "\r" + (" " * @status_width) + "\r"
      @io.flush
      @status_shown = false
    end

    private def apply(action)
      case action
      when :think_open  then @md.finish; @mode = :think
      when :think_close then @mode = :normal
      when :tool_open   then @md.finish; @tool_chars = 0; @mode = :tool
      when :tool_close  then clear_status; @mode = :normal
      end
    end

    # Longest suffix of s that is a (shorter) prefix of some tag — held back in
    # case the tag completes on the next token.
    private def partial_tag_suffix(s : String) : Int32
      max = 0
      TAGS.each_key do |tag|
        (1...tag.size).each do |k|
          max = k if k <= s.size && k > max && s[(s.size - k)..] == tag[0, k]
        end
      end
      max
    end
  end

  # Bytes above which a whole-file read returns a head plus a line count instead of the file.
  #
  # The old tool truncated at MAX_READ_BYTES mid-line and said only "... [truncated]" -- no total, no
  # hint that a range was possible -- so the model could not do better even in principle. Observed on
  # a 2000-line source file: one read_file spent 16124 tokens, a quarter of a 64K window, to answer a
  # question an outline would have served.
  #
  # The design is ported from bar-bot's coding agent, which had already solved this in Crystal, and it
  # matches what a survey of established harnesses found: they enforce context economy through BOUNDED
  # READ DEFAULTS in the tool itself rather than through a "grep before reading" rule in the system
  # prompt. A rule the model may ignore is weaker than a tool that cannot overspend.
  LARGE_FILE_BYTES = 20_000
  HEAD_LINES       =    200

  # Directories never worth walking for source. A plain "**/*" walk of this repo descends into .git
  # (thousands of loose objects), into lib/ (every installed shard's full source) and into any build
  # output, which is slow and buries the real hits.
  NOISE_DIRS = {".git", "node_modules", "dist", "build", "vendor", "target", ".cache", "lib", "bin", ".shards"}

  # Most matching lines returned before the rest are summarized as a count.
  MAX_MATCHES = 120

  private def self.noise_path?(rel : String) : Bool
    rel.split('/').any? { |seg| NOISE_DIRS.includes?(seg) }
  end

  private def self.relativize(path : String) : String
    ws = workspace
    path.starts_with?("#{ws}/") ? path[(ws.size + 1)..] : path
  end

  # Set by the SIGINT handler, read by the generation loop.
  #
  # Ctrl-C used to kill the process outright, which on this model means losing a loaded 9 GB of
  # weights, the KV cache and the whole conversation because a turn went somewhere unhelpful -- and a
  # 27B prefill takes long enough that wanting out mid-turn is routine, not exceptional. A flag the
  # decoder checks each token turns that into abandoning one turn.
  @@cancelled = false

  # Whether a turn is in flight. The SIGINT handler needs this to decide between cancelling the turn
  # and treating the press as "I want out": the same key means different things at a prompt and
  # mid-generation, and guessing wrong either strands the user or discards their work.
  @@generating = false

  def self.generating? : Bool
    @@generating
  end

  def self.generating=(v : Bool)
    @@generating = v
  end

  def self.cancelled? : Bool
    @@cancelled
  end

  def self.cancel!
    @@cancelled = true
  end

  def self.clear_cancel!
    @@cancelled = false
  end

  # Bracketed paste. A terminal told to enable it wraps pasted text in these markers, which is the
  # only reliable way to tell "the user pasted four lines" from "the user sent four messages" --
  # timing heuristics misfire on a slow terminal and on a fast typist alike.
  PASTE_START = "\e[200~"
  PASTE_END   = "\e[201~"

  def self.enable_bracketed_paste
    return unless STDIN.tty?
    STDERR.print "\e[?2004h"
    STDERR.flush
  end

  def self.disable_bracketed_paste
    return unless STDIN.tty?
    STDERR.print "\e[?2004l"
    STDERR.flush
  end

  # Read one submission, splicing a bracketed paste into the surrounding typed text.
  #
  # A paste marker is a DELIMITER inside a line, not a mode for the whole submission. The first version
  # treated it as a mode: on seeing PASTE_START it kept only the bytes after it, which silently DROPPED
  # everything the user typed before the paste. Observed: "analyze <paste>2606.14164v1.pdf</paste>, see
  # what we can add" arrived as just "2606.14164v1.pdf" -- the instruction around the pasted filename
  # was thrown away, so the agent had a bare filename and no task.
  #
  # The markers are now simply removed wherever they fall, and content is accumulated until every
  # opened paste has closed. A paste entirely within one line needs no extra reads; a multi-line paste
  # keeps reading until the count balances. Returns nil on EOF (Ctrl-D).
  def self.read_submission : String?
    line = read_line_edited
    return if line.nil?
    return line.chomp unless line.includes?(PASTE_START) || line.includes?(PASTE_END)

    buf = line
    # Keep reading while more paste-starts have been seen than paste-ends: the paste spans lines.
    while buf.split(PASTE_START).size > buf.split(PASTE_END).size
      nxt = read_line_edited(continued: true)
      break if nxt.nil?
      buf += "\n#{nxt}"
    end

    text = buf.gsub(PASTE_START, "").gsub(PASTE_END, "").chomp
    lines = text.count('\n') + 1
    STDERR.puts "  #{"pasted #{lines} line(s), #{text.bytesize} B".colorize(:dark_gray)}" if lines > 1
    text
  end

  # The prompt readline draws, and where history is kept across sessions.
  PROMPT       = "#{"You".colorize(:light_green).bold} \u276f "
  HISTORY_FILE = File.join(ENV["HOME"]? || ".", ".shainet_agent_history")
  MAX_HISTORY  = 1000

  @@history_loaded = false
  @@last_history : String? = nil
  # Our own copy of the session's entries, because the crystal-readline shard binds add_history but
  # exposes NO way to read the history list back for saving.
  @@history = [] of String

  # Read one edited line: arrow-key history, in-line editing (Ctrl-A/E/W, left/right), and a persistent
  # history file -- the biggest gap the earlier harness review named, since a coding agent has you
  # retyping paths and commands constantly. Uses the crystal-readline shard (a libreadline binding), so
  # this is line editing the terminal already knows how to do rather than a reimplementation.
  #
  # Falls back to plain gets when stdin is not a tty (piped input, a test harness): readline needs a
  # terminal, and forcing it on a pipe would hang. A `continued` line of a multi-line paste draws a
  # blank prompt and is not recorded on its own -- the whole submission is one history entry.
  #
  # Returns nil on EOF (Ctrl-D), which the shard returns as nil.
  def self.read_line_edited(continued : Bool = false) : String?
    unless STDIN.tty?
      line = gets
      return line.nil? ? nil : line.chomp
    end
    load_history unless @@history_loaded
    prompt = continued ? "" : PROMPT
    # add_history: true lets libreadline record this line for arrow-key recall. A continuation line
    # passes false. We also keep our own copy in @@history for persistence, since the shard exposes no
    # way to read libreadline's history list back out for saving.
    add = !continued
    line = begin
      Readline.readline(prompt, add_history: add)
    rescue
      nil
    end
    return if line.nil?
    if add && !line.strip.empty? && line != @@last_history
      @@history << line
      @@last_history = line
    end
    line
  end

  private def self.load_history
    @@history_loaded = true
    return unless File.exists?(HISTORY_FILE)
    File.each_line(HISTORY_FILE) do |l|
      next if l.strip.empty?
      # Seed libreadline's own history directly -- the shard has no public add method, but its
      # LibReadline binding does, and that is what makes the file's entries reachable with up-arrow.
      LibReadline.add_history(l)
      @@history << l
      @@last_history = l
    end
  rescue
    # A corrupt or unreadable history file must never stop the agent starting.
  end

  # Append the session's entries, capped so the file cannot grow without bound. Called at exit.
  def self.save_history
    return unless @@history_loaded
    lines = @@history.last(MAX_HISTORY)
    File.write(HISTORY_FILE, lines.join("\n") + "\n") unless lines.empty?
  rescue
    # History is a convenience; failing to persist it must not crash a clean exit.
  end

  # One dimmed line describing what a tool returned.
  #
  # Results previously went only into the model's context, so the human watching saw the CALL and then
  # nothing: a read that came back empty, a search that matched nothing, and a search that matched a
  # hundred lines were indistinguishable on screen. That is the difference between watching an agent
  # work and watching a spinner. An error shows its first line, because the reason a retry is happening
  # is exactly what a person needs in order to intervene.
  def self.summarize_result(result : String) : String
    first = result.lines.first?.to_s.strip
    return "↳ (empty)" if result.strip.empty?
    if first.starts_with?("Error")
      return "↳ #{first[0, 100]}"
    end
    lines = result.count('\n') + 1
    kb = result.bytesize >= 1024 ? " · #{(result.bytesize / 1024.0).round(1)} KB" : " · #{result.bytesize} B"
    # A file read already states its own range in the header, and a no-match search says so; echoing
    # a line count over those adds nothing, so show their own first line instead.
    if first.starts_with?("// ") || first.starts_with?("No matches") || first.starts_with?("No files")
      "↳ #{first[0, 100]}"
    else
      "↳ #{lines} line(s)#{kb}"
    end
  end

  # Structural regex metacharacters. A bare '.' is deliberately NOT here: it appears in almost every
  # literal search ("agent.cr", "File.read") and as a regex it still matches the literal, so treating
  # it as an intent signal would fire constantly for no gain.
  REGEX_METACHARS = /[|()\[\]+*?{}^$]/

  # Search file contents, returning "path:line: text" lines.
  #
  # git grep first: it is far faster than walking the tree in-process and it honours .gitignore, so
  # generated and vendored files do not drown the real hits. It is not always available -- the
  # workspace may not be a repo, or git may not be installed -- so the in-process walk stays as the
  # fallback rather than as dead code, and both paths are held to the same output budget.
  #
  # A zero-hit LITERAL search whose pattern looks like a regex is retried as one. Matching literally by
  # default is right -- it cannot produce surprising hits -- but observed behaviour is that the model
  # writes an alternation without setting regex and then misreads the empty result: in one session it
  # searched "puts|print|colorize|…", got nothing, concluded the path parameter rejects files (it does
  # not -- that same query matches 46 lines), changed the path AND dropped the alternation together,
  # and credited the wrong change. Two calls wasted and a false belief to reason from. The retry only
  # ever fires when the strict answer was already empty, so it cannot mask a real result, and it says
  # what it did so the next call is better formed.
  def self.search_contents(root : String, pattern : String, glob : String?,
                           context : Int32, regex : Bool) : String
    hit = git_grep(root, pattern, glob, context, regex)
    hit ||= walk_grep(root, pattern, glob, context, regex)
    return hit unless hit == "No matches." && !regex && pattern.matches?(REGEX_METACHARS)

    retried = git_grep(root, pattern, glob, context, true) ||
              walk_grep(root, pattern, glob, context, true)
    return hit if retried == "No matches."
    "(no literal match; retried as a regular expression — pass regex=true to do this directly)\n#{retried}"
  end

  # Returns nil when git grep could not run, so the caller falls back.
  private def self.git_grep(root : String, pattern : String, glob : String?,
                            context : Int32, regex : Bool) : String?
    rel = relativize(root)
    args = ["grep", "-nI", "--no-color", "--untracked", "-i"]
    # An explicit path is a request to look THERE, so ignored files under it count. Without this a
    # gitignored file named directly came back "No matches." -- git grep exits 1 for it -- which the
    # model read as the tool ignoring its path (seen with a *.log-ignored run.log). Workspace-wide
    # searches keep .gitignore so build output does not drown the real hits.
    args << "--no-exclude-standard" unless rel == "." || rel.empty?
    args += ["-C", context.to_s] if context > 0
    args << (regex ? "-E" : "-F")
    args += ["-e", pattern, "--"]
    if glob && !glob.empty?
      spec = glob.includes?("/") ? glob : "**/#{glob}"
      prefix = (rel == "." || rel.empty?) ? "" : "#{rel.rstrip('/')}/"
      args << ":(glob)#{prefix}#{spec}"
    elsif rel != "." && !rel.empty?
      args << rel
    end

    stdout = IO::Memory.new
    stderr = IO::Memory.new
    status = Process.run("git", args, output: stdout, error: stderr, chdir: workspace)
    # Exit 1 is "no matches", which is a real answer. Anything else (128 = not a repo, or git
    # missing) means the search did not happen, so say so by returning nil.
    return unless status.exit_code == 0 || status.exit_code == 1
    bound_matches(stdout.to_s.lines.map(&.chomp).reject(&.empty?))
  rescue
    nil
  end

  private def self.walk_grep(root : String, pattern : String, glob : String?,
                             context : Int32, regex : Bool) : String
    re = begin
      regex ? Regex.new(pattern, Regex::Options::IGNORE_CASE) : Regex.new(Regex.escape(pattern), Regex::Options::IGNORE_CASE)
    rescue ex
      return "Error: invalid regex: #{ex.message}"
    end

    files = Dir.exists?(root) ? Dir.glob(File.join(root, "**", "*")) : [root]
    hits = [] of String
    files.each do |f|
      break if hits.size >= MAX_MATCHES
      rel = relativize(f)
      next if noise_path?(rel)
      next if Dir.exists?(f) || !File.exists?(f) || File.size(f) > 2_000_000
      next if glob && !glob.empty? && !File.match?(glob.includes?("/") ? glob : "**/#{glob}", rel)
      begin
        lines = File.read_lines(f)
        lines.each_with_index do |line, i|
          next unless re.matches?(line)
          lo = Math.max(0, i - context)
          hi = Math.min(lines.size - 1, i + context)
          (lo..hi).each { |j| hits << "#{rel}:#{j + 1}: #{lines[j].strip}" }
          break if hits.size >= MAX_MATCHES
        end
      rescue
        # unreadable or binary
      end
    end
    bound_matches(hits)
  end

  # Hold match output to the same budget as a file read: long lines cut, total bounded, and the
  # overflow reported as a count with the remedy rather than silently dropped.
  private def self.bound_matches(lines : Array(String)) : String
    return "No matches." if lines.empty?
    kept = lines.first(MAX_MATCHES)
    body = kept.map do |l|
      l.size > MAX_LINE_CHARS ? "#{l[0, MAX_LINE_CHARS]} …[+#{l.size - MAX_LINE_CHARS} chars]" : l
    end
    text = body.join("\n")
    if text.bytesize > LARGE_FILE_BYTES
      acc = [] of String
      used = 0
      body.each do |l|
        break if used + l.bytesize + 1 > LARGE_FILE_BYTES
        used += l.bytesize + 1
        acc << l
      end
      text = acc.join("\n")
      return "#{text}\n… output truncated at #{acc.size} of #{lines.size} matching lines; narrow the pattern, the glob or the path."
    end
    if lines.size > kept.size
      "#{text}\n… and #{lines.size - kept.size} more matches; narrow the pattern, the glob or the path."
    else
      text
    end
  end

  # Find files by NAME. Returns paths only, so the model can pick what to read.
  def self.glob_files(root : String, pattern : String) : String
    spec = File.join(root, pattern)
    found = Dir.glob(spec).reject do |p|
      rel = relativize(p)
      noise_path?(rel) || Dir.exists?(p)
    end
    return "No files match #{pattern}." if found.empty?
    rels = found.map { |p| relativize(p) }.sort!
    shown = rels.first(MAX_MATCHES)
    head = "#{rels.size} file(s) match #{pattern}"
    body = shown.join("\n")
    if rels.size > shown.size
      "#{head} (showing #{shown.size}):\n#{body}\n… narrow the pattern to see the rest."
    else
      "#{head}:\n#{body}"
    end
  end

  # Longest tool argument value shown on the call line.
  #
  # The call line used to print every argument through `inspect` in full, so a write_file or
  # apply_patch carrying a whole file scrolled the entire payload past the user -- the one moment they
  # most need to SEE what is about to happen is the moment the screen fills with it. The name and the
  # size are what a person checks; the body is already shown by the confirmation diff.
  ARG_PREVIEW_CHARS = 72

  # Format tool-call arguments for the one-line display: short values verbatim, long ones cut with
  # their real size, so a large patch reads as "patch=\"*** Begin Patch…\" (+4182 chars)".
  def self.format_args(args : Hash(String, String)) : String
    args.map do |k, v|
      if v.size > ARG_PREVIEW_CHARS
        "#{k}=#{v[0, ARG_PREVIEW_CHARS].inspect} (+#{v.size - ARG_PREVIEW_CHARS} chars)"
      else
        "#{k}=#{v.inspect}"
      end
    end.join(", ")
  end

  # Longest single line returned before it is cut.
  #
  # A line-count cap alone does NOT bound the response: 200 lines of minified JavaScript, generated
  # code or an embedded data blob is still enormous, so the head limit above would be satisfied while
  # the budget was blown anyway. Claude Code truncates each line to 2000 characters for this reason.
  # 500 is tighter because our window is 64K rather than 200K, and a source line past 500 characters
  # is nearly always machine-written.
  MAX_LINE_CHARS = 500

  # Render a file for the model: line-numbered, with a header saying which lines these are of how many.
  #
  # The header is what makes paging possible. A model that sees "lines 1-200 of 2093" knows both that
  # there is more and exactly how to ask for it; one handed a silent truncation knows neither.
  def self.render_file(display_path : String, content : String,
                       start_line : Int32?, end_line : Int32?) : String
    lines = content.split('\n')
    total = lines.size

    if start_line || end_line
      s = Math.max(1, start_line || 1)
      e = Math.min(total, end_line || total)
      return "Error: line #{s} is past the end of #{display_path} (#{total} lines)" if s > total
      return "Error: end_line #{e} is before start_line #{s}" if e < s
      kept = fit_lines(lines, s - 1, e - s + 1)
      last = s + kept - 1
      note = last < e ? "; clipped at line #{last} to stay inside the context budget, ask for the rest" : ""
      return "// #{display_path} — lines #{s}-#{last} of #{total}#{note}\n" \
             "#{number_lines(lines[(s - 1), kept], s, last)}"
    end

    if content.bytesize > LARGE_FILE_BYTES
      kept = fit_lines(lines, 0, HEAD_LINES)
      "// #{display_path} — lines 1-#{kept} of #{total}; file is large, so this is only the head. " \
      "Call read_file again with start_line/end_line for a specific range, or use search to locate " \
      "what you need first.\n#{number_lines(lines[0, kept], 1, kept)}"
    else
      "// #{display_path} — #{total} lines\n#{number_lines(lines, 1, total)}"
    end
  end

  # How many lines starting at `from` fit inside the byte budget, up to `want` of them.
  #
  # A line COUNT cap does not bound the response on its own, and neither does a per-line cap: 200
  # lines of 500 characters is still 100 KB, about 26K tokens, which would defeat the whole point on a
  # 64K window. Only counting the bytes actually does it. Measured on 400 lines of 2000 characters --
  # the shape of minified or generated code -- this is what takes an 800 KB file to a bounded reply
  # instead of a 104 KB one.
  #
  # At least one line is always returned, so a single enormous line still yields something rather than
  # an empty response.
  private def self.fit_lines(lines : Array(String), from : Int32, want : Int32) : Int32
    budget = LARGE_FILE_BYTES
    used = 0
    kept = 0
    while kept < want && (from + kept) < lines.size
      line = lines[from + kept]
      cost = Math.min(line.bytesize, MAX_LINE_CHARS) + 10 # +10 for the "NNNN | " gutter
      break if kept > 0 && used + cost > budget
      used += cost
      kept += 1
    end
    kept
  end

  private def self.number_lines(slice : Array(String), from : Int32, upto : Int32) : String
    width = upto.to_s.size
    slice.map_with_index do |line, i|
      shown = if line.size > MAX_LINE_CHARS
                "#{line[0, MAX_LINE_CHARS]} …[+#{line.size - MAX_LINE_CHARS} chars]"
              else
                line
              end
      "#{(from + i).to_s.rjust(width)} | #{shown}"
    end.join("\n")
  end

  # Did this reply START a tool call without finishing it?
  #
  # This is the precise signal, and it beats guessing from prose. The observed failure was the model
  # emitting its preamble inside a <think> block and then getting 81 characters into the call before
  # generation stopped -- right after </parameter>, with </function></tool_call> missing. The parser
  # needs the closing tag, so it returned nothing and the loop treated a half-written call as the
  # final answer.
  #
  # Counting the tags is unambiguous where prose heuristics are not: an earlier version of this looked
  # for "let's" / "i'll" openers and missed the real case entirely, because once the <think> block was
  # stripped the remaining text began with "<tool_call>".
  def self.truncated_tool_call?(text : String) : Bool
    opens = text.split("<tool_call>").size - 1
    closes = text.split("</tool_call>").size - 1
    opens > closes
  end

  # Did the model reply with nothing usable at all -- no call, no answer?
  #
  # Distinct from a truncated call: here it never started one and said nothing either, so there is
  # nothing to salvage and nothing to show the user.
  def self.empty_reply?(visible : String) : Bool
    visible.strip.empty?
  end

  def self.parse_tool_calls(text : String) : Array(ToolCall)
    text = text.scrub # never run regex on invalid UTF-8 (broken-model output)
    calls = [] of ToolCall
    text.scan(/<tool_call>(.*?)<\/tool_call>/m) do |m|
      body = m[1]
      fmatch = body.match(/<function=([^>\s]+)>(.*)/m)
      next unless fmatch
      name = fmatch[1].strip
      next if name.empty? || name == "none" # ignore placeholder/hallucinated calls
      args = {} of String => String
      fmatch[2].scan(/<parameter=([^>\s]+)>\n?(.*?)\n?<\/parameter>/m) do |pm|
        args[pm[1].strip] = pm[2]
      end
      calls << ToolCall.new(name, args)
    end
    calls
  end

  class Agent
    @im_start : Int32
    @im_end : Int32
    @nl : Array(Int32)
    @stop_ids : Array(Int32)
    @system_block : String

    # Optional bottom-line status bar. Nil when not on a tty, so every update site can call it
    # unconditionally without a second tty check.
    property status_bar : AgentDemo::StatusBar? = nil
    @messages : Array(Message)
    @recent_calls : Array(String)
    @sampler : SHAInet::Sampler
    getter max_context : Int32

    def initialize(@net : SHAInet::Network, @tokenizer : SHAInet::BPETokenizer, @tools : Array(Tool), @max_context : Int32 = 65536)
      im_start = @tokenizer.vocab["<|im_start|>"]?
      im_end = @tokenizer.vocab["<|im_end|>"]?
      raise "model is not ChatML (<|im_start|>/<|im_end|> missing); this agent targets Qwen3-style models" unless im_start && im_end
      @im_start = im_start.not_nil!
      @im_end = im_end.not_nil!
      @nl = @tokenizer.encode("\n")
      @stop_ids = [@im_end]
      ["<|endoftext|>", "<|end_of_text|>"].each { |n| (id = @tokenizer.vocab[n]?) && @stop_ids << id }
      @system_block = AgentDemo.render_tools_block(@tools)
      @recent_calls = [] of String
      # Qwen3.8's recommended thinking-mode sampling (model card): temperature 1.0, top_p 0.95,
      # top_k 20, no presence or repetition penalty. The previous 0.3 was close to greedy, which Qwen
      # warns degrades thinking-mode output and causes endless repetition -- the loops this agent hit.
      @sampler = SHAInet::Sampler.new(temperature: QwenThinking::TEMPERATURE, top_k: QwenThinking::TOP_K,
        top_p: QwenThinking::TOP_P)
      @messages = [] of Message
    end

    # <|im_start|>{role}\n{content}<|im_end|>\n
    private def render_message(role : String, content : String) : Array(Int32)
      ids = [@im_start]
      ids.concat(@tokenizer.encode("#{role}\n#{content}"))
      ids << @im_end
      ids.concat(@nl)
      ids
    end

    # Same framing, but with the assistant's body taken VERBATIM from the tokens the model
    # emitted. This is what makes the KV prefix match: during generation the cache holds
    # `<|im_start|>assistant\n` followed by exactly these ids, so the cache is a true prefix
    # of the next prompt and only the tail has to be prefilled.
    private def render_assistant(body : Array(Int32)) : Array(Int32)
      ids = [@im_start]
      ids.concat(@tokenizer.encode("assistant\n"))
      ids.concat(body)
      ids << @im_end
      ids.concat(@nl)
      ids
    end

    private def build_prompt : Array(Int32)
      ids = transcript_ids
      ids << @im_start
      ids.concat(@tokenizer.encode("assistant\n"))
      ids
    end

    # Qwen3.8's template opens the system message with the reasoning-effort line; xhigh is its default.
    THINKING = QwenThinking.mode(nil)

    private def system_ids : Array(Int32)
      body = "#{@system_block}\n\n<WORKSPACE>\nAll relative paths resolve " \
             "against #{AgentDemo.workspace}\nThis is a #{AgentDemo.host_os} host. You are already in that " \
             "directory — do not cd to it, and never search the filesystem for it.\n</WORKSPACE>"
      render_message("system", QwenThinking.with_instruction(body, THINKING))
    end

    # Tokens every prompt carries regardless of the conversation: the system message (with the tool
    # descriptions, ~2.4K tokens) and the assistant primer. Compaction targets are measured on top of
    # this; a target below it can only be met by deleting the whole conversation.
    def overhead_tokens : Int32
      system_ids.size + 1 + @tokenizer.encode("assistant\n").size
    end

    # The system message and every message so far, without the assistant primer. build_prompt and the
    # in-context summary both start with exactly these ids, which is what lets a summary reuse the KV
    # cache instead of re-prefilling the conversation.
    private def transcript_ids : Array(Int32)
      # The workspace is appended HERE rather than baked into @system_block, which is built once at
      # construction: /workspace can change the root mid-session, and a stale root in the system
      # message is worse than none.
      #
      # Stating it at all matters because the model does not otherwise know where it is. Observed: it
      # tried `cd /Users/...` -- a macOS path, on Linux -- and then `find / -name agent.cr`, a scan of
      # the entire filesystem, to work out its own location. Every path it handles is relative to this
      # directory, so it is the one fact the prompt cannot afford to omit.
      ids = system_ids
      @messages.each do |m|
        if m.role == "tool"
          ids.concat(render_message("user", "<tool_response>\n#{m.content}\n</tool_response>"))
        elsif body = m.ids
          ids.concat(render_assistant(body))
        else
          ids.concat(render_message(m.role, m.content))
        end
      end
      ids
    end

    # Current prompt size in tokens (full transcript + assistant primer).
    def context_tokens : Int32
      build_prompt.size
    end

    def reset
      @messages.clear
      # Also drop the KV cache. Prefix reuse would stay correct without this -- a shorter
      # prompt cannot be an extension of the old cache, so it would fall back to a full
      # prefill -- but there is no reason to keep the old conversation's cache resident.
      reset_cache!
    end

    # Bound a tool result in TOKENS, not bytes.
    #
    # The tools already cap their output at MAX_READ_BYTES, but bytes are the wrong unit: the
    # binding constraint is the context window, and the bytes-per-token ratio varies wildly.
    # 64 KB of English is roughly 16k tokens; 64 KB of a compiled binary tokenized to 60534
    # in one observed run, which then tried to prefill against an 8192-token window.
    # Compaction cannot rescue that either, because it drops the OLDEST messages while the
    # offending one is the newest.
    #
    # A quarter of the window leaves room for the transcript, the primer and a reply.
    private def clamp_tool_result(result : String) : String
      budget = @max_context // 4
      ids = @tokenizer.encode(result)
      return result if ids.size <= budget
      kept = @tokenizer.decode(ids[0, budget]).scrub
      "#{kept}\n... [truncated: #{ids.size} tokens exceeded the #{budget}-token tool budget]"
    end

    # Tokens one message contributes to the prompt.
    private def message_tokens(m : Message) : Int32
      if m.role == "tool"
        render_message("user", "<tool_response>\n#{m.content}\n</tool_response>").size
      else
        render_message(m.role, m.content).size
      end
    end

    # Longest summary the model may write, and the instruction it gets.
    SUMMARY_TOKENS      = 1024
    SUMMARY_INSTRUCTION = "Write a summary of the conversation so far. It will REPLACE the older " \
                          "messages, so it must carry everything still needed to continue: the user's " \
                          "goals and requests, decisions made, every file path read or changed, commands " \
                          "and their outcomes, important tool results, and any unfinished work. If the " \
                          "conversation starts with an earlier summary, carry its facts forward -- they " \
                          "exist nowhere else. Do not call any tools. Be concise and factual. Output " \
                          "only the summary."

    # Below this an in-context summary is treated as a misfire. Observed once: a 35-token "summary"
    # that recorded nothing, after which the model could not say which files it had read.
    MIN_SUMMARY_TOKENS = 48

    # The assistant primer with an EMPTY think block: Qwen's way of asking for an answer without
    # reasoning first. A summary does not need it, and reasoning would spend the token budget before
    # the summary starts.
    private def no_think_primer : Array(Int32)
      ids = [@im_start]
      ids.concat(@tokenizer.encode("assistant\n<think>\n\n</think>\n\n"))
      ids
    end

    # Summarize `head` -- the messages about to be dropped -- with the model itself, so history is
    # condensed rather than lost.
    #
    # The first choice is to ask IN the conversation: the current transcript plus one instruction.
    # The KV cache already holds that transcript, so only the instruction is prefilled. The previous
    # version asked in a fresh prompt (its own system message, the head re-rendered as text), which
    # threw the cache away and re-prefilled the whole old history -- tens of thousands of tokens at
    # exactly the moment the context was too large -- and could itself exceed the window.
    #
    # When the transcript leaves no room for the instruction and a summary (one oversized message can
    # do that), fall back to summarizing the head on its own, in bounded chunks.
    private def summarize(head : Array(Message)) : String
      instr = render_message("user", SUMMARY_INSTRUCTION)
      base = transcript_ids
      fits = base.size + instr.size + no_think_primer.size + SUMMARY_TOKENS <= @max_context
      # Only worth it when the cache already holds (nearly all of) the transcript. After the OOM
      # recovery has cleared the cache, asking in-context would re-prefill the very transcript that
      # just ran out of memory; the bounded standalone path is the safer one there.
      cached = 0
      limit = Math.min(@cache_ids.size, base.size)
      while cached < limit && @cache_ids[cached] == base[cached]
        cached += 1
      end
      if fits && cached * 10 >= base.size * 9
        prompt = base + instr + no_think_primer
        summary = clean_summary(generate_from(prompt, SUMMARY_TOKENS)[0])
        return summary if @tokenizer.encode(summary).size >= MIN_SUMMARY_TOKENS
        STDERR.puts "  [agent] in-context summary came back empty; summarizing the old messages directly".colorize(:dark_gray)
      end
      summarize_standalone(head)
    end

    # Fallback summary over the head alone, as a rolling summary over chunks that each fit half the
    # window: every call is bounded no matter how large the history is.
    private def summarize_standalone(head : Array(Message)) : String
      budget = @max_context // 2
      summary = ""
      chunk = String::Builder.new
      chunk_tokens = 0
      flush = ->(text : String) do
        intro = summary.empty? ? "" : "Summary of the conversation before this part:\n#{summary}\n\n"
        instruction = "#{intro}Conversation:\n#{text}\n\n#{SUMMARY_INSTRUCTION}"
        prompt = render_message("system", "You write concise, faithful conversation summaries.")
        prompt.concat(render_message("user", instruction))
        prompt.concat(no_think_primer)
        summary = clean_summary(generate_from(prompt, SUMMARY_TOKENS)[0])
      end
      head.each do |m|
        line = "#{m.role}: #{transcript_text(m)}\n\n"
        t = @tokenizer.encode(line).size
        if chunk_tokens > 0 && chunk_tokens + t > budget
          flush.call(chunk.to_s)
          chunk = String::Builder.new
          chunk_tokens = 0
        end
        chunk << line
        chunk_tokens += t
      end
      flush.call(chunk.to_s) if chunk_tokens > 0
      summary.empty? ? TRUNCATION_MARKER : summary
    end

    # A message as the standalone summarizer sees it: reasoning dropped, and long bodies (tool output
    # above all) cut to their start and end, which is where the useful part of a log or a file is.
    private def transcript_text(m : Message) : String
      text = m.content.gsub(/<think>.*?<\/think>/m, "").strip
      ids = @tokenizer.encode(text)
      return text if ids.size <= 1500
      "#{@tokenizer.decode(ids[0, 1000]).scrub}\n... [#{ids.size - 1500} tokens omitted] ...\n" \
      "#{@tokenizer.decode(ids[-500..]).scrub}"
    end

    private def clean_summary(text : String) : String
      text.gsub(/<think>.*?<\/think>/m, "").gsub(/<\/?think>/, "")
        .gsub(/<tool_call>.*?(<\/tool_call>|\z)/m, "").strip
    end

    # Compact the transcript toward `target` tokens: keep a recent tail verbatim and replace the older
    # head with a model-written summary, hard-trimming as a last resort. Returns tokens removed.
    #
    # `force` compacts even when the transcript is already under `target`, which is what /compact
    # means: without it the command returned 0 for any session below 75% of the window -- that is,
    # it did nothing in exactly the sessions where someone thinks to run it.
    #
    # Never raises. A summary that fails (out of memory, interrupted) falls back to dropping the head
    # with a marker; the old version let the exception out of the turn and took the agent down.
    def compact!(target : Int32, force : Bool = false) : Int32
      before = context_tokens
      return 0 if @messages.size <= 1
      return 0 if before <= target && !force

      # Keep the most recent messages within ~half of what the target leaves for the conversation;
      # summarize the rest.
      keep_budget = Math.max(0, target - overhead_tokens) // 2
      tail = [] of Message
      tail_tokens = 0
      @messages.reverse_each do |m|
        t = message_tokens(m)
        break if !tail.empty? && tail_tokens + t > keep_budget
        tail.unshift(m)
        tail_tokens += t
      end
      head = @messages[0, @messages.size - tail.size]

      if head.empty?
        # Nothing old enough to summarize — fall back to plain truncation.
        return truncate!(target)
      end

      STDERR.puts "  [agent] summarizing #{head.size} older message(s) to free context...".colorize(:dark_gray)
      summary = begin
        summarize(head)
      rescue ex
        STDERR.puts "  [agent] summary failed (#{ex.message}); dropping the older messages instead".colorize(:yellow)
        reset_cache!
        nil
      end
      if summary
        STDERR.puts "  [agent] summary: #{@tokenizer.encode(summary).size} tokens".colorize(:dark_gray)
        summary.each_line { |l| STDERR.puts "    │ #{l}".colorize(:dark_gray) }
        @messages = [Message.new("user", "#{SUMMARY_PREFIX}\n#{summary}")] + tail
      else
        @messages = [Message.new("user", TRUNCATION_MARKER)] + tail
      end
      # If the summary + tail still overflow, drop oldest until it fits.
      truncate!(target)
      before - context_tokens
    end

    # Plain sliding-window drop of oldest messages with a marker. Returns tokens removed.
    #
    # A leading summary is kept: it is the condensed form of everything already dropped, so trimming
    # starts with the oldest message AFTER it. Dropping it first -- as this did -- meant a compaction
    # whose summary plus tail ran over the target deleted the summary it had just written.
    private def truncate!(target : Int32) : Int32
      before = context_tokens
      keep = @messages.first?.try(&.content.starts_with?(SUMMARY_PREFIX)) ? 1 : 0
      trimmed = false
      while context_tokens > target && @messages.size > keep + 1
        @messages.delete_at(keep)
        trimmed = true
      end
      if trimmed && keep == 0 && (@messages.empty? || @messages.first.content != TRUNCATION_MARKER)
        @messages.unshift(Message.new("user", TRUNCATION_MARKER))
      end
      before - context_tokens
    end

    SUMMARY_PREFIX = "[Summary of earlier conversation]"

    TRUNCATION_MARKER = "[Earlier conversation was truncated to fit the context window.]"

    # Tokens a turn keeps free for the model's reply before compacting. A quarter of the window: the
    # turn itself is bounded only by the window (there is no per-reply token cap), so this is a
    # headroom heuristic, not a hard reply size -- it keeps a session from starting a turn with almost
    # no room for the reasoning, the answer and a tool round-trip.
    def reply_reserve : Int32
      @max_context // 4
    end

    # One-line status: context usage, VRAM, expert-cache hit rate.
    def status : String
      parts = ["ctx #{context_tokens}/#{@max_context} tok"]
      if info = SHAInet::CUDA.memory_info
        used = info[:total] - info[:free]
        mb = 1024.0 * 1024.0
        parts << "VRAM #{(used / mb).round}/#{(info[:total] / mb).round} MB (#{(100.0 * used / info[:total]).round}%)"
      end
      if ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1" && SHAInet::CUDA.fully_available?
        cs = SHAInet::Q4HostMatrix.cache_stats
        if cs[:hits] + cs[:misses] > 0
          parts << "cache #{(cs[:hit_rate] * 100).round}% hit"
        end
      end
      "[#{parts.join(" · ")}]"
    end

    # Tokens currently held in the KV cache, in order. This is what makes turn 2 onwards
    # cheap: the cache already contains the whole conversation so far, so only the new tail
    # has to be prefilled.
    @cache_ids = [] of Int32

    # Clearing the KV cache and forgetting what it held MUST happen together. If they drift,
    # a later turn believes the cache still holds a prefix that is gone and reuses it, which
    # is silently wrong output rather than an error.
    private def reset_cache!
      @net.clear_cache!
      @cache_ids.clear
    end

    # Prefill `prompt`, reusing whatever the KV cache already holds.
    #
    # This used to clear the cache and re-prefill the entire transcript on EVERY call, and
    # the tool loop calls it once per step -- so one request with three tool calls paid four
    # full prefills of a growing context. At 16k that is around 405 s each, which is the
    # multi-minute silence between an answer and the next action. It was never hidden
    # reasoning; it was prefill with nothing to show.
    #
    # build_prompt appends, so consecutive prompts share a long prefix. Reuse is only taken
    # when the cache is EXACTLY a prefix of the new prompt: a KV cache cannot drop a suffix,
    # so a partial match cannot be salvaged. If the assistant's text re-tokenizes differently
    # from the tokens that were generated, the common prefix simply ends early and this falls
    # back to a full prefill -- slower, never wrong.
    private def prefill(prompt : Array(Int32)) : SHAInet::SimpleMatrix
      shared = 0
      limit = Math.min(@cache_ids.size, prompt.size)
      while shared < limit && @cache_ids[shared] == prompt[shared]
        shared += 1
      end

      if shared == @cache_ids.size && shared > 0 && shared < prompt.size
        reused = shared
        tail = prompt[shared..]
        STDERR.puts "  reusing #{reused} cached tok · prefilling #{tail.size}".colorize(:dark_gray)
        logits = run_prefill(tail, reused, prompt.size)
        record_prefilled(prompt)
        logits
      else
        STDERR.puts "  prefilling #{prompt.size} tok from scratch".colorize(:dark_gray) if prompt.size > 512
        reset_cache!
        logits = run_prefill(prompt, 0, prompt.size)
        record_prefilled(prompt)
        logits
      end
    end

    # Record what the KV cache now holds -- unless the prefill was interrupted, in which case throw the
    # cache away.
    #
    # An interrupted prefill leaves the cache holding FEWER tokens than the prompt, so claiming the
    # whole prompt would make the next turn "reuse" a prefix that is not physically there. That is
    # silent corruption of the kind this codebase has already produced once (a clear_cache! that missed
    # the recurrent blocks), and it is the exact hole an audit flagged as the next one of its family:
    # nothing asserts that a prefix reuser's cache length matches the prefix it claims. An aborted turn
    # does not need to be fast, it needs to leave the next one correct.
    private def record_prefilled(prompt : Array(Int32))
      if AgentDemo.cancelled?
        reset_cache!
        STDERR.puts "  #{"cache cleared after the interrupted prefill".colorize(:dark_gray)}"
      else
        @cache_ids = prompt.dup
      end
    end

    # Feed tokens to the model in slices so there is something to show. The KV cache
    # accumulates across calls, so slicing is equivalent to one call.
    private def run_prefill(ids : Array(Int32), already : Int32, total : Int32) : SHAInet::SimpleMatrix
      slice = (ENV["SHAINET_PREFILL_SLICE"]? || "0").to_i
      slice = ids.size if slice <= 0
      logits = nil
      i = 0
      t0 = Time.monotonic
      # Per-layer progress callback
      @net.prefill_progress = ->(layer_idx : Int32, total_layers : Int32) do
        elapsed = (Time.monotonic - t0).total_seconds
        pct = layer_idx * 100 // total_layers
        STDERR.print "\r  prefill layer #{layer_idx}/#{total_layers} (#{pct}%) · #{elapsed.round(0).to_i}s".colorize(:dark_gray)
        STDERR.flush
        AgentDemo.pump_input
        nil
      end
      while i < ids.size
        n = Math.min(slice, ids.size - i)
        logits = @net.run(ids[i, n], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
        i += n
        done = already + i
        elapsed = (Time.monotonic - t0).total_seconds
        STDERR.print "\r  prefill #{done}/#{total} tok (#{done * 100 // total}%) · #{elapsed.round(0).to_i}s".colorize(:dark_gray)
        STDERR.flush
        # Prefill is the OTHER long blocking phase, and the one most worth escaping: a cold 60K prefill
        # runs for minutes, and realizing the prompt was wrong 20 seconds in should not mean waiting it
        # out. Same reason as the decode loop -- without a yield the signal fiber never runs -- and the
        # cost is one yield per chunk, not per token.
        Fiber.yield
        if AgentDemo.cancelled?
          STDERR.print "\r\033[K"
          STDERR.puts "  #{"interrupted during prefill".colorize(:yellow)}"
          break
        end
      end
      @net.prefill_progress = nil
      STDERR.print "\r\033[K"
      logits.not_nil!
    end

    # Generate from an explicit prompt. When echo is set, the user-facing prose is streamed
    # live (with <think> and tool-call markup filtered out).
    #
    # The only ceiling is `max_tokens` clamped to the room left in the context window. A turn passes
    # the whole window (see `generate`), so the model may think for as long as the window holds and
    # stops for real only on a stop token, a user interrupt (Ctrl-C/steer), or the window filling.
    # Summaries pass a small `max_tokens` (SUMMARY_TOKENS) to stay bounded.
    private def generate_from(prompt : Array(Int32), max_tokens : Int32, echo : Bool = false) : Tuple(String, Array(Int32))
      # Never generate past the KV budget: the cache is pinned at @max_context, and running into it
      # mid-reply raises out of the decode loop and loses the turn.
      max_tokens = Math.min(max_tokens, @max_context - prompt.size)
      logits = prefill(prompt)
      generated = [] of Int32
      renderer = echo ? AgentDemo::StreamRenderer.new(STDERR) : nil
      prev = ""
      gen_started = Time.instant
      max_tokens.times do
        # Let the scheduler run the signal-handling fiber.
        #
        # Without this the Ctrl-C handler NEVER RUNS. Crystal catches the signal in a C handler that
        # only writes to a pipe; the block registered with Process.on_terminate is invoked by a
        # dedicated fiber reading that pipe. This loop is single-threaded and CPU/GPU bound with no
        # other yield point, so the scheduler never gets control and that fiber never runs -- measured
        # directly: 40 iterations of heavy compute with a signal already pending, handler never fired;
        # the same loop with this line fired on the first iteration. The earlier version of this
        # feature was therefore inert, and pressing Ctrl-C repeatedly did nothing until SIGQUIT dumped
        # core.
        #
        # Costs 786 ns, against roughly 85 ms per token.
        Fiber.yield
        # Abandon the turn on Ctrl-C. Checked per token rather than per tool step so a long answer
        # stops promptly; the partial text stays on screen and in history, which is what makes the
        # next instruction ("no, not that file") land in context that explains itself.
        if AgentDemo.cancelled?
          STDERR.puts "\n  #{"interrupted".colorize(:yellow)}"
          break
        end
        # Steering: a line typed during generation stops it and is delivered as the next message.
        # Polled every 8 tokens rather than every one because it is a syscall, and 8 tokens is well
        # under a second of latency at this model's speed.
        AgentDemo.pump_input
        if AgentDemo.steer_pending?
          STDERR.puts "\n  #{"steering".colorize(:cyan)}"
          break
        end
        # Live rate on the bar. This is the number the user actually wants during a long answer, and it
        # did not exist anywhere before: prefill had a progress line, decode had nothing but the text.
        if (bar = @status_bar) && generated.size % 8 == 0 && generated.size > 0
          secs = (Time.instant - gen_started).total_seconds
          rate = secs > 0 ? generated.size / secs : 0.0
          # @cache_ids is what the KV cache holds (prompt plus tokens generated so far). context_tokens
          # rebuilds and re-tokenizes the whole transcript, which is too much work for every 8 tokens.
          bar.text = " ctx #{@cache_ids.size}/#{@max_context} · #{generated.size} tok · " \
                     "#{rate.round(1)} tok/s · #{secs.round(0).to_i}s"
        end
        row = logits.rows - 1
        id = @sampler.sample(logits, row)
        break if id < 0 || @stop_ids.includes?(id)
        break unless logits[row, id].finite?
        generated << id
        # Decoded unconditionally, not just when echoing: the tool-call state above is derived from
        # it, so a non-echoing call (/ask) needs it too.
        full = AgentDemo.complete_utf8(@tokenizer.decode(generated))
        if r = renderer
          r.feed(full.byte_slice(prev.bytesize)) if full.bytesize > prev.bytesize
        end
        prev = full
        logits = @net.run([id], stealth: true, return_matrix: true).as(SHAInet::SimpleMatrix)
        # The sampled token is now in the cache too, so the next turn can reuse it.
        @cache_ids << id
      end
      renderer.try(&.finish)
      # scrub: a broken model can emit tokens that decode to invalid UTF-8, which
      # would crash downstream regex/parsing. The raw ids go back too: the decoded text is
      # for humans and parsing, the ids are what the KV cache actually holds.
      {@tokenizer.decode(generated).scrub, generated}
    end

    # Generate one assistant turn from the current transcript, streaming the filtered prose live.
    #
    # The turn is bounded ONLY by the context window, not by a fixed per-reply token cap. A 27B
    # reasoning model routinely writes a <think> block of many thousands of tokens before it answers;
    # a sub-window cap cut it off mid-thought and the turn ended with no answer. The window is the real
    # budget, and compaction (see chat) keeps room free for the reply and the tool round-trips. A model
    # that never stops is bounded by the window and by Ctrl-C, not by truncating every good turn.
    private def generate : Tuple(String, Array(Int32))
      generate_from(build_prompt, @max_context, echo: true)
    end

    # Handle one user input: run the tool loop until the model answers plainly.
    def chat(input : String)
      @messages << Message.new("user", input)
      @recent_calls.clear
      # One OOM recovery per turn: a second failure after compacting is not a transient.
      compacted_for_oom = false
      empty_nudges = 0
      step = 0
      loop do
        step += 1
        # A cancel during a tool step must also end the LOOP, not just the token stream it was in --
        # otherwise the decoder returns early and the loop calmly starts the next step.
        break if AgentDemo.cancelled?
        # The LAST allowed step still generates, it just may not call tools. Breaking before
        # generation (which is what this did) threw the turn away: the user got the cap
        # message and no answer, after the agent had already done the work. So tell the model
        # its budget is gone and let it write a closing reply from what it has.
        last_step = step >= MAX_TOOL_STEPS
        if step > MAX_TOOL_STEPS
          break
        end
        if last_step
          STDERR.puts "  [agent] tool budget reached (#{MAX_TOOL_STEPS} steps); asking for a final answer".colorize(:dark_gray)
          STDERR.puts "  (raise it with AGENT_MAX_TOOL_STEPS)".colorize(:dark_gray)
          @messages << Message.new("user",
            "You have used all #{MAX_TOOL_STEPS} tool steps for this turn. Do not call any " \
            "more tools. Answer now with what you already have, and say plainly what you " \
            "could not finish.")
        end

        # Keep the prompt AND its reply within the context budget. Compacting only once the prompt
        # alone exceeded the window let a 63K prompt start a 4K reply against a 64K KV budget: the
        # cache filled mid-reply, the turn died with "generation failed", and the next turn -- same
        # size, still under the window -- did it again. Compact to 75% to leave room for the reply
        # and a few tool round-trips before the next compaction.
        if context_tokens + reply_reserve > @max_context
          removed = compact!((@max_context * 0.75).to_i)
          STDERR.puts "  [agent] context compacted (−#{removed} tokens)" if removed > 0
        end

        STDERR.print "\n#{"Agent".colorize(:light_cyan).bold} ❯ "
        STDERR.flush
        begin
          text, text_ids = generate # streams the filtered prose inline
        rescue ex
          # An allocation failure is recoverable, and losing the turn to it is the wrong answer.
          #
          # Auto-compaction only triggers above @max_context, but VRAM can run out BELOW it -- a
          # reported session died at 15401 of 16384 tokens, failing a 64 MB allocation with 79 MB
          # free, throwing away several minutes of tool work. The cache and context are exactly what
          # the memory is holding, so drop both hard and try once more before giving up.
          oom = ex.message.to_s.downcase.includes?("memory allocation") ||
                ex.message.to_s.downcase.includes?("out of memory")
          if oom && !compacted_for_oom
            compacted_for_oom = true
            reset_cache!
            # Compact below where it ACTUALLY failed, not to half the context.
            #
            # The bug: this compacted to @max_context * 0.5. VRAM runs out well below the context limit
            # -- the KV reserve plus the transient prefill workspace can exhaust the card at a token
            # count the context budget still considers fine. Observed: OOM at 17817 tokens with a 40960
            # context, so the 20480 target was ABOVE the failure point, compact! removed 0 (before <=
            # target returns early), and the identical prompt was retried -- straight into an illegal
            # memory access that corrupted CUDA state, because the first OOM had already left the
            # context half-allocated in a way reset_cache! does not fully undo.
            #
            # Target 60% of what failed, so the retry is materially smaller with headroom for the
            # workspace this time.
            failed_at = context_tokens
            target = (failed_at * 0.6).to_i
            removed = compact!(target)
            if removed == 0
              # Could not shrink below the failure point (one huge message, or already minimal).
              # Retrying would hit the same wall and the same CUDA corruption, so stop cleanly.
              STDERR.puts "\n  [agent] out of GPU memory at #{failed_at} tokens and cannot compact " \
                          "below it (one message may be too large). Try /clear or a shorter request, " \
                          "or restart with a smaller SHAINET_AGENT_CONTEXT.".colorize(:yellow)
              break
            end
            STDERR.puts "\n  [agent] out of GPU memory at #{failed_at} tokens; " \
                        "compacted to #{context_tokens} (−#{removed}) and retrying".colorize(:yellow)
            next
          end
          STDERR.puts "\n  [agent] generation failed: #{ex.message}".colorize(:red)
          STDERR.puts "  (out of GPU memory? try /clear, a shorter request, or a smaller SHAINET_EXPERT_CACHE_MB)".colorize(:dark_gray)
          reset_cache!
          break
        end
        STDERR.puts ""
        calls = AgentDemo.parse_tool_calls(text)

        # If the model generated only think tags or whitespace, the user sees nothing.
        visible = text.gsub(/<think>.*?<\/think>/m, "").gsub(/<tool_call>.*?<\/tool_call>/m, "").strip
        if visible.empty? && calls.empty?
          STDERR.puts "  [agent] model returned no visible text".colorize(:dark_gray)
        end

        # Keep the assistant's output (incl. any tool_call markup) verbatim.
        @messages << Message.new("assistant", text.strip, text_ids)

        # A steer interrupted this generation: deliver it and go round again, with the partial output
        # kept above so the model can see what it was doing when it was corrected. No tool calls are
        # run from a generation that was cut off mid-sentence -- the call may be half-written, and the
        # user's correction may well be about that very call.
        if steer = AgentDemo.take_steer
          @messages << Message.new("user", steer)
          STDERR.puts "  #{"→ #{steer}".colorize(:cyan)}"
          next
        end

        # A short reply with no tool call is almost always a DROPPED call, not an answer.
        #
        # This model's own chat template says reasoning may come "in natural language BEFORE the
        # function call, but NOT after", so a turn that ends on a one-line preamble has stopped in
        # the middle of the format it was told to use. Observed verbatim: asked to analyse a file,
        # the model replied "Let's take a look at the file." and ended the turn -- the loop treated
        # that as the final answer, printed it, and handed the prompt back having done nothing.
        #
        # Nudging costs one extra generation; accepting it costs the whole request. Bounded so a
        # model that genuinely wants to answer briefly is not badgered, and skipped on the last step
        # where there would be no chance to act on a call anyway.
        truncated = AgentDemo.truncated_tool_call?(text)
        if calls.empty? && !last_step && empty_nudges < EMPTY_REPLY_NUDGE_LIMIT &&
           (truncated || AgentDemo.empty_reply?(visible))
          empty_nudges += 1
          if truncated
            STDERR.puts "  [agent] tool call was cut off mid-write; asking for it again".colorize(:yellow)
            @messages << Message.new("tool",
              "Your last reply began a <tool_call> but stopped before closing it, so nothing ran. " \
              "Send the SAME call again, complete, as the entire reply -- every <parameter> closed, " \
              "then </function>, then </tool_call>. Write nothing after the closing </tool_call>.")
          else
            STDERR.puts "  [agent] reply was empty; asking the model to act or answer".colorize(:yellow)
            @messages << Message.new("tool",
              "Your last reply was empty, so nothing ran and the user saw nothing. Either emit a " \
              "tool call as the entire reply, or give the complete answer.")
          end
          next
        end

        break if calls.empty?
        # On the final step the reply IS the answer. Running its tool calls would spend the
        # results and then break on the next iteration, throwing them away unanswered.
        break if last_step

        calls.each do |c|
          STDERR.puts "  #{"⚒ #{c.name}".colorize(:yellow)}(#{AgentDemo.format_args(c.args)})".colorize(:dark_gray)

          # Break identical repeated calls.
          #
          # Nothing else in the loop can. The repetition penalty runs with window 20 over the
          # CURRENT message's tokens, but a tool call is ~45 tokens and each retry is a separate
          # generate() call, so the penalty never sees that the same call was already made last
          # turn. Observed in practice: a model asked for a path that did not exist, then reissued
          # one malformed list_directory ten times in a row, each round costing a full prefill and
          # ~94 tokens of context, until the step budget ran out.
          sig = "#{c.name}(#{c.args.to_a.sort_by(&.[0]).map { |k, v| "#{k}=#{v}" }.join(",")})"
          repeats = @recent_calls.count(sig)
          if repeats >= REPEAT_CALL_LIMIT
            STDERR.puts "  [agent] same call #{repeats + 1}x; telling the model to change approach".colorize(:yellow)
            @recent_calls.clear
            @messages << Message.new("tool",
              "This exact call was already made #{repeats + 1} times and returned the same result " \
              "each time. Repeating it will not help. Change approach: check the argument against " \
              "the tool's description (list_directory takes a DIRECTORY path, not a shell command " \
              "-- use run_command for shell), try a different tool, or answer with what you have " \
              "and say what you could not determine.")
            next
          end
          @recent_calls << sig
          @recent_calls.shift if @recent_calls.size > RECENT_CALL_MEMORY

          tool = @tools.find { |t| t.name == c.name }
          result =
            if tool
              begin
                tool.call(c.args)
              rescue ex
                "Error running #{c.name}: #{ex.message}"
              end
            else
              "Error: unknown tool #{c.name}"
            end
          # Tool output is arbitrary bytes -- a compiler's or a binary's output is not always UTF-8 --
          # and every regex downstream (the tokenizer's pre-split, the result summary, the markdown
          # renderer) raises on invalid UTF-8 rather than skipping it. Observed: a run_command whose
          # output held a stray 0x80 byte killed the whole agent with "Regex match error". Scrubbed
          # once here, where untrusted text enters the conversation.
          result = result.scrub
          @messages << Message.new("tool", clamp_tool_result(result))
          STDERR.puts "    #{AgentDemo.summarize_result(result).colorize(:dark_gray)}"
        end
      end
    end
  end

  # ASCII splash banner.
  def self.print_banner(io : IO, subtitle : String)
    # ameba:disable Style/HeredocEscape
    art = <<-'ASCII'
       ____  _   _    _    ___            _
      / ___|| | | |  / \  |_ _|_ __   ___| |_
      \___ \| |_| | / _ \  | || '_ \ / _ \ __|
       ___) |  _  |/ ___ \ | || | | |  __/ |_
      |____/|_| |_/_/   \_\___|_| |_|\___|\__|
      ASCII
    io.puts
    io.puts art.colorize(:light_cyan).bold
    io.puts "  ▸ Agent ".colorize(:cyan).bold.to_s + "· #{subtitle}".colorize(:dark_gray).to_s
    io.puts
  end

  # System prompt describing the tools in the Qwen3-Coder XML format.
  def self.render_tools_block(tools : Array(Tool)) : String
    String.build do |s|
      s << "You are a helpful AI coding assistant that can interact with the user's"
      s << " computer to solve tasks.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
      tools.each do |t|
        s << "\n<function>\n<name>" << t.name << "</name>"
        s << "\n<description>" << t.description << "</description>\n<parameters>"
        t.params.each do |p|
          s << "\n<parameter>\n<name>" << p.name << "</name>\n<type>" << p.type
          s << "</type>\n<description>" << p.description << "</description>\n</parameter>"
        end
        s << "\n</parameters>\n</function>"
      end
      s << "\n</tools>"
      s << "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
      s << "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
      s << "</parameter>\n</function>\n</tool_call>\n\n"
      s << "<IMPORTANT>\n- Function calls MUST be wrapped in <tool_call></tool_call> with an inner"
      s << " <function=...></function> block.\n- Provide any reasoning BEFORE the call, never after.\n"
      s << "- If no function is needed, just answer normally.\n</IMPORTANT>\n\n"
      # Locating before reading is the difference between three targeted reads and paging a whole
      # file. Asked what to improve in a 1389-line file, the model read the head and then every
      # remaining range in sequence -- five calls, about 24K tokens -- when glob plus search would
      # have pointed at the handful of ranges that mattered. The bounded read stops any single call
      # from being huge; only knowing WHERE to look stops the total from being huge.
      s << "<SEARCH_FIRST>\nTo work on code you have not read, LOCATE before you read:\n"
      s << "1. glob to see which files exist (e.g. pattern='src/**/*.cr').\n"
      s << "2. search for the symbol, message or pattern you care about — it returns path:line, and\n"
      s << "   context=3 shows the surrounding lines.\n"
      s << "3. read_file with start_line/end_line around the lines search pointed at.\n"
      s << "Reading a whole large file, or paging one range after another until you have all of it,"
      s << " spends the context you need for the actual work. Search is cheap; a full read is not.\n"
      s << "</SEARCH_FIRST>"
    end
  end
end

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
model_dir = ARGV[0]?

# Resolve the model path: Ollama name ("qwen3.8:27b"), GGUF file, or directory.
gguf_mode = false
if model_dir && SHAInet::OllamaResolve.ollama_name?(model_dir)
  resolved = SHAInet::OllamaResolve.resolve(model_dir)
  if resolved
    STDERR.puts "Resolved Ollama '#{model_dir}' -> #{resolved}"
    model_dir = resolved
    gguf_mode = true
  else
    STDERR.puts "Ollama model '#{model_dir}' not found. Is it pulled?"
    exit 1
  end
elsif model_dir && File.file?(model_dir)
  gguf_mode = true
end

unless model_dir && (Dir.exists?(model_dir) || File.file?(model_dir.not_nil!))
  STDERR.puts "Usage: agent <model | gguf-file | ollama-name>"
  STDERR.puts "  agent ~/models/Qwen3.8-27B/.q4"
  STDERR.puts "  agent qwen3.8:27b"
  exit 1
end

# Colour when a human is watching, and only then. A tty is not sufficient on its own: NO_COLOR is the
# cross-tool convention for "I am a human on a tty and I still do not want escapes" (screen readers,
# logging a session to a file through `script`, a terminal with a palette that renders dark_gray
# unreadable), and TERM=dumb terminals do not interpret the sequences at all.
Colorize.enabled = STDERR.tty? && ENV["NO_COLOR"]?.nil? && ENV["TERM"]? != "dumb"
STDERR.sync = true # stream tokens as they arrive (no buffering)
AgentDemo.print_banner(STDERR, "#{File.basename(model_dir)} · local coding agent on Network#run")

STDERR.puts "Loading model from #{model_dir}...".colorize(:dark_gray)

# Decide the context BEFORE loading, and tell the loader about it.
#
# These were two independent defaults that happened to agree, and when they stopped agreeing the
# failure was an OOM mid-conversation rather than anything legible. The loader sizes its VRAM reserve
# from SHAINET_MAX_CONTEXT so the KV cache has somewhere to grow; the agent separately decided how
# many tokens it would allow. Loading first and reading the agent's context afterwards meant the
# reserve was always built for the LOADER's default no matter what the agent went on to permit -- so
# raising the agent's window silently produced a model with no room for it.
#
# A long default because this architecture is built for it (the GGUF declares 262144) and because a
# 16K window is the difference between an agent that can hold a session and one that compacts every
# few turns. The loader places what fits and leaves the rest on the host, so a smaller card degrades
# in speed rather than failing.
#
# 128K with the q4 KV cache: 4-bit rows with rotated keys take 144 bytes per head per position at
# head_dim 256 against fp8's 256, so 128K costs 2304 MB, about what fp8 took at 72K. Measured on the
# 27B on a 16 GB card: all 64 layers stay on the GPU, a 127.8K-token prefill completes with 402 MB
# free, and four notes hidden at 5-92% depth are all recalled; at 30K the next-token KL against an f16
# cache is 0.0018 (fp8: 0.0006). SHAINET_KV_DTYPE=fp8 with SHAINET_AGENT_CONTEXT=65536 is the previous
# setting.
agent_context = (ENV["SHAINET_AGENT_CONTEXT"]? || "131072").to_i
ENV["SHAINET_MAX_CONTEXT"] ||= agent_context.to_s
ENV["SHAINET_KV_DTYPE"] ||= "q4"
# A 30B takes about nine minutes here. Without a progress line that is indistinguishable
# from a hang, and the layer loop is where nearly all of it goes.
load_started = Time.monotonic
SHAInet::HFLoader.progress = ->(done : Int32, total : Int32) do
  elapsed = (Time.monotonic - load_started).total_seconds
  eta = done > 0 ? (elapsed / done) * (total - done) : 0.0
  STDERR.print "\r  layer #{done}/#{total} (#{done * 100 // total}%) " \
               "· #{elapsed.round.to_i}s elapsed · ~#{eta.round.to_i}s left".colorize(:dark_gray)
  STDERR.print "\n" if done == total
end
t0 = Time.monotonic
# Q4 + MoE offload are the only configuration a large MoE model actually runs in on a
# single consumer GPU: without them a 30B-A3B does not fit at all. So they are the
# DEFAULT here rather than opt-in. SHAINET_Q8=1 and SHAINET_MOE_OFFLOAD=0 opt out.
ENV["SHAINET_MOE_OFFLOAD"] = "1" unless ENV.has_key?("SHAINET_MOE_OFFLOAD")
quantize = SHAInet::CUDA.fully_available? && !ENV["SHAINET_FP32"]?
bits = ENV.fetch("SHAINET_Q8", "0") == "1" ? 8 : 4
offload = ENV.fetch("SHAINET_MOE_OFFLOAD", "0") == "1"
net = SHAInet::HFLoader.load(model_dir, quantize: quantize, bits: bits)
net.use_kv_cache = true
if gguf_mode
  STDERR.print "Extracting tokenizer from GGUF... ".colorize(:dark_gray)
  STDERR.flush
  tokenizer = SHAInet::GGUF.extract_tokenizer(model_dir)
  STDERR.puts "#{tokenizer.vocab.size} tokens".colorize(:dark_gray)
else
  tokenizer_path = ENV["SHAINET_TOKENIZER_PATH"]? || File.join(model_dir, "tokenizer.json")
  unless File.exists?(tokenizer_path)
    STDERR.puts "Error: tokenizer.json not found."
    exit 1
  end
  tokenizer = SHAInet::BPETokenizer.from_hf(tokenizer_path)
end
has_moe = net.hidden_layers.any? { |l| l.is_a?(SHAInet::LlamaBlock) && l.as(SHAInet::LlamaBlock).ffn.is_a?(SHAInet::MoEFF) }
mode = ENV["SHAINET_FP32"]? ? "fp32" : "Q#{bits}"
mode += " (MoE offload)" if offload && has_moe
STDERR.puts "Loaded in #{(Time.monotonic - t0).total_seconds.round(1)}s · #{mode} (vocab #{tokenizer.vocab.size})"

# Size the expert cache to leave headroom for the model + prefill activations.
# Only relevant for MoE models, where a fraction of experts are active per token and the rest
# are cached in VRAM for reuse. Dense models (Qwen3.5, Qwen3-0.6B) have no experts to cache,
# so reserving VRAM for this wastes 2+ GB that could serve KV context instead.
# Override with SHAINET_EXPERT_CACHE_MB (0 disables).
# Set before the load, so the loader's VRAM reserve and this window cannot disagree.
max_context = agent_context

has_moe = net.hidden_layers.any? { |l| l.is_a?(SHAInet::LlamaBlock) && l.as(SHAInet::LlamaBlock).ffn.is_a?(SHAInet::MoEFF) }

# The reserve has to cover what GROWS with context -- the KV cache above all -- not just a
# flat allowance. Measured on Qwen3-Coder-30B-A3B: with a fixed 6 GB reserve the cache took
# 7881 MB and a 16384-token prefill peaked at 15140 MB of 16376, leaving 1.2 GB. That is
# where a multi-turn agent runs out. Capping the cache instead put the same prefill at
# 11455 MB, and 24576 tokens fit in 11069 MB.
#
# 128 KiB per token is deliberately generous: fp16 KV on this 48-layer 30B is about 96
# KiB/token, and the slack covers the prefill workspaces. A smaller model over-reserves,
# which costs it cache it did not need anyway since it is not near the VRAM limit.
# SHAINET_EXPERT_CACHE_MB overrides the whole calculation (0 disables the cache).
if has_moe && offload && !ENV["SHAINET_EXPERT_CACHE_MB"]? && (info = SHAInet::CUDA.memory_info)
  kv_reserve = max_context.to_u64 * 128_u64 * 1024_u64
  reserve = 6_u64 * 1024 * 1024 * 1024 + kv_reserve
  free = info[:free]
  budget_mb = free > reserve ? ((free - reserve) // (1024_u64 * 1024_u64)) : 0_u64
  ENV["SHAINET_EXPERT_CACHE_MB"] = budget_mb.to_s
  STDERR.puts "  Expert cache budget: #{budget_mb} MB " \
              "(free #{free // (1024*1024)} MB − 6GB − #{kv_reserve // (1024*1024)} MB for #{max_context} tok context)".colorize(:dark_gray)
end

agent = AgentDemo::Agent.new(net, tokenizer, AgentDemo.build_tools, max_context)
STDERR.puts "Ready · tools: #{AgentDemo.build_tools.map(&.name).join(", ")} · max context #{max_context} tok".colorize(:green)
STDERR.puts "Commands: /context   /compact   /clear   /ask   /workspace   /help    (Ctrl-C interrupts a turn, Ctrl-D exits)".colorize(:dark_gray)

# Ctrl-C interrupts the TURN; twice in a row at an idle prompt exits.
#
# The default handler kills the process, which here throws away 9 GB of loaded weights, the KV cache
# and the conversation -- a brutal price for "not that file". Trapping it means a wrong turn costs the
# turn. A second press while already idle is an explicit request to leave, so honour it rather than
# leaving the user hunting for the exit.
idle_interrupt = false
Process.on_terminate do |reason|
  if reason.interrupted?
    if AgentDemo.generating?
      AgentDemo.cancel!
    elsif idle_interrupt
      STDERR.puts "\nbye 👋".colorize(:cyan)
      AgentDemo.disable_bracketed_paste
      exit 0
    else
      idle_interrupt = true
      STDERR.print "\n  #{"press Ctrl-C again to exit, or Ctrl-D".colorize(:dark_gray)}\n"
    end
  else
    # A real termination request (TERM, or the terminal going away) is not a change of mind about one
    # turn -- do not swallow it into a cancel the user is not there to see.
    AgentDemo.disable_bracketed_paste
    exit 0
  end
end

AgentDemo.enable_bracketed_paste
BAR = AgentDemo::StatusBar.new(STDERR)
BAR.install
# Registered before the paste reset so the terminal is restored even on an abrupt exit: a process that
# dies with a scroll region still set leaves the user's shell scrolling inside a box.
at_exit { BAR.uninstall }
# The trap runs in a fiber, so it fires when the scheduler next gets control: every generated token
# yields, and readline (which handles the resize for its own line) returns before the next prompt,
# where the loop calls refresh as well.
Signal::WINCH.trap { BAR.refresh }
# Text typed during a turn but never sent comes back pre-filled at the next prompt.
LibRLHook.rl_startup_hook = -> { AgentDemo.readline_prefill }
at_exit { AgentDemo.disable_bracketed_paste }
at_exit { AgentDemo.save_history }
agent.status_bar = BAR

loop do
  # No manual "You ❯" print here: readline draws PROMPT itself so its line editing works within it.
  # On a pipe (no tty) read_submission falls back to gets, and a leading newline keeps turns spaced.
  BAR.refresh
  STDERR.print "\n" if STDIN.tty?
  # A line sent with Enter after the turn's last generation had already finished: nothing consumed it
  # as steering, so it is the next message.
  if late = AgentDemo.take_steer
    STDERR.puts "#{AgentDemo::PROMPT}#{late}"
    input = late
  else
    input = AgentDemo.read_submission
  end
  break if input.nil?
  input = input.scrub
  idle_interrupt = false
  AgentDemo.clear_cancel!
  input = input.strip
  next if input.empty?

  case input
  when "/help"
    STDERR.puts "  /context  show context size, VRAM and cache usage".colorize(:dark_gray)
    STDERR.puts "  /compact  summarize + trim the conversation history now".colorize(:dark_gray)
    STDERR.puts "  /clear    reset the conversation".colorize(:dark_gray)
    STDERR.puts "  /ask      revoke 'allow all' and confirm each tool call again".colorize(:dark_gray)
    STDERR.puts "  /workspace  show the workspace root, or /workspace <dir> to change it".colorize(:dark_gray)
    STDERR.puts "  /help     this message".colorize(:dark_gray)
    next
  when "/ask"
    # The way back from answering "a". Without this, one keystroke silently approves every
    # write and shell command for the rest of the session with no way to reconsider. It also
    # forgets approved out-of-workspace locations, since those were granted by the same
    # keystroke and leaving them behind would make the revocation only partly true.
    outside = AgentDemo::WorkspacePath.approved_outside_count
    if AgentDemo.allow_all? || outside > 0
      AgentDemo.allow_all = false
      AgentDemo::WorkspacePath.reset_outside_approvals!
      note = outside > 0 ? " (also forgot #{outside} approved path(s) outside the workspace)" : ""
      STDERR.puts "  will confirm each tool call again#{note}".colorize(:dark_gray)
    else
      STDERR.puts "  already confirming each tool call".colorize(:dark_gray)
    end
    next
  when "/context"
    STDERR.puts "  #{agent.status}".colorize(:dark_gray)
    next
  when "/compact"
    # Forced: summarize everything but the most recent quarter of the transcript, whatever its size.
    overhead = agent.overhead_tokens
    target = Math.min(overhead + (agent.context_tokens - overhead) // 2, (agent.max_context * 0.75).to_i)
    removed = agent.compact!(target, force: true)
    STDERR.puts "  compacted −#{removed} tokens · #{agent.status}".colorize(:dark_gray)
    next
  when "/clear"
    agent.reset
    STDERR.puts "  conversation cleared".colorize(:dark_gray)
    next
  when "/workspace"
    # No argument: show where the file tools are confined.
    STDERR.puts "  workspace: #{AgentDemo.workspace}"
    next
  when /^\/workspace\s+(.+)$/
    # Switch the sandbox root. The old root's outside approvals are forgotten with it: they
    # were granted against a different sandbox, and keeping them would let the new workspace
    # inherit permissions the user never asked for here.
    arg = input[/^\/workspace\s+(.+)$/, 1].strip
    begin
      AgentDemo.switch_workspace!(arg)
      outside = AgentDemo::WorkspacePath.approved_outside_count
      AgentDemo::WorkspacePath.reset_outside_approvals!
      note = outside > 0 ? " (forgot #{outside} approved path(s) outside the old workspace)" : ""
      STDERR.puts "  workspace → #{AgentDemo.workspace}#{note}"
    rescue ex : ArgumentError
      STDERR.puts "  #{ex.message}"
    end
    next
  else
    # A mistyped command used to go to the MODEL as an ordinary prompt: "/cotext" spent a whole
    # generation explaining that it does not know what /cotext means. Anything starting with a slash
    # was meant for the program, so say it is unknown rather than charging a turn for the typo.
    if input.starts_with?('/')
      STDERR.puts "  unknown command #{input.split(' ').first} · try /help".colorize(:dark_gray)
      next
    end
  end

  AgentDemo.generating = true
  AgentDemo.steer_mode_on
  begin
    agent.chat(input)
  ensure
    AgentDemo.steer_mode_off
    AgentDemo.generating = false
  end
  STDERR.puts "  #{agent.status}".colorize(:dark_gray)
end
STDERR.puts "\nbye 👋".colorize(:cyan)
SHAInet::Profile.report(STDERR)
