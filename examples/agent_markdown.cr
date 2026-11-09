require "colorize"

module AgentDemo
  # The longest prefix of a streamed decode that ends on a whole UTF-8 character.
  #
  # One emoji is four bytes and the tokenizer may emit them across two tokens. Decoding after the
  # first and scrubbing turned the half character into U+FFFD, which went to the screen at once, and
  # the stream's character offsets were off from then on: "🔴 High" arrived as "��� High". Holding
  # back an incomplete tail (at most 3 bytes) lets the next token finish it. Bytes that are invalid
  # for good, not merely incomplete, are still scrubbed, so the result is always valid.
  def self.complete_utf8(s : String) : String
    return s if s.valid_encoding?
    bytes = s.to_slice
    (1..3).each do |k|
      break if k > bytes.size
      t = String.new(bytes[0, bytes.size - k])
      return t if t.valid_encoding?
    end
    s.scrub
  end

  # Terminal columns a string occupies. Wide East Asian text and emoji take two, combining marks,
  # variation selectors and zero-width joiners none -- which is what makes a table holding "🔴 High"
  # line up. An approximation of wcwidth: terminals disagree at the edges anyway.
  module TermWidth
    def self.of(s : String) : Int32
      w = 0
      s.each_char { |c| w += char(c) }
      w
    end

    def self.char(c : Char) : Int32
      cp = c.ord
      return 0 if cp == 0x200D || (0xFE00..0xFE0F).includes?(cp) || (0x200B..0x200F).includes?(cp)
      return 0 if cp < 32 || (0x7F...0xA0).includes?(cp)
      return 0 if c.mark?
      return 2 if wide?(cp)
      1
    end

    EMOJI_BMP = Set{0x231A, 0x231B, 0x23E9, 0x23EA, 0x23EB, 0x23EC, 0x23F0, 0x23F3, 0x25FD, 0x25FE,
                    0x2614, 0x2615, 0x267F, 0x2693, 0x26A1, 0x26AA, 0x26AB, 0x26BD, 0x26BE, 0x26C4,
                    0x26C5, 0x26CE, 0x26D4, 0x26EA, 0x26F2, 0x26F3, 0x26F5, 0x26FA, 0x26FD, 0x2705,
                    0x270A, 0x270B, 0x2728, 0x274C, 0x274E, 0x2753, 0x2754, 0x2755, 0x2757, 0x2795,
                    0x2796, 0x2797, 0x27B0, 0x27BF, 0x2B1B, 0x2B1C, 0x2B50, 0x2B55}

    private def self.wide?(cp : Int32) : Bool
      (0x1100..0x115F).includes?(cp) || (0x2E80..0x303E).includes?(cp) || (0x3041..0x33FF).includes?(cp) ||
        (0x3400..0x4DBF).includes?(cp) || (0x4E00..0x9FFF).includes?(cp) || (0xA000..0xA4CF).includes?(cp) ||
        (0xAC00..0xD7A3).includes?(cp) || (0xF900..0xFAFF).includes?(cp) || (0xFE30..0xFE4F).includes?(cp) ||
        (0xFF00..0xFF60).includes?(cp) || (0xFFE0..0xFFE6).includes?(cp) || (0x2648..0x2653).includes?(cp) ||
        (0x1F300..0x1F64F).includes?(cp) || (0x1F680..0x1F6FF).includes?(cp) || (0x1F7E0..0x1F7FF).includes?(cp) ||
        (0x1F900..0x1F9FF).includes?(cp) || (0x1FA70..0x1FAFF).includes?(cp) || (0x20000..0x3FFFD).includes?(cp) ||
        EMOJI_BMP.includes?(cp)
    end
  end

  # Inline markdown -> styled runs: **bold**, *italic*, `code`, ~~strike~~ and [text](url).
  #
  # Streaming: feed may stop anywhere, so a marker that could still grow ("*" before a possible
  # second "*", an open "[" before its "](url)") is held until the next feed decides it. `final`
  # (end of line) resolves everything held as literal text. Underscores are deliberately not markup:
  # they live inside identifiers (snake_case), and mangling those in an answer about code is worse
  # than missing the odd _italic_.
  class MdInline
    BOLD   =   1
    ITALIC =   2
    CODE   =   4
    STRIKE =   8
    LINK   =  16
    URL    =  32
    H1     =  64
    HEAD   = 128
    QUOTE  = 256
    TH     = 512

    alias Run = Tuple(String, Int32)

    getter flags = 0
    @hold = ""
    @prev = ' '

    def reset
      @flags = 0
      @hold = ""
      @prev = ' '
    end

    def feed(s : String, final : Bool = false) : Array(Run)
      res = [] of Run
      buf = String::Builder.new
      text = @hold + s
      @hold = ""
      chars = text.chars
      n = chars.size
      flush = ->(f : Int32) do
        str = buf.to_s
        res << {str, f} unless str.empty?
        buf = String::Builder.new
        nil
      end
      i = 0
      while i < n
        c = chars[i]
        if @flags & CODE != 0
          if c == '`'
            flush.call(@flags)
            @flags &= ~CODE
          else
            buf << c
          end
          @prev = c
          i += 1
          next
        end
        case c
        when '`'
          flush.call(@flags)
          @flags |= CODE
          i += 1
        when '*', '~'
          if i + 1 >= n && !final
            @hold = chars[i..].join
            break
          end
          double = i + 1 < n && chars[i + 1] == c
          if double
            flush.call(@flags)
            @flags ^= (c == '*' ? BOLD : STRIKE)
            i += 2
          elsif c == '*' && @flags & ITALIC != 0 && !@prev.whitespace?
            flush.call(@flags)
            @flags &= ~ITALIC
            i += 1
          elsif c == '*' && i + 1 < n && !chars[i + 1].whitespace? && (@prev.whitespace? || "([{\"'".includes?(@prev))
            flush.call(@flags)
            @flags |= ITALIC
            i += 1
          else
            buf << c
            @prev = c
            i += 1
          end
        when '['
          rest = chars[i..].join
          if m = rest.match(/\A\[([^\]\n]*)\]\(([^)\s]*)\)/)
            flush.call(@flags)
            res << {m[1], @flags | LINK}
            res << {" (#{m[2]})", URL} unless m[2].empty? || m[2] == m[1]
            i += m[0].size
            @prev = ')'
          elsif !final && rest.size < 400 && rest.matches?(/\A\[[^\]\n]*(\]\(?[^)\s]*)?\z/)
            @hold = rest
            break
          else
            buf << c
            @prev = c
            i += 1
          end
        else
          buf << c
          @prev = c
          i += 1
        end
      end
      flush.call(@flags)
      res
    end
  end

  # Streaming markdown renderer for the agent's answers.
  #
  # Prose streams as it arrives with inline styles applied on the fly; block syntax is decided from
  # the first few characters of each line (headings, lists, quotes, rules), so nothing waits for the
  # whole answer. Two blocks are line-buffered: fenced code (each line is highlighted whole) and
  # tables, which are held until their last row and then drawn with box characters, column widths
  # measured in terminal cells and long cells word-wrapped to fit the terminal.
  #
  # With color off (output not a terminal) the text passes through untouched, so piped output is
  # still the model's markdown.
  class Markdown
    property? col0 : Bool

    def initialize(@io : IO, @width : Proc(Int32) = -> { 100 }, @color : Bool = Colorize.enabled?,
                   @col0 : Bool = true)
      @state = :start
      @lbuf = ""
      @base = 0
      @inline = MdInline.new
      @cur = -1
      @in_code = false
      @fence = ""
      @lang = ""
      @table = [] of String
      @table_status = false
    end

    def feed(text : String)
      unless @color
        @io << text
        @col0 = text.ends_with?('\n') unless text.empty?
        return
      end
      text.each_char do |c|
        if c == '\n'
          end_line
          next
        end
        case @state
        when :start
          @lbuf += c
          classify(false)
        when :inline
          write_runs(@inline.feed(c.to_s))
        else
          @lbuf += c
        end
      end
      @io.flush
    end

    # End of the answer (or a switch to thinking / a tool call): settle whatever is held.
    def finish
      return unless @color
      case @state
      when :start
        classify(true) unless @lbuf.empty?
        hold_line(false) if @state == :hold
        finish_inline if @state == :inline
      when :hold
        hold_line(false)
      when :inline
        finish_inline
      end
      flush_table
      if @in_code
        newline unless @col0
        @io << dim("╰─") << "\n"
        @in_code = false
        @col0 = true
      end
      @state = :start
      @lbuf = ""
      @io.flush
    end

    private def finish_inline
      write_runs(@inline.feed("", final: true))
      @io << "\e[0m"
      @cur = -1
      @inline.reset
      @base = 0
    end

    private def end_line
      case @state
      when :start
        classify(true)
        if @state == :hold
          hold_line(true)
        elsif @state == :inline
          finish_inline
          newline
        else
          newline
        end
      when :hold
        hold_line(true)
      when :inline
        finish_inline
        newline
      end
      @state = :start
      @lbuf = ""
    end

    private def newline
      @io << "\n"
      @col0 = true
    end

    # Decide what the current line is once enough of it is known.
    private def classify(final : Bool)
      l = @lbuf
      if @in_code
        @state = :hold
        return
      end
      s = l.lstrip
      if s.empty?
        if final
          flush_table
          @state = :done
        end
        return
      end
      if s.starts_with?("```") || s.starts_with?("~~~") || s.starts_with?('|')
        @state = :hold
        return
      end
      decided = final || s.includes?(' ') || s.size >= 10 || !"#-*+>0123456789`~_".includes?(s[0])
      return unless decided
      flush_table
      if final && l.matches?(/\A\s*([-*_])(\s*\1){2,}\s*\z/)
        block_start
        @io << dim("─" * Math.min(@width.call - 1, 60))
        @col0 = false
        @state = :done
        return
      end
      @state = :inline
      @inline.reset
      if m = l.match(/\A(\s*)(\#{1,6})\s+(.*)\z/m)
        block_start
        @base = m[2].size == 1 ? (MdInline::HEAD | MdInline::H1) : MdInline::HEAD
        write_runs(@inline.feed(m[3]))
      elsif m = l.match(/\A(\s*)([-*+])\s+(.*)\z/m)
        lvl = m[1].size // 2
        @io << m[1] << colored(["•", "◦", "▪"][lvl % 3], "36") << " "
        @col0 = false
        write_runs(@inline.feed(m[3]))
      elsif m = l.match(/\A(\s*)(\d{1,3})([.)])\s+(.*)\z/m)
        @io << m[1] << colored("#{m[2]}#{m[3]}", "36") << " "
        @col0 = false
        write_runs(@inline.feed(m[4]))
      elsif m = l.match(/\A(\s*)>\s?(.*)\z/m)
        @io << m[1] << dim("▎ ")
        @col0 = false
        @base = MdInline::QUOTE
        write_runs(@inline.feed(m[2]))
      else
        write_runs(@inline.feed(l))
      end
    end

    # A whole buffered line: code-block body, a fence, or a table row.
    private def hold_line(had_newline : Bool)
      l = @lbuf
      s = l.strip
      if @in_code
        if s.starts_with?(@fence) && s.delete(@fence[0]).empty?
          @io << dim("╰─")
          @in_code = false
          @col0 = false
          newline if had_newline
        else
          @io << dim("│ ") << Highlight.line(l.gsub('\t', "    "), @lang)
          @col0 = false
          newline if had_newline
        end
      elsif s.starts_with?("```") || s.starts_with?("~~~")
        flush_table
        block_start
        @fence = s[0, 3]
        @lang = s[3..].strip.downcase
        @in_code = true
        @io << dim("╭─ #{@lang.empty? ? "code" : @lang}")
        @col0 = false
        newline if had_newline
      else
        @table << l
        show_table_status
        flush_table unless had_newline
      end
    end

    private def show_table_status
      @io << "\r\e[2K" << dim("  ⋯ table (#{@table.size} rows)")
      @table_status = true
    end

    private def flush_table
      return if @table.empty?
      if @table_status
        @io << "\r\e[2K"
        @table_status = false
        @col0 = true
      end
      rows = @table.map { |r| split_cells(r) }
      @table.clear
      if rows.size >= 2 && separator?(rows[1])
        block_start
        draw_table(rows[0], rows[1], rows[2..])
      else
        rows.each do |cells|
          write_runs(MdInline.new.feed("| " + cells.join(" | ") + " |", final: true))
          @io << "\e[0m"
          @cur = -1
          newline
        end
      end
    end

    private def split_cells(row : String) : Array(String)
      s = row.strip
      s = s[1..] if s.starts_with?('|')
      s = s[0...-1] if s.ends_with?('|') && !s.ends_with?("\\|")
      cells = [] of String
      cur = String::Builder.new
      in_code = false
      prev = ' '
      s.each_char do |c|
        if c == '`'
          in_code = !in_code
        end
        if c == '|' && !in_code && prev != '\\'
          cells << cur.to_s.strip
          cur = String::Builder.new
        else
          cur << c
        end
        prev = c
      end
      cells << cur.to_s.strip
      cells.map(&.gsub("\\|", "|"))
    end

    private def separator?(cells : Array(String)) : Bool
      cells.all?(&.matches?(/\A:?-{1,}:?\z/))
    end

    private def draw_table(head : Array(String), sep : Array(String), body : Array(Array(String)))
      ncols = ([head.size] + body.map(&.size)).max
      aligns = Array.new(ncols) do |i|
        a = sep[i]? || "-"
        a.starts_with?(':') && a.ends_with?(':') ? :center : (a.ends_with?(':') ? :right : :left)
      end
      parse = ->(cells : Array(String), base : Int32) do
        Array.new(ncols) do |i|
          MdInline.new.feed(cells[i]? || "", final: true).map { |(t, f)| {t, f | base} }
        end
      end
      hrows = parse.call(head, MdInline::TH)
      brows = body.map { |r| parse.call(r, 0) }
      all = [hrows] + brows
      natural = Array.new(ncols) { |i| all.max_of { |r| run_width(r[i]) }.clamp(1, Int32::MAX) }
      words = Array.new(ncols) do |i|
        all.max_of { |r| r[i].max_of? { |(t, _)| t.split(/\s+/).max_of? { |wd| TermWidth.of(wd) } || 0 } || 0 }
      end
      avail = @width.call - (3 * ncols + 1)
      widths = fit(natural, words, avail)

      @io << dim("┌" + widths.map { |w| "─" * (w + 2) }.join("┬") + "┐") << "\n"
      draw_row(hrows, widths, aligns)
      @io << dim("├" + widths.map { |w| "─" * (w + 2) }.join("┼") + "┤") << "\n"
      tall = brows.any? { |r| r.each_with_index.any? { |cell, i| run_width(cell) > widths[i] } }
      brows.each_with_index do |r, ri|
        if tall && ri > 0
          @io << dim("├" + widths.map { |w| "─" * (w + 2) }.join("┼") + "┤") << "\n"
        end
        draw_row(r, widths, aligns)
      end
      @io << dim("└" + widths.map { |w| "─" * (w + 2) }.join("┴") + "┘") << "\n"
      @col0 = true
    end

    # Fit the columns into `avail` cells. Columns give up their slack first -- width beyond their
    # longest word (capped at 32) -- most slack first, so prose wraps before identifiers are cut.
    # Only if that is not enough do the widest columns shrink below their longest word, never under
    # 4 cells (a terminal narrower than that just wraps the table, which is still readable).
    private def fit(natural : Array(Int32), words : Array(Int32), avail : Int32) : Array(Int32)
      w = natural.dup
      floor = words.map_with_index { |m, i| Math.min(m.clamp(4, 32), natural[i]) }
      while w.sum > avail
        i = (0...w.size).max_by { |j| w[j] - floor[j] }
        break if w[i] <= floor[i]
        w[i] -= 1
      end
      while w.sum > avail
        i = w.index!(w.max)
        break if w[i] <= 4
        w[i] -= 1
      end
      w
    end

    private def draw_row(cells : Array(Array(MdInline::Run)), widths : Array(Int32), aligns : Array(Symbol))
      wrapped = cells.map_with_index { |c, i| wrap(c, widths[i]) }
      height = wrapped.max_of(&.size).clamp(1, Int32::MAX)
      height.times do |li|
        @io << dim("│")
        wrapped.each_with_index do |lines, ci|
          line = lines[li]? || [] of MdInline::Run
          pad = widths[ci] - run_width(line)
          left, right = case aligns[ci]
                        when :right  then {pad, 0}
                        when :center then {pad // 2, pad - pad // 2}
                        else              {0, pad}
                        end
          @io << " " << " " * left
          @cur = -1
          line.each { |(t, f)| @io << sgr(f) << t }
          @io << "\e[0m" << " " * right << " " << dim("│")
        end
        @io << "\n"
      end
      @cur = -1
    end

    # Word-wrap styled runs to `w` cells; words longer than a line are split.
    private def wrap(runs : Array(MdInline::Run), w : Int32) : Array(Array(MdInline::Run))
      lines = [[] of MdInline::Run]
      used = 0
      runs.each do |(text, f)|
        text.split(/(\s+)/).each do |tok|
          next if tok.empty?
          tw = TermWidth.of(tok)
          if tok.blank?
            next if used == 0
            if used + tw <= w
              lines.last << {" ", f}
              used += 1
            end
            next
          end
          if used + tw > w && used > 0
            # Drop the trailing space the previous word left.
            if (last = lines.last.last?) && last[0] == " "
              lines.last.pop
            end
            lines << [] of MdInline::Run
            used = 0
          end
          if tw <= w
            lines.last << {tok, f}
            used += tw
          else
            # Too long for any line: split, preferring to break after :: / - _ . so identifiers and
            # paths come apart at their joints rather than mid-name.
            rest = tok
            until rest.empty?
              room = w - used
              if room <= 0
                lines << [] of MdInline::Run
                used = 0
                room = w
              end
              if TermWidth.of(rest) <= room
                lines.last << {rest, f}
                used += TermWidth.of(rest)
                break
              end
              cut = 0
              cw = 0
              rest.each_char_with_index do |c, ci|
                cw += TermWidth.char(c)
                break if cw > room
                cut = ci + 1
              end
              cut = 1 if cut == 0 && used == 0
              if cut == 0
                lines << [] of MdInline::Run
                used = 0
                next
              end
              joint = rest[0, cut].rindex(/[:\/\-_.](?=[^:\/\-_.])/)
              cut = joint + 1 if joint && joint + 1 >= cut // 2
              lines.last << {rest[0, cut], f}
              used += TermWidth.of(rest[0, cut])
              rest = rest[cut..]
              unless rest.empty?
                lines << [] of MdInline::Run
                used = 0
              end
            end
          end
        end
      end
      lines
    end

    private def run_width(runs : Array(MdInline::Run)) : Int32
      runs.sum { |(t, _)| TermWidth.of(t) }
    end

    # Blocks (headings, rules, tables, code) start on their own line.
    private def block_start
      newline unless @col0
    end

    private def write_runs(runs : Array(MdInline::Run))
      runs.each do |(t, f)|
        next if t.empty?
        f |= @base
        if f != @cur
          @io << sgr(f)
          @cur = f
        end
        @io << t
        @col0 = false
      end
    end

    private def sgr(f : Int32) : String
      codes = ["0"]
      codes << "1" if f & (MdInline::BOLD | MdInline::HEAD | MdInline::TH) != 0
      codes << "3" if f & (MdInline::ITALIC | MdInline::QUOTE) != 0
      codes << "9" if f & MdInline::STRIKE != 0
      codes << "4" if f & (MdInline::LINK | MdInline::H1) != 0
      codes << "2" if f & (MdInline::URL | MdInline::QUOTE) != 0
      if f & MdInline::CODE != 0
        codes << "38;5;179"
      elsif f & MdInline::LINK != 0
        codes << "34"
      elsif f & MdInline::HEAD != 0
        codes << "36"
      end
      "\e[#{codes.join(';')}m"
    end

    private def dim(s : String) : String
      "\e[2m#{s}\e[0m"
    end

    private def colored(s : String, code : String) : String
      "\e[#{code}m#{s}\e[0m"
    end
  end

  # One-line syntax highlighting for fenced code: comments, strings, numbers, keywords and
  # Capitalized names. Per line and stateless (a multi-line string or block comment is only colored
  # on its first line) -- enough to make code readable at a glance, which is the job here.
  module Highlight
    HASH_COMMENT = Set{"crystal", "cr", "ruby", "rb", "python", "py", "sh", "bash", "shell", "zsh", "console",
                       "yaml", "yml", "toml", "perl", "r", "make", "makefile", "dockerfile", "nim", "ini", "conf"}
    DASH_COMMENT = Set{"sql", "lua", "haskell", "hs"}
    KEYWORDS     = Set.new(%w[def end if else elsif elif unless while until for in do return class module struct
      enum lib fun macro require include extend private protected public self super nil true false and or not
      then case when begin rescue ensure raise yield lambda proc import from as with try except finally pass
      break continue let const var function new delete typeof instanceof async await export default interface
      type implements package static void int char float double long short unsigned signed bool auto namespace
      using template typename sizeof switch fn mut impl trait pub use match loop where mod echo fi done esac
      local null undefined this None True False abstract alias annotation getter setter property record])
    SQL_KEYWORDS = Set.new(%w[select from where insert into update delete create table join left right inner on
      group by order having limit values set and or not null as distinct count])
    TOKEN = /(?<str>"(?:\\.|[^"\\])*"?|'(?:\\.|[^'\\])*'?|`[^`]*`?)|(?<num>\b\d[\w.]*)|(?<sym>(?<![:\w]):[a-z_]\w*[?!]?)|(?<word>[A-Za-z_]\w*[?!]?)/

    def self.line(text : String, lang : String) : String
      case lang
      when "diff", "patch"
        return color(text, "32") if text.starts_with?('+')
        return color(text, "31") if text.starts_with?('-')
        return color(text, "36") if text.starts_with?("@@")
        return text
      when "json", "jsonc"
        return text.gsub(/("(?:\\.|[^"\\])*")(\s*:)?|\b(-?\d[\d.eE+-]*|true|false|null)\b/) do |s, m|
          if m[1]?
            m[2]? ? color(m[1], "36") + m[2] : color(m[1], "32")
          else
            color(s, "33")
          end
        end
      end
      code, comment = split_comment(text, lang)
      sql = lang == "sql"
      res = code.gsub(TOKEN) do |s, m|
        if m["str"]?
          color(s, "32")
        elsif m["num"]?
          color(s, "33")
        elsif m["sym"]?
          color(s, "33")
        elsif sql ? SQL_KEYWORDS.includes?(s.downcase) : KEYWORDS.includes?(s)
          color(s, "35")
        elsif s[0].uppercase?
          color(s, "36")
        else
          s
        end
      end
      comment.empty? ? res : res + "\e[2;3m#{comment}\e[0m"
    end

    # Split off a trailing comment, ignoring comment markers inside string literals.
    private def self.split_comment(text : String, lang : String) : Tuple(String, String)
      markers = if HASH_COMMENT.includes?(lang)
                  ["#"]
                elsif DASH_COMMENT.includes?(lang)
                  ["--"]
                elsif lang.empty?
                  ["//", "#"]
                else
                  ["//"]
                end
      quote = nil
      prev = ' '
      i = 0
      while i < text.size
        c = text[i]
        if q = quote
          quote = nil if c == q && prev != '\\'
        elsif c == '"' || c == '\''
          quote = c
        else
          markers.each do |mk|
            next unless text[i..].starts_with?(mk)
            # "#{" is interpolation, not a comment, in Crystal and Ruby.
            next if mk == "#" && text[i + 1]? == '{'
            # In an unknown language only treat '#' as a comment at the start of a word.
            next if mk == "#" && lang.empty? && !prev.whitespace? && i > 0
            return {text[0, i], text[i..]}
          end
        end
        prev = c
        i += 1
      end
      {text, ""}
    end

    private def self.color(s : String, code : String) : String
      "\e[#{code}m#{s}\e[0m"
    end
  end
end
