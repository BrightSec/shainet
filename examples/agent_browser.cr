require "json"
require "uri"
require "base64"
require "http/client"
require "http/web_socket"
require "file_utils"

# Browser access for the agent, over the Chrome DevTools Protocol.
#
# No shard: a headless Chrome is launched as a child process and driven through its DevTools
# WebSocket. The agent gets three tools -- `browse` (open a URL, read the rendered text), `browser_eval`
# (run a JS expression against the loaded page, which covers clicking, DOM reads and extraction) and
# `browser_screenshot`. Chrome is launched lazily on the first call and killed at exit, with a throwaway
# profile, so nothing touches the user's own Chrome or its logins.
module AgentDemo
  class BrowserError < Exception
  end

  # Full path to a Chrome/Chromium binary, or nil when none is installed. SHAINET_CHROME overrides.
  # Resolved once (the result, including "not found", is cached).
  @@chrome_path : String? = nil
  @@chrome_checked = false

  def self.chrome_path : String?
    return @@chrome_path if @@chrome_checked
    @@chrome_checked = true
    names = [ENV["SHAINET_CHROME"]?, "google-chrome-stable", "google-chrome",
             "chromium", "chromium-browser", "chrome"].compact
    names.each do |n|
      if p = Process.find_executable(n)
        return @@chrome_path = p
      end
    end
    @@chrome_path = nil
  end

  # One headless Chrome, one page, driven over CDP. Commands are request/response correlated by id;
  # events are ignored (navigation waits on document.readyState instead, which needs no event plumbing).
  class CDP
    @next_id = 0
    @pending = {} of Int32 => Channel(JSON::Any)
    @session_id : String = ""
    @closed = false

    def initialize(@proc : Process, @profile_dir : String, ws_url : String)
      @ws = HTTP::WebSocket.new(URI.parse(ws_url))
      @ws.on_message { |raw| dispatch(raw) }
      @ws.on_close { |_| fail_pending("the browser connection closed") }
      spawn do
        @ws.run
      rescue
      ensure
        fail_pending("the browser connection ended")
      end
    end

    # Launch Chrome with a throwaway profile, find its DevTools port, and attach a page.
    #
    # Chrome's SUID sandbox is misconfigured on some hosts (chrome-sandbox not owned root/4755), and
    # there Chrome aborts at startup rather than run unsandboxed. We keep the sandbox where it works
    # and retry once with --no-sandbox only after such an early abort. SHAINET_CHROME_NO_SANDBOX=1
    # goes straight to the unsandboxed launch.
    def self.launch : CDP
      path = AgentDemo.chrome_path || raise BrowserError.new(
        "no Chrome or Chromium found; install one or set SHAINET_CHROME to its path")
      return launch_with(path, ["--no-sandbox"]) if ENV["SHAINET_CHROME_NO_SANDBOX"]? == "1"
      begin
        launch_with(path, [] of String)
      rescue BrowserError
        launch_with(path, ["--no-sandbox"])
      end
    end

    private def self.launch_with(path : String, extra : Array(String)) : CDP
      dir = File.join(Dir.tempdir, "shainet-agent-chrome-#{Process.pid}-#{Random.rand(100000)}")
      Dir.mkdir_p(dir)
      args = ["--headless=new", "--remote-debugging-port=0", "--user-data-dir=#{dir}",
              "--no-first-run", "--no-default-browser-check", "--disable-gpu",
              "--disable-dev-shm-usage", "--window-size=1280,900"] + extra + ["about:blank"]
      proc = Process.new(path, args, output: Process::Redirect::Close, error: Process::Redirect::Close)
      begin
        port = wait_for_port(dir, proc)
        body = HTTP::Client.get("http://127.0.0.1:#{port}/json/version").body
        ws_url = JSON.parse(body)["webSocketDebuggerUrl"].as_s
        cdp = new(proc, dir, ws_url)
        cdp.open_page
        cdp
      rescue ex
        proc.terminate(graceful: false) rescue nil
        FileUtils.rm_rf(dir) rescue nil
        raise ex.is_a?(BrowserError) ? ex : BrowserError.new("could not start Chrome: #{ex.message}")
      end
    end

    # Chrome writes the chosen port to DevToolsActivePort (first line) once it is listening.
    private def self.wait_for_port(dir : String, proc : Process) : Int32
      file = File.join(dir, "DevToolsActivePort")
      deadline = Time.monotonic + 20.seconds
      loop do
        raise BrowserError.new("Chrome exited before it was ready") if proc.terminated?
        if File.exists?(file) && (line = File.read_lines(file).first?) && (n = line.to_i?)
          return n
        end
        raise BrowserError.new("Chrome did not open a DevTools port in time") if Time.monotonic > deadline
        sleep 0.05.seconds
      end
    end

    # Open a page target and attach to it with a flat session, so every later command names it.
    def open_page
      target = call("Target.createTarget", %({"url":"about:blank"}))
      target_id = target["targetId"].as_s
      attached = call("Target.attachToTarget", %({"targetId":#{target_id.to_json},"flatten":true}))
      @session_id = attached["sessionId"].as_s
      page_call("Page.enable")
    end

    private def dispatch(raw : String)
      msg = JSON.parse(raw)
      if (id = msg["id"]?.try(&.as_i?)) && (ch = @pending.delete(id))
        ch.send(msg)
      end
    rescue
      # A frame we cannot parse is not worth tearing the connection down for.
    end

    private def fail_pending(reason : String)
      return if @pending.empty?
      err = JSON.parse(%({"error":{"message":#{reason.to_json}}}))
      @pending.each_value { |ch| ch.send(err) rescue nil }
      @pending.clear
    end

    # Send a CDP command and wait for its reply. `params` is a JSON object string; `session_id` scopes
    # it to a target (the page, by default). Returns the `result` object, or raises on a CDP error.
    def call(method : String, params : String = "{}", session_id : String? = nil,
             timeout : Time::Span = 30.seconds) : JSON::Any
      raise BrowserError.new("the browser is closed") if @closed
      id = (@next_id += 1)
      ch = Channel(JSON::Any).new(1)
      @pending[id] = ch
      msg = String.build do |s|
        s << %({"id":) << id << %(,"method":) << method.to_json << %(,"params":) << params
        s << %(,"sessionId":) << session_id.to_json if session_id
        s << '}'
      end
      @ws.send(msg)
      select
      when reply = ch.receive
        if err = reply["error"]?
          raise BrowserError.new(err["message"]?.try(&.as_s) || err.to_json)
        end
        reply["result"]? || JSON::Any.new(nil)
      when timeout(timeout)
        @pending.delete(id)
        raise BrowserError.new("CDP #{method} timed out after #{timeout.total_seconds.to_i}s")
      end
    end

    private def page_call(method : String, params : String = "{}", timeout : Time::Span = 30.seconds) : JSON::Any
      call(method, params, @session_id, timeout)
    end

    # Navigate the page and wait until the document finishes loading (readyState "complete").
    def navigate(url : String, timeout : Time::Span = 30.seconds)
      r = page_call("Page.navigate", ({url: url}).to_json, timeout)
      if err = r["errorText"]?.try(&.as_s)
        raise BrowserError.new("could not load #{url}: #{err}")
      end
      deadline = Time.monotonic + timeout
      until eval_string("document.readyState") == "complete"
        raise BrowserError.new("page did not finish loading in #{timeout.total_seconds.to_i}s") if Time.monotonic > deadline
        sleep 0.1.seconds
      end
    end

    # Evaluate a JS expression in the page and return its value (by value, awaiting a promise).
    def eval_json(expression : String, timeout : Time::Span = 30.seconds) : JSON::Any
      r = page_call("Runtime.evaluate",
        ({expression: expression, returnByValue: true, awaitPromise: true}).to_json, timeout)
      if ex = r["exceptionDetails"]?
        detail = ex.dig?("exception", "description").try(&.as_s) || ex["text"]?.try(&.as_s) || ex.to_json
        raise BrowserError.new("JavaScript error: #{detail}")
      end
      r.dig?("result", "value") || JSON::Any.new(nil)
    end

    def eval_string(expression : String) : String
      v = eval_json(expression)
      v.as_s? || v.to_json
    end

    # Capture a PNG of the current page to `path`.
    def screenshot(path : String)
      r = page_call("Page.captureScreenshot", %({"format":"png"}))
      data = r["data"]?.try(&.as_s) || raise BrowserError.new("no screenshot data returned")
      File.write(path, Base64.decode(data))
    end

    def close
      return if @closed
      @closed = true
      @ws.close rescue nil
      @proc.terminate(graceful: false) rescue nil
      @proc.wait rescue nil
      FileUtils.rm_rf(@profile_dir) rescue nil
    end
  end

  @@browser : CDP? = nil
  @@browser_atexit = false

  # The shared headless browser, launched on first use. Raises BrowserError if Chrome cannot start.
  def self.browser : CDP
    b = @@browser
    return b if b
    unless @@browser_atexit
      @@browser_atexit = true
      at_exit { AgentDemo.close_browser }
    end
    @@browser = CDP.launch
  end

  def self.close_browser
    @@browser.try(&.close)
    @@browser = nil
  end

  # Output under this many bytes is returned inline; anything larger is written to a temp file so a
  # single big page or query result cannot swamp the model's context.
  BROWSER_INLINE_LIMIT  = 8_000
  BROWSER_PREVIEW_BYTES = 2_000

  # Return small browser output inline; for large output, write the full text (header included) to a
  # temp file and return a short notice plus a head preview, pointing the model at read_file to page
  # through the rest. `header` is metadata shown inline either way (empty for none); `kind` names the
  # temp file.
  def self.deliver_browser_output(header : String, body : String, kind : String) : String
    full = header.empty? ? body : "#{header}\n\n#{body}"
    return full if full.bytesize <= BROWSER_INLINE_LIMIT

    path = File.join(Dir.tempdir, "shainet-#{kind}-#{Time.utc.to_unix_ms}.txt")
    begin
      File.write(path, full)
    rescue ex
      # If the dump fails, fall back to an inline, byte-capped slice rather than losing the output.
      return "#{full.byte_slice(0, BROWSER_INLINE_LIMIT)}\n... [truncated; could not write temp file: #{ex.message}]"
    end
    lines = full.count('\n') + 1
    String.build do |s|
      s << header << "\n\n" unless header.empty?
      s << "[Full output is " << full.bytesize << " bytes (" << lines << " lines), written to " << path
      s << ".\nRead it with read_file \"" << path << "\" using start_line/end_line ranges; "
      s << "do not try to read it all at once.]\n\n--- preview (first " << BROWSER_PREVIEW_BYTES << " bytes) ---\n"
      s << body.byte_slice(0, BROWSER_PREVIEW_BYTES)
      s << "\n..." if body.bytesize > BROWSER_PREVIEW_BYTES
    end
  end

  # The browser tools, or none when no Chrome is installed (so the model is not offered a dead tool).
  def self.browser_tools : Array(Tool)
    return [] of Tool unless chrome_path

    [
      Tool.new(
        "browse",
        "Open a URL in a headless Chrome and return the page's title, final URL and visible text " \
        "(after JavaScript runs). Use this to read web pages, docs or JSON/HTML APIs. http(s) only; " \
        "the browser keeps its page between calls, so browser_eval afterwards acts on what you opened. " \
        "A large page is written to a temp file and only a preview is returned -- read the file with " \
        "read_file in line ranges rather than expecting the whole page inline.",
        [ToolParam.new("url", "string", "The http(s) URL to open.")]
      ) do |args|
        url = (args["url"]? || "").strip
        if url.empty?
          next "Error: empty url"
        elsif !(url.starts_with?("http://") || url.starts_with?("https://"))
          next "Error: only http:// and https:// URLs are supported (got #{url})"
        end
        begin
          b = AgentDemo.browser
          b.navigate(url)
          title = b.eval_string("document.title")
          final = b.eval_string("location.href")
          text = b.eval_string("document.body ? document.body.innerText : ''")
          AgentDemo.deliver_browser_output("// #{final}\n// title: #{title}", text, "page")
        rescue ex : AgentDemo::BrowserError
          "Error: #{ex.message}"
        end
      end,
      Tool.new(
        "browser_eval",
        "Evaluate a JavaScript expression in the current browser page and return its value as JSON. " \
        "This is how you interact: read the DOM (document.querySelector('h1').innerText), click " \
        "(document.querySelector('#submit').click()), fill a field (el.value='x'), or extract data. " \
        "Call browse first to load a page. A promise is awaited; the value is returned by value. " \
        "A large result is written to a temp file and only a preview is returned -- read the file " \
        "with read_file in line ranges.",
        [ToolParam.new("expression", "string", "A JavaScript expression; its value is returned.")]
      ) do |args|
        expr = args["expression"]? || ""
        next "Error: empty expression" if expr.empty?
        begin
          result = AgentDemo.browser.eval_json(expr).to_json
          AgentDemo.deliver_browser_output("", result, "eval")
        rescue ex : AgentDemo::BrowserError
          "Error: #{ex.message}"
        end
      end,
      Tool.new(
        "browser_screenshot",
        "Save a PNG screenshot of the current browser page and return its file path. Load a page with " \
        "browse first.",
        [ToolParam.new("path", "string", "Output file path (default: a temp .png).", false)]
      ) do |args|
        path = args["path"]?.try(&.strip)
        path = nil if path.try(&.empty?)
        dest = path || File.join(Dir.tempdir, "shainet-screenshot-#{Time.utc.to_unix_ms}.png")
        begin
          AgentDemo.browser.screenshot(dest)
          "Saved screenshot to #{dest}"
        rescue ex : AgentDemo::BrowserError
          "Error: #{ex.message}"
        end
      end,
    ]
  end
end
