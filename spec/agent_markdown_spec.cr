require "./spec_helper"
require "../examples/agent_markdown"

private def render(text : String, width = 80, chunk = 0) : String
  io = IO::Memory.new
  md = AgentDemo::Markdown.new(io, -> { width }, true, true)
  if chunk > 0
    i = 0
    while i < text.size
      md.feed(text[i, chunk])
      i += chunk
    end
  else
    md.feed(text)
  end
  md.finish
  io.to_s
end

private def plain(s : String) : String
  s.gsub(/\e\[[0-9;]*[mK]/, "").gsub(/\r[^\n┌]*/, "")
end

describe AgentDemo::Markdown do
  it "renders the same however the stream is chunked" do
    text = "Some **bold** and `code` and [a](http://x).\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```cr\nx = 1\n```\n"
    whole = render(text)
    render(text, chunk: 1).should eq(whole)
    render(text, chunk: 3).should eq(whole)
  end

  it "draws tables whose rows line up in terminal cells, emoji and wide text included" do
    text = "| # | Severity | Notes |\n|---|---|---|\n| 1 | 🔴 High | a fairly long note that has to wrap somewhere |\n| 2 | 🟡 Low | 日本語 |\n"
    lines = plain(render(text, width: 40)).lines.select { |l| l.starts_with?('│') || l.starts_with?('┌') }
    lines.size.should be > 3
    lines.map { |l| AgentDemo::TermWidth.of(l) }.uniq!.size.should eq(1)
    lines.each { |l| AgentDemo::TermWidth.of(l).should be <= 40 }
  end

  it "styles inline markup but leaves arithmetic and identifiers alone" do
    res = render("**b** `c` 2 * 3 a*b*c snake_case_name\n")
    res.should contain("\e[0;1mb")
    plain(res).should eq("b c 2 * 3 a*b*c snake_case_name\n")
  end

  it "passes text through untouched when color is off" do
    io = IO::Memory.new
    md = AgentDemo::Markdown.new(io, -> { 80 }, false)
    md.feed("# T\n| a |\n**x**")
    md.finish
    io.to_s.should eq("# T\n| a |\n**x**")
  end
end

describe "AgentDemo.complete_utf8" do
  it "holds back a character split across tokens instead of printing U+FFFD" do
    full = "ok 🔴"
    half = String.new(full.to_slice[0, full.bytesize - 2])
    AgentDemo.complete_utf8(half).should eq("ok ")
    AgentDemo.complete_utf8(full).should eq(full)
  end
end

describe AgentDemo::TermWidth do
  it "counts emoji and CJK as two cells and combining marks as none" do
    AgentDemo::TermWidth.of("🔴 High").should eq(7)
    AgentDemo::TermWidth.of("日本").should eq(4)
    AgentDemo::TermWidth.of("e\u0301").should eq(1)
  end
end
