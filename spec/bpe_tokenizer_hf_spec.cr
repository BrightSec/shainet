require "./spec_helper"

# Byte-level (HF / GGUF) BPE. Exact parity with llama.cpp was checked against the real Qwen3.5
# vocabulary out of band; these pin the behaviours that parity depends on with a tiny vocabulary.

# Every byte's GPT-2 stand-in as ids 0..255, then each merge's result, in rank order.
private def byte_tokenizer(merges : Array(Tuple(String, String))) : SHAInet::BPETokenizer
  tok = SHAInet::BPETokenizer.new
  tok.hf_mode = true
  256.times do |b|
    t = SHAInet::BPETokenizer.byte_to_unicode(b.to_u8).to_s
    tok.vocab[t] = b
    tok.inv_vocab << t
  end
  merges.each_with_index do |(a, b), rank|
    m = a + b
    unless tok.vocab.has_key?(m)
      tok.vocab[m] = tok.inv_vocab.size
      tok.inv_vocab << m
    end
    tok.merges << {a, b}
    tok.merges_map[{a, b}] = m
    tok.merges_rank[{a, b}] = rank
  end
  tok
end

describe "BPETokenizer (byte-level)" do
  it "never merges across a pre-tokenizer split" do
    # "Ġ" is the stand-in for a space. "oĠ" spans the end of "go" and the space that starts " go",
    # which the qwen35 split puts in different pieces.
    tok = byte_tokenizer([{"o", "Ġ"}, {"g", "o"}, {"Ġ", "go"}])
    tok.pre_tokenizer = "qwen35"
    ids = tok.encode("go go")
    ids.map { |i| tok.inv_vocab[i] }.should eq(["go", "Ġgo"])
    tok.decode(ids).should eq("go go")
  end

  it "merges lowest rank first, leftmost on a tie" do
    tok = byte_tokenizer([{"a", "a"}, {"b", "c"}, {"a", "b"}])
    tok.pre_tokenizer = "qwen35"
    tok.encode("aaa").map { |i| tok.inv_vocab[i] }.should eq(["aa", "a"])
    # "ab" outranks nothing here: "bc" (rank 1) merges before "ab" (rank 2) can.
    tok.encode("abc").map { |i| tok.inv_vocab[i] }.should eq(["a", "bc"])
  end

  it "matches added literal tokens whole, wherever they appear" do
    tok = byte_tokenizer([] of Tuple(String, String))
    tok.pre_tokenizer = "qwen35"
    tok.vocab["<think>"] = 1000
    tok.add_literal_token("<think>", 1000)
    ids = tok.encode("a<think>b")
    ids.should eq([tok.vocab["a"], 1000, tok.vocab["b"]])
  end
end

describe "BPETokenizer (byte-level) with invalid UTF-8" do
  it "encodes instead of raising" do
    tok = byte_tokenizer([] of Tuple(String, String))
    tok.pre_tokenizer = "qwen35"
    bad = String.new(Bytes[0x61, 0x80, 0x62])
    tok.encode(bad).should_not be_empty
  end
end
