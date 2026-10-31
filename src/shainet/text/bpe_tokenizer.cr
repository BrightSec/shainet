# No CUDA imports needed - CPU-only implementation
require "json"

module SHAInet
  # Simple byte-pair encoding tokenizer. It can train a vocabulary
  # from text and encode/decode using the learned merges.
  class BPETokenizer
    Log = ::Log.for(self)

    property vocab = Hash(String, Int32).new
    property inv_vocab = [] of String
    property merges = [] of Tuple(String, String)
    property merges_map = Hash(Tuple(String, String), String).new
    property merges_rank = Hash(Tuple(String, String), Int32).new
    property? hf_mode : Bool = false
    # Byte-level BPE models split text with a regex BEFORE merging, and merges never cross a split.
    # nil keeps the old whole-text behaviour for tokenizers whose split is unknown.
    property pre_regex : Regex? = nil

    # Pre-tokenizer regexes by GGUF `tokenizer.ggml.pre` name, as in llama.cpp's llama-vocab.cpp
    # (the tokenizer.json originals; PCRE2 takes them as written).
    PRE_REGEXES = {
      "qwen35" => %q{(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+},
      "qwen2"  => %q{(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+},
      "llama-bpe" => %q{(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+},
      "gpt2"   => %q{'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+},
    }

    # Select the pre-tokenizer for a GGUF `tokenizer.ggml.pre` value. Unknown names leave it unset.
    def pre_tokenizer=(name : String)
      if pat = PRE_REGEXES[name]?
        @pre_regex = Regex.new(pat)
      else
        Log.warn { "BPE: unknown pre-tokenizer '#{name}'; encoding without a pre-split" }
      end
    end

    # Added tokens matched literally before the pre-split, e.g. Qwen's <tool_call> and <think>
    # (GGUF token type USER_DEFINED, or non-special added_tokens in tokenizer.json). Control tokens
    # such as <|im_end|> are deliberately NOT here: like llama.cpp with parse_special off, text that
    # spells them stays text, so a file or tool output cannot inject chat structure.
    @literal_ids = Hash(String, Int32).new
    @literal_regex : Regex? = nil

    def add_literal_token(token : String, id : Int32)
      return if token.empty?
      @literal_ids[token] = id
      # Longest first, so a token that prefixes another cannot win.
      @literal_regex = Regex.union(@literal_ids.keys.sort_by { |t| -t.bytesize })
    end

    # Merge table keyed by token-id pair, built on first use: (a << 32 | b) -> {rank, merged id}.
    @id_merges : Hash(Int64, Tuple(Int32, Int32))? = nil
    # BPE result per pre-token piece. Text repeats heavily (words, indentation, punctuation), so most
    # pieces after the first few thousand are hits.
    @piece_cache = Hash(String, Array(Int32)).new
    PIECE_CACHE_MAX = 200_000

    # GPT-2 byte-to-unicode mapping (lazily built). Maps a printable Unicode
    # codepoint back to the original raw byte it represents in byte-level BPE.
    @@unicode_to_byte : Hash(Char, UInt8)?
    @@byte_to_unicode : Hash(UInt8, Char)?

    def self.unicode_to_byte(ch : Char) : UInt8
      map = (@@unicode_to_byte ||= build_byte_map)
      map[ch]? || 0_u8
    end

    def self.byte_to_unicode(b : UInt8) : Char
      map = (@@byte_to_unicode ||= build_byte_to_unicode_map)
      map[b]
    end

    private def self.build_byte_to_unicode_map : Hash(UInt8, Char)
      # Inverse of build_byte_map
      byte_map = (@@unicode_to_byte ||= build_byte_map)
      inv = Hash(UInt8, Char).new
      byte_map.each { |ch, b| inv[b] = ch }
      inv
    end

    private def self.build_byte_map : Hash(Char, UInt8)
      # Printable byte ranges that map to themselves
      bs = [] of Int32
      (('!'.ord)..('~'.ord)).each { |b| bs << b }
      (('¡'.ord)..('¬'.ord)).each { |b| bs << b }
      (('®'.ord)..('ÿ'.ord)).each { |b| bs << b }
      cs = bs.dup
      n = 0
      (0..255).each do |b|
        unless bs.includes?(b)
          bs << b
          cs << (256 + n)
          n += 1
        end
      end
      map = Hash(Char, UInt8).new
      bs.each_with_index { |b, i| map[cs[i].chr] = b.to_u8 }
      map
    end

    # Load a pre-trained tokenizer from a HuggingFace tokenizer.json file
    def self.from_hf(path : String) : BPETokenizer
      tok = new
      tok.hf_mode = true
      data = JSON.parse(File.read(path))
      model = data["model"]

      # Load vocab
      model["vocab"].as_h.each do |token, id|
        tok.vocab[token] = id.as_i
      end

      # Build inverse vocab
      max_id = tok.vocab.values.max? || 0
      tok.inv_vocab.concat(Array(String).new(max_id + 1, ""))
      tok.vocab.each { |token, id| tok.inv_vocab[id] = token }

      # Load merges with rank (handles both "a b" string and ["a","b"] array formats)
      if merges = model["merges"]?
        merges.as_a.each_with_index do |m, rank|
          parts = if m.as_a?
                    a = m.as_a
                    next unless a.size == 2
                    [a[0].as_s, a[1].as_s]
                  else
                    m.as_s.split(' ', 2)
                  end
          next unless parts.size == 2
          pair = {parts[0], parts[1]}
          merged = parts[0] + parts[1]
          tok.merges << pair
          tok.merges_map[pair] = merged
          tok.merges_rank[pair] = rank
        end
      end

      if pat = hf_split_regex(data["pre_tokenizer"]?)
        tok.pre_regex = Regex.new(pat)
      end

      # Load added tokens (special tokens)
      if added = data["added_tokens"]?
        added.as_a.each do |t|
          token = t["content"].as_s
          id = t["id"].as_i
          tok.vocab[token] = id
          tok.add_literal_token(token, id) unless t["special"]?.try(&.as_bool?)
          while tok.inv_vocab.size <= id
            tok.inv_vocab << ""
          end
          tok.inv_vocab[id] = token
        end
      end

      tok
    end

    # The Split regex of a tokenizer.json pre_tokenizer, alone or inside a Sequence.
    private def self.hf_split_regex(pre : JSON::Any?) : String?
      return nil unless pre && (h = pre.as_h?)
      case h["type"]?.try(&.as_s?)
      when "Split"
        h["pattern"]?.try { |pt| pt["Regex"]?.try(&.as_s?) }
      when "Sequence"
        h["pretokenizers"]?.try(&.as_a?).try &.each do |sub|
          if r = hf_split_regex(sub)
            return r
          end
        end
        nil
      end
    end

    # Train the tokenizer vocabulary from the given text using the
    # byte-pair encoding algorithm. `vocab_size` determines how many
    # unique tokens will be created at most.
    def train(text : String, vocab_size : Int32)
      word_freqs = Hash(String, Int32).new(0)
      text.split(/\s+/).each { |w| word_freqs[w] += 1 }

      corpus = Array(Array(String)).new(word_freqs.size)
      freqs = Array(Int32).new(word_freqs.size)
      word_freqs.each do |word, count|
        tokens = word.chars.map(&.to_s) + ["</w>"]
        corpus << tokens
        freqs << count
        tokens.each { |t| add_token(t) }
      end

      rounds = 0
      while @vocab.size < vocab_size
        rounds += 1
        if rounds % 100 == 0
          Log.debug { "Round #{rounds}: Merges done: #{@merges.size}, vocabulary size: #{@vocab.size}" }
          Log.debug { "Progress: #{(@vocab.size.to_f / vocab_size * 100.0).round(2)}%" }
        end
        # Use CPU-only pair counting - faster and simpler than GPU version
        pair_counts = cpu_pair_counts(corpus, freqs)
        break if pair_counts.empty?
        heap = PairHeap.new
        pair_counts.each do |pair, count|
          heap.push PairHeap::Node.new(pair, count)
        end
        best_node = heap.pop
        break unless best_node
        best_pair_ids = best_node.pair

        token_a = @inv_vocab[best_pair_ids[0]]
        token_b = @inv_vocab[best_pair_ids[1]]
        new_token = String.build do |io|
          io << token_a
          io << token_b
        end
        corpus.each do |tokens|
          i = 0
          size = tokens.size
          while i < size - 1
            if tokens[i] == token_a && tokens[i + 1] == token_b
              tokens[i] = new_token
              j = i + 1
              while j < size - 1
                tokens[j] = tokens[j + 1]
                j += 1
              end
              tokens.pop
              size -= 1
            else
              i += 1
            end
          end
        end
        pair = {token_a, token_b}
        @merges << pair
        @merges_map[pair] = new_token
        @merges_rank[pair] = @merges.size - 1
        add_token(new_token)
      end
    end

    # Encode a string into token IDs. Unknown tokens are added to the
    # vocabulary using a greedy BPE merging strategy.
    def encode(text : String) : Array(Int32)
      if hf_mode?
        return encode_hf(text)
      end
      words = text.split(/\s+/)
      ids = Array(Int32).new(words.size)
      words.each do |word|
        tokens = encode_tokens(word)
        tokens.each { |t| ids << add_token(t) }
      end
      ids
    end

    # HF-style encode: byte-level BPE with GPT-2 unicode mapping.
    #
    # With a pre-tokenizer the text is split first and each piece merged on its own, which is both
    # what the model was trained on and what keeps this fast. Without one the whole text was a single
    # "word": merges could cross word boundaries (token sequences the model never saw), and each
    # merge rescanned the whole array, so a 39K-token transcript took 11.8 s to encode.
    private def encode_hf(text : String) : Array(Int32)
      ids = Array(Int32).new
      return ids if text.empty?
      if lit = @literal_regex
        pos = 0
        while m = lit.match(text, pos)
          encode_span(text[pos...m.begin(0)], ids) if m.begin(0) > pos
          ids << @literal_ids[m[0]]
          pos = m.end(0)
        end
        encode_span(text[pos..], ids) if pos < text.size
      else
        encode_span(text, ids)
      end
      ids
    end

    private def encode_span(text : String, ids : Array(Int32))
      if re = @pre_regex
        text.scan(re) { |m| encode_piece(m[0], ids) }
      else
        encode_piece(text, ids)
      end
    end

    private def encode_piece(piece : String, ids : Array(Int32))
      if cached = @piece_cache[piece]?
        ids.concat(cached)
        return
      end
      res = bpe_piece(piece)
      @piece_cache.clear if @piece_cache.size >= PIECE_CACHE_MAX
      @piece_cache[piece] = res if piece.bytesize <= 256
      ids.concat(res)
    end

    # Byte-level BPE of one piece over token ids: a heap of candidate merges keyed (rank, position),
    # so the lowest-rank pair merges first and ties go to the leftmost, as in HF and llama.cpp.
    # Stale heap entries are skipped by re-checking the pair's rank when popped (a rank names exactly
    # one pair, so an equal rank means the same pair is still there).
    private def bpe_piece(piece : String) : Array(Int32)
      merges = id_merges
      bids = byte_ids
      syms = Array(Int32).new(piece.bytesize)
      piece.each_byte do |b|
        id = bids[b]
        syms << id if id >= 0
      end
      n = syms.size
      return syms if n <= 1 || merges.empty?
      nxt = Array(Int32).new(n) { |i| i + 1 < n ? i + 1 : -1 }
      prv = Array(Int32).new(n) { |i| i - 1 }
      alive = Array(Bool).new(n, true)
      heap = [] of Int64
      push = ->(i : Int32) do
        j = nxt[i]
        if j >= 0
          if m = merges[(syms[i].to_i64 << 32) | syms[j].to_i64]?
            heap_push(heap, (m[0].to_i64 << 32) | i.to_i64)
          end
        end
        nil
      end
      (0...n - 1).each { |i| push.call(i) }
      until heap.empty?
        key = heap_pop(heap)
        rank = (key >> 32).to_i32
        i = (key & 0xFFFFFFFF).to_i32
        next unless alive[i]
        j = nxt[i]
        next if j < 0
        m = merges[(syms[i].to_i64 << 32) | syms[j].to_i64]?
        next unless m && m[0] == rank
        syms[i] = m[1]
        alive[j] = false
        k = nxt[j]
        nxt[i] = k
        prv[k] = i if k >= 0
        push.call(prv[i]) if prv[i] >= 0
        push.call(i)
      end
      res = Array(Int32).new
      i = 0
      while i >= 0
        res << syms[i]
        i = nxt[i]
      end
      res
    end

    @byte_ids : Array(Int32)? = nil

    # Vocabulary id of each raw byte's GPT-2 unicode stand-in (-1 when absent).
    private def byte_ids : Array(Int32)
      @byte_ids ||= Array(Int32).new(256) { |b| @vocab[BPETokenizer.byte_to_unicode(b.to_u8).to_s]? || -1 }
    end

    private def id_merges : Hash(Int64, Tuple(Int32, Int32))
      @id_merges ||= begin
        h = Hash(Int64, Tuple(Int32, Int32)).new(initial_capacity: @merges_rank.size)
        @merges_rank.each do |pair, rank|
          a = @vocab[pair[0]]?
          b = @vocab[pair[1]]?
          m = @vocab[pair[0] + pair[1]]?
          next unless a && b && m
          h[(a.to_i64 << 32) | b.to_i64] = {rank, m}
        end
        h
      end
    end

    private def heap_push(h : Array(Int64), v : Int64)
      h << v
      i = h.size - 1
      while i > 0
        p = (i - 1) >> 1
        break if h[p] <= v
        h[i] = h[p]
        i = p
      end
      h[i] = v
    end

    private def heap_pop(h : Array(Int64)) : Int64
      top = h[0]
      last = h.pop
      return top if h.empty?
      i = 0
      n = h.size
      loop do
        l = 2 * i + 1
        break if l >= n
        c = (l + 1 < n && h[l + 1] < h[l]) ? l + 1 : l
        break if h[c] >= last
        h[i] = h[c]
        i = c
      end
      h[i] = last
      top
    end

    private def encode_tokens(word : String) : Array(String)
      tokens = word.chars.map(&.to_s) + ["</w>"]
      return tokens if @merges_rank.empty?

      pairs = get_pairs(tokens)
      while !pairs.empty?
        best_pair = nil
        best_rank = Int32::MAX
        pairs.each do |p|
          if rank = @merges_rank[p]?
            if rank < best_rank
              best_rank = rank
              best_pair = p
            end
          end
        end
        break unless best_pair
        merge_tokens!(tokens, best_pair)
        break if tokens.size <= 1
        pairs = get_pairs(tokens)
      end

      tokens
    end

    private def get_pairs(tokens : Array(String)) : Array(Tuple(String, String))
      pairs = Array(Tuple(String, String)).new(tokens.size - 1)
      (0...tokens.size - 1).each do |i|
        pairs << {tokens[i], tokens[i + 1]}
      end
      pairs
    end

    # Decode an array of token IDs back into a string.
    def decode(ids : Array(Int32)) : String
      if hf_mode?
        # Byte-level BPE: each token char maps back to a raw byte (GPT-2
        # bytes_to_unicode reverse), then the byte stream is UTF-8 decoded.
        bytes = Array(UInt8).new
        ids.each do |id|
          token = @inv_vocab[id]? || ""
          token.each_char do |ch|
            bytes << BPETokenizer.unicode_to_byte(ch)
          end
        end
        return String.new(Slice.new(bytes.to_unsafe, bytes.size))
      end
      String.build do |io|
        current = String::Builder.new
        ids.each do |id|
          token = @inv_vocab[id]? || ""
          if token.ends_with?("</w>")
            current << token[0, token.size - 4]
            io << current.to_s
            io << ' '
            current = String::Builder.new
          else
            current << token
          end
        end
      end.rstrip
    end

    private def merge_tokens!(tokens : Array(String), pair : Tuple(String, String))
      new_token = @merges_map[pair]?
      new_token ||= String.build do |io|
        io << pair[0]
        io << pair[1]
      end
      i = 0
      size = tokens.size
      while i < size - 1
        if tokens[i] == pair[0] && tokens[i + 1] == pair[1]
          tokens[i] = new_token
          j = i + 1
          while j < size - 1
            tokens[j] = tokens[j + 1]
            j += 1
          end
          tokens.pop
          size -= 1
        else
          i += 1
        end
      end
    end

    private def cpu_pair_counts(corpus, freqs)
      pc = Hash(Tuple(Int32, Int32), Int32).new(0)
      corpus.each_with_index do |tokens, idx|
        freq = freqs[idx]
        (0...(tokens.size - 1)).each do |i|
          if id1 = @vocab[tokens[i]]?
            if id2 = @vocab[tokens[i + 1]]?
              pc[{id1, id2}] += freq
            end
          end
        end
      end
      pc
    end

    private def add_token(token : String) : Int32
      if id = @vocab[token]?
        id
      else
        new_id = @vocab.size
        @vocab[token] = new_id
        @inv_vocab << token
        new_id
      end
    end
  end

  # Simple max-heap used to select the best pair during training
  class PairHeap
    struct Node
      getter pair : Tuple(Int32, Int32)
      getter count : Int32

      def initialize(@pair : Tuple(Int32, Int32), @count : Int32)
      end
    end

    def initialize
      @data = [] of Node
    end

    def push(node : Node)
      @data << node
      sift_up(@data.size - 1)
    end

    def pop : Node?
      return if @data.empty?
      max = @data[0]
      last = @data.pop
      if !@data.empty?
        @data[0] = last
        sift_down(0)
      end
      max
    end

    private def sift_up(idx : Int32)
      while idx > 0
        parent = (idx - 1) // 2
        if @data[parent].count < @data[idx].count
          @data[parent], @data[idx] = @data[idx], @data[parent]
          idx = parent
        else
          break
        end
      end
    end

    private def sift_down(idx : Int32)
      size = @data.size
      loop do
        left = idx * 2 + 1
        right = left + 1
        largest = idx
        largest = left if left < size && @data[left].count > @data[largest].count
        largest = right if right < size && @data[right].count > @data[largest].count
        break if largest == idx
        @data[idx], @data[largest] = @data[largest], @data[idx]
        idx = largest
      end
    end
  end
end
