require "../math/simple_matrix"

module SHAInet
  # Token sampler over raw logits stored in a `SimpleMatrix` row.
  #
  # Designed for autoregressive decoding: the instance holds its configuration
  # and reuses internal top-k buffers, so steady-state sampling allocates
  # nothing per token. All hot loops read/write the backing `Array(Float32)`
  # through raw pointers to avoid per-element bounds checks (the vocab is large,
  # e.g. 128K, so a single sort or 128K `#[]` calls dominated decode time).
  #
  # Example:
  # ```
  # sampler = SHAInet::Sampler.new(temperature: 0.6, top_k: 40, repetition_penalty: 1.2)
  # sampler.apply_repetition_penalty!(logits, generated_ids, window: 20)
  # next_id = sampler.sample(logits)
  # ```
  class Sampler
    property temperature : Float64
    property top_k : Int32
    property repetition_penalty : Float64
    # Nucleus sampling: after top-k, keep the smallest set of tokens whose probability sums to at
    # least top_p. 1.0 disables it.
    property top_p : Float64
    property rng : Random

    @ids : Array(Int32)
    @vals : Array(Float64)

    def initialize(@temperature : Float64 = 1.0, @top_k : Int32 = 40,
                   @repetition_penalty : Float64 = 1.0, @rng : Random = Random.new,
                   @top_p : Float64 = 1.0)
      raise ArgumentError.new("top_k must be >= 1") unless @top_k >= 1
      raise ArgumentError.new("temperature must be > 0") unless @temperature > 0.0
      raise ArgumentError.new("top_p must be in (0, 1]") unless 0.0 < @top_p <= 1.0
      # Reused across calls; capacity fixed at top_k so no per-token growth.
      @ids = Array(Int32).new(@top_k)
      @vals = Array(Float64).new(@top_k)
    end

    # OpenAI presence/frequency penalties: subtract `presence` once from every token already generated
    # and `frequency` per occurrence. `counts` maps token id to how often it has been generated; the
    # caller keeps it up to date, so this costs one pass over the distinct tokens.
    def apply_presence_frequency!(logits : SimpleMatrix, counts : Hash(Int32, Int32),
                                  presence : Float64, frequency : Float64,
                                  row : Int32 = logits.rows - 1) : Nil
      return if presence == 0.0 && frequency == 0.0
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      counts.each do |id, n|
        next unless 0 <= id < cols
        ptr[id] = (ptr[id] - presence - frequency * n).to_f32
      end
    end

    # Greedy argmax over a logits row, skipping non-finite values (NaN/Inf).
    # Stateless — no temperature or sampling involved.
    def self.greedy(logits : SimpleMatrix, row : Int32 = logits.rows - 1) : Int32
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      best_id = 0
      best = -Float32::INFINITY
      j = 0
      while j < cols
        v = ptr[j]
        if v.finite? && v > best
          best = v
          best_id = j
        end
        j += 1
      end
      best_id
    end

    # Apply a repetition penalty in-place to the logits of recently emitted
    # tokens: `v > 0 ? v / penalty : v * penalty`. Only the last `window`
    # ids are considered (default: all of `recent_ids`), so the caller can pass
    # the full history without slicing/allocating.
    def apply_repetition_penalty!(logits : SimpleMatrix, recent_ids : Array(Int32),
                                  window : Int32 = recent_ids.size,
                                  row : Int32 = logits.rows - 1) : Nil
      return if @repetition_penalty == 1.0
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      p = @repetition_penalty
      n = recent_ids.size
      start = n - window
      start = 0 if start < 0
      i = start
      while i < n
        id = recent_ids.unsafe_fetch(i)
        if 0 <= id < cols
          v = ptr[id]
          ptr[id] = (v > 0 ? v / p : v * p).to_f32
        end
        i += 1
      end
    end

    # Sample a token id from a logits row using temperature + top-k softmax.
    #
    # Selection is an O(vocab) partial selection into a sorted k-buffer (no full
    # sort). NaN logits are never selected. With `top_k == 1` this is equivalent
    # to greedy. Uses the instance `rng` so callers can seed for determinism.
    def sample(logits : SimpleMatrix, row : Int32 = logits.rows - 1) : Int32
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      k = @top_k
      t = @temperature

      ids = @ids
      vals = @vals
      ids.clear
      vals.clear

      # Partial top-k selection over the full vocab (descending by value).
      j = 0
      while j < cols
        v = ptr[j].to_f64 / t
        unless v.nan?
          sz = vals.size
          if sz < k
            pos = vals.bsearch_index { |x| x < v } || sz
            vals.insert(pos, v)
            ids.insert(pos, j)
          elsif v > vals.unsafe_fetch(k - 1)
            pos = vals.bsearch_index { |x| x < v } || sz
            vals.insert(pos, v)
            ids.insert(pos, j)
            vals.pop
            ids.pop
          end
        end
        j += 1
      end

      szk = vals.size
      return 0 if szk == 0 # all-NaN row; nothing to sample

      # Buffers are stable now (no more insert/pop) — take raw pointers.
      vp = vals.to_unsafe
      ip = ids.to_unsafe

      # Softmax over the selected logits, reusing `vals` to hold exp values.
      max_val = vp[0] # descending order, so element 0 is the max
      sum = 0.0
      i = 0
      while i < szk
        e = Math.exp(vp[i] - max_val)
        vp[i] = e
        sum += e
        i += 1
      end
      # Nucleus cut: values are in descending order, so the kept set is a prefix. Always at least one.
      if @top_p < 1.0
        keep = 0
        acc = 0.0
        while keep < szk
          acc += vp[keep]
          keep += 1
          break if acc >= @top_p * sum
        end
        szk = keep
        sum = acc
      end
      inv_sum = 1.0 / sum

      # Inverse-CDF sample.
      r = @rng.rand
      cumulative = 0.0
      best_id = ip[szk - 1]
      i = 0
      while i < szk
        cumulative += vp[i] * inv_sum
        if r <= cumulative
          best_id = ip[i]
          break
        end
        i += 1
      end
      best_id
    end
  end
end
