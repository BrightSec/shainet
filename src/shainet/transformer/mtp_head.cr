module SHAInet
  # Multi-token-prediction (MTP / NextN) draft head for Qwen3.5 / Qwen3.8, as llama.cpp runs it
  # (src/models/qwen35.cpp, graph_mtp).
  #
  # The head is one extra decoder block trained to predict the token AFTER the next one. At position
  # p it reads two things: the embedding of token x_p and the target's final hidden state h_{p-1}
  # (after output_norm), i.e. the state that predicted x_p. Then
  #
  #   x = eh_proj([enorm(embed(x_p)) | hnorm(h_{p-1})])     # 2d -> d
  #   y = full-attention block(x)                            # own KV cache, same shape as the trunk's
  #   h' = shared_head_norm(y);  logits = output(h')         # the trunk's LM head
  #
  # and h' stands in for the target's hidden state when the head is chained to draft further tokens.
  #
  # Its KV cache holds one row per position of the conversation, in lockstep with the trunk's: the
  # trunk's every forward pass is followed by `catch_up` over the same tokens, each paired with the
  # previous position's target hidden state (`pending_h` carries the last one across calls).
  class MTPHead
    getter block : LlamaBlock
    getter eh_proj : QuantizedWeight
    getter enorm : RMSNorm
    getter hnorm : RMSNorm
    getter head_norm : RMSNorm
    getter d_model : Int32

    # Target hidden state of the last position the trunk processed (zeros before the first).
    getter pending_h : SimpleMatrix

    def initialize(@block, @eh_proj, @enorm, @hnorm, @head_norm, @d_model)
      @pending_h = SimpleMatrix.new(1, @d_model)
    end

    def cached_length : Int32
      @block.cached_length
    end

    def clear_cache!
      @block.clear_cache!
      @pending_h = SimpleMatrix.new(1, @d_model)
    end

    # Cut the head's KV cache back to `positions` and set the hidden state the next position pairs
    # with.
    def truncate!(positions : Int32, pending : SimpleMatrix)
      @block.truncate_cache!(positions)
      @pending_h = pending
    end

    def pending_h=(h : SimpleMatrix)
      @pending_h = h
    end

    # Feed the head the tokens the trunk just processed, with `hidden` the trunk's [n, d] final
    # hidden rows for them: token i pairs with hidden row i-1, and token 0 with pending_h.
    def catch_up(tokens : Array(Int32), hidden : SimpleMatrix, embed : EmbeddingLayer) : Nil
      n = tokens.size
      return if n == 0
      raise ArgumentError.new("catch_up: #{n} tokens but #{hidden.rows} hidden rows") unless hidden.rows == n
      h = SimpleMatrix.new(n, @d_model)
      h.data.to_unsafe.copy_from(@pending_h.data.to_unsafe, @d_model)
      if n > 1
        (h.data.to_unsafe + @d_model).copy_from(hidden.data.to_unsafe, (n - 1).to_i64 * @d_model)
      end
      forward(tokens, h, embed)
      last = SimpleMatrix.new(1, @d_model)
      last.data.to_unsafe.copy_from(hidden.data.to_unsafe + (n - 1).to_i64 * @d_model, @d_model)
      @pending_h = last
    end

    # One head step over `tokens` paired row by row with `h`; returns the normed output rows h'
    # (the input of the LM head, and the h of the next chained step). Appends to the KV cache.
    def forward(tokens : Array(Int32), h : SimpleMatrix, embed : EmbeddingLayer) : SimpleMatrix
      n = tokens.size
      d = @d_model
      e = Profile.measure("mtp.embed") { embed.embed_cpu(tokens) }
      en = @enorm.forward(e)
      hn = @hnorm.forward(h)
      cat = CudaMatrix.new(n, 2 * d)
      # [enorm(e) | hnorm(h)] per row: llama.cpp concatenates e first (ggml_concat dim 0).
      n.times do |r|
        (cat.raw_data.to_unsafe + r.to_i64 * 2 * d).copy_from(en.data.to_unsafe + r.to_i64 * d, d)
        (cat.raw_data.to_unsafe + r.to_i64 * 2 * d + d).copy_from(hn.data.to_unsafe + r.to_i64 * d, d)
      end
      cat.mark_host_modified!
      cat.sync_to_device!("mtp_cat")
      x = Profile.measure("mtp.eh_proj") { @eh_proj.gemv(cat) }
      cat.free!
      y = Profile.measure("mtp.block") { run_block(x) }
      x.free!
      @head_norm.forward(y)
    end

    private def run_block(x : CudaMatrix) : SimpleMatrix
      if @block.block_device_capable?
        res = x.rows > 1 ? @block.forward_cached_device_multi(x) : @block.forward_cached_device(x)
        if res
          res.sync_from_device!("mtp_block_out") if res.device_dirty?
          sm = SimpleMatrix.new(res.rows, res.cols)
          sm.data.to_unsafe.copy_from(res.raw_data.to_unsafe, res.rows * res.cols)
          return sm
        end
      end
      x.sync_from_device!("mtp_block_in") if x.device_dirty?
      sm = SimpleMatrix.new(x.rows, x.cols)
      sm.data.to_unsafe.copy_from(x.raw_data.to_unsafe, x.rows * x.cols)
      @block.forward_cached(sm)
    end

    # The head's part of a cache snapshot (Network#snapshot_cache).
    class Snapshot
      getter kv : LlamaBlock::KVSnapshot
      getter pending_h : SimpleMatrix

      def initialize(@kv, @pending_h)
      end

      def bytes : Int64
        @kv.bytes + @pending_h.data.size.to_i64 * 4
      end
    end

    def snapshot : Snapshot
      Snapshot.new(@block.snapshot_kv, @pending_h.clone)
    end

    def can_restore?(snap : Snapshot) : Bool
      @block.can_restore_kv?(snap.kv)
    end

    def restore!(snap : Snapshot) : Nil
      @block.restore_kv!(snap.kv)
      @pending_h = snap.pending_h.clone
    end
  end
end
