module SHAInet
  # A Gated DeltaNet token-mixer block: the linear-attention layer type in a qwen3_5 hybrid
  # stack, standing in for LlamaBlock wherever layer_types says "linear_attention".
  #
  # Macro shape follows LlamaBlock exactly, so the two are interchangeable in a stack:
  #
  #   x -> norm1 -> mixer -> +x  ->  norm2 -> ffn -> +
  #
  # Only the mixer differs. Per the paper's block design (arXiv:2412.06464 section 3.4):
  #
  #   q, k: linear proj -> short conv -> SiLU -> L2 normalize
  #   v:    linear proj -> short conv -> SiLU
  #   a, b: linear proj only
  #   out:  gated delta rule -> norm -> SiLU gate -> output proj
  #
  # Their ablation says the short conv and the output gate both matter (removing either costs
  # more than removing the output norm), and that L2 normalization is essential -- so none of
  # those pieces are optional.
  #
  # WHAT THIS BUYS on a 16 GB card: no KV cache. The mixer's whole memory is a [d_v, d_k] state
  # per head plus a kernel-1 conv window, neither of which grows with context. Softmax attention
  # costs 96 KiB per position on the 30B, which is what caps context at 32768 here.
  #
  # UNVERIFIED against a checkpoint. The gate parameterization below is the documented Mamba2
  # form, but the paper says only "we use Mamba2's parameterization for alpha but omit it for
  # brevity", and there is no reference implementation on this machine to diff against. The
  # shapes, the state carry and the prefill/decode equivalence ARE specified and tested; the
  # exact gate formula needs a real Qwen3.5 checkpoint to confirm, which is why the loader still
  # refuses to build one of these from weights.
  class GatedDeltaNetBlock < MatrixLayer
    include QuantizedProjection

    getter d_model : Int32
    getter num_v_heads : Int32
    getter num_k_heads : Int32
    getter head_k : Int32
    getter head_v : Int32
    getter conv_kernel : Int32

    getter norm1 : RMSNorm
    getter norm2 : RMSNorm
    getter out_norm : RMSNorm
    getter ffn : SwiGLUFF

    # The five large projections carry the block's whole parameter cost and are quantized.
    #
    # w_alpha and w_beta are [d_model, num_v_heads] -- 0.5 MB against 100 MB for the rest -- so they
    # are NOT quantized, but they do move to the device as fp32. Computing them as a hand-rolled
    # triple loop cost 0.328 s per layer at a 1216-token prefill, 47% of the mixer once the
    # recurrence moved to the GPU, for what is simply two GEMMs. Precision matters here: alpha feeds
    # exp(-exp(a_log) * softplus(...)), so a Q4 rounding of the projection moves the decay itself.
    property w_q : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_k : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_v : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_o : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_gate : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_alpha : SimpleMatrix | CudaMatrix
    property w_beta : SimpleMatrix | CudaMatrix
    # Mamba2's per-head decay parameters. a_log is stored in log space because the decay must
    # stay in (0,1) for the state to be stable, and exp(-exp(a_log)*dt) is in (0,1) for any
    # real a_log and any dt > 0 -- no clamping needed.
    getter a_log : Array(Float64)
    getter dt_bias : Array(Float64)

    # Which key head each value head reads, which is a property of the WEIGHT LAYOUT and so
    # differs by loader -- it is not a free choice.
    #
    # false (grouped, `h // heads_per_k`): the SafeTensors layout. HFLoader's
    # split_head_interleaved deliberately rearranges the fused projection so value heads for key
    # head kh land contiguously at kh*heads_per_k, matching HF's repeat_interleave.
    #
    # true (tiled, `h % num_k_heads`): the GGUF layout. llama.cpp widens q/k from num_k_heads to
    # num_v_heads with ggml_repeat, which TILES rather than interleaves, so value head h reads key
    # head h % num_k_heads. Verified elementwise against llama.cpp's own dumped attn_output at
    # layers 0, 10 and 21: cosine 1.000000 and magnitude ratio 1.000000 for tiled, against 0.74,
    # 0.16 and 0.26 for grouped.
    property? k_head_tiled : Bool

    def initialize(@d_model : Int32, ff_hidden : Int32,
                   @num_v_heads : Int32 = 32, @num_k_heads : Int32 = 16,
                   @head_k : Int32 = 128, @head_v : Int32 = 128,
                   @conv_kernel : Int32 = 4, eps : Float64 = 1e-6, allocate : Bool = true)
      super(@d_model, SHAInet.none)
      raise ArgumentError.new("num_v_heads must be positive") unless @num_v_heads > 0
      raise ArgumentError.new("num_k_heads must be positive") unless @num_k_heads > 0
      # Value heads are grouped over key heads, as in grouped-query attention: several value
      # heads share one key head's state dimension.
      raise ArgumentError.new("num_v_heads (#{@num_v_heads}) must be divisible by num_k_heads (#{@num_k_heads})") unless @num_v_heads % @num_k_heads == 0

      @eps = eps
      @norm1 = RMSNorm.new(@d_model, eps)
      @norm2 = RMSNorm.new(@d_model, eps)
      # PER-HEAD output norm, over head_v rather than over the concatenated v_dim.
      #
      # Established from the real checkpoint, not from the paper: Qwen3.5-9B's
      # linear_attn.norm.weight is [128], which is head_v, against a v_dim of 4096. One weight
      # vector is shared across all 32 value heads and applied to each head's slice
      # independently. Normalizing the whole concatenation instead would couple the heads
      # through one shared RMS and silently change every output -- a wrong-but-plausible design
      # that no shape check would have caught.
      @out_norm = RMSNorm.new(@head_v, eps)
      @ffn = SwiGLUFF.new(@d_model, ff_hidden, allocate: allocate)

      k_dim = @num_k_heads * @head_k
      v_dim = @num_v_heads * @head_v
      if allocate
        @w_q = SimpleMatrix.new(@d_model, k_dim)
        @w_k = SimpleMatrix.new(@d_model, k_dim)
        @w_v = SimpleMatrix.new(@d_model, v_dim)
        @w_o = SimpleMatrix.new(v_dim, @d_model)
        @w_gate = SimpleMatrix.new(@d_model, v_dim)
        @w_alpha = SimpleMatrix.new(@d_model, @num_v_heads)
        @w_beta = SimpleMatrix.new(@d_model, @num_v_heads)
      else
        @w_q = SimpleMatrix.new(0, 0)
        @w_k = SimpleMatrix.new(0, 0)
        @w_v = SimpleMatrix.new(0, 0)
        @w_o = SimpleMatrix.new(0, 0)
        @w_gate = SimpleMatrix.new(0, 0)
        @w_alpha = SimpleMatrix.new(0, 0)
        @w_beta = SimpleMatrix.new(0, 0)
      end
      @a_log = Array(Float64).new(@num_v_heads, 0.0)
      @dt_bias = Array(Float64).new(@num_v_heads, 0.0)
      @k_head_tiled = false

      @conv_q = ShortConv.new(k_dim, @conv_kernel)
      @conv_k = ShortConv.new(k_dim, @conv_kernel)
      @conv_v = ShortConv.new(v_dim, @conv_kernel)

      @state = Array(SimpleMatrix?).new(@num_v_heads, nil)
      @conv_state_q = nil
      @conv_state_k = nil
      @conv_state_v = nil
    end

    getter conv_q : ShortConv
    getter conv_k : ShortConv
    getter conv_v : ShortConv

    @state : Array(SimpleMatrix?)
    @conv_state_q : SimpleMatrix?
    @conv_state_k : SimpleMatrix?
    @conv_state_v : SimpleMatrix?

    # Drop all recurrent state, so the next forward starts a fresh sequence.
    #
    # The analogue of clearing a KV cache, and it must clear the conv windows TOO: leaving them
    # behind would let the first positions of a new sequence see the tail of the previous one,
    # which is a silent correctness bug rather than a visible failure.
    def clear_cache!
      @num_v_heads.times { |h| @state[h] = nil }
      @conv_state_q = nil
      @conv_state_k = nil
      @conv_state_v = nil
      # The device state must go too, or a new sequence would continue from the previous one's
      # accumulated state -- silently wrong output rather than a visible failure, and exactly the
      # bug the conv-window clearing above exists to prevent.
      @dev_state.try(&.free!)
      @dev_state = nil
      # ...and so must the DEVICE conv windows, which are a separate set of buffers from the three
      # host ones above. Missing them left exactly the hole this method exists to close, on the only
      # path that actually runs when the stack is GPU-resident: clearing the host windows while the
      # device kept its own. Symptom was a second prompt in the same process decoding an endless run
      # of role tags instead of an answer, while the identical prompt in a fresh process answered
      # correctly.
      #
      # Zeroed and re-uploaded rather than freed, because the shapes do not change between sequences
      # and reallocating per sequence would churn device memory for nothing.
      @conv_state_devs.each_value do |m|
        m.zero!
        m.mark_host_modified!
        m.sync_to_device!("gdn_conv_state_clear")
      end
    end

    # One saved copy of the recurrent state, for rewinding a sequence to the point it was taken.
    #
    # Recurrent state cannot be truncated the way a KV cache can: every token is folded into it, so
    # the only way back to an earlier position is a copy made at that position. Device state is copied
    # to HOST memory, so the copy costs no VRAM, and the buffers are reused between saves.
    @saved = false
    @saved_state = [] of SimpleMatrix?
    @saved_conv_q : SimpleMatrix?
    @saved_conv_k : SimpleMatrix?
    @saved_conv_v : SimpleMatrix?
    @saved_dev_state : Bytes?
    @saved_dev_present = false
    @saved_conv_devs = {} of Symbol => Bytes

    # Save the current recurrent state (host and device) into this block's single save slot.
    def save_state!
      @saved_state = @state.map(&.try(&.clone))
      @saved_conv_q = @conv_state_q.try(&.clone)
      @saved_conv_k = @conv_state_k.try(&.clone)
      @saved_conv_v = @conv_state_v.try(&.clone)
      if m = @dev_state
        @saved_dev_state = GatedDeltaNetBlock.device_to_host(m, @saved_dev_state)
        @saved_dev_present = true
      else
        @saved_dev_present = false
      end
      @saved_conv_devs.select! { |role, _| @conv_state_devs.has_key?(role) }
      @conv_state_devs.each do |role, m|
        @saved_conv_devs[role] = GatedDeltaNetBlock.device_to_host(m, @saved_conv_devs[role]?)
      end
      @saved = true
    end

    # Replace the current state with the saved one. Returns false, changing nothing, when nothing was
    # saved. The slot is kept, so the same point can be restored again.
    def restore_saved_state! : Bool
      return false unless @saved
      @num_v_heads.times { |h| @state[h] = @saved_state[h]?.try(&.clone) }
      @conv_state_q = @saved_conv_q.try(&.clone)
      @conv_state_k = @saved_conv_k.try(&.clone)
      @conv_state_v = @saved_conv_v.try(&.clone)
      # Device buffers are written in place: same shapes every time, so no reallocation per request.
      if @saved_dev_present && (bytes = @saved_dev_state)
        GatedDeltaNetBlock.host_to_device(device_state, bytes)
      else
        @dev_state.try(&.free!)
        @dev_state = nil
      end
      @conv_state_devs.each do |role, m|
        if bytes = @saved_conv_devs[role]?
          GatedDeltaNetBlock.host_to_device(m, bytes)
        else
          # Created after the save, so at the saved point it did not exist yet: fresh means zero.
          m.zero!
          m.mark_host_modified!
          m.sync_to_device!("gdn_conv_state_restore")
        end
      end
      true
    end

    # Forget the saved state.
    def drop_saved_state!
      @saved = false
    end

    # Copy a device matrix's float32 contents into `buf` (reused when it is the right size).
    protected def self.device_to_host(m : CudaMatrix, buf : Bytes?) : Bytes
      size = m.rows.to_i64 * m.cols * 4
      buf = Bytes.new(size) unless buf && buf.size == size
      ptr = m.device_ptr || raise "GDN state has no device buffer"
      r = CUDA.memcpy(buf.to_unsafe.as(Pointer(Void)), ptr.as(Pointer(Void)), size.to_u64, CUDA::MemcpyKind::DeviceToHost)
      raise "GDN state copy to host failed (#{r})" unless r.nil? || r == 0
      buf
    end

    protected def self.host_to_device(m : CudaMatrix, buf : Bytes) : Nil
      size = m.rows.to_i64 * m.cols * 4
      raise "GDN state size changed (#{buf.size} saved, #{size} now)" unless buf.size == size
      ptr = m.device_ptr || raise "GDN state has no device buffer"
      r = CUDA.memcpy(ptr.as(Pointer(Void)), buf.to_unsafe.as(Pointer(Void)), size.to_u64, CUDA::MemcpyKind::HostToDevice)
      raise "GDN state copy to device failed (#{r})" unless r.nil? || r == 0
      # The device copy is now the newer one; the host mirror is stale.
      m.mark_device_dirty!
    end

    # Bytes of recurrent state held, for comparison against a KV cache.
    def state_bytes : Int64
      per_head = (@head_v * @head_k * 4).to_i64
      conv = ((@num_k_heads * @head_k * 2 + @num_v_heads * @head_v) * (@conv_kernel - 1) * 4).to_i64
      per_head * @num_v_heads + conv
    end

    # SiLU, x * sigmoid(x).
    private def silu(x : Float64) : Float64
      x / (1.0 + Math.exp(-x))
    end

    private def softplus(x : Float64) : Float64
      # log1p(exp(x)) guarded for large x, where exp overflows and the function is ~x.
      x > 20.0 ? x : Math.log1p(Math.exp(x))
    end

    # Per-head alpha and beta for each position.
    #
    # Mamba2's parameterization, which the paper adopts without restating:
    #
    #   dt    = softplus(x W_alpha + dt_bias)
    #   alpha = exp(-exp(a_log) * dt)      in (0,1) for any real a_log, dt > 0
    #   beta  = sigmoid(x W_beta)          in (0,1)
    #
    # Both ranges are structural rather than clamped, which matters: an alpha at or above 1
    # would let the state grow without bound over a long context, and a beta outside (0,1) would
    # break the delta rule's interpretation as a blend of old and new.
    def gates(normed : SimpleMatrix) : {Array(Array(Float64)), Array(Array(Float64))}
      seq = normed.rows
      # Two GEMMs, then elementwise. Previously a triple loop over seq x heads x d_model, which at a
      # 1216-token prefill was 0.328 s per layer -- 47% of the mixer once the recurrence moved to the
      # device -- for 159M iterations that cuBLAS does in milliseconds.
      a_proj = gpu_matmul(normed, @w_alpha)
      b_proj = gpu_matmul(normed, @w_beta)

      alpha = Array(Array(Float64)).new(@num_v_heads) { Array(Float64).new(seq, 0.0) }
      beta = Array(Array(Float64)).new(@num_v_heads) { Array(Float64).new(seq, 0.0) }
      @num_v_heads.times do |h|
        decay = Math.exp(@a_log[h])
        bias = @dt_bias[h]
        ah = alpha[h]
        bh = beta[h]
        seq.times do |t|
          ah[t] = Math.exp(-decay * softplus(a_proj[t, h].to_f64 + bias))
          bh[t] = 1.0 / (1.0 + Math.exp(-b_proj[t, h].to_f64))
        end
      end

      {alpha, beta}
    end

    # The key head value head `h` reads. See k_head_tiled for why this depends on the loader.
    def k_head_for(h : Int32, heads_per_k : Int32) : Int32
      @k_head_tiled ? h % @num_k_heads : h // heads_per_k
    end

    # The whole mixer on the device: one upload in, one download out.
    #
    # The host path runs each projection through gpu_matmul, which syncs the result back after every
    # one. At 288 matmuls per decode token those syncs measured 64 ms -- more than the matmul
    # kernels themselves -- and the elementwise stages between them (conv, SiLU, the gates, the
    # per-head norm, the output gate) each walked a SimpleMatrix element by element in Crystal.
    #
    # Returns nil when anything it needs is missing, so the host path above stays the reference
    # implementation and a CPU-only build keeps working. SHAINET_GDN_RESIDENT=0 forces the host
    # path, which is how a spec A/Bs the two in one process.
    private def mix_resident(normed : SimpleMatrix) : SimpleMatrix?
      # Check BEFORE touching resident_buf: a CudaMatrix cannot be constructed without CUDA, so
      # allocating first raised instead of declining on a CPU-only build.
      return unless mixer_resident_capable?
      seq = normed.rows
      xd = resident_buf(:x, seq, @d_model)
      xd.raw_data.to_unsafe.copy_from(normed.data.to_unsafe, seq * @d_model)
      xd.mark_host_modified!
      xd.sync_to_device!("gdn_res_in")
      rd = mix_resident_device(xd)
      return unless rd
      rd.sync_from_device!("gdn_res_out") if rd.device_dirty?
      out = SimpleMatrix.new(seq, @d_model)
      out.data.to_unsafe.copy_from(rd.raw_data.to_unsafe, seq * @d_model)
      out
    end

    # Can the mixer run entirely on the device? Checked separately so forward_resident can decide
    # before it commits to the device chain.
    private def mixer_resident_capable? : Bool
      return false unless @@resident_enabled
      return false unless CUDA.fully_available? && CUDA.gated_delta_rule_available?
      return false unless CUDA.gdn_mixer_kernels_available?
      @w_q.is_a?(QuantizedWeight) && @w_k.is_a?(QuantizedWeight) &&
        @w_v.is_a?(QuantizedWeight) && @w_gate.is_a?(QuantizedWeight) &&
        @w_o.is_a?(QuantizedWeight) &&
        @w_alpha.is_a?(CudaMatrix) && @w_beta.is_a?(CudaMatrix)
    end

    # The mixer with a device input and a device output, so a caller already on the device does not
    # round-trip through the host to reach it.
    private def mix_resident_device(xd : CudaMatrix) : CudaMatrix?
      return unless mixer_resident_capable?
      wq = @w_q.as(QuantizedWeight)
      wk = @w_k.as(QuantizedWeight)
      wv = @w_v.as(QuantizedWeight)
      wg = @w_gate.as(QuantizedWeight)
      wo = @w_o.as(QuantizedWeight)
      wa = @w_alpha.as(CudaMatrix)
      wb = @w_beta.as(CudaMatrix)

      seq = xd.rows
      k_dim = @num_k_heads * @head_k
      v_dim = @num_v_heads * @head_v

      qd = resident_buf(:q, seq, k_dim)
      kd = resident_buf(:k, seq, k_dim)
      vd = resident_buf(:v, seq, v_dim)
      gd = resident_buf(:gd, seq, v_dim)
      # All four projections read xd, so they share one int8 quantization of it. The output gate
      # (gd) is only consumed at the end of the mixer, but computing it here is the same product
      # into its own buffer, and it lets the share cover four GEMMs instead of three.
      GGUFMatrix.with_shared_input(xd) do
        wq.gemv_into(xd, qd)
        wk.gemv_into(xd, kd)
        wv.gemv_into(xd, vd)
        wg.gemv_into(xd, gd)
      end

      qc = resident_buf(:qc, seq, k_dim)
      kc = resident_buf(:kc, seq, k_dim)
      vc = resident_buf(:vc, seq, v_dim)
      # Nine launches (three conv, three state rolls, three SiLU) collapse to two when the fused
      # kernel is present. SiLU is written in mul_sigmoid's exact form inside the conv, so the
      # numbers do not move; the unfused path below stays as the fallback.
      fused = CUDA.short_conv_silu3(
        qc.device_ptr.not_nil!, qd.device_ptr.not_nil!,
        conv_state_dev(:q, k_dim), conv_weight_dev(:q, @conv_q),
        kc.device_ptr.not_nil!, kd.device_ptr.not_nil!,
        conv_state_dev(:k, k_dim), conv_weight_dev(:k, @conv_k),
        vc.device_ptr.not_nil!, vd.device_ptr.not_nil!,
        conv_state_dev(:v, v_dim), conv_weight_dev(:v, @conv_v),
        seq, k_dim, k_dim, v_dim, @conv_kernel, true)
      unless fused
        CUDA.short_conv(qc.device_ptr.not_nil!, qd.device_ptr.not_nil!,
          conv_state_dev(:q, k_dim), conv_weight_dev(:q, @conv_q), seq, k_dim, @conv_kernel)
        CUDA.short_conv(kc.device_ptr.not_nil!, kd.device_ptr.not_nil!,
          conv_state_dev(:k, k_dim), conv_weight_dev(:k, @conv_k), seq, k_dim, @conv_kernel)
        CUDA.short_conv(vc.device_ptr.not_nil!, vd.device_ptr.not_nil!,
          conv_state_dev(:v, v_dim), conv_weight_dev(:v, @conv_v), seq, v_dim, @conv_kernel)
        # mul_sigmoid(x, x) is x * sigmoid(x), i.e. SiLU in place.
        CUDA.mul_sigmoid(qc.device_ptr.not_nil!, qc.device_ptr.not_nil!, seq * k_dim)
        CUDA.mul_sigmoid(kc.device_ptr.not_nil!, kc.device_ptr.not_nil!, seq * k_dim)
        CUDA.mul_sigmoid(vc.device_ptr.not_nil!, vc.device_ptr.not_nil!, seq * v_dim)
      end
      qc.mark_device_dirty!
      kc.mark_device_dirty!
      vc.mark_device_dirty!

      ap = resident_buf(:ap, seq, @num_v_heads)
      bp = resident_buf(:bp, seq, @num_v_heads)
      # One token: a dedicated kernel computes both projections in one pass. cuBLAS takes ~31 us
      # per call at this shape, 96 calls a token across the stack.
      ab_done = seq == 1 && CUDA.gdn_ab_proj(xd.device_ptr.not_nil!, wa.device_ptr.not_nil!,
        wb.device_ptr.not_nil!, ap.device_ptr.not_nil!, bp.device_ptr.not_nil!, @d_model, @num_v_heads)
      unless ab_done
        h = GatedDeltaNetBlock.shared_cublas
        # ap is row-major [seq, heads], i.e. column-major [heads, seq]; w_alpha is row-major
        # [d_model, heads], i.e. column-major [heads, d_model]. So this is a plain N,N product
        # W'(heads x d_model) * x'(d_model x seq) -- NO transpose. Using the transposed form here
        # silently mixed the head and model axes and moved the <|im_start|> logit by 4%.
        CUDA.gemm(h, wa.device_ptr.not_nil!, xd.device_ptr.not_nil!,
          ap.device_ptr.not_nil!, @num_v_heads, seq, @d_model, @num_v_heads, @d_model, @num_v_heads)
        CUDA.gemm(h, wb.device_ptr.not_nil!, xd.device_ptr.not_nil!,
          bp.device_ptr.not_nil!, @num_v_heads, seq, @d_model, @num_v_heads, @d_model, @num_v_heads)
      end
      ad = resident_buf(:ad, seq, @num_v_heads)
      bd = resident_buf(:bd, seq, @num_v_heads)
      CUDA.gdn_gates(ad.device_ptr.not_nil!, bd.device_ptr.not_nil!,
        ap.device_ptr.not_nil!, bp.device_ptr.not_nil!,
        gate_param_dev(:a_log, @a_log), gate_param_dev(:dt_bias, @dt_bias),
        seq, @num_v_heads)

      od = resident_buf(:od, seq, v_dim)
      CUDA.gated_delta_rule(
        qc.device_ptr.not_nil!, kc.device_ptr.not_nil!, vc.device_ptr.not_nil!,
        ad.device_ptr.not_nil!, bd.device_ptr.not_nil!,
        device_state.device_ptr.not_nil!, od.device_ptr.not_nil!,
        seq, @num_v_heads, @num_k_heads, @head_k, @head_v, @num_v_heads // @num_k_heads,
        (1.0 / Math.sqrt(@head_k.to_f64)).to_f32, @k_head_tiled)

      # Per-head RMS norm with the shared [head_v] weight, in place.
      CUDA.head_rmsnorm_rows(od.device_ptr.not_nil!, ssm_gamma_dev, seq, @num_v_heads,
        @head_v, @eps.to_f32)

      # swiglu_forward(dst, gate, up) = silu(gate) * up, which is the output gate exactly.
      gated = resident_buf(:gated, seq, v_dim)
      CUDA.swiglu_forward(gated.device_ptr.not_nil!, gd.device_ptr.not_nil!,
        od.device_ptr.not_nil!, seq * v_dim)
      gated.mark_device_dirty!

      rd = resident_buf(:rd, seq, @d_model)
      wo.gemv_into(gated, rd)
      rd
    end

    # Reusable device buffers, one per role, SHARED across every block.
    #
    # Per-block buffers OOM'd: at a 301-token prefill one block's set is ~40 MB, and 40 device
    # blocks wanted 1.6 GB on a card with ~1.5 GB spare. The blocks run strictly one at a time, so
    # one set serves all of them, exactly like the GEMM dequant scratch.
    #
    # Keyed by ROLE ONLY, and reallocated when the shape changes. Keying by {role, rows, cols}
    # instead kept a full set per distinct sequence length, which is unbounded for a caller whose
    # prompts vary: after one 301-token prefill and one 1400-token prefill a 1400-token run OOM'd
    # even with a 2048 MB reserve. A generation does one prefill shape then a steady rows=1 shape,
    # so eviction costs two reallocations and then nothing.
    @@res_bufs = {} of Symbol => CudaMatrix

    private def resident_buf(role : Symbol, rows : Int32, cols : Int32) : CudaMatrix
      if existing = @@res_bufs[role]?
        return existing if existing.rows == rows && existing.cols == cols
        existing.free!
      end
      @@res_bufs[role] = CudaMatrix.new(rows, cols)
    end

    def self.release_resident_buffers!
      @@res_bufs.each_value(&.free!)
      @@res_bufs.clear
    end

    @conv_state_devs = {} of Symbol => CudaMatrix
    @conv_weight_devs = {} of Symbol => CudaMatrix
    @gate_param_devs = {} of Symbol => CudaMatrix
    @ssm_gamma_dev : CudaMatrix?
    @eps : Float64 = 1e-6

    # ONE cuBLAS handle for every block. A handle carries its own workspace, so the per-block
    # version of this allocated 40+ of them and cublasCreate started failing outright once the
    # layer split left less VRAM spare -- at a 1024 MB reserve it died on the first mixer call.
    @@cublas : CUDA::LibCUBLAS::Handle?

    protected def self.shared_cublas : CUDA::LibCUBLAS::Handle
      @@cublas ||= CUDA.create_handle
    end

    private def conv_state_dev(role : Symbol, channels : Int32) : Pointer(Float32)
      m = @conv_state_devs[role] ||= begin
        c = CudaMatrix.new(channels, @conv_kernel - 1)
        c.zero!
        c.sync_to_device!("gdn_conv_state_init")
        c
      end
      m.device_ptr.not_nil!
    end

    private def conv_weight_dev(role : Symbol, conv : ShortConv) : Pointer(Float32)
      m = @conv_weight_devs[role] ||= begin
        w = CudaMatrix.new(conv.channels, conv.kernel)
        conv.channels.times { |c| conv.kernel.times { |j| w[c, j] = conv.weight[c, j] } }
        w.mark_host_modified!
        w.sync_to_device!("gdn_conv_w")
        w
      end
      m.device_ptr.not_nil!
    end

    private def gate_param_dev(role : Symbol, src : Array(Float64)) : Pointer(Float32)
      m = @gate_param_devs[role] ||= begin
        v = CudaMatrix.new(1, src.size)
        src.each_with_index { |x, i| v[0, i] = x.to_f32 }
        v.mark_host_modified!
        v.sync_to_device!("gdn_gate_param")
        v
      end
      m.device_ptr.not_nil!
    end

    private def ssm_gamma_dev : Pointer(Float32)
      m = @ssm_gamma_dev ||= begin
        g = @out_norm.gamma
        v = CudaMatrix.new(1, g.cols)
        g.cols.times { |i| v[0, i] = g[0, i] }
        v.mark_host_modified!
        v.sync_to_device!("gdn_ssm_gamma")
        v
      end
      m.device_ptr.not_nil!
    end

    # The mixer: normed input in, [seq, d_model] out.
    #
    # `chunk` selects the operator form. Above 1 it uses the chunked parallel path for prefill;
    # at 1 it is the sequential recurrence, which is all decode can use anyway. Both produce the
    # same numbers, which is specified rather than assumed.
    def mix(normed : SimpleMatrix, chunk : Int32 = 64) : SimpleMatrix
      if res = Profile.measure("gdn.resident") { mix_resident(normed) }
        return res
      end

      seq = normed.rows
      k_dim = @num_k_heads * @head_k
      v_dim = @num_v_heads * @head_v

      q_lin = Profile.measure("gdn.project") { project(normed, @w_q, k_dim) }
      k_lin = Profile.measure("gdn.project") { project(normed, @w_k, k_dim) }
      v_lin = Profile.measure("gdn.project") { project(normed, @w_v, v_dim) }

      q_c = k_c = v_c = uninitialized SimpleMatrix
      Profile.measure("gdn.conv") do
        q_c, @conv_state_q = @conv_q.forward(q_lin, @conv_state_q)
        k_c, @conv_state_k = @conv_k.forward(k_lin, @conv_state_k)
        v_c, @conv_state_v = @conv_v.forward(v_lin, @conv_state_v)
      end

      Profile.measure("gdn.silu") do
        seq.times do |t|
          k_dim.times { |j| q_c[t, j] = silu(q_c[t, j].to_f64); k_c[t, j] = silu(k_c[t, j].to_f64) }
          v_dim.times { |j| v_c[t, j] = silu(v_c[t, j].to_f64) }
        end
      end

      alpha, beta = Profile.measure("gdn.gates") { gates(normed) }
      heads_per_k = @num_v_heads // @num_k_heads

      if dev = Profile.measure("gdn.recurrence_dev") { device_mix(q_c, k_c, v_c, alpha, beta, seq, k_dim, v_dim, heads_per_k) }
        mixed = dev
      else
        mixed = SimpleMatrix.new(seq, v_dim, 0.0)
        @num_v_heads.times do |h|
          kh = k_head_for(h, heads_per_k)
          qh = SimpleMatrix.new(seq, @head_k, 0.0)
          khm = SimpleMatrix.new(seq, @head_k, 0.0)
          vh = SimpleMatrix.new(seq, @head_v, 0.0)
          seq.times do |t|
            @head_k.times do |j|
              qh[t, j] = q_c[t, kh * @head_k + j]
              khm[t, j] = k_c[t, kh * @head_k + j]
            end
            @head_v.times { |j| vh[t, j] = v_c[t, h * @head_v + j] }
          end
          # L2 on q and k only, per the block design. The paper's ablation calls it essential, and
          # the recurrence's stability depends on ||k|| = 1.
          GatedDeltaNet.l2_normalize!(qh)
          GatedDeltaNet.l2_normalize!(khm)
          # Then scale q by 1/sqrt(head_k), as the reference kernel does unconditionally:
          #
          #   query = query / (query.shape[-1] ** 0.5)
          #
          # This is NOT cosmetic and it is NOT removed by the output norm, which is what I first
          # assumed. RMSNorm is scale-invariant only while the variance dominates its epsilon, and
          # here it does not: with the scale applied the core output's variance is ~1.9e-6 against an
          # eps of 1e-6, so eps is about half of it. Dropping the scale makes the state 128x larger in
          # variance, moves it out of the epsilon-dominated regime the trained weights expect, and
          # changes every value the norm produces.
          #
          # Measured against a numpy transcription of the reference on the real checkpoint, layer 0:
          # core rms 0.015739 without the scale against 0.001391 with it, a ratio of 11.31 = sqrt(128),
          # which is how the missing factor was identified.
          inv = 1.0 / Math.sqrt(@head_k.to_f64)
          seq.times { |t| @head_k.times { |j| qh[t, j] = qh[t, j].to_f64 * inv } }

          out_h, st = if chunk > 1
                        GatedDeltaNet.chunked(qh, khm, vh, alpha[h], beta[h], @state[h], chunk: chunk)
                      else
                        GatedDeltaNet.recurrent(qh, khm, vh, alpha[h], beta[h], @state[h])
                      end
          @state[h] = st
          seq.times { |t| @head_v.times { |j| mixed[t, h * @head_v + j] = out_h[t, j] } }
        end
      end

      # Output norm PER HEAD, then the SiLU gate, then project back to d_model.
      #
      # Each head's head_v slice is normalized on its own with the shared weight, matching the
      # checkpoint's [head_v] norm tensor. Doing it over the whole v_dim would make one head's
      # magnitude affect every other head's output.
      normed_mix = SimpleMatrix.new(seq, v_dim, 0.0)
      Profile.measure("gdn.out_norm") do
        @num_v_heads.times do |h|
          slice = SimpleMatrix.new(seq, @head_v, 0.0)
          seq.times { |t| @head_v.times { |j| slice[t, j] = mixed[t, h * @head_v + j] } }
          normed_slice = @out_norm.forward(slice)
          seq.times { |t| @head_v.times { |j| normed_mix[t, h * @head_v + j] = normed_slice[t, j] } }
        end
      end

      gate = Profile.measure("gdn.project") { project(normed, @w_gate, v_dim) }
      Profile.measure("gdn.gate_mul") do
        seq.times do |t|
          v_dim.times { |j| normed_mix[t, j] = normed_mix[t, j].to_f64 * silu(gate[t, j].to_f64) }
        end
      end
      Profile.measure("gdn.out_proj") { project(normed_mix, @w_o, @d_model) }
    end

    # Move the block's projections to the device, optionally quantized.
    #
    # The mixer's recurrence stays on the host deliberately. Per token it is ~2M FLOP (32 heads
    # over a 128x128 state) against ~600M for the projections, so a device kernel for it would
    # chase 0.3% of the work while the GEMMs are what make the block unusable in fp32. Quantizing
    # the projections is also what lets a hybrid stack load at all: they are 100 MB per layer
    # against 0.5 MB for everything else here.
    def to_gpu!(quantize : Bool = false, bits : Int32 = 8, offload : Bool = false)
      return unless CUDA.fully_available?
      if quantize
        @w_q = to_quant(@w_q, bits, offload)
        @w_k = to_quant(@w_k, bits, offload)
        @w_v = to_quant(@w_v, bits, offload)
        @w_o = to_quant(@w_o, bits, offload)
        @w_gate = to_quant(@w_gate, bits, offload)
      else
        raise ArgumentError.new("dense offload requires quantization (offload is Q4-only)") if offload
        @w_q = @w_q.as(SimpleMatrix).to_cuda if @w_q.is_a?(SimpleMatrix)
        @w_k = @w_k.as(SimpleMatrix).to_cuda if @w_k.is_a?(SimpleMatrix)
        @w_v = @w_v.as(SimpleMatrix).to_cuda if @w_v.is_a?(SimpleMatrix)
        @w_o = @w_o.as(SimpleMatrix).to_cuda if @w_o.is_a?(SimpleMatrix)
        @w_gate = @w_gate.as(SimpleMatrix).to_cuda if @w_gate.is_a?(SimpleMatrix)
      end
      # fp32 on the device, never quantized -- see the property comment.
      @w_alpha = @w_alpha.as(SimpleMatrix).to_cuda if @w_alpha.is_a?(SimpleMatrix)
      @w_beta = @w_beta.as(SimpleMatrix).to_cuda if @w_beta.is_a?(SimpleMatrix)
      @norm1.to_gpu!
      @norm2.to_gpu!
      @out_norm.to_gpu!
      @ffn.to_gpu!(quantize, bits, offload)
    end

    # True once the projections are quantized, so a caller can tell a loaded-and-quantized block
    # from one still holding fp32 without inspecting weight types itself.
    def quantized? : Bool
      @w_q.is_a?(QuantizedWeight) && @w_k.is_a?(QuantizedWeight) && @w_v.is_a?(QuantizedWeight) &&
        @w_o.is_a?(QuantizedWeight) && @w_gate.is_a?(QuantizedWeight)
    end

    # The recurrence on the device: one kernel launch for the whole sequence, all heads.
    #
    # Returns nil when the kernel is unavailable, so the host path above stays the reference
    # implementation and the CPU-only build keeps working. The two are pinned equivalent by spec.
    #
    # The kernel does the per-head slicing, the L2 normalization and the q scale itself, so nothing
    # here reshapes per head -- that copying cost as much as the arithmetic it fed. What remains is
    # one upload of q/k/v and the gates, one download of the output, and the state kept on the
    # device across calls so a decode step does not re-upload it.
    private def device_mix(q_c : SimpleMatrix, k_c : SimpleMatrix, v_c : SimpleMatrix,
                           alpha : Array(Array(Float64)), beta : Array(Array(Float64)),
                           seq : Int32, k_dim : Int32, v_dim : Int32,
                           heads_per_k : Int32) : SimpleMatrix?
      # SHAINET_GDN_DEVICE=0 forces the host path. Read per call rather than memoized so a spec can
      # A/B the two within one process, which is how their equivalence is asserted.
      return unless @@device_enabled
      return unless CUDA.gated_delta_rule_available?

      qd = CudaMatrix.new(seq, k_dim)
      kd = CudaMatrix.new(seq, k_dim)
      vd = CudaMatrix.new(seq, v_dim)
      ad = CudaMatrix.new(seq, @num_v_heads)
      bd = CudaMatrix.new(seq, @num_v_heads)
      od = CudaMatrix.new(seq, v_dim)
      begin
        qd.raw_data.to_unsafe.copy_from(q_c.data.to_unsafe, seq * k_dim)
        kd.raw_data.to_unsafe.copy_from(k_c.data.to_unsafe, seq * k_dim)
        vd.raw_data.to_unsafe.copy_from(v_c.data.to_unsafe, seq * v_dim)
        seq.times do |t|
          @num_v_heads.times do |h|
            ad[t, h] = alpha[h][t]
            bd[t, h] = beta[h][t]
          end
        end
        qd.sync_to_device!("gdn_q")
        kd.sync_to_device!("gdn_k")
        vd.sync_to_device!("gdn_v")
        ad.sync_to_device!("gdn_alpha")
        bd.sync_to_device!("gdn_beta")

        st = device_state
        CUDA.gated_delta_rule(
          qd.device_ptr.not_nil!, kd.device_ptr.not_nil!, vd.device_ptr.not_nil!,
          ad.device_ptr.not_nil!, bd.device_ptr.not_nil!,
          st.device_ptr.not_nil!, od.device_ptr.not_nil!,
          seq, @num_v_heads, @num_k_heads, @head_k, @head_v, heads_per_k,
          (1.0 / Math.sqrt(@head_k.to_f64)).to_f32, @k_head_tiled)
        od.mark_device_dirty!
        od.sync_from_device!("gdn_out")

        result = SimpleMatrix.new(seq, v_dim)
        result.data.to_unsafe.copy_from(od.raw_data.to_unsafe, seq * v_dim)
        result
      ensure
        qd.free!
        kd.free!
        vd.free!
        ad.free!
        bd.free!
        od.free!
      end
    end

    # The [num_v_heads, head_k, head_v] recurrent state, resident on the device.
    #
    # Allocated zeroed on first use, which is exactly the fresh-sequence state, and reused across
    # calls so a decode step carries the state without a round trip. clear_cache! frees it, so a new
    # sequence starts from zeros again rather than from the previous one's tail.
    private def device_state : CudaMatrix
      st = @dev_state
      return st if st
      st = CudaMatrix.new(@num_v_heads * @head_k, @head_v)
      st.zero!
      st.sync_to_device!("gdn_state_init")
      @dev_state = st
    end

    @dev_state : CudaMatrix?

    private def project(x : SimpleMatrix, w : SimpleMatrix | CudaMatrix | QuantizedWeight, out_dim : Int32) : SimpleMatrix
      dst = gpu_matmul(x, w)
      raise "projection produced #{dst.cols} columns, expected #{out_dim}" unless dst.cols == out_dim
      dst
    end

    def forward(x : SimpleMatrix) : SimpleMatrix
      if r = Profile.measure("gdn.block_resident") { forward_resident(x) }
        return r
      end
      n1 = Profile.measure("gdn.norm") { @norm1.forward(x) }
      mixed = mix(n1)
      h = Profile.measure("gdn.residual") { x + mixed }
      n2 = Profile.measure("gdn.norm") { @norm2.forward(h) }
      ff = ffn_forward(n2)
      Profile.measure("gdn.residual") { h + ff }
    end

    # Single-token step, for decode. Same math, chunk 1.
    def forward_cached(x : SimpleMatrix) : SimpleMatrix
      if r = Profile.measure("gdn.block_resident") { forward_resident(x) }
        return r
      end
      n1 = Profile.measure("gdn.norm") { @norm1.forward(x) }
      mixed = mix(n1, chunk: 1)
      h = Profile.measure("gdn.residual") { x + mixed }
      n2 = Profile.measure("gdn.norm") { @norm2.forward(h) }
      ff = ffn_forward(n2)
      Profile.measure("gdn.residual") { h + ff }
    end

    # The whole block device-in, device-out, so the network's device chain does not have to break
    # at a linear-attention layer.
    #
    # Before this the chain ended at every Gated DeltaNet block: the activation came home, the block
    # ran, and the next block uploaded it again. In a hybrid stack that is 48 of 64 layers, so the
    # activation crossed the bus about 96 times per token on top of the mixer's own transfers.
    #
    # Returns nil unless every stage can run on the device, so the caller falls back to the host
    # path and that stays the reference implementation.
    # Whether the network's device chain can pass through this block without coming home.
    def device_chain_capable? : Bool
      return false unless mixer_resident_capable?
      ffn = @ffn
      return false unless ffn.is_a?(SwiGLUFF) && ffn.gate_proj.is_a?(QuantizedWeight)
      @norm1.gamma.is_a?(CudaMatrix) && @norm2.gamma.is_a?(CudaMatrix) && CUDA.kernels_available?
    end

    # How many tokens the resident path processes at once.
    #
    # Its workspace is about a dozen buffers scaled by the row count -- at 1400 rows roughly 630 MB
    # once the SwiGLU trio is counted -- so sizing it to the whole prompt made a long prefill fail
    # for want of VRAM and forced a larger reserve, which in turn kept layers on the host. The
    # recurrence already carries its conv and recurrent state across calls, which is exactly what a
    # decode step relies on, so a chunked prefill computes the same thing as a whole-sequence one.
    # Read ONCE: these gate capability checks that run for all 48 GDN blocks on every token, and a
    # Crystal ENV lookup is a getenv linear scan of the environment plus a String allocation.
    @@resident_enabled : Bool = ENV.fetch("SHAINET_GDN_RESIDENT", "1") != "0"
    @@device_enabled : Bool = ENV.fetch("SHAINET_GDN_DEVICE", "1") != "0"

    # Rows of prompt this block processes per pass during prefill.
    #
    # This is what sets M for every matmul in the block, and M is what amortizes the dequant: the
    # quantized weight is dequantized into fp32 scratch once per pass regardless of how many rows use
    # it, so a small chunk pays that cost over few rows. Measured on one real 5120x17408 IQ2_XXS
    # weight through gemm_into, dequant included: M=512 gives 12.8 TFLOP/s and M=1024 gives 19.3.
    #
    # It was a hardcoded 256, which silently capped M for 48 of the 64 layers and is why raising
    # Network.prefill_rows measured as doing nothing at all (512/1024/2048 all within noise) -- the
    # network-level chunk never reached the matmul.
    #
    # Bigger is faster but costs workspace VRAM, which is what caps context, so it is tunable rather
    # than simply raised: SHAINET_GDN_CHUNK.
    @@resident_chunk : Int32 = (ENV["SHAINET_GDN_CHUNK"]?.try(&.to_i?) || 256)

    def self.resident_chunk : Int32
      @@resident_chunk
    end

    def self.resident_chunk=(v : Int32)
      @@resident_chunk = v
    end

    def forward_cached_device(xd : CudaMatrix) : CudaMatrix?
      return unless device_chain_capable?
      seq = xd.rows
      return forward_cached_device_chunk(xd) if seq <= @@resident_chunk

      d = @d_model
      full = resident_buf(:blk_full, seq, d)
      off = 0
      while off < seq
        n = seq - off
        n = @@resident_chunk if n > @@resident_chunk
        src = resident_buf(:blk_slice, n, d)
        CUDA.copy_device_to_device(src.device_ptr.not_nil!,
          xd.device_ptr.not_nil! + off * d, (n.to_u64 * d * 4).to_u64)
        src.mark_device_dirty!
        got = forward_cached_device_chunk(src)
        return unless got
        CUDA.copy_device_to_device(full.device_ptr.not_nil! + off * d,
          got.device_ptr.not_nil!, (n.to_u64 * d * 4).to_u64)
        off += n
      end
      full.mark_device_dirty!
      full
    end

    # The whole block device-in, device-out for one chunk of at most RESIDENT_CHUNK rows.
    #
    # Before this the chain ended at every Gated DeltaNet block: the activation came home, the block
    # ran, and the next block uploaded it again. In a hybrid stack that is 48 of 64 layers, so the
    # activation crossed the bus about 96 times per token on top of the mixer's own transfers.
    #
    # Returns nil unless every stage can run on the device, so the caller falls back to the host
    # path and that stays the reference implementation.
    private def forward_cached_device_chunk(xd : CudaMatrix) : CudaMatrix?
      return unless mixer_resident_capable?
      ffn = @ffn
      return unless ffn.is_a?(SwiGLUFF) && ffn.gate_proj.is_a?(QuantizedWeight)
      g1 = @norm1.gamma
      g2 = @norm2.gamma
      return unless g1.is_a?(CudaMatrix) && g2.is_a?(CudaMatrix)
      return unless CUDA.kernels_available?

      seq = xd.rows
      d = @d_model
      eps = @eps.to_f32

      # The residual accumulates in a buffer this block owns: xd belongs to the caller's chain and
      # the next block still needs it intact until this one returns.
      acc = resident_buf(:blk_acc, seq, d)
      CUDA.copy_device_to_device(acc.device_ptr.not_nil!, xd.device_ptr.not_nil!,
        (seq.to_u64 * d * 4).to_u64)
      acc.mark_device_dirty!

      nd = resident_buf(:blk_n, seq, d)
      CUDA.rms_norm_forward(nd.device_ptr.not_nil!, acc.device_ptr.not_nil!,
        g1.device_ptr.not_nil!, seq, d, eps)
      nd.mark_device_dirty!

      md = mix_resident_device(nd)
      return unless md
      CUDA.add_inplace(acc.device_ptr.not_nil!, md.device_ptr.not_nil!, seq * d)

      CUDA.rms_norm_forward(nd.device_ptr.not_nil!, acc.device_ptr.not_nil!,
        g2.device_ptr.not_nil!, seq, d, eps)
      nd.mark_device_dirty!

      fd = resident_buf(:blk_ffn, seq, d)
      ffn.forward_device_batch(nd, fd)
      CUDA.add_inplace(acc.device_ptr.not_nil!, fd.device_ptr.not_nil!, seq * d)
      acc.mark_device_dirty!

      # The chain's next block reads this, so hand back a buffer it can keep: acc is reused on the
      # next call, so copy into a distinct output slot.
      outd = resident_buf(:blk_out, seq, d)
      CUDA.copy_device_to_device(outd.device_ptr.not_nil!, acc.device_ptr.not_nil!,
        (seq.to_u64 * d * 4).to_u64)
      outd.mark_device_dirty!
      outd
    end

    # The whole block on the device: one upload in, one download out.
    #
    # Even with the mixer resident, the block still crossed the bus four times per layer -- the
    # mixer uploaded and downloaded, then the FFN did it again -- and ran both RMS norms and both
    # residual adds element by element on the host. Chaining them on the device leaves one upload
    # and one download for the whole block.
    #
    # Returns nil unless every stage can run on the device, so the host path above stays the
    # reference implementation.
    private def forward_resident(x : SimpleMatrix) : SimpleMatrix?
      return unless mixer_resident_capable?
      ffn = @ffn
      return unless ffn.is_a?(SwiGLUFF) && ffn.gate_proj.is_a?(QuantizedWeight)
      g1 = @norm1.gamma
      g2 = @norm2.gamma
      return unless g1.is_a?(CudaMatrix) && g2.is_a?(CudaMatrix)
      return unless CUDA.kernels_available?

      seq = x.rows
      d = @d_model
      eps = @eps.to_f32

      xd = resident_buf(:blk_x, seq, d)
      xd.raw_data.to_unsafe.copy_from(x.data.to_unsafe, seq * d)
      xd.mark_host_modified!
      xd.sync_to_device!("gdn_blk_in")

      nd = resident_buf(:blk_n, seq, d)
      CUDA.rms_norm_forward(nd.device_ptr.not_nil!, xd.device_ptr.not_nil!,
        g1.device_ptr.not_nil!, seq, d, eps)
      nd.mark_device_dirty!

      md = mix_resident_device(nd)
      return unless md
      # residual: xd += mixer output
      CUDA.add_inplace(xd.device_ptr.not_nil!, md.device_ptr.not_nil!, seq * d)
      xd.mark_device_dirty!

      CUDA.rms_norm_forward(nd.device_ptr.not_nil!, xd.device_ptr.not_nil!,
        g2.device_ptr.not_nil!, seq, d, eps)
      nd.mark_device_dirty!

      fd = resident_buf(:blk_ffn, seq, d)
      ffn.forward_device_batch(nd, fd)
      CUDA.add_inplace(xd.device_ptr.not_nil!, fd.device_ptr.not_nil!, seq * d)
      xd.mark_device_dirty!

      xd.sync_from_device!("gdn_blk_out")
      out = SimpleMatrix.new(seq, d)
      out.data.to_unsafe.copy_from(xd.raw_data.to_unsafe, seq * d)
      out
    end

    # Run the FFN via the device batch path when the weights are quantized, avoiding the
    # host-side SwiGLU loop (9.6 s of a 103 s prefill on the 27B). The device path streams
    # each Q4 weight once, applies SiLU on-device via the CUDA kernel, and reads back the
    # result -- identical numerics, ~10x faster for the activation.
    private def ffn_forward(normed : SimpleMatrix) : SimpleMatrix
      ffn = @ffn
      return ffn.forward(normed) unless ffn.gate_proj.is_a?(QuantizedWeight) && CUDA.fully_available?

      n = normed.rows
      d = normed.cols
      out_cols = ffn.down_proj.as(QuantizedWeight).cols

      # Upload normed to device.
      xd = CudaMatrix.new(n, d)
      xd.raw_data.to_unsafe.copy_from(normed.data.to_unsafe, n * d)
      xd.mark_host_modified!
      xd.sync_to_device!("gdn_ffn_in")

      od = CudaMatrix.new(n, out_cols)
      ffn.forward_device_batch(xd, od)
      od.sync_from_device!("gdn_ffn_out")

      result = SimpleMatrix.new(n, out_cols)
      result.data.to_unsafe.copy_from(od.raw_data.to_unsafe, n * out_cols)
      xd.free!
      od.free!
      result
    end

    def backward(d_out : SimpleMatrix) : SimpleMatrix
      raise NotImplementedError.new("GatedDeltaNetBlock is inference-only for now")
    end

    def apply_gradients(lr : Float64)
      raise NotImplementedError.new("GatedDeltaNetBlock is inference-only for now")
    end
  end
end
