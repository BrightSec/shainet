require "../basic/matrix_layer"

module SHAInet
  # LLaMA-style transformer block with KV cache for efficient generation.
  # Supports Grouped Query Attention (GQA).
  class LlamaBlock < MatrixLayer
    include QuantizedProjection

    getter norm1 : RMSNorm
    getter norm2 : RMSNorm
    getter ffn : SwiGLUFF | MoEFF
    getter num_heads : Int32
    getter num_kv_heads : Int32
    getter head_dim : Int32
    getter d_model : Int32
    # Total width of the concatenated query heads = num_heads * head_dim. Equals
    # d_model for LLaMA/Qwen2, but Qwen3 uses an explicit head_dim so this can be
    # larger (e.g. 32*128=4096 with d_model=2048). w_q is [d_model, q_dim] and
    # w_o is [q_dim, d_model]; the per-head attention output has width q_dim.
    getter q_dim : Int32
    # Qwen3 QK-norm: per-head RMSNorm applied to Q and K (over head_dim) right
    # before RoPE. nil = disabled (the LLaMA/Qwen2 default — zero overhead).
    property q_norm : Array(Float32)?
    property k_norm : Array(Float32)?
    @qk_norm_eps : Float64 = 1e-6
    property rope_theta : Float64
    # Optional precomputed inverse frequencies (size head_dim/2). When set,
    # these override the default theta^(-2i/d) computation (used for LLaMA 3
    # rope_scaling). nil means use the default.
    property rope_freqs : Array(Float32)?

    # How many leading dimensions of each head RoPE rotates. nil means all of head_dim, which is
    # every architecture supported before Qwen3.5.
    #
    # Qwen3.5 sets partial_rotary_factor 0.25 on a head_dim of 256: only the first 64 dimensions
    # are rotated and the other 192 pass through unrotated. Rotating all 256 is not a small error
    # -- it applies position-dependent phases to 192 dimensions that were trained to carry
    # position-independent content, and it changes the inverse-frequency spacing for the 64 that
    # should be rotated. Measured effect: the 9B produced fluent-magnitude logits whose argmax was
    # uniformly rare tokens, with a perfectly healthy residual stream, so nothing but the output
    # text showed it.
    property rotary_dim : Int32?

    property w_q : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_k : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_v : SimpleMatrix | CudaMatrix | QuantizedWeight
    property w_o : SimpleMatrix | CudaMatrix | QuantizedWeight

    # Optional attention output gate, for Qwen3.5's "Gated Attention" full-attention layers.
    #
    # Those layers pack q and the gate into ONE q_proj of [2 * q_dim, d_model]: the checkpoint's
    # q_proj is [8192, 4096] where num_attention_heads * head_dim is only 4096. The gate
    # multiplies the attention output element-wise (through SiLU) before w_o projects it down,
    # mirroring what in_proj_z does in the linear-attention block.
    #
    # nil for every other architecture, where attention is ungated and this costs one nil check.
    property w_gate_attn : (SimpleMatrix | CudaMatrix | QuantizedWeight)?

    # Optional Q/K/V projection biases. Qwen2-style architectures add a bias to
    # the query/key/value projections; LLaMA/Mistral do not. Kept as host-side
    # fp32 vectors and added to the projection output before RoPE, so they work
    # identically on the fp32, CUDA, and Q8 weight paths. nil means "no bias"
    # (the LLaMA default — zero overhead, behaviour unchanged). Sizes: b_q is
    # d_model; b_k/b_v are num_kv_heads * head_dim. o_proj has no bias in Qwen2.
    property b_q : Array(Float32)?
    property b_k : Array(Float32)?
    property b_v : Array(Float32)?

    # KV cache: stored per kv_head as [seq_len, head_dim] growing matrices
    @k_cache : Array(Array(Float32)) # [num_kv_heads][seq_len * head_dim]
    @v_cache : Array(Array(Float32))
    @cache_len : Int32 = 0

    # GPU-resident mirror of the KV cache, laid out [num_kv_heads, capacity,
    # head_dim] per tensor. The CPU cache stays the source of truth: the GPU
    # copy is appended incrementally each token and fully re-uploaded from the
    # mirror whenever capacity grows. Only used when CUDA kernels are loaded.
    #
    # Stored as an untyped pointer because the element type is chosen at first
    # allocation: fp16 halves the cache's VRAM footprint, and at long context
    # the cache is the single largest consumer (0.375 MB per token for a 48-layer
    # 8-KV-head model in fp32). Arithmetic stays fp32 in both cases — the kernels
    # convert on load — so only the stored precision differs.
    @gpu_k_cache : Pointer(Void) = Pointer(Void).null
    @gpu_v_cache : Pointer(Void) = Pointer(Void).null
    @gpu_cache_cap : Int32 = 0
    # nil until the first device allocation decides the cache dtype.
    @kv_dtype : KVDtype? = nil
    # Context budget: when set the device cache is allocated once at exactly this
    # many positions rather than doubling into it. Resolved from
    # SHAINET_KV_MAX_CONTEXT on first read; the checked flag keeps a deliberate
    # nil from re-reading the environment on every forward pass.
    @kv_max_context : Int32? = nil
    @kv_max_context_checked : Bool = false
    # Persistent device scratch for the attention hot path. SHARED across all
    # blocks (class-level): blocks run sequentially, so one grow-only scratch set
    # serves every layer instead of each of N layers holding its own ~(heads×
    # seq×seq) workspace (which was ~N× redundant and exhausted VRAM on long
    # prefills). The KV cache above stays per-instance.
    @@gpu_staging : Pointer(Float32) = Pointer(Float32).null
    @@gpu_staging_cap : Int32 = 0
    @@gpu_attn_out : Pointer(Float32) = Pointer(Float32).null
    @@gpu_attn_out_cap : Int32 = 0
    @@gpu_attn_ws : Pointer(Float32) = Pointer(Float32).null
    @@gpu_attn_ws_cap : Int32 = 0
    @@staging_host : Array(Float32) = Array(Float32).new
    @gpu_attn_avail : Bool?
    # Force the CPU attention path even when CUDA is available (tests/fallback).
    property? force_cpu_attention : Bool = false

    # Ceiling on the attention workspace, in fp32 elements (256 MB). Prefill is
    # chunked so num_heads * chunk_tokens * total_len stays under this; without
    # chunking the workspace grows as num_heads * seq^2, which is the single
    # sharpest VRAM cliff on a long prompt.
    ATTN_WS_BUDGET_FLOATS = 64_i64 * 1024 * 1024
    # Upper bound on chunk size regardless of budget. Larger chunks stop paying
    # off once the per-launch work saturates the device.
    ATTN_CHUNK_MAX = 256

    def initialize(@d_model : Int32, @num_heads : Int32, ff_hidden : Int32,
                   eps : Float64 = 1e-6, @rope_theta : Float64 = 10000.0,
                   num_kv_heads : Int32? = nil, head_dim : Int32? = nil,
                   moe_experts : Int32? = nil, moe_top_k : Int32 = 8,
                   moe_norm_topk : Bool = true, moe_ff_hidden : Int32? = nil,
                   moe_offload : Bool = false, allocate : Bool = true)
      super(@d_model, SHAInet.none)
      # num_kv_heads defaults to num_heads (no grouped-query attention). It cannot
      # be written as `@num_kv_heads : Int32 = @num_heads` in the parameter list:
      # a default argument that reads another ivar assigned in the same list counts
      # as a use-before-initialize, which makes @num_heads nilable and makes the
      # whole constructor uninstantiable unless the caller passes num_kv_heads
      # explicitly. Every existing caller happened to pass it, so the default was
      # never exercised.
      @num_kv_heads = num_kv_heads || @num_heads
      raise ArgumentError.new("num_heads must be divisible by num_kv_heads") unless @num_kv_heads > 0 && @num_heads % @num_kv_heads == 0
      # head_dim defaults to d_model/num_heads (LLaMA/Qwen2). Qwen3 passes it
      # explicitly (e.g. 128), so q_dim = num_heads*head_dim may differ from d_model.
      if hd = head_dim
        raise ArgumentError.new("head_dim must be positive") unless hd > 0
        @head_dim = hd
      else
        raise ArgumentError.new("d_model must be divisible by num_heads") unless @d_model % @num_heads == 0
        @head_dim = @d_model // @num_heads
      end
      @q_dim = @num_heads * @head_dim
      kv_dim = @num_kv_heads * @head_dim
      @qk_norm_eps = eps
      @norm1 = RMSNorm.new(@d_model, eps)
      @norm2 = RMSNorm.new(@d_model, eps)
      # Mixture-of-Experts FFN (Qwen3-MoE) when moe_experts is given; otherwise a
      # single dense SwiGLU (LLaMA/Mistral/Qwen2/Qwen3-dense).
      @ffn = if ne = moe_experts
               MoEFF.new(@d_model, moe_ff_hidden || ff_hidden, ne, moe_top_k, moe_norm_topk, moe_offload)
             else
               SwiGLUFF.new(@d_model, ff_hidden, allocate: allocate)
             end
      if allocate
        @w_q = SimpleMatrix.new(@d_model, @q_dim)
        @w_k = SimpleMatrix.new(@d_model, kv_dim)
        @w_v = SimpleMatrix.new(@d_model, kv_dim)
        @w_o = SimpleMatrix.new(@q_dim, @d_model)
      else
        @w_q = SimpleMatrix.new(0, 0)
        @w_k = SimpleMatrix.new(0, 0)
        @w_v = SimpleMatrix.new(0, 0)
        @w_o = SimpleMatrix.new(0, 0)
      end
      @k_cache = Array.new(@num_kv_heads) { Array(Float32).new }
      @v_cache = Array.new(@num_kv_heads) { Array(Float32).new }
    end

    # Persistent single-row GEMV workspaces for decode (M=1), keyed by width.
    # Reused across tokens to avoid per-call cudaMalloc/cudaFree churn. Never
    # freed during inference, so they cannot be GC-collected mid-GEMM.

    def clear_cache!
      @k_cache.each(&.clear)
      @v_cache.each(&.clear)
      @cache_len = 0
      @host_kv_stale = false
      @host_cache_reserved = false
      # Device cache buffers are kept; stale positions are rewritten by the
      # append kernel before they ever become visible to attention.
    end

    # Positions currently held in the KV cache.
    def cached_length : Int32
      cached_positions
    end

    # Drop every cached position at or after `positions`, so the next forward continues from there.
    #
    # The device cache is append-only at @cache_len, so moving @cache_len back is the whole rewind:
    # the stale positions past it are overwritten before attention can see them, the same guarantee
    # clear_cache! relies on. The host mirror, when it is the authoritative copy, is cut to match.
    def truncate_cache!(positions : Int32)
      have = cached_positions
      unless 0 <= positions <= have
        raise ArgumentError.new("cannot truncate a #{have}-position KV cache to #{positions}")
      end
      unless @host_kv_stale
        want = positions * @head_dim
        {@k_cache, @v_cache}.each do |cache|
          cache.each { |arr| arr.truncate(0, want) if arr.size > want }
        end
      end
      @cache_len = positions
    end

    def finalize
      # Only free the per-instance KV cache; the attention scratch is class-level
      # and shared across all blocks, so it must not be freed per instance.
      {@gpu_k_cache, @gpu_v_cache}.each do |p|
        CUDA.free(p) unless p.null?
      end
    end

    # When offload is true the (Q4-only) dense weights are kept in host RAM and
    # streamed to the GPU on demand, which is what moves the dense model ceiling
    # off the 16 GB VRAM budget and onto host RAM. Unlike MoE experts, every one
    # of these weights is touched on EVERY token, so there is no sparsity to
    # amortize the transfer: this trades decode speed for capacity.
    def to_gpu!(quantize : Bool = false, bits : Int32 = 8, offload : Bool = false)
      return unless CUDA.fully_available?
      if quantize
        @w_q = to_quant(@w_q, bits, offload)
        @w_k = to_quant(@w_k, bits, offload)
        @w_v = to_quant(@w_v, bits, offload)
        @w_o = to_quant(@w_o, bits, offload)
        # The output gate is a full [d_model, q_dim] projection, so it is quantized with the rest.
        # Keeping it on the host forced gated attention layers off the device prefill path, which
        # measured 269.4 s per layer at a 1216-token prefill against 0.384 s on it.
        if wg = @w_gate_attn
          @w_gate_attn = to_quant(wg, bits, offload)
        end
      else
        raise ArgumentError.new("dense offload requires quantization (offload is Q4-only)") if offload
        # Only promote host weights; leave existing CudaMatrix/QuantizedWeight as-is.
        @w_q = @w_q.as(SimpleMatrix).to_cuda if @w_q.is_a?(SimpleMatrix)
        @w_k = @w_k.as(SimpleMatrix).to_cuda if @w_k.is_a?(SimpleMatrix)
        @w_v = @w_v.as(SimpleMatrix).to_cuda if @w_v.is_a?(SimpleMatrix)
        @w_o = @w_o.as(SimpleMatrix).to_cuda if @w_o.is_a?(SimpleMatrix)
      end
      @norm1.to_gpu!
      @norm2.to_gpu!
      # A dense SwiGLU takes the offload flag. MoE experts have their own
      # (moe_offload) flag decided at construction, and its router stays fp32,
      # so there is nothing for dense offload to do there.
      case f = @ffn
      when SwiGLUFF then f.to_gpu!(quantize, bits, offload)
      else               f.to_gpu!(quantize, bits)
      end
    end

    def apply_gradients(lr : Float64)
    end

    def backward(d_out : SimpleMatrix) : SimpleMatrix
      raise "LlamaBlock backward pass not yet implemented"
    end

    def backward(d_out : CudaMatrix) : CudaMatrix
      raise "LlamaBlock backward pass not yet implemented"
    end

    # CPU forward — full sequence (no cache, for prefill or training)
    def forward(x : SimpleMatrix) : SimpleMatrix
      normed = @norm1.forward(x)
      attn = attention_full_cpu(normed)
      h = x + attn
      normed2 = @norm2.forward(h)
      ff_out = @ffn.forward(normed2)
      h + ff_out
    end

    # CPU forward with KV cache — only processes new tokens
    def forward_cached(x : SimpleMatrix) : SimpleMatrix
      normed = Profile.measure("block.norm") { @norm1.forward(x) }
      attn = attention_cached_cpu(normed)
      h = Profile.measure("block.residual") { x + attn }
      normed2 = Profile.measure("block.norm") { @norm2.forward(h) }
      ff_out = @ffn.forward(normed2)
      Profile.measure("block.residual") { h + ff_out }
    end

    # Device-resident workspaces for the block chain, one set per block.
    @dev_n1 : CudaMatrix? = nil
    @dev_n2 : CudaMatrix? = nil
    @dev_attn : CudaMatrix? = nil
    @dev_h : CudaMatrix? = nil
    @dev_ff_out : CudaMatrix? = nil

    @@block_device : Bool? = nil

    # Device-resident block chain is the default; SHAINET_BLOCK_DEVICE=0 forces the
    # host path, which is how the A/B is taken and how the fallback stays exercised.
    def self.block_device_enabled? : Bool
      flag = @@block_device
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_BLOCK_DEVICE", "1") != "0"
      @@block_device = flag
      flag
    end

    # Pass nil to forget the decision and re-read the environment.
    def self.block_device_enabled=(value : Bool?)
      @@block_device = value
    end

    # True when the whole block can run with the activation staying on the device:
    # both norms have their kernel and gamma there, the attention projections and
    # o_proj are quantized, and the FFN has a device path.
    def block_device_capable? : Bool
      return false unless self.class.block_device_enabled?
      return false unless CUDA.fully_available? && CUDA.block_device_kernels_available?
      return false unless CUDA.attention_device_kernels_available?
      return false unless @norm1.device_capable? && @norm2.device_capable?
      return false unless @w_q.is_a?(QuantizedWeight) && @w_k.is_a?(QuantizedWeight) &&
                          @w_v.is_a?(QuantizedWeight) && @w_o.is_a?(QuantizedWeight)
      # A gated attention layer (Qwen3.5) used to decline outright, because the gate was applied on
      # the two HOST paths and the PREFILL device path but NOT in attention_cached_device, which ran
      # w_o on ungated attention. So the device path silently dropped the gate on every decode step,
      # and the symptom was that a kv-cached generation disagreed with the same prompt run uncached
      # -- visible only by comparing the two, since each looked plausible alone.
      #
      # attention_cached_device now applies it too, so the requirement is that the gate be
      # REACHABLE, not absent. It still refuses rather than proceeding ungated.
      return false unless device_attn_gate_reachable?
      return false unless gpu_attention?
      ffn = @ffn
      case ffn
      when MoEFF    then ffn.device_row_capable?
      when SwiGLUFF then ffn.device_resident_capable?
      else               false
      end
    end

    # Decode forward for a single token whose activation never leaves the device.
    #
    # The host path spends, per layer: two host RMSNorm loops, two host residual adds,
    # three uploads of the same normed row for q/k/v, a readback of the o_proj result
    # and a readback of the FFN result. All of those are gone here. What remains is
    # inherent to the attention implementation: q/k/v come back because RoPE, the KV
    # host mirror and the attention staging are host-side, so this does NOT claim to
    # make attention device-resident.
    #
    # `x` must be device-resident; the returned matrix is this block's own workspace,
    # so the caller must consume or hand it straight to the next block.
    def forward_cached_device(x : CudaMatrix) : CudaMatrix
      dm = @d_model
      n1 = (@dev_n1 ||= CudaMatrix.new(1, dm))
      n2 = (@dev_n2 ||= CudaMatrix.new(1, dm))
      attn_dev = (@dev_attn ||= CudaMatrix.new(1, dm))
      h = (@dev_h ||= CudaMatrix.new(1, dm))

      Profile.measure("block.dev_norm") { @norm1.forward_into(x, n1) }
      attention_cached_device(n1, attn_dev)

      # h = x + attn, then h += ffn(norm2(h)), both residuals on the device.
      Profile.measure("block.dev_residual") do
        CUDA.memcpy(h.device_ptr.not_nil!.as(Pointer(Void)), x.device_ptr.not_nil!.as(Pointer(Void)),
          dm.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToDevice)
        CUDA.add_inplace(h.device_ptr.not_nil!, attn_dev.device_ptr.not_nil!, dm)
        h.mark_device_dirty!
      end

      Profile.measure("block.dev_norm") { @norm2.forward_into(h, n2) }
      ffn = @ffn
      ff_out = case ffn
               when MoEFF then ffn.forward_device_row(n2)
               else
                 # Dense SwiGLU: one expert, same device chain, no readback.
                 ff_buf = (@dev_ff_out ||= CudaMatrix.new(1, dm))
                 ffn.as(SwiGLUFF).forward_device(n2, ff_buf)
               end
      Profile.measure("block.dev_residual") do
        CUDA.add_inplace(h.device_ptr.not_nil!, ff_out.device_ptr.not_nil!, dm)
        h.mark_device_dirty!
      end
      h
    end

    # Device-resident constants, uploaded once per block on first use.
    @dev_q : CudaMatrix? = nil
    @dev_k : CudaMatrix? = nil
    @dev_v : CudaMatrix? = nil
    @dev_attn_out : CudaMatrix? = nil
    @dev_inv_freq : CudaMatrix? = nil
    @dev_b_q : CudaMatrix? = nil
    @dev_b_k : CudaMatrix? = nil
    @dev_b_v : CudaMatrix? = nil
    @dev_q_norm : CudaMatrix? = nil
    @dev_k_norm : CudaMatrix? = nil

    # True once the device cache holds positions the host mirror's CONTENTS do not.
    # Length parity is still maintained, so a later prefill chunk indexes correctly;
    # only the values are absent, and the CPU attention path refuses to run against
    # them rather than silently attending to zeros.
    @host_kv_stale : Bool = false
    @host_cache_reserved : Bool = false

    # Bytes of host RAM the fp32 KV mirror is holding. Exposed so "the device path does
    # not touch it" is assertable directly rather than inferred. Counts capacity, not
    # length, because an eager reservation is exactly the cost this is watching for.
    def host_kv_bytes : UInt64
      total = 0_u64
      {@k_cache, @v_cache}.each do |cache|
        cache.each { |arr| total += (arr.size.to_u64 * 4_u64) }
      end
      total
    end

    def host_kv_stale? : Bool
      @host_kv_stale
    end

    private def upload_vec(src : Array(Float32)) : CudaMatrix
      m = CudaMatrix.new(1, src.size)
      m.raw_data.to_unsafe.copy_from(src.to_unsafe, src.size)
      m.mark_host_modified!
      m.sync_to_device!("dev_const")
      m
    end

    private def dev_inv_freq_ptr : Pointer(Float32)
      m = @dev_inv_freq
      unless m
        # Sized to the ROTARY width: the kernel reads inv_freq[0, rot_dim/2), so a head_dim-sized
        # buffer would be right by accident here and wrong if the two ever diverge in the other
        # direction.
        half = rot_dim // 2
        freqs = Array(Float32).new(half) { |i| inv_freq(i) }
        m = upload_vec(freqs)
        @dev_inv_freq = m
      end
      m.device_ptr.not_nil!
    end

    # Fully device-resident attention for one decode token.
    #
    # This is the last plumbing step. Previously q/k/v were read back to the host
    # every layer because RoPE, the QK-norm and the KV append all ran there, which
    # left three synchronous readbacks per layer as 50.7% of a 48-layer decode step.
    # Now the projections write device buffers, the bias add, QK-norm and RoPE are
    # device kernels, K/V are appended into the device cache by the existing append
    # kernel (which already read from device memory), the attention kernel consumes
    # the device Q directly, and o_proj writes a device buffer. Nothing crosses PCIe.
    private def attention_cached_device(x : CudaMatrix, dst : CudaMatrix) : Nil
      head_dim = @head_dim
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      pos = @cache_len
      kv_dim = @num_kv_heads * head_dim
      total_len = pos + 1

      qd = (@dev_q ||= CudaMatrix.new(1, @q_dim))
      kd = (@dev_k ||= CudaMatrix.new(1, kv_dim))
      vd = (@dev_v ||= CudaMatrix.new(1, kv_dim))
      ao = (@dev_attn_out ||= CudaMatrix.new(1, @q_dim))

      Profile.measure("attn.dev_qkv") do
        GGUFMatrix.with_shared_input(x) do
          @w_q.as(QuantizedWeight).gemv_into(x, qd)
          @w_k.as(QuantizedWeight).gemv_into(x, kd)
          @w_v.as(QuantizedWeight).gemv_into(x, vd)
        end
      end

      Profile.measure("attn.dev_prep") do
        if b = @b_q
          @dev_b_q ||= upload_vec(b)
          CUDA.add_inplace(qd.device_ptr.not_nil!, @dev_b_q.not_nil!.device_ptr.not_nil!, @q_dim)
        end
        if b = @b_k
          @dev_b_k ||= upload_vec(b)
          CUDA.add_inplace(kd.device_ptr.not_nil!, @dev_b_k.not_nil!.device_ptr.not_nil!, kv_dim)
        end
        if b = @b_v
          @dev_b_v ||= upload_vec(b)
          CUDA.add_inplace(vd.device_ptr.not_nil!, @dev_b_v.not_nil!.device_ptr.not_nil!, kv_dim)
        end
        # Qwen3 QK-norm, then RoPE, in the same order as the host path.
        if g = @q_norm
          @dev_q_norm ||= upload_vec(g)
          CUDA.head_rmsnorm(qd.device_ptr.not_nil!, @dev_q_norm.not_nil!.device_ptr.not_nil!,
            @num_heads, head_dim, @qk_norm_eps.to_f32)
        end
        if g = @k_norm
          @dev_k_norm ||= upload_vec(g)
          CUDA.head_rmsnorm(kd.device_ptr.not_nil!, @dev_k_norm.not_nil!.device_ptr.not_nil!,
            @num_kv_heads, head_dim, @qk_norm_eps.to_f32)
        end
        ifr = dev_inv_freq_ptr
        CUDA.rope_forward(qd.device_ptr.not_nil!, ifr, pos, @num_heads, head_dim, rot_dim)
        CUDA.rope_forward(kd.device_ptr.not_nil!, ifr, pos, @num_kv_heads, head_dim, rot_dim)
      end

      # Staging and workspace sized for a single query token, for whichever attention path is in use.
      @@gpu_staging, @@gpu_staging_cap = grow_dev_buf(@@gpu_staging, @@gpu_staging_cap, 2 * kv_dim + @q_dim)
      @@gpu_attn_out, @@gpu_attn_out_cap = grow_dev_buf(@@gpu_attn_out, @@gpu_attn_out_cap, @q_dim)
      @@gpu_attn_ws, @@gpu_attn_ws_cap = grow_dev_buf(@@gpu_attn_ws, @@gpu_attn_ws_cap,
        attn_ws_floats(1, total_len), cap_limit: ATTN_WS_BUDGET_FLOATS.to_i32)
      ensure_gpu_cache!(total_len, 1)

      Profile.measure("attn.dev_append") do
        # The append kernel takes K then V from one contiguous device blob, so stage
        # them with device-to-device copies (a few KB, no PCIe).
        bytes = kv_dim.to_u64 * 4_u64
        CUDA.memcpy(@@gpu_staging.as(Pointer(Void)), kd.device_ptr.not_nil!.as(Pointer(Void)),
          bytes, CUDA::MemcpyKind::DeviceToDevice)
        CUDA.memcpy((@@gpu_staging + kv_dim).as(Pointer(Void)), vd.device_ptr.not_nil!.as(Pointer(Void)),
          bytes, CUDA::MemcpyKind::DeviceToDevice)
        append_kv(1, pos)
      end

      # The host mirror is NOT extended here. It used to receive placeholder zeros purely
      # to keep its length in step, which cost 192 KiB per token per layer of host RAM to
      # store values nothing reads. context_length already reports cache_len once
      # host_kv_stale is set, and the host paths refuse outright rather than indexing a
      # mirror the device has moved past.
      @host_kv_stale = true
      @cache_len = total_len

      Profile.measure("attn.dev_kernel") do
        attend_kv(qd.device_ptr.not_nil!, 1, pos, @num_heads // @num_kv_heads, scale)
      end

      Profile.measure("attn.dev_oproj") do
        CUDA.memcpy(ao.device_ptr.not_nil!.as(Pointer(Void)), @@gpu_attn_out.as(Pointer(Void)),
          @q_dim.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToDevice)
        ao.mark_device_dirty!
        # The gate belongs HERE, between attention and w_o, exactly as the two host paths and the
        # prefill device path apply it. It was missing: this path ran w_o on ungated attention, so a
        # Qwen3.5 stack silently lost the gate on every decode step. That is why
        # block_device_capable? refused a gated layer outright, and why relaxing that refusal
        # without this line turned 'system' at 24.16 into '2' at 19.42.
        raise "device attention gate could not be applied" unless apply_device_attn_gate!(x, ao)
        @w_o.as(QuantizedWeight).gemv_into(ao, dst)
      end
    end

    # Whether the single-token device attention can apply this layer's gate. Checked by
    # block_device_capable? so a layer whose gate is not reachable declines the device path instead
    # of running ungated.
    private def device_attn_gate_reachable? : Bool
      wg = @w_gate_attn
      return true if wg.nil?
      return false unless CUDA.mul_sigmoid_available?
      wg.is_a?(QuantizedWeight) || (wg.is_a?(CudaMatrix) && wg.rows == @d_model && wg.cols == @q_dim)
    end

    # GPU forward — full sequence
    def forward(x : CudaMatrix) : CudaMatrix
      normed = @norm1.forward(x)
      attn = attention_full_gpu(normed)
      h = x + attn
      normed2 = @norm2.forward(h)
      ff_out = @ffn.forward(normed2)
      h + ff_out
    end

    # --- CPU attention with full recompute ---
    # Multiply the attention output by SiLU(x * w_gate_attn), in place.
    #
    # No-op unless the layer is gated, so every other architecture pays one nil check. The gate
    # is computed from the NORMED block input, the same source as q/k/v, because it arrives fused
    # into the same projection.
    # Multiply the attention output by its gate, elementwise, before w_o.
    #
    # SIGMOID, not SiLU. HF's gated attention applies `attn_output * torch.sigmoid(gate)`, and
    # SiLU(g) = g * sigmoid(g) carries an extra factor of the pre-activation, which is unbounded.
    # Measured on Qwen3.5-9B: with SiLU the model scored 12.06 nats against 12.42 for chance and
    # its top logits sat near 5.5, while simply DELETING the gate scored 10.01 with logits near
    # 14.4 -- a gate that made the model worse than no gate at all is how the extra factor showed
    # up. The control model's healthy logits are 16-19, for scale.
    #
    # SHAINET_ATTN_GATE_ACT=silu restores the old behaviour, which is how the A/B was taken.
    private def apply_attn_gate!(output : SimpleMatrix, x : SimpleMatrix, row_offset : Int32) : Nil
      wg = @w_gate_attn
      return if wg.nil?
      rows = output.rows
      cols = output.cols
      # The gate is a full projection and is quantized with the other weights, so it goes through
      # the shared dispatcher rather than being indexed directly. row_offset selects this call's
      # slice of x when the caller is working on a window of a longer sequence.
      xs = if row_offset == 0 && x.rows == rows
             x
           else
             slice = SimpleMatrix.new(rows, x.cols)
             rows.times { |t| x.cols.times { |i| slice[t, i] = x[row_offset + t, i] } }
             slice
           end
      gate = gpu_matmul(xs, wg)
      raise "attention gate produced #{gate.cols} columns, expected #{cols}" unless gate.cols == cols
      silu = self.class.attn_gate_silu?
      rows.times do |t|
        cols.times do |j|
          acc = gate[t, j].to_f64
          g = 1.0 / (1.0 + Math.exp(-acc))
          output[t, j] = output[t, j].to_f64 * (silu ? acc * g : g)
        end
      end
    end

    @@attn_gate_silu : Bool? = nil

    def self.attn_gate_silu? : Bool
      flag = @@attn_gate_silu
      return flag unless flag.nil?
      @@attn_gate_silu = ENV.fetch("SHAINET_ATTN_GATE_ACT", "sigmoid") == "silu"
    end

    private def attention_full_cpu(x : SimpleMatrix) : SimpleMatrix
      seq_len = x.rows
      head_dim = @head_dim
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32

      q_full = gpu_matmul(x, @w_q) # [seq, d_model]
      k_full = gpu_matmul(x, @w_k) # [seq, kv_dim]
      v_full = gpu_matmul(x, @w_v) # [seq, kv_dim]
      add_bias!(q_full, @b_q)
      add_bias!(k_full, @b_k)
      add_bias!(v_full, @b_v)
      apply_head_rmsnorm!(q_full, @num_heads, @q_norm)
      apply_head_rmsnorm!(k_full, @num_kv_heads, @k_norm)

      output = SimpleMatrix.new(seq_len, @q_dim)
      heads_per_kv = @num_heads // @num_kv_heads

      @num_heads.times do |h|
        q_col = h * head_dim
        kv_h = h // heads_per_kv
        kv_col = kv_h * head_dim

        # Extract + RoPE
        q_h = extract_head(q_full, seq_len, q_col, head_dim)
        k_h = extract_head(k_full, seq_len, kv_col, head_dim)
        v_h = extract_head(v_full, seq_len, kv_col, head_dim)
        apply_rope!(q_h, 0)
        apply_rope!(k_h, 0)

        # Attention: Q * K^T * scale, causal mask, softmax, * V
        causal_attention!(output, q_h, k_h, v_h, q_col, scale)
      end

      apply_attn_gate!(output, x, 0)
      gpu_matmul(output, @w_o)
    end

    # --- CPU attention with KV cache (incremental) ---
    private def attention_cached_cpu(x : SimpleMatrix) : SimpleMatrix
      # Device-resident prefill first: it keeps Q/K/V, bias, QK-norm, RoPE and w_o on the
      # device instead of reading three projections back and rebuilding a staging blob on
      # the host. w_o is already applied there, so the result is returned as it is.
      if dev_out = attention_prefill_device(x)
        return dev_out
      end

      new_tokens = x.rows
      head_dim = @head_dim
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      start_pos = @cache_len

      # Project Q/K/V — use GPU GEMM if weights are on device
      q_full = gpu_matmul(x, @w_q)
      k_new = gpu_matmul(x, @w_k)
      v_new = gpu_matmul(x, @w_v)
      # Qwen2-style projection biases (no-op when unset, i.e. LLaMA).
      add_bias!(q_full, @b_q)
      add_bias!(k_new, @b_k)
      add_bias!(v_new, @b_v)
      # Qwen3 QK-norm before RoPE (no-op when unset). K is normalized here, then
      # rotated and appended below; Q is normalized here, then rotated in the
      # per-head attention loop.
      apply_head_rmsnorm!(q_full, @num_heads, @q_norm)
      apply_head_rmsnorm!(k_new, @num_kv_heads, @k_norm)

      # Apply RoPE to new K at insert time (HF half-split), then append to cache.
      #
      # rot_dim, not head_dim: this inline copy of apply_rope! has to honour partial rotary too,
      # or a Qwen3.5 layer rotates its cached K over all 256 dimensions while its Q is rotated
      # over 64. Nothing downstream can detect that -- the shapes are identical either way.
      half = rot_dim // 2
      # This is the only path that populates the host mirror, so it is where the
      # reservation belongs: the device paths never read it and must not pay for it.
      if budget = kv_max_context
        reserve_host_cache!(budget) unless @host_cache_reserved
        @host_cache_reserved = true
      end
      Profile.measure("attn.rope_kv_append") do
        @num_kv_heads.times do |kv_h|
          kv_col = kv_h * head_dim
          new_tokens.times do |t|
            pos = start_pos + t
            rotated = Array(Float32).new(head_dim, 0.0_f32)
            half.times do |i|
              freq = inv_freq(i)
              angle = (pos * freq).to_f32
              cos_val = Math.cos(angle).to_f32
              sin_val = Math.sin(angle).to_f32
              x0 = k_new[t, kv_col + i].to_f32
              x1 = k_new[t, kv_col + i + half].to_f32
              rotated[i] = x0 * cos_val - x1 * sin_val
              rotated[i + half] = x1 * cos_val + x0 * sin_val
            end
            rotated.each { |val| @k_cache[kv_h] << val }
            head_dim.times { |d| @v_cache[kv_h] << v_new[t, kv_col + d].to_f32 }
          end
        end
      end

      total_len = @cache_len + new_tokens
      @cache_len = total_len
      output = SimpleMatrix.new(new_tokens, @q_dim)

      if gpu_attention?
        attention_heads_gpu(q_full, output, new_tokens, start_pos, total_len, scale)
      else
        attention_heads_cpu(q_full, output, new_tokens, start_pos, total_len, scale)
      end

      apply_attn_gate!(output, x, 0)
      gpu_matmul(output, @w_o)
    end

    # Device-resident prefill attention is the default; SHAINET_PREFILL_ATTN_DEVICE=0
    # forces the host path, which is how the A/B is taken and how the fallback stays
    # exercised.
    @@prefill_attn_device : Bool? = nil

    def self.prefill_attn_device_enabled? : Bool
      flag = @@prefill_attn_device
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_PREFILL_ATTN_DEVICE", "1") != "0"
      @@prefill_attn_device = flag
      flag
    end

    # Pass nil to forget the decision and re-read the environment.
    def self.prefill_attn_device_enabled=(value : Bool?)
      @@prefill_attn_device = value
    end

    # Device-resident PREFILL attention. Returns nil when it does not apply, and the
    # caller falls back to the host path.
    #
    # The host path projects Q/K/V with a GPU GEMM, reads all three back, applies bias,
    # QK-norm and RoPE on the CPU, writes K/V into the host mirror, and then
    # attention_heads_gpu rebuilds a blob from that mirror and uploads it again. Here
    # none of that leaves the device: the projections write into device workspaces, the
    # multi-row kernels do bias, QK-norm and RoPE in place, and pack_kv_heads writes the
    # exact layout append_kv already reads, so the KV cache code is untouched.
    #
    # The only remaining transfer is the attention OUTPUT, which the host block chain
    # still consumes. Removing that one needs a device-resident multi-token block chain
    # and is deliberately not attempted here.
    #
    # Ordering matters and matches the host path exactly: bias, then QK-norm, then RoPE.
    private def attention_prefill_device_core(xd : CudaMatrix) : CudaMatrix?
      # Any token count, including ONE. Restricting this to rows > 1 left a real hole:
      # forward_cached with a single token falls through to the host staging path, which
      # after a device prefill would index a mirror the device has moved past. While the
      # mirror was still padded with placeholder zeros that read silently wrong; now it
      # would read past the end. Handling one row here keeps the whole of forward_cached
      # on one side of the fence. forward_cached_device has its own decode path and does
      # not come through here.
      return unless self.class.prefill_attn_device_enabled?
      return unless gpu_attention?
      return unless CUDA.prefill_attn_kernels_available?
      # A gated attention layer is allowed here: the gate is applied to the attention output before
      # w_o by apply_device_attn_gate!, which both callers invoke. It must NOT be skipped -- doing so
      # diverged the real 9B's first attention layer by 71% between its cached and uncached paths --
      # so a gate that cannot be applied on the device makes this decline instead.
      if wg = @w_gate_attn
        return unless wg.is_a?(QuantizedWeight) || wg.is_a?(CudaMatrix)
        return unless CUDA.mul_sigmoid_available?
      end
      wq = @w_q
      wk = @w_k
      wv = @w_v
      return unless wq.is_a?(QuantizedWeight) && wk.is_a?(QuantizedWeight) && wv.is_a?(QuantizedWeight)

      n = xd.rows
      head_dim = @head_dim
      kv_dim = @num_kv_heads * head_dim
      qdim = @q_dim
      start_pos = @cache_len
      total_len = start_pos + n
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      heads_per_kv = @num_heads // @num_kv_heads

      xd.sync_to_device!("attn_prefill_in") unless xd.device_dirty?

      qd = attn_ws(@@attn_ws_q, n, qdim)
      kd = attn_ws(@@attn_ws_k, n, kv_dim)
      vd = attn_ws(@@attn_ws_v, n, kv_dim)

      Profile.measure("attn.dev_qkv") do
        GGUFMatrix.with_shared_input(xd) do
          wq.gemv_into(xd, qd)
          wk.gemv_into(xd, kd)
          wv.gemv_into(xd, vd)
        end
      end

      Profile.measure("attn.dev_prep") do
        if b = @b_q
          @dev_b_q ||= upload_vec(b)
          CUDA.add_bias_rows(qd.device_ptr.not_nil!, @dev_b_q.not_nil!.device_ptr.not_nil!, n, qdim)
        end
        if b = @b_k
          @dev_b_k ||= upload_vec(b)
          CUDA.add_bias_rows(kd.device_ptr.not_nil!, @dev_b_k.not_nil!.device_ptr.not_nil!, n, kv_dim)
        end
        if b = @b_v
          @dev_b_v ||= upload_vec(b)
          CUDA.add_bias_rows(vd.device_ptr.not_nil!, @dev_b_v.not_nil!.device_ptr.not_nil!, n, kv_dim)
        end
        if g = @q_norm
          @dev_q_norm ||= upload_vec(g)
          CUDA.head_rmsnorm_rows(qd.device_ptr.not_nil!, @dev_q_norm.not_nil!.device_ptr.not_nil!,
            n, @num_heads, head_dim, @qk_norm_eps.to_f32)
        end
        if g = @k_norm
          @dev_k_norm ||= upload_vec(g)
          CUDA.head_rmsnorm_rows(kd.device_ptr.not_nil!, @dev_k_norm.not_nil!.device_ptr.not_nil!,
            n, @num_kv_heads, head_dim, @qk_norm_eps.to_f32)
        end

        ifr = dev_inv_freq_ptr
        CUDA.rope_forward_rows(qd.device_ptr.not_nil!, ifr, start_pos, n, @num_heads, head_dim, rot_dim)
        CUDA.rope_forward_rows(kd.device_ptr.not_nil!, ifr, start_pos, n, @num_kv_heads, head_dim, rot_dim)
        qd.mark_device_dirty!
        kd.mark_device_dirty!
        vd.mark_device_dirty!
      end

      # The attention result is [tokens, q_dim], NOT [tokens, d_model]: q_dim is
      # num_heads * head_dim, which on a 30B-A3B is 4096 against a d_model of 2048. It
      # stays on the device and w_o projects it down there, so whatever leaves the device
      # is the smaller d_model-wide tensor.
      aq = attn_ws(@@attn_ws_aq, n, qdim)

      # The tiled prefill kernels need no workspace, so when they will serve this chunk none of the
      # workspace-driven limits apply: not the score-row budget in attn_chunk_tokens, not the split
      # cap, not the split workspace itself. Those were sized for kernels that stage scores in global
      # memory; applied to the tiled kernel they cut a 1024-token chunk to ~12 queries per call at 55K
      # of context -- against 64-query tiles -- and grew a 48 -> 97 MB buffer nothing read. attend_kv
      # sizes the workspace itself if a kernel declines and a workspace path runs after all.
      tiled = n > 1 && use_tiled_prefill_attn? && head_dim % 16 == 0 && head_dim <= 256
      max_chunk = tiled ? n : attn_chunk_tokens(total_len)
      # The split path's workspace scales with chunk * splits, so cap the chunk to keep it inside a
      # fixed budget. Without this the workspace grows with context and no reserve is ever enough.
      if !tiled && (cap = split_attn_chunk_cap(total_len))
        max_chunk = cap if cap < max_chunk
      end
      max_chunk = n if n < max_chunk
      kv_cap = @num_kv_heads * max_chunk * head_dim
      q_cap = max_chunk * qdim
      # Staging now holds K and V only: Q is already on the device, so it does not pass
      # through here any more.
      @@gpu_staging, @@gpu_staging_cap = grow_dev_buf(@@gpu_staging, @@gpu_staging_cap, 2 * kv_cap)
      @@gpu_attn_out, @@gpu_attn_out_cap = grow_dev_buf(@@gpu_attn_out, @@gpu_attn_out_cap, q_cap)
      # Sized for whichever attention path use_split_attn? selects. The split path stores per-split
      # partials (heads * tokens * splits * (head_dim + 2)) instead of a full score row
      # (heads * tokens * total_len), which at 24 heads, a 256-token chunk and 16k of context is
      # ~203 MB rather than ~384 MB -- and the score row was the largest single allocation in the
      # process, the one that OOM'd a 16 GB card at 9.5k tokens.
      unless tiled
        ws_floats = attn_ws_floats(max_chunk, total_len)
        @@gpu_attn_ws, @@gpu_attn_ws_cap = grow_dev_buf(@@gpu_attn_ws, @@gpu_attn_ws_cap, ws_floats)
      end
      ensure_gpu_cache!(total_len, max_chunk)

      aqp = aq.device_ptr.not_nil!

      off = 0
      while off < n
        c = max_chunk < (n - off) ? max_chunk : (n - off)
        base_pos = start_pos + off
        kvf = @num_kv_heads * c * head_dim

        Profile.measure("attn.dev_pack") do
          CUDA.pack_kv_heads(@@gpu_staging, kd.device_ptr.not_nil! + off * kv_dim,
            c, @num_kv_heads, head_dim)
          CUDA.pack_kv_heads(@@gpu_staging + kvf, vd.device_ptr.not_nil! + off * kv_dim,
            c, @num_kv_heads, head_dim)
        end
        Profile.measure("attn.kernels") do
          append_kv(c, base_pos)
          attend_kv(qd.device_ptr.not_nil! + off * qdim, c, base_pos, heads_per_kv, scale)
        end
        # Device to device: the chunk's result is collected on the card, where it used to
        # be copied out to the host once per chunk.
        Profile.measure("attn.dev_collect") do
          CUDA.memcpy((aqp + off * qdim).as(Pointer(Void)), @@gpu_attn_out.as(Pointer(Void)),
            (c * qdim).to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToDevice)
        end

        off += c
      end
      aq.mark_device_dirty!

      # As in device decode: the host mirror is not extended. context_length reports
      # cache_len once host_kv_stale is set, and the host attention paths refuse rather
      # than index a mirror whose rows were never written.
      @host_kv_stale = true
      @cache_len = total_len

      aq
    end

    # Host-facing wrapper: device attention, w_o applied ON the device, and ONE readback
    # of the d_model-wide result. It used to hand back the q_dim-wide tensor and let the
    # caller run w_o through gpu_matmul, which read q_dim out and uploaded it again -- on
    # this 30B that is 4096 wide against d_model's 2048, so this halves the transfer and
    # removes a round trip.
    private def attention_prefill_device(x : SimpleMatrix) : SimpleMatrix?
      wo = @w_o
      return unless wo.is_a?(QuantizedWeight)
      xd = attn_ws(@@attn_ws_x, x.rows, @d_model)
      Profile.measure("attn.dev_upload") do
        xd.raw_data.to_unsafe.copy_from(x.data.to_unsafe, x.rows * @d_model)
        xd.mark_host_modified!
        xd.sync_to_device!("attn_prefill_in")
      end
      aq = attention_prefill_device_core(xd)
      return unless aq
      return unless apply_device_attn_gate!(xd, aq)

      ao = attn_ws(@@attn_ws_o, aq.rows, @d_model)
      Profile.measure("attn.dev_wo") { wo.gemv_into(aq, ao) }
      Profile.measure("attn.d2h") do
        ao.sync_from_device!("attn_out") if ao.device_dirty?
        ao.to_simple
      end
    end

    # Device chain wrapper: same core, and the result never leaves the card.
    private def attention_prefill_device_into(xd : CudaMatrix, dst : CudaMatrix) : Bool
      wo = @w_o
      return false unless wo.is_a?(QuantizedWeight)
      aq = attention_prefill_device_core(xd)
      return false unless aq
      return false unless apply_device_attn_gate!(xd, aq)
      Profile.measure("attn.dev_wo") { wo.gemv_into(aq, dst) }
      true
    end

    # attn_out *= sigmoid(x * w_gate_attn), on the device, before w_o.
    #
    # Returns false when a gate exists but cannot be applied here, so the caller declines the whole
    # device path rather than silently returning ungated attention. That silent drop is the bug this
    # replaces: reachable only with quantized weights, so every fp32 equivalence check passed while
    # the Q4 ones diverged by 1.25 relative.
    private def apply_device_attn_gate!(xd : CudaMatrix, aq : CudaMatrix) : Bool
      wg = @w_gate_attn
      return true if wg.nil?
      return false unless CUDA.mul_sigmoid_available?
      gd = attn_ws(@@attn_ws_gate, aq.rows, aq.cols)
      case wg
      when QuantizedWeight then wg.gemv_into(xd, gd)
      when CudaMatrix
        return false unless wg.rows == xd.cols && wg.cols == aq.cols
        prod = xd * wg
        CUDA.copy_device_to_device(gd.device_ptr.not_nil!, prod.device_ptr.not_nil!,
          (aq.rows.to_u64 * aq.cols.to_u64 * 4_u64))
        gd.mark_device_dirty!
        prod.free!
      else
        return false
      end
      Profile.measure("attn.dev_gate") do
        CUDA.mul_sigmoid(aq.device_ptr.not_nil!, gd.device_ptr.not_nil!, aq.rows * aq.cols)
        aq.mark_device_dirty!
      end
      true
    end

    # Rows per prefill chunk through the block chain. Two jobs, in tension:
    #
    # It decouples workspace VRAM from prompt length. Measured, a 20480-token prefill died
    # on a 160 MB cudaMalloc, and 167772160 / 4 / 20480 is exactly 2048 -- d_model. About a
    # dozen [prompt_length, d_model] workspaces live across the chain, the attention path
    # and the MoE prefill, roughly 2 GB at 20k against 1236 MB of headroom left at 16k.
    # Chunked, they are all sized to this instead.
    #
    # It also sets how many tokens each expert receives per chunk (chunk * top_k / experts),
    # and so the M of every expert GEMM. 8192 with MoEFF.tile_rows at 512 measured 29.4 s
    # against 33.3 s at chunk 2048 tile 128 for a 4096-token prefill; the two only pay off
    # together, since at chunk 2048 an expert only gets 128 tokens and a larger tile has
    # nothing to fill it with.
    #
    # Bigger is therefore faster but costs workspace VRAM, which is what caps context.
    # SHAINET_PREFILL_CHUNK overrides it.
    #
    # Settable at runtime too: with a fixed value a spec-sized prompt is one chunk and the
    # chunking would never be exercised.
    @@prefill_chunk : Int32? = nil

    def self.prefill_chunk : Int32
      v = @@prefill_chunk
      return v if v
      # 8192 meant "never chunk in practice", so the attention workspaces sized to the whole prompt
      # and a long prefill needed VRAM in proportion to it. Measured on Qwen3.8-27B at the 2048 MB
      # default reserve: a 2048-token prefill FAILED outright with 8192, and completes with 256
      # leaving 808 MB spare. It is also not slower -- a 1400-token prefill went 20.0 to 25.9 per
      # second, since a chunk that fits cache beats one that does not.
      #
      # The knock-on is layers: the reserve is what decides how many stay on the host, so bounding
      # the workspace is what lets it come down. At a 1536 MB reserve this places 55 layers instead
      # of 52 and still survives a 2048-token prefill.
      v = (ENV["SHAINET_PREFILL_CHUNK"]? || "256").to_i
      v = 1 if v < 1
      @@prefill_chunk = v
      v
    end

    # Pass nil to forget the value and re-read the environment.
    def self.prefill_chunk=(value : Int32?)
      @@prefill_chunk = value
    end

    # Multi-token device block chain: the whole block for a PREFILL, with no host round
    # trip. The single-row forward_cached_device above is the decode equivalent; this is
    # the same shape with a row dimension.
    #
    # The rows are processed in chunks, IN PLACE. That is sound because the block is causal
    # and per-token everywhere it is not: chunk B's attention sees chunk A's KV because A
    # was appended first, and the norms, the FFN and the residuals treat each token
    # independently. Once a chunk's rows have been consumed they can be overwritten, so one
    # full-length buffer carries the activation and every workspace is chunk-sized.
    #
    # Returns nil when it does not apply, and the caller keeps the host chain.
    #
    # x is UPDATED IN PLACE and returned, so the caller must not expect its input back
    # unchanged.
    def forward_cached_device_multi(x : CudaMatrix) : CudaMatrix?
      return unless x.rows > 1
      return unless self.class.prefill_attn_device_enabled?
      return unless block_device_capable?
      return unless CUDA.prefill_attn_kernels_available?
      return unless @w_o.is_a?(QuantizedWeight)

      dm = @d_model
      n = x.rows
      xp = x.device_ptr.not_nil!
      chunk = self.class.prefill_chunk
      chunk = n if n < chunk

      off = 0
      while off < n
        c = chunk < (n - off) ? chunk : (n - off)
        total = c * dm

        xc = attn_ws(@@blk_ws_x, c, dm)
        n1 = attn_ws(@@blk_ws_n1, c, dm)
        n2 = attn_ws(@@blk_ws_n2, c, dm)
        attn_dev = attn_ws(@@blk_ws_attn, c, dm)
        h = attn_ws(@@blk_ws_h, c, dm)

        # Bring this chunk's rows into a chunk-sized buffer, device to device.
        CUDA.memcpy(xc.device_ptr.not_nil!.as(Pointer(Void)),
          (xp + off * dm).as(Pointer(Void)),
          total.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToDevice)
        xc.mark_device_dirty!

        Profile.measure("block.dev_norm") { @norm1.forward_into(xc, n1) }
        return unless attention_prefill_device_into(n1, attn_dev)

        # h = xc + attn, then h += ffn(norm2(h)), both residuals on the device.
        Profile.measure("block.dev_residual") do
          CUDA.memcpy(h.device_ptr.not_nil!.as(Pointer(Void)), xc.device_ptr.not_nil!.as(Pointer(Void)),
            total.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToDevice)
          CUDA.add_inplace(h.device_ptr.not_nil!, attn_dev.device_ptr.not_nil!, total)
          h.mark_device_dirty!
        end

        Profile.measure("block.dev_norm") { @norm2.forward_into(h, n2) }

        ffn = @ffn
        ff_out = case ffn
                 when MoEFF
                   ffn.forward_device_batch(n2)
                 else
                   buf = attn_ws(@@blk_ws_ff, c, dm)
                   ffn.as(SwiGLUFF).forward_device_batch(n2, buf)
                 end
        return unless ff_out

        Profile.measure("block.dev_residual") do
          CUDA.add_inplace(h.device_ptr.not_nil!, ff_out.device_ptr.not_nil!, total)
          h.mark_device_dirty!
        end

        # Write the finished chunk back over its own rows. Safe in place: later chunks read
        # only their own rows, and this chunk's input is no longer needed.
        CUDA.memcpy((xp + off * dm).as(Pointer(Void)),
          h.device_ptr.not_nil!.as(Pointer(Void)),
          total.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToDevice)

        off += c
      end

      x.mark_device_dirty!
      x
    end

    @@blk_ws_x = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@blk_ws_n1 = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@blk_ws_n2 = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@blk_ws_attn = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@blk_ws_h = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@blk_ws_ff = Hash(Tuple(Int32, Int32), CudaMatrix).new

    # Bounded per-shape workspaces for the prefill attention tensors.
    #
    # CLASS-level, shared across every layer, and that is not incidental: at a 4096-token
    # prefill these are ~117 MB per layer (q alone is 4096 x 4096 x 4 = 67 MB), so holding
    # a set per layer costs ~5.6 GB across 48 layers. Measured the wrong way round first:
    # per-instance pushed VRAM to 15882 MB of 16376 and the next prefill failed a 64 MB
    # allocation. Only one layer runs at a time, so one set is enough.
    #
    # Also CAPPED per shape, with the least recently used freed: prompt length varies per
    # turn, and keying a device buffer cache on a per-call size without a bound is what
    # leaked several GB in the MoE prefill path.
    # Three shapes, not two: a prompt length, the single row a decode step uses, and one
    # spare so a changing prompt length does not evict the decode shape on every turn.
    ATTN_WS_SHAPES = 3

    @@attn_ws_x = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@attn_ws_q = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@attn_ws_k = Hash(Tuple(Int32, Int32), CudaMatrix).new
    @@attn_ws_v = Hash(Tuple(Int32, Int32), CudaMatrix).new
    # Attention result in q_dim space, before w_o projects it to d_model.
    @@attn_ws_aq = Hash(Tuple(Int32, Int32), CudaMatrix).new
    # Post-w_o result, d_model wide: what the host path reads back and what the device
    # chain hands to the residual add.
    @@attn_ws_o = Hash(Tuple(Int32, Int32), CudaMatrix).new

    # Qwen3.5's attention output gate, q_dim wide. Same shape set as attn_ws_aq, so it is bounded by
    # the architecture rather than by how long the process runs, and it is counted and released with
    # the rest.
    @@attn_ws_gate = Hash(Tuple(Int32, Int32), CudaMatrix).new

    # Total VRAM held by the shared attention workspaces, so the bound is assertable.
    def self.attn_workspace_bytes : UInt64
      total = 0_u64
      [@@attn_ws_x, @@attn_ws_q, @@attn_ws_k, @@attn_ws_v, @@attn_ws_aq, @@attn_ws_o, @@attn_ws_gate,
       @@blk_ws_x, @@blk_ws_n1, @@blk_ws_n2, @@blk_ws_attn, @@blk_ws_h, @@blk_ws_ff].each do |cache|
        cache.each_value { |m| total += (m.rows.to_u64 * m.cols.to_u64 * 4_u64) }
      end
      total
    end

    def self.release_attn_workspaces! : Nil
      [@@attn_ws_x, @@attn_ws_q, @@attn_ws_k, @@attn_ws_v, @@attn_ws_aq, @@attn_ws_o, @@attn_ws_gate,
       @@blk_ws_x, @@blk_ws_n1, @@blk_ws_n2, @@blk_ws_attn, @@blk_ws_h, @@blk_ws_ff].each do |cache|
        cache.each_value(&.free!)
        cache.clear
      end
    end

    private def attn_ws(cache : Hash(Tuple(Int32, Int32), CudaMatrix), rows : Int32, cols : Int32) : CudaMatrix
      key = {rows, cols}
      if existing = cache[key]?
        cache.delete(key)
        cache[key] = existing
        return existing
      end
      while cache.size >= ATTN_WS_SHAPES
        oldest = cache.first_key
        cache[oldest].free!
        cache.delete(oldest)
      end
      cache[key] = CudaMatrix.new(rows, cols)
    end

    # --- CPU head loop: scores -> softmax -> AV, per query head/token ---
    # Refuses to run against a host mirror whose contents the device path has moved
    # past. Length parity is kept so prefill still indexes correctly, but the values
    # live only on the device, and attending to the placeholder zeros would be
    # silently wrong. Callers switching back to the CPU path must clear_cache! first.
    private def attention_heads_cpu(q_full : SimpleMatrix, output : SimpleMatrix,
                                    new_tokens : Int32, start_pos : Int32,
                                    total_len : Int32, scale : Float32)
      if @host_kv_stale
        raise RuntimeError.new(
          "KV cache lives on the device after device-resident decode; the CPU attention " \
          "path would read placeholder rows. Call clear_cache! before switching paths.")
      end
      head_dim = @head_dim
      heads_per_kv = @num_heads // @num_kv_heads

      # rot_dim, not head_dim: Q must be rotated over the same width as the cached K, or a
      # partial-rotary layer rotates 256 dimensions of Q against 64 of K.
      half = rot_dim // 2
      dm = @q_dim
      qptr = q_full.data.to_unsafe
      optr = output.data.to_unsafe
      # Reusable scratch buffers (avoid per-head/per-token allocations).
      q_rot = Array(Float32).new(head_dim, 0.0_f32)
      scores = Array(Float32).new(total_len, 0.0_f32)
      out = Array(Float32).new(head_dim, 0.0_f32)
      # Raw pointers bypass Array bounds-checks in the hot inner loops and let
      # the compiler vectorize the dot-product / weighted-sum reductions.
      qrp = q_rot.to_unsafe
      scp = scores.to_unsafe
      outp = out.to_unsafe

      @num_heads.times do |h|
        q_col = h * head_dim
        kv_h = h // heads_per_kv
        kptr = @k_cache[kv_h].to_unsafe
        vptr = @v_cache[kv_h].to_unsafe

        new_tokens.times do |i|
          pos = start_pos + i
          qrow = i * dm + q_col

          # RoPE-rotate this token's Q head directly into q_rot (HF half-split).
          idx = 0
          while idx < half
            freq = inv_freq(idx)
            angle = (pos * freq).to_f32
            c = Math.cos(angle).to_f32
            s = Math.sin(angle).to_f32
            x0 = qptr[qrow + idx]
            x1 = qptr[qrow + idx + half]
            qrp[idx] = x0 * c - x1 * s
            qrp[idx + half] = x1 * c + x0 * s
            idx += 1
          end

          # scores[j] = scale * (q_rot · K_cache[j]); track max for stable softmax.
          visible = start_pos + i + 1
          max_val = -Float32::INFINITY
          j = 0
          while j < visible
            kbase = j * head_dim
            dot = 0.0_f32
            d = 0
            while d < head_dim
              dot += qrp[d] * kptr[kbase + d]
              d += 1
            end
            sv = dot * scale
            scp[j] = sv
            max_val = sv if sv > max_val
            j += 1
          end

          # softmax over visible positions
          exp_sum = 0.0_f32
          j = 0
          while j < visible
            e = Math.exp((scp[j] - max_val).to_f64).to_f32
            scp[j] = e
            exp_sum += e
            j += 1
          end
          inv_sum = 1.0_f32 / exp_sum

          # out = sum_j (softmax_j) * V_cache[j]
          d = 0
          while d < head_dim
            outp[d] = 0.0_f32
            d += 1
          end
          j = 0
          while j < visible
            w = scp[j] * inv_sum
            vbase = j * head_dim
            d = 0
            while d < head_dim
              outp[d] += w * vptr[vbase + d]
              d += 1
            end
            j += 1
          end

          orow = i * dm + q_col
          d = 0
          while d < head_dim
            optr[orow + d] = outp[d]
            d += 1
          end
        end
      end
    end

    # --- GPU head loop: fused scores/softmax/AV over the device KV cache ---
    # One staging upload carries the new K/V rows plus the RoPE'd Q, then two
    # kernels (cache append + attention) run on-device and only the attention
    # output [new_tokens, d_model] is copied back.
    private def attention_heads_gpu(q_full : SimpleMatrix, output : SimpleMatrix,
                                    new_tokens : Int32, start_pos : Int32,
                                    total_len : Int32, scale : Float32)
      # This path stages K/V out of the host mirror. Once a device path has run, the
      # mirror is no longer extended at all, so indexing it here would read PAST its end
      # rather than merely read stale values. Refuse loudly instead.
      if @host_kv_stale
        raise RuntimeError.new(
          "KV cache lives on the device; the host staging path would read past the end " \
          "of the mirror. Call clear_cache! before switching paths.")
      end
      head_dim = @head_dim
      dm = @q_dim
      # rot_dim, not head_dim: Q must be rotated over the same width as the cached K, or a
      # partial-rotary layer rotates 256 dimensions of Q against 64 of K.
      half = rot_dim // 2
      heads_per_kv = @num_heads // @num_kv_heads

      # Prefill is processed in bounded chunks of query tokens. The attention
      # workspace is O(num_heads * tokens * total_len), so attending a whole
      # long prompt in one launch is quadratic in the prompt length (a 32k
      # prefill with 32 heads would need ~200 GB of scratch). Chunking caps the
      # workspace at ATTN_WS_BUDGET_FLOATS and also bounds the staging and
      # output buffers, which were previously linear in the full prompt length.
      max_chunk = attn_chunk_tokens(total_len)
      # The split path's workspace scales with chunk * splits, so cap the chunk to keep it inside a
      # fixed budget. Without this the workspace grows with context and no reserve is ever enough.
      if cap = split_attn_chunk_cap(total_len)
        max_chunk = cap if cap < max_chunk
      end
      max_chunk = new_tokens if new_tokens < max_chunk

      chunk_floats = max_chunk * head_dim   # floats per kv_head per tensor
      kv_cap = @num_kv_heads * chunk_floats # staging size of K (and of V)
      q_cap = max_chunk * dm
      staging_floats = 2 * kv_cap + q_cap
      ws_floats = attn_ws_floats(max_chunk, total_len)

      @@gpu_staging, @@gpu_staging_cap = grow_dev_buf(@@gpu_staging, @@gpu_staging_cap, staging_floats)
      @@gpu_attn_out, @@gpu_attn_out_cap = grow_dev_buf(@@gpu_attn_out, @@gpu_attn_out_cap, q_cap)
      @@gpu_attn_ws, @@gpu_attn_ws_cap = grow_dev_buf(@@gpu_attn_ws, @@gpu_attn_ws_cap, ws_floats,
        cap_limit: ATTN_WS_BUDGET_FLOATS.to_i32)

      st = @@staging_host
      if st.size < staging_floats
        st = Array(Float32).new(staging_floats, 0.0_f32)
        @@staging_host = st
      end
      stp = st.to_unsafe

      # Allocated last: growing the cache re-uploads the host mirror through the
      # append kernel, which needs the staging buffers above already sized.
      ensure_gpu_cache!(total_len, max_chunk)

      qptr = q_full.data.to_unsafe
      outp = output.data.to_unsafe
      # cos/sin depend only on (pos, rotation index), so compute once per token.
      cosv = Array(Float32).new(half, 0.0_f32)
      sinv = Array(Float32).new(half, 0.0_f32)
      cp = cosv.to_unsafe
      sp = sinv.to_unsafe

      off = 0
      while off < new_tokens
        n = Math.min(max_chunk, new_tokens - off)
        base_pos = start_pos + off
        cf = n * head_dim
        kvf = @num_kv_heads * cf
        qf = n * dm

        Profile.measure("attn.stage_host") do
          # This chunk's new K/V rows: each kv_head's slice is contiguous in the
          # CPU mirror (RoPE already applied to K at insert).
          tail = base_pos * head_dim
          @num_kv_heads.times do |kv_h|
            (stp + kv_h * cf).copy_from(@k_cache[kv_h].to_unsafe + tail, cf)
            (stp + kvf + kv_h * cf).copy_from(@v_cache[kv_h].to_unsafe + tail, cf)
          end

          # RoPE-rotate this chunk's Q (HF half-split) into the staging blob,
          # token-major and re-based to the chunk's own first row.
          qst = stp + 2 * kvf
          n.times do |i|
            pos = base_pos + i
            half.times do |r|
              angle = (pos * inv_freq(r)).to_f32
              cp[r] = Math.cos(angle).to_f32
              sp[r] = Math.sin(angle).to_f32
            end
            src_row = (off + i) * dm
            dst_row = i * dm
            @num_heads.times do |h|
              sb = src_row + h * head_dim
              db = dst_row + h * head_dim
              r = 0
              while r < half
                x0 = qptr[sb + r]
                x1 = qptr[sb + r + half]
                qst[db + r] = x0 * cp[r] - x1 * sp[r]
                qst[db + r + half] = x1 * cp[r] + x0 * sp[r]
                r += 1
              end
            end
          end
        end

        Profile.measure("attn.h2d") do
          CUDA.memcpy(@@gpu_staging.as(Pointer(Void)), stp.as(Pointer(Void)),
            (2 * kvf + qf).to_u64 * 4_u64, CUDA::MemcpyKind::HostToDevice)
        end
        Profile.measure("attn.kernels") do
          append_kv(n, base_pos)
          attend_kv(@@gpu_staging + 2 * kvf, n, base_pos, heads_per_kv, scale)
        end
        # Synchronous D2H read-back also orders after both kernels above.
        Profile.measure("attn.d2h") do
          CUDA.memcpy((outp + off * dm).as(Pointer(Void)), @@gpu_attn_out.as(Pointer(Void)),
            qf.to_u64 * 4_u64, CUDA::MemcpyKind::DeviceToHost)
        end

        off += n
      end
    end

    # Query tokens attended per kernel launch. Sized so the attention scratch
    # (num_heads * tokens * total_len floats) stays within ATTN_WS_BUDGET_FLOATS,
    # capped at ATTN_CHUNK_MAX. Override with SHAINET_ATTN_CHUNK.
    #
    # Public so the budget math can be asserted directly rather than inferred
    # from an allocation side effect.
    def attn_chunk_tokens(total_len : Int32) : Int32
      if env = ENV["SHAINET_ATTN_CHUNK"]?
        forced = env.to_i?
        return forced if forced && forced > 0
      end
      per_token = @num_heads.to_i64 * total_len.to_i64
      return ATTN_CHUNK_MAX if per_token <= 0
      n = (ATTN_WS_BUDGET_FLOATS // per_token).to_i32
      return 1 if n < 1
      n > ATTN_CHUNK_MAX ? ATTN_CHUNK_MAX : n
    end

    # GPU attention is used whenever the CUDA kernels are loadable and the
    # caller has not forced the CPU path (property or SHAINET_CPU_ATTENTION=1).
    # Memoized: fully_available? dlopens.
    private def gpu_attention? : Bool
      return false if force_cpu_attention?
      avail = @gpu_attn_avail
      if avail.nil?
        avail = !ENV["SHAINET_CPU_ATTENTION"]? && CUDA.fully_available?
        @gpu_attn_avail = avail
      end
      avail
    end

    # Storage type of the device KV cache. Arithmetic is fp32 for all three -- the kernels convert on
    # load -- so only the stored precision and the VRAM it costs differ.
    #
    # Q4 is the exception: packed 4-bit rows (q4_0 groups of 32 with fp16 scales, keys stored after a
    # fixed Hadamard rotation), 144 bytes per row at head_dim 256 against fp8's 256. Its rows are not
    # a whole number of bytes per element, so sizes go through row_bytes, never an element size.
    enum KVDtype
      F32
      F16
      FP8
      Q4

      # Bytes one cached position takes per KV head, in K or V.
      def row_bytes(head_dim : Int32) : UInt64
        case self
        in .f32? then 4_u64 * head_dim
        in .f16? then 2_u64 * head_dim
        in .fp8? then 1_u64 * head_dim
        in .q4?  then (((head_dim // 32) * 2 + 15) // 16 * 16 + head_dim // 2).to_u64
        end
      end
    end

    # The KV dtype the environment asks for, downgraded to what the loaded kernels support.
    #
    # SHAINET_KV_DTYPE=f32|f16|fp8|q4 selects it; without it the older SHAINET_KV_FP16=0 still means
    # fp32, and the default is fp16. One function rather than a per-caller reading so the loader,
    # which sizes the VRAM reserve from it, and every block, which allocates from it, cannot disagree.
    def self.resolve_kv_dtype : KVDtype
      want = case ENV["SHAINET_KV_DTYPE"]?.try(&.downcase)
             when "q4", "q4_0", "int4" then KVDtype::Q4
             when "fp8", "f8", "e4m3"  then KVDtype::FP8
             when "f16", "fp16"        then KVDtype::F16
             when "f32", "fp32"        then KVDtype::F32
             when nil, ""              then ENV.fetch("SHAINET_KV_FP16", "1") == "0" ? KVDtype::F32 : KVDtype::F16
             else
               Log.warn { "unknown SHAINET_KV_DTYPE=#{ENV["SHAINET_KV_DTYPE"]}; using f16" }
               KVDtype::F16
             end
      if want.q4? && !(CUDA.kv_q4_kernels_available? && ENV.fetch("SHAINET_SPLIT_ATTN", "1") != "0")
        # The q4 cache has no single-block attention kernel, so it needs the split path too.
        Log.warn { "SHAINET_KV_DTYPE=q4 needs the q4 kernels and the split attention path; using fp8" }
        want = KVDtype::FP8
      end
      want = KVDtype::F16 if want.fp8? && !CUDA.kv_fp8_kernels_available?
      want = KVDtype::F32 if want.f16? && !CUDA.kv_f16_kernels_available?
      want
    end

    # Fixed at the first allocation and kept for the buffers' lifetime, so a resize never has to
    # reinterpret existing contents.
    def kv_dtype : KVDtype
      @kv_dtype ||= LlamaBlock.resolve_kv_dtype
    end

    # Whether the device KV cache is kept in fp16.
    def kv_cache_fp16? : Bool
      kv_dtype.f16?
    end

    def kv_cache_fp8? : Bool
      kv_dtype.fp8?
    end

    def kv_cache_q4? : Bool
      kv_dtype.q4?
    end

    # Current device footprint of this block's KV cache in bytes (K and V), at
    # the allocated capacity rather than the used length.
    def kv_cache_bytes : UInt64
      return 0_u64 if @gpu_cache_cap == 0
      2_u64 * @num_kv_heads.to_u64 * @gpu_cache_cap.to_u64 * kv_row_bytes
    end

    # Bytes per cached position per KV head for the active dtype.
    private def kv_row_bytes : UInt64
      kv_dtype.row_bytes(@head_dim)
    end

    # Ensure the device KV cache holds at least total_len positions per
    # kv_head.
    #
    # With a context budget configured (see #kv_max_context) the cache is
    # allocated ONCE at exactly that size. Without one it grows by doubling,
    # which overshoots badly: a measured 4128-token context allocated 8192 slots,
    # so half the cache was never used.
    #
    # On (re)allocation the CPU mirror, which already contains the new tokens, is
    # re-uploaded into the fresh buffers. `max_chunk` bounds the staging used by
    # that re-upload, so this must be called after the staging buffers have been
    # sized for that chunk.
    private def ensure_gpu_cache!(total_len : Int32, max_chunk : Int32)
      return if @gpu_cache_cap >= total_len

      new_cap = if budget = kv_max_context
                  if total_len > budget
                    raise ArgumentError.new(
                      "context length #{total_len} exceeds the configured KV budget #{budget}; " \
                      "raise kv_max_context (or SHAINET_KV_MAX_CONTEXT) or shorten the prompt")
                  end
                  budget
                else
                  Math.max(256, Math.max(total_len, @gpu_cache_cap * 2))
                end

      bytes = @num_kv_heads.to_u64 * new_cap.to_u64 * kv_row_bytes
      old_k = @gpu_k_cache
      old_v = @gpu_v_cache
      old_cap = @gpu_cache_cap
      kp = Pointer(Void).null
      vp = Pointer(Void).null
      CUDA.malloc(pointerof(kp), bytes)
      CUDA.malloc(pointerof(vp), bytes)

      # Carry the existing cache across DEVICE TO DEVICE rather than re-uploading the
      # host mirror. Two reasons: it avoids a PCIe round trip on every growth, and it
      # frees the device decode path from having to maintain a host mirror at all.
      # The copy is per kv_head because the row stride is `cap`, so a single block
      # copy would land every head at the wrong offset after the capacity changes.
      #
      # CLAMPED to the OLD capacity: on the host path the mirror is appended to before
      # the device append runs, so cached_positions can already exceed what the old
      # device buffer holds, and copying that many rows would read past its end. The
      # rows beyond are written by the append kernel immediately after anyway.
      used = Math.min(cached_positions, old_cap)
      if used > 0 && !old_k.null? && !old_v.null?
        rb = kv_row_bytes
        span = used.to_u64 * rb
        @num_kv_heads.times do |h|
          dst_off = h.to_u64 * new_cap.to_u64 * rb
          src_off = h.to_u64 * old_cap.to_u64 * rb
          CUDA.memcpy((kp.as(Pointer(UInt8)) + dst_off).as(Pointer(Void)),
            (old_k.as(Pointer(UInt8)) + src_off).as(Pointer(Void)),
            span, CUDA::MemcpyKind::DeviceToDevice)
          CUDA.memcpy((vp.as(Pointer(UInt8)) + dst_off).as(Pointer(Void)),
            (old_v.as(Pointer(UInt8)) + src_off).as(Pointer(Void)),
            span, CUDA::MemcpyKind::DeviceToDevice)
        end
      end

      CUDA.free(old_k) unless old_k.null?
      CUDA.free(old_v) unless old_v.null?
      @gpu_k_cache = kp
      @gpu_v_cache = vp
      @gpu_cache_cap = new_cap
    end

    # Positions currently held in the cache. The device path is authoritative and
    # does not fill the host mirror's contents, so this reads @cache_len there and
    # falls back to the mirror's length only when the mirror is still the truth.
    private def cached_positions : Int32
      return @cache_len if @host_kv_stale
      @num_kv_heads > 0 ? @k_cache[0].size // @head_dim : 0
    end

    # Upper bound on cached positions. When set, the device KV cache is sized to
    # exactly this once instead of doubling into it, which removes the overshoot
    # (up to 2x) that doubling leaves behind, and removes the repeated
    # free/malloc/re-upload cycle on the way there. The host mirror reserves the
    # same capacity so it stops reallocating as it grows.
    #
    # Defaults from SHAINET_KV_MAX_CONTEXT; nil keeps the doubling behaviour.
    def kv_max_context : Int32?
      cached = @kv_max_context
      return cached if cached
      return if @kv_max_context_checked
      @kv_max_context_checked = true
      if raw = ENV["SHAINET_KV_MAX_CONTEXT"]?
        if v = raw.to_i?
          @kv_max_context = v if v > 0
        end
      end
      @kv_max_context
    end

    # Set the context budget explicitly. Reserves host mirror capacity right
    # away; the device buffers are sized on the next forward pass. Raises if the
    # device cache has already grown past the new budget, since shrinking it
    # would discard cached positions the caller may still be attending to.
    def kv_max_context=(value : Int32?)
      if v = value
        raise ArgumentError.new("kv_max_context must be positive") unless v > 0
        if @gpu_cache_cap > v
          raise ArgumentError.new(
            "device KV cache is already #{@gpu_cache_cap} positions; " \
            "clear_cache! and free it before lowering the budget to #{v}")
        end
        # Deliberately NOT reserving the host mirror here. The mirror exists only for the
        # host attention paths, and both prefill and decode are device-resident by
        # default, so reserving at configuration time allocated host RAM for values
        # nothing reads: positions * head_dim per kv_head per cache, which for a 48-layer
        # 30B at a 32k budget is about 6.4 GB. The host path reserves on its first append
        # instead, which is the same guarantee (no reallocation mid-generation) paid for
        # only when it is used.
      end
      @kv_max_context_checked = true
      @kv_max_context = value
    end

    # Grow the host mirror's capacity without changing its length, so appending
    # up to `positions` tokens does not reallocate mid-generation.
    private def reserve_host_cache!(positions : Int32)
      want = positions.to_i64 * @head_dim
      return if want <= 0
      @num_kv_heads.times do |kv_h|
        {@k_cache, @v_cache}.each do |cache|
          arr = cache[kv_h]
          next if arr.size >= want
          grown = Array(Float32).new(want)
          grown.concat(arr)
          cache[kv_h] = grown
        end
      end
    end

    # Dispatch the cache-append kernel for the active cache dtype. Staging is
    # always fp32 and lives at the head of @@gpu_staging.
    private def append_kv(n : Int32, base_pos : Int32)
      case kv_dtype
      in .q4?
        CUDA.kv_cache_append_q4(@@gpu_staging, @gpu_k_cache.as(Pointer(UInt8)),
          @gpu_v_cache.as(Pointer(UInt8)), n, base_pos, @num_kv_heads,
          @head_dim, @gpu_cache_cap)
      in .fp8?
        CUDA.kv_cache_append_fp8(@@gpu_staging, @gpu_k_cache.as(Pointer(UInt8)),
          @gpu_v_cache.as(Pointer(UInt8)), n, base_pos, @num_kv_heads,
          @head_dim, @gpu_cache_cap)
      in .f16?
        CUDA.kv_cache_append_f16(@@gpu_staging, @gpu_k_cache.as(Pointer(UInt16)),
          @gpu_v_cache.as(Pointer(UInt16)), n, base_pos, @num_kv_heads,
          @head_dim, @gpu_cache_cap)
      in .f32?
        CUDA.kv_cache_append_f32(@@gpu_staging, @gpu_k_cache.as(Pointer(Float32)),
          @gpu_v_cache.as(Pointer(Float32)), n, base_pos, @num_kv_heads,
          @head_dim, @gpu_cache_cap)
      end
    end

    # Ceiling on the SPLIT attention workspace, in fp32 elements (64 MB).
    #
    # The split partials are heads * chunk * splits * (head_dim + 2), and splits grows as
    # context / ATTN_SPLIT_SCORES -- so at a fixed query chunk the workspace grows with context, just
    # as the single-block score row did. That is what made a full-context session fail no matter how
    # large the reserve was: at 15979 tokens the workspace wanted 131 MB (and grow_dev_buf asked for
    # 148 MB) while 118 MB was free, and every megabyte added to the reserve was absorbed by the same
    # growth. Holding chunk * splits under a budget makes the workspace CONSTANT in context, which is
    # what lets the reserve be a fixed term rather than a losing race.
    ATTN_SPLIT_WS_BUDGET_FLOATS = 16_i64 * 1024 * 1024

    # Largest query chunk that keeps the split workspace inside its budget at this context length.
    # Returns nil when the split path is not in use, so callers keep attn_chunk_tokens unchanged --
    # that function's budget arithmetic is pinned by specs and is still correct for the single-block
    # path.
    def split_attn_chunk_cap(total_len : Int32) : Int32?
      return unless use_split_attn?
      splits = (total_len + ATTN_SPLIT_SCORES_HOST - 1) // ATTN_SPLIT_SCORES_HOST
      return if splits <= 0
      per_token = @num_heads.to_i64 * splits.to_i64 * (@head_dim + 2).to_i64
      return if per_token <= 0
      n = (ATTN_SPLIT_WS_BUDGET_FLOATS // per_token).to_i32
      n < 1 ? 1 : n
    end

    # Mirrors ATTN_SPLIT_SCORES in the CUDA source. Kept here so the chunk cap can be computed
    # without a device round trip; a mismatch would only make the cap conservative, not wrong.
    ATTN_SPLIT_SCORES_HOST = 512

    # Whether the split-KV attention path is in use, decided ONCE.
    #
    # Sizing and dispatch must agree: the split path needs a partials buffer
    # (heads * tokens * splits * (head_dim + 2)) while the single-block path needs a full score row
    # (heads * tokens * total_len), and the score row is far larger. If the workspace were sized for
    # the split path but a later dispatch fell back to the single-block kernel, that kernel would
    # write past the end of the buffer. So this predicate is the only place the choice is made, and
    # both the allocation and the launch read it.
    private def use_split_attn? : Bool
      flag = @@use_split_attn
      return flag unless flag.nil?
      flag = ENV.fetch("SHAINET_SPLIT_ATTN", "1") != "0" && CUDA.attention_split_kv_available?
      @@use_split_attn = flag
      flag
    end

    @@use_split_attn : Bool? = nil

    # SHAINET_TILED_PREFILL_ATTN=0 keeps prefill on the split-KV kernel, for A/B comparison.
    @@use_tiled_prefill_attn : Bool = ENV.fetch("SHAINET_TILED_PREFILL_ATTN", "1") != "0"

    private def use_tiled_prefill_attn? : Bool
      @@use_tiled_prefill_attn
    end

    # Floats the attention workspace needs for the chosen path.
    private def attn_ws_floats(tokens : Int32, total_len : Int32) : Int32
      if use_split_attn?
        need = CUDA.attention_split_ws_floats(tokens, @num_heads, @head_dim, total_len)
        return need if need > 0
      end
      @num_heads * tokens * total_len
    end

    # Dispatch the attention kernel for the active cache dtype.
    #
    # Prefers the split-KV ("flash-decoding") path, which adds the KV length as a grid dimension. The
    # single-block kernels below parallelize over (head, token) only, so at decode -- one token -- they
    # run num_heads blocks, about 2.6% of this card, and each block walks the whole KV length while
    # staging its score row in GLOBAL memory. Measured at the real shape (24 heads, head_dim 256), the
    # split path is 1.37x at 1k context rising to 6.87x at 14k, and stays nearly FLAT as context grows
    # (0.298 ms to 0.944 ms across a 14x range) where the single-block kernel scales linearly. With 16
    # attention layers that is ~104 ms of per-token attention at 14k becoming ~15 ms.
    private def attend_kv(q : Pointer(Float32), n : Int32, base_pos : Int32,
                          heads_per_kv : Int32, scale : Float32)
      # The q4 cache holds keys rotated by a fixed orthogonal transform, so q gets the same rotation
      # (q.k is unchanged). In place: q is dead after attention on every path that calls this.
      CUDA.kv_q4_rotate_q(q, n, @num_heads, @head_dim) if kv_dtype.q4?

      # Multi-token chunks (prefill) take the tiled kernel: it reads each K/V row once per 16 queries
      # rather than once per query. It declines shapes it cannot serve, which falls through to the
      # paths below. Its output lands in @@gpu_attn_out like theirs; it does not use the workspace.
      if n > 1 && use_tiled_prefill_attn?
        suffix = case kv_dtype
                 in .q4?  then "q4"
                 in .fp8? then "fp8"
                 in .f16? then "f16"
                 in .f32? then "f32"
                 end
        return if CUDA.attention_prefill(suffix, q, @gpu_k_cache, @gpu_v_cache, @@gpu_attn_out,
                    n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
      end

      # The kernels below need a workspace; the tiled one does not, so prefill no longer sizes it up
      # front (see attention_prefill_device_core). Guarantee it here instead, which costs a compare
      # when it is already large enough and keeps a declined tiled call from overrunning a buffer
      # sized for a different path.
      @@gpu_attn_ws, @@gpu_attn_ws_cap = grow_dev_buf(@@gpu_attn_ws, @@gpu_attn_ws_cap,
        attn_ws_floats(n, base_pos + n))
      if use_split_attn?
        ok = case kv_dtype
             in .q4?
               CUDA.attention_split_kv_q4(q, @gpu_k_cache.as(Pointer(UInt8)),
                 @gpu_v_cache.as(Pointer(UInt8)), @@gpu_attn_out, @@gpu_attn_ws,
                 n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
             in .fp8?
               CUDA.attention_split_kv_fp8(q, @gpu_k_cache.as(Pointer(UInt8)),
                 @gpu_v_cache.as(Pointer(UInt8)), @@gpu_attn_out, @@gpu_attn_ws,
                 n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
             in .f16?
               CUDA.attention_split_kv_f16(q, @gpu_k_cache.as(Pointer(UInt16)),
                 @gpu_v_cache.as(Pointer(UInt16)), @@gpu_attn_out, @@gpu_attn_ws,
                 n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
             in .f32?
               CUDA.attention_split_kv_f32(q, @gpu_k_cache.as(Pointer(Float32)),
                 @gpu_v_cache.as(Pointer(Float32)), @@gpu_attn_out, @@gpu_attn_ws,
                 n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
             end
        # The workspace was sized for THIS path, so falling through to the single-block kernel would
        # overrun it. The predicate already confirmed the kernels load, so a false here is a real
        # fault rather than a capability miss.
        raise "split-KV attention kernel failed after reporting available" unless ok
        return
      end

      case kv_dtype
      in .q4?
        # resolve_kv_dtype only picks q4 with the split path on, so this is not reachable.
        raise "the q4 KV cache requires the split attention path"
      in .fp8?
        CUDA.attention_kv_fp8(q, @gpu_k_cache.as(Pointer(UInt8)),
          @gpu_v_cache.as(Pointer(UInt8)), @@gpu_attn_out, @@gpu_attn_ws,
          n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
      in .f16?
        CUDA.attention_kv_f16(q, @gpu_k_cache.as(Pointer(UInt16)),
          @gpu_v_cache.as(Pointer(UInt16)), @@gpu_attn_out, @@gpu_attn_ws,
          n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
      in .f32?
        CUDA.attention_kv_f32(q, @gpu_k_cache.as(Pointer(Float32)),
          @gpu_v_cache.as(Pointer(Float32)), @@gpu_attn_out, @@gpu_attn_ws,
          n, base_pos, @num_heads, heads_per_kv, @head_dim, @gpu_cache_cap, scale)
      end
    end

    # Past this size, stop doubling and grow by a margin instead (fp32 elements; 16M = 64 MB).
    #
    # Doubling is free insurance while a buffer is small, but the attention scratch is the largest
    # single allocation in the process and doubling it wastes VRAM in hundreds of megabytes. Measured
    # on a 9543-token prefill: the workspace needed 24 heads x 256 chunk x 9543 = 58.6M floats
    # (234 MB) and the doubling asked for 100.7M (384 MB), which is the allocation that OOM'd a 16 GB
    # card. Above the threshold there are only a handful of reallocations over a whole run, so the
    # churn doubling avoids is not worth paying 2x peak for.
    LARGE_BUF_FLOATS = 16 * 1024 * 1024

    # Pure size policy for grow_dev_buf, split out so it can be asserted directly
    # without allocating anything or driving a GPU.
    #
    # Doubles to avoid realloc churn, but `cap_limit` clamps the doubling so a
    # budgeted buffer cannot overshoot its budget: without the clamp a buffer
    # sitting just under its limit doubles straight past it, making the effective
    # ceiling 2x the stated one. A single request larger than the limit still
    # wins, since under-allocating would corrupt the kernel's writes.
    def self.next_buf_cap(cur_cap : Int32, needed : Int32, cap_limit : Int32? = nil) : Int32
      if needed >= LARGE_BUF_FLOATS
        # An eighth of headroom absorbs the next few token-by-token growths without another
        # allocation, while capping the waste at 12.5% instead of 100%.
        grown = needed + needed // 8
        grown = cap_limit if cap_limit && grown > cap_limit
        return Math.max(needed, grown)
      end
      doubled = cur_cap * 2
      doubled = cap_limit if cap_limit && doubled > cap_limit
      Math.max(needed, doubled)
    end

    # Grow-only device buffer; frees and reallocates only when too small.
    private def grow_dev_buf(ptr : Pointer(Float32), cur_cap : Int32, needed : Int32,
                             cap_limit : Int32? = nil) : {Pointer(Float32), Int32}
      return {ptr, cur_cap} if !ptr.null? && cur_cap >= needed
      CUDA.free(ptr.as(Pointer(Void))) unless ptr.null?
      new_cap = LlamaBlock.next_buf_cap(cur_cap, needed, cap_limit)
      np = Pointer(Float32).null
      CUDA.malloc(pointerof(np).as(Pointer(Pointer(Void))), new_cap.to_u64 * 4_u64)
      {np, new_cap}
    end

    # Diagnostics for the shared attention scratch. Chunked prefill must hold the
    # workspace to O(num_heads * chunk * total_len) rather than the unchunked
    # O(num_heads * seq^2); these let a spec assert that bound directly instead
    # of trusting it.
    def self.attn_ws_floats : Int32
      @@gpu_attn_ws_cap
    end

    def self.attn_staging_floats : Int32
      @@gpu_staging_cap
    end

    # --- GPU attention (full sequence, stays on device) ---
    private def attention_full_gpu(x : CudaMatrix) : CudaMatrix
      seq_len = x.rows
      head_dim = @head_dim
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32

      # Big matmuls on GPU
      q_full = x * @w_q.as(CudaMatrix) # cuBLAS SGEMM
      k_full = x * @w_k.as(CudaMatrix)
      v_full = x * @w_v.as(CudaMatrix)

      # Bring to CPU for per-head RoPE + causal softmax (small matrices)
      q_full.sync_from_device!("attn_q") if q_full.device_dirty?
      k_full.sync_from_device!("attn_k") if k_full.device_dirty?
      v_full.sync_from_device!("attn_v") if v_full.device_dirty?
      add_bias!(q_full, @b_q)
      add_bias!(k_full, @b_k)
      add_bias!(v_full, @b_v)
      apply_head_rmsnorm!(q_full, @num_heads, @q_norm)
      apply_head_rmsnorm!(k_full, @num_kv_heads, @k_norm)

      output = SimpleMatrix.new(seq_len, @q_dim)
      heads_per_kv = @num_heads // @num_kv_heads

      @num_heads.times do |h|
        q_col = h * head_dim
        kv_h = h // heads_per_kv
        kv_col = kv_h * head_dim

        q_h = SimpleMatrix.new(seq_len, head_dim)
        k_h = SimpleMatrix.new(seq_len, head_dim)
        v_h = SimpleMatrix.new(seq_len, head_dim)
        seq_len.times do |s|
          head_dim.times do |d|
            q_h[s, d] = q_full[s, q_col + d]
            k_h[s, d] = k_full[s, kv_col + d]
            v_h[s, d] = v_full[s, kv_col + d]
          end
        end

        apply_rope!(q_h, 0)
        apply_rope!(k_h, 0)
        causal_attention!(output, q_h, k_h, v_h, q_col, scale)
      end

      # Output projection on GPU: convert to CudaMatrix, then SGEMM
      result = CudaMatrix.new(seq_len, @q_dim)
      seq_len.times { |i| @q_dim.times { |j| result[i, j] = output[i, j] } }
      result.sync_to_device!("attn_concat")
      result * @w_o.as(CudaMatrix)
    end

    # --- Helper: add a per-column bias vector to every row, in-place (host) ---
    private def add_bias!(m : SimpleMatrix, b : Array(Float32)?)
      return unless b
      raise ArgumentError.new("bias size #{b.size} does not match matrix cols #{m.cols}") unless b.size == m.cols
      bp = b.to_unsafe
      data = m.data.to_unsafe
      cols = m.cols
      r = 0
      while r < m.rows
        base = r * cols
        c = 0
        while c < cols
          data[base + c] += bp[c]
          c += 1
        end
        r += 1
      end
    end

    # CudaMatrix variant: used by the full-sequence GPU path after the
    # projection result has been synced back to host for per-head processing.
    private def add_bias!(m : CudaMatrix, b : Array(Float32)?)
      return unless b
      raise ArgumentError.new("bias size #{b.size} does not match matrix cols #{m.cols}") unless b.size == m.cols
      m.rows.times { |r| m.cols.times { |c| m[r, c] = (m[r, c].to_f32 + b[c]) } }
    end

    # --- Helper: Qwen3 QK-norm. Per-head RMSNorm over head_dim with a learned
    # weight, applied in-place to every (row, head) slice of `m`
    # [rows, nheads*head_dim], before RoPE. No-op when `weight` is nil (the
    # LLaMA/Qwen2 default), so existing models are byte-identical.
    private def apply_head_rmsnorm!(m : SimpleMatrix, nheads : Int32, weight : Array(Float32)?)
      return unless w = weight
      hd = @head_dim
      raise ArgumentError.new("qk_norm weight size #{w.size} != head_dim #{hd}") unless w.size == hd
      eps = @qk_norm_eps
      wp = w.to_unsafe
      data = m.data.to_unsafe
      cols = m.cols
      m.rows.times do |r|
        nheads.times do |h|
          base = r * cols + h * hd
          ss = 0.0_f32
          d = 0
          while d < hd
            v = data[base + d]
            ss += v * v
            d += 1
          end
          inv = (1.0 / Math.sqrt(ss / hd + eps)).to_f32
          d = 0
          while d < hd
            data[base + d] = data[base + d] * inv * wp[d]
            d += 1
          end
        end
      end
    end

    # CudaMatrix variant (full-sequence GPU path, after the projection has been
    # synced to host for per-head processing). Operates on the host mirror.
    private def apply_head_rmsnorm!(m : CudaMatrix, nheads : Int32, weight : Array(Float32)?)
      return unless w = weight
      hd = @head_dim
      raise ArgumentError.new("qk_norm weight size #{w.size} != head_dim #{hd}") unless w.size == hd
      eps = @qk_norm_eps
      m.rows.times do |r|
        nheads.times do |h|
          col0 = h * hd
          ss = 0.0_f32
          hd.times { |d| v = m[r, col0 + d].to_f32; ss += v * v }
          inv = (1.0 / Math.sqrt(ss / hd + eps)).to_f32
          hd.times { |d| m[r, col0 + d] = (m[r, col0 + d].to_f32 * inv * w[d]) }
        end
      end
    end

    # --- Helper: extract head slice ---
    private def extract_head(full : SimpleMatrix, rows : Int32, col_start : Int32, cols : Int32) : SimpleMatrix
      m = SimpleMatrix.new(rows, cols)
      rows.times { |r| cols.times { |c| m[r, c] = full[r, col_start + c] } }
      m
    end

    # --- Helper: inverse frequency for rotation index i (0..head_dim/2) ---
    # Dimensions actually rotated, clamped to head_dim and to an even count (RoPE rotates pairs).
    private def rot_dim : Int32
      rd = @rotary_dim
      return @head_dim if rd.nil? || rd <= 0 || rd > @head_dim
      rd - (rd % 2)
    end

    private def inv_freq(i : Int32) : Float32
      if freqs = @rope_freqs
        freqs[i]
      else
        # The exponent denominator is the ROTARY width, not head_dim: HF computes inv_freq over
        # the rotated slice, so using head_dim here would space the frequencies wrongly even if
        # the right dimensions were rotated.
        (1.0 / (@rope_theta ** (2.0 * i / rot_dim))).to_f32
      end
    end

    # --- Helper: apply RoPE in-place (HF half-split convention) ---
    private def apply_rope!(m : SimpleMatrix, start_pos : Int32)
      half = rot_dim // 2
      m.rows.times do |pos|
        actual_pos = pos + start_pos
        half.times do |i|
          freq = inv_freq(i)
          angle = (actual_pos * freq).to_f32
          cos_val = Math.cos(angle).to_f32
          sin_val = Math.sin(angle).to_f32
          x0 = m[pos, i].to_f32
          x1 = m[pos, i + half].to_f32
          m[pos, i] = x0 * cos_val - x1 * sin_val
          m[pos, i + half] = x1 * cos_val + x0 * sin_val
        end
      end
    end

    # --- Helper: causal attention, writes into output at q_col ---
    private def causal_attention!(output : SimpleMatrix, q_h : SimpleMatrix, k_h : SimpleMatrix, v_h : SimpleMatrix, q_col : Int32, scale : Float32)
      seq_len = q_h.rows
      k_t = k_h.transpose
      scores = q_h * k_t

      attn_weights = SimpleMatrix.new(seq_len, k_h.rows)
      seq_len.times do |i|
        max_val = -Float32::INFINITY
        (0..i).each { |j| sv = scores[i, j].to_f32 * scale; max_val = sv if sv > max_val }
        exp_sum = 0.0_f32
        (0..i).each do |j|
          e = Math.exp((scores[i, j].to_f32 * scale - max_val).to_f64).to_f32
          attn_weights[i, j] = e
          exp_sum += e
        end
        (0..i).each { |j| attn_weights[i, j] = attn_weights[i, j].to_f32 / exp_sum }
      end

      attn_out = attn_weights * v_h
      seq_len.times { |s| @head_dim.times { |d| output[s, q_col + d] = attn_out[s, d] } }
    end

    # --- Helper: cache flat array to matrix ---
    private def cache_to_matrix(cache : Array(Float32), rows : Int32, cols : Int32) : SimpleMatrix
      m = SimpleMatrix.new(rows, cols)
      rows.times { |r| cols.times { |c| m[r, c] = cache[r * cols + c] } }
      m
    end
  end

  alias LlamaLayer = LlamaBlock
end
