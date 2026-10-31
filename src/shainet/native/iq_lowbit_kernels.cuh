// Native GEMV + row-dequant for the low-bit i-quant tail: Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M, IQ1_S.
//
// These were previously dequantized on the CPU at load time and re-quantized to symmetric Q4. That
// worked but cost VRAM rather than saving it, because Q4 is WIDER than what it replaced: the 928 MB
// of low-bit tensors in the IQ3_S build inflated to about 1740 MB, and the bulk pool had already
// uploaded the originals, so both copies were resident. On the IQ3_XXS build, which leans much
// harder on these types, 3317 MB inflated to roughly 6219 MB and the model needed MORE memory than
// the larger IQ3_S file. VRAM is exactly what buys context, so the transcode was standing between us
// and a longer window.
//
// Structure. Every one of these formats stores 256 values per block as 32 groups of 8, which is one
// group per lane of a warp -- the same decomposition the IQ3 kernels use. So each type needs only a
// per-lane "decode 8 values" function, and the GEMV and the row-dequant are both generated from it
// by the macro at the bottom. They cannot drift apart, which matters because the prefill path uses
// the dequant form and the decode path the GEMV, and a discrepancy between them would show up only
// as a long prompt disagreeing with a short one.
//
// Codebook placement follows what was measured on the IQ3 kernels: constant memory serializes
// divergent per-lane lookups, so the grids that carry real traffic are staged into shared memory
// once per block. The IQ1 grid is 16 KB: the persistent GEMV stages it too, while the row-dequant,
// which launches a block per row slice and could not amortize it, reads the constant copy.
// Q2_K needs no codebook at all; it is a plain 2-bit scale-and-offset.

#ifndef SHAINET_IQ_LOWBIT_KERNELS_CUH
#define SHAINET_IQ_LOWBIT_KERNELS_CUH

#define SHAINET_IQ1S_DELTA_F 0.125f

// ---------------------------------------------------------------------------------------------
// Per-lane decoders. Each writes the 8 values belonging to (ib32, lpos) of one 256-value block.
// ---------------------------------------------------------------------------------------------

// Q2_K: scales[16], qs[64], d, dmin. 16-value groups rather than 8, so a lane covers half a group.
//
// The reference walks n over {0,128} and j over 0..3 with a sub-block flip inside, taking a fresh
// scale byte each time. Flattened, value v has n = v/128, j = (v%128)/32, sub = (v%32)/16 and
// l = v%16, with the scale index 8n + 2j + sub and the shift 2j. All 8 values a lane owns share
// n, j and sub because 8 divides 16, so those are computed once.
__device__ __forceinline__ void decode_q2k_8(const unsigned char* __restrict__ block,
                                             int ib32, int lpos, float* v) {
    const unsigned char* scales = block;
    const unsigned char* qs = block + 16;
    const float d = __half2float(*((const __half*)(block + 80)));
    const float dmin = __half2float(*((const __half*)(block + 82)));

    const int v0 = ib32 * 8 + lpos * 8 / 8 * 0 + (ib32 * 0);  // placeholder, replaced below
    (void)v0;
    const int base = (ib32 * 4 + lpos) * 8;   // this lane's first value index, 0..255
    const int n = base >> 7;                  // 0 or 1
    const int idx = base & 127;
    const int j = idx >> 5;                   // 0..3
    const int sub = (idx & 31) >> 4;          // 0 or 1
    const int l0 = idx & 15;                  // 0..15, start within the 16-value group
    const unsigned char sc = scales[8 * n + 2 * j + sub];
    const float dl = d * (float)(sc & 0xF);
    const float ml = dmin * (float)(sc >> 4);
    const unsigned char* q = qs + 32 * n + 16 * sub;
    const int shift = 2 * j;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        v[t] = dl * (float)((q[l0 + t] >> shift) & 3) - ml;
    }
}

// IQ2_XXS: d, qs[64]. Each 32-value sub-block consumes 8 bytes: four grid indices in the low word
// and a packed scale plus four 7-bit sign selectors in the high word.
__device__ __forceinline__ void decode_iq2xxs_8(const unsigned char* __restrict__ block,
                                                const unsigned long long* __restrict__ grid,
                                                const unsigned char* __restrict__ ksigns,
                                                int ib32, int lpos, float* v) {
    const float d = __half2float(*((const __half*)(block + 0)));
    const unsigned char* qs = block + 2;
    const unsigned char* aux8 = qs + 8 * ib32;
    unsigned int aux1;
    memcpy(&aux1, qs + 8 * ib32 + 4, sizeof(unsigned int));
    const float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
    const unsigned char signs = ksigns[(aux1 >> (7 * lpos)) & 127];
    const unsigned long long g = grid[aux8[lpos]];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float mag = (float)((unsigned char)((g >> (8 * t)) & 0xFFULL));
        v[t] = db * mag * ((signs & (1u << t)) ? -1.0f : 1.0f);
    }
}

// IQ2_XS: d, qs[32 x uint16], scales[8]. Nine index bits and seven sign bits share each uint16.
__device__ __forceinline__ void decode_iq2xs_8(const unsigned char* __restrict__ block,
                                               const unsigned long long* __restrict__ grid,
                                               const unsigned char* __restrict__ ksigns,
                                               int ib32, int lpos, float* v) {
    const float d = __half2float(*((const __half*)(block + 0)));
    const unsigned short* qs = (const unsigned short*)(block + 2);
    const unsigned char* scales = block + 2 + 64;
    const unsigned char sc = scales[ib32];
    const float db = d * (0.5f + (float)((lpos >> 1) ? (sc >> 4) : (sc & 0xf))) * 0.25f;
    const unsigned short q = qs[4 * ib32 + lpos];
    const unsigned long long g = grid[q & 511];
    const unsigned char signs = ksigns[q >> 9];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float mag = (float)((unsigned char)((g >> (8 * t)) & 0xFFULL));
        v[t] = db * mag * ((signs & (1u << t)) ? -1.0f : 1.0f);
    }
}

// IQ2_S: d, qs[32] then signs[32], qh[8], scales[8]. The tenth index bit comes from qh.
__device__ __forceinline__ void decode_iq2s_8(const unsigned char* __restrict__ block,
                                              const unsigned long long* __restrict__ grid,
                                              const unsigned char* __restrict__ kmask,
                                              int ib32, int lpos, float* v) {
    const float d = __half2float(*((const __half*)(block + 0)));
    const unsigned char* qs = block + 2;
    const unsigned char* signs = qs + 32;
    const unsigned char* qh = block + 2 + 64;
    const unsigned char* scales = block + 2 + 64 + 8;
    const unsigned char sc = scales[ib32];
    const float dl = d * (0.5f + (float)((lpos >> 1) ? (sc >> 4) : (sc & 0xf))) * 0.25f;
    const int gi = qs[4 * ib32 + lpos] | (((int)qh[ib32] << (8 - 2 * lpos)) & 0x300);
    const unsigned long long g = grid[gi];
    const unsigned char sg = signs[4 * ib32 + lpos];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float mag = (float)((unsigned char)((g >> (8 * t)) & 0xFFULL));
        v[t] = dl * mag * ((sg & kmask[t]) ? -1.0f : 1.0f);
    }
}

// IQ1_M: qs[32], qh[16], sc[4 x uint16]. No fp16 scale field of its own -- the block scale is
// reassembled from one nibble of each of the four scale words. Codebook values are SIGNED, and each
// group carries a +/- delta rather than a sign mask.
__device__ __forceinline__ void decode_iq1m_8(const unsigned char* __restrict__ block,
                                              const uint64_t* __restrict__ grid,
                                              int ib32, int lpos, float* v) {
    const unsigned char* qs = block + 0;
    const unsigned char* qh = block + 32;
    const unsigned short* sc = (const unsigned short*)(block + 48);
    const unsigned short scale_u16 =
        (unsigned short)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float d = __half2float(*((const __half*)&scale_u16));

    const unsigned short scw = sc[ib32 / 2];
    const int sh = 6 * (ib32 % 2);
    const float dl = (lpos < 2)
        ? d * (float)(2 * ((scw >> (sh + 0)) & 0x7) + 1)
        : d * (float)(2 * ((scw >> (sh + 3)) & 0x7) + 1);

    const unsigned char* q = qs + 4 * ib32;
    const unsigned char* h = qh + 2 * ib32;
    const int hb = h[lpos >> 1];
    const int idx = q[lpos] | (((lpos & 1) ? (hb << 4) : (hb << 8)) & 0x700);
    const float delta = ((lpos & 1) ? (hb & 0x80) : (hb & 0x08)) ? -SHAINET_IQ1S_DELTA_F
                                                                 : SHAINET_IQ1S_DELTA_F;
    const unsigned long long g = grid[idx];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float gv = (float)((signed char)((g >> (8 * t)) & 0xFFULL));
        v[t] = dl * (gv + delta);
    }
}

// IQ1_S: d, qs[32], qh[8 x uint16]. Scale, three high index bits per group and the delta sign all
// live in one uint16 per 32-value sub-block.
__device__ __forceinline__ void decode_iq1s_8(const unsigned char* __restrict__ block,
                                              const uint64_t* __restrict__ grid,
                                              int ib32, int lpos, float* v) {
    const float d = __half2float(*((const __half*)(block + 0)));
    const unsigned char* qs = block + 2;
    const unsigned short* qh = (const unsigned short*)(block + 2 + 32);
    const unsigned short h = qh[ib32];
    const float dl = d * (float)(2 * ((h >> 12) & 7) + 1);
    const float delta = (h & 0x8000) ? -SHAINET_IQ1S_DELTA_F : SHAINET_IQ1S_DELTA_F;
    const int idx = qs[4 * ib32 + lpos] | (((h >> (3 * lpos)) & 7) << 8);
    const unsigned long long g = grid[idx];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float gv = (float)((signed char)((g >> (8 * t)) & 0xFFULL));
        v[t] = dl * (gv + delta);
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel generation. One GEMV and one row-dequant per type, from the decoder above.
//
// STAGE names the shared-memory preamble and DECODE the call, so a type that needs no codebook
// (Q2_K) and one that needs an 8 KB grid (IQ2_S) instantiate from the same template. The GEMV and
// the row-dequant take SEPARATE stage/decode arguments because their grids differ: the GEMV is
// persistent (a few blocks per SM walking rows with a grid stride, as the IQ3 kernels do), so it can
// afford to stage even IQ1's 16 KB codebook once per block, while the row-dequant launches a block
// per (row, block-slice) and must not.
//
// The GEMV loop also loads its activations as two float4 and is unrolled so several blocks' loads
// are in flight per lane -- the same two changes that took the IQ3 kernels from ~140 to ~350 GB/s.
// ---------------------------------------------------------------------------------------------

#define SHAINET_IQ_LOWBIT_KERNELS(NAME, BYTES, GSTAGE, GDECODE, STAGE, DECODE)                    \
__global__ void gemv_##NAME##_kernel(const float* __restrict__ x,                                 \
                                     const unsigned char* __restrict__ w,                         \
                                     float* __restrict__ y,                                       \
                                     int M, int N, int K) {                                       \
    GSTAGE                                                                                        \
    const int lane = threadIdx.x & 31;                                                            \
    const int wpb = blockDim.x >> 5;                                                              \
    const int m = blockIdx.y;                                                                     \
    if (m >= M) return;                                                                           \
    const int nblocks = K / GGUF_QK_K;                                                            \
    const long row_bytes = (long)nblocks * (BYTES);                                               \
    const float* xrow = x + (long)m * K;                                                          \
    const int ib32 = lane >> 2;                                                                   \
    const int lpos = lane & 3;                                                                    \
    for (int n = blockIdx.x * wpb + (threadIdx.x >> 5); n < N; n += gridDim.x * wpb) {            \
        const unsigned char* wrow = w + (long)n * row_bytes;                                      \
        float acc = 0.0f;                                                                         \
        _Pragma("unroll 4")                                                                        \
        for (int b = 0; b < nblocks; ++b) {                                                       \
            const unsigned char* block = wrow + (long)b * (BYTES);                                \
            float v[8];                                                                           \
            GDECODE                                                                               \
            const float4* xp = (const float4*)(xrow + b * GGUF_QK_K + ib32 * 32 + lpos * 8);      \
            const float4 xa = xp[0], xb = xp[1];                                                  \
            acc += v[0] * xa.x + v[1] * xa.y + v[2] * xa.z + v[3] * xa.w                           \
                 + v[4] * xb.x + v[5] * xb.y + v[6] * xb.z + v[7] * xb.w;                          \
        }                                                                                         \
        _Pragma("unroll")                                                                          \
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, s);             \
        if (lane == 0) y[(long)m * N + n] = acc;                                                  \
    }                                                                                             \
}                                                                                                 \
                                                                                                  \
__global__ void dequant_##NAME##_rows_kernel(const unsigned char* __restrict__ w,                 \
                                             float* __restrict__ out,                             \
                                             int row0, int rows, int K) {                         \
    STAGE                                                                                         \
    const int r = blockIdx.x;                                                                     \
    if (r >= rows) return;                                                                        \
    const int nblocks = K / GGUF_QK_K;                                                            \
    const long row_bytes = (long)nblocks * (BYTES);                                               \
    const unsigned char* wrow = w + (long)(row0 + r) * row_bytes;                                 \
    float* orow = out + (long)r * K;                                                              \
    for (int b = blockIdx.y; b < nblocks; b += gridDim.y) {                                       \
        const unsigned char* block = wrow + (long)b * (BYTES);                                    \
        for (int lane = threadIdx.x; lane < 32; lane += blockDim.x) {                             \
            const int ib32 = lane >> 2;                                                           \
            const int lpos = lane & 3;                                                            \
            float v[8];                                                                           \
            DECODE                                                                                \
            float* op = orow + b * GGUF_QK_K + ib32 * 32 + lpos * 8;                              \
            _Pragma("unroll")                                                                      \
            for (int t = 0; t < 8; ++t) op[t] = v[t];                                             \
        }                                                                                         \
    }                                                                                             \
}

// Shared-memory staging preambles. __syncthreads() is safe in both templates: every thread of the
// block reaches it before any early return that depends on the row index.
#define STAGE_NONE

#define STAGE_G256_SIGNS                                                                          \
    __shared__ unsigned long long s_g[256];                                                       \
    __shared__ unsigned char s_ks[128];                                                           \
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_g[i] = d_iq2xxs_grid[i];                \
    for (int i = threadIdx.x; i < 128; i += blockDim.x) s_ks[i] = d_ksigns_iq2xs[i];              \
    __syncthreads();

#define STAGE_G512_SIGNS                                                                          \
    __shared__ unsigned long long s_g[512];                                                       \
    __shared__ unsigned char s_ks[128];                                                           \
    for (int i = threadIdx.x; i < 512; i += blockDim.x) s_g[i] = d_iq2xs_grid[i];                 \
    for (int i = threadIdx.x; i < 128; i += blockDim.x) s_ks[i] = d_ksigns_iq2xs[i];              \
    __syncthreads();

#define STAGE_G1024_MASK                                                                          \
    __shared__ unsigned long long s_g[1024];                                                      \
    __shared__ unsigned char s_km[8];                                                             \
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_g[i] = d_iq2s_grid[i];                 \
    if (threadIdx.x < 8) s_km[threadIdx.x] = (unsigned char)(1u << threadIdx.x);                  \
    __syncthreads();

// IQ1's 16 KB grid, for the persistent GEMV only (see the macro comment).
#define STAGE_G2048_IQ1                                                                           \
    __shared__ uint64_t s_g[2048];                                                                \
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) s_g[i] = d_iq1s_grid[i];                 \
    __syncthreads();

#define SHAINET_IQ_LOWBIT_SAME(NAME, BYTES, STAGE, DECODE) \
    SHAINET_IQ_LOWBIT_KERNELS(NAME, BYTES, STAGE, DECODE, STAGE, DECODE)

SHAINET_IQ_LOWBIT_SAME(q2k_lb, 84, STAGE_NONE, decode_q2k_8(block, ib32, lpos, v);)
SHAINET_IQ_LOWBIT_SAME(iq2xxs, 66, STAGE_G256_SIGNS, decode_iq2xxs_8(block, s_g, s_ks, ib32, lpos, v);)
SHAINET_IQ_LOWBIT_SAME(iq2xs, 74, STAGE_G512_SIGNS, decode_iq2xs_8(block, s_g, s_ks, ib32, lpos, v);)
SHAINET_IQ_LOWBIT_SAME(iq2s, 82, STAGE_G1024_MASK, decode_iq2s_8(block, s_g, s_km, ib32, lpos, v);)
SHAINET_IQ_LOWBIT_KERNELS(iq1m, 56,
                          STAGE_G2048_IQ1, decode_iq1m_8(block, s_g, ib32, lpos, v);,
                          STAGE_NONE, decode_iq1m_8(block, d_iq1s_grid, ib32, lpos, v);)
SHAINET_IQ_LOWBIT_KERNELS(iq1s, 50,
                          STAGE_G2048_IQ1, decode_iq1s_8(block, s_g, ib32, lpos, v);,
                          STAGE_NONE, decode_iq1s_8(block, d_iq1s_grid, ib32, lpos, v);)

#endif // SHAINET_IQ_LOWBIT_KERNELS_CUH
