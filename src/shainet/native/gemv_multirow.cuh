// Multi-row GEMV: y[M,N] = x[M,K] * dequant(W[N,K]) for 2 <= M <= GEMV_MR_MAX_ROWS.
//
// This is the speculative-decoding verify pass: the target model runs the last committed token
// plus a few drafted ones, so every projection sees a handful of rows. Neither existing path fits:
//   - the GEMV kernels take M through blockIdx.y, so each row re-reads and re-decodes the whole
//     weight -- M times the bytes of a decode step;
//   - the batched path (int8 MMQ, fp16 or TF32 GEMM) decodes the weight into a scratch slice and
//     multiplies with a tensor-core tile sized for hundreds of rows. Its fixed cost per weight is the
//     decode plus the slice round trip, which at 2-8 rows made a 5-token pass cost 5.5x a 1-token
//     step (255 ms against 46.5 ms on the 27B).
//
// What llama.cpp does for batches up to 8 (mmvq), and so what this does: quantize the activation
// rows to int8 once (one scale per 32 values, the MMQ prefill's own scheme), then read each weight
// block once, decode it to int8 and dp4a it against every row.
//
// The int8 activations matter more than the dp4a. A first version kept them fp32: every warp then
// pulls all NR rows out of L1/L2 for every weight block, N * K * NR * 4 bytes in total -- 2.85 GB for
// an 8-row ffn_up against 38 MB of weight -- and the kernel scaled almost linearly with NR (M=8 cost
// ~4.5x M=1 on real IQ3_S weights). int8 cuts that traffic by 4x, and each warp also computes COLS
// adjacent output columns from one register copy of the rows, which divides it again.
//
// Precision: the activations lose what int8 per-32 absmax loses, as in the MMQ prefill and in
// llama.cpp's decode. Single-row decode keeps its fp32 path.
//
// Layout: a 256-value block is 32 groups of eight, and lane l owns group l (sub-block ib32 = l/4,
// position lpos = l%4), i.e. values ib32*32 + lpos*8 .. +7 -- the layout of the existing IQ GEMVs.
// Each type supplies a decoder that turns its eight values into eight int8 q plus (d, m) such that
// value = d * q + m. Every row's eight int8 activations for that group sit in one 32-value scale
// block (b*8 + ib32), so the dot product is xs * (d * dot(q, xq) + m * sum(xq)).

#ifndef SHAINET_GEMV_MULTIROW_CUH
#define SHAINET_GEMV_MULTIROW_CUH

#define GEMV_MR_MAX_ROWS 8

extern "C++" {

__device__ __forceinline__ int mr_pack4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((unsigned int)(d & 0xff) << 24);
}

// Four unsigned magnitudes in g, negated byte-wise where bit t of the 4-bit sign field is set.
// Magnitudes are at most 63 in every grid used here, so a byte-wise two's-complement negate is exact.
__device__ __forceinline__ int mr_signed4(unsigned int g, unsigned int signs4) {
    const unsigned int mask = ((signs4 & 1) ? 0x000000ffu : 0u) | ((signs4 & 2) ? 0x0000ff00u : 0u) |
                              ((signs4 & 4) ? 0x00ff0000u : 0u) | ((signs4 & 8) ? 0xff000000u : 0u);
    return (int)__vsub4(g ^ mask, mask);
}

// ---------------------------------------------------------------------------------------------
// Decoders: block -> q0 (values 0-3), q1 (values 4-7) as packed int8, plus d and m.
// ---------------------------------------------------------------------------------------------

// IQ3_S: d, qs[64], qh[8], signs[32], scales[4].
__device__ __forceinline__ void mr_iq3s(const unsigned char* __restrict__ block, const unsigned int* __restrict__ grid,
                                        int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd = __half2float(*((const __half*)(block + 0)));
    const unsigned char* qs = block + 2;
    const unsigned char* qh = block + 2 + 64;
    const unsigned char* signs = block + 2 + 64 + 8;
    const unsigned char* scales = block + 2 + 64 + 8 + 32;
    const int ls = (scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xf;
    d = dd * (1 + 2 * ls);
    m = 0.0f;
    const unsigned int sg = signs[4 * ib32 + lpos];
    const unsigned int qhb = qh[ib32];
    const unsigned short q2 = *((const unsigned short*)(qs + 8 * ib32 + 2 * lpos));
    q0 = mr_signed4(grid[(q2 & 0xff) | ((qhb << (8 - 2 * lpos)) & 256)], sg & 0xf);
    q1 = mr_signed4(grid[(q2 >> 8) | ((qhb << (7 - 2 * lpos)) & 256)], sg >> 4);
}

// IQ3_XXS: d, qs[64] grid indices, then the scale-and-sign words.
__device__ __forceinline__ void mr_iq3xxs(const unsigned char* __restrict__ block, const unsigned int* __restrict__ grid,
                                          const unsigned char* __restrict__ ksigns,
                                          int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd = __half2float(*((const __half*)(block + 0)));
    const unsigned char* qs = block + 2;
    // The 98-byte block stride only guarantees 2-byte alignment.
    const unsigned short* sp = (const unsigned short*)(qs + 64 + 4 * ib32);
    const unsigned int aux32 = (unsigned int)sp[0] | ((unsigned int)sp[1] << 16);
    d = dd * (0.5f + (aux32 >> 28)) * 0.5f;
    m = 0.0f;
    const unsigned int sg = ksigns[(aux32 >> (7 * lpos)) & 127];
    const unsigned short q2 = *((const unsigned short*)(qs + 8 * ib32 + 2 * lpos));
    q0 = mr_signed4(grid[q2 & 0xff], sg & 0xf);
    q1 = mr_signed4(grid[q2 >> 8], sg >> 4);
}

// IQ4_XS: d, scales_h, scales_l[4], qs[128]. Sub-block ib32 owns qs[16*ib32 .. +16]; its low nibbles
// are values 0-15 and its high nibbles 16-31, so lpos 0/1 read low nibbles and 2/3 high.
__device__ __forceinline__ void mr_iq4xs(const unsigned char* __restrict__ block, const signed char* __restrict__ kv,
                                         int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd = __half2float(*((const __half*)(block + 0)));
    const unsigned short scales_h = *((const unsigned short*)(block + 2));
    const unsigned char* scales_l = block + 4;
    const int ls = ((scales_l[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xF) | (((scales_h >> (2 * ib32)) & 3) << 4);
    d = dd * (float)(ls - 32);
    m = 0.0f;
    // 8 + 16*ib32 + 8*(lpos&1) is a multiple of 8 and so is the 136-byte stride.
    const uint2 q = *((const uint2*)(block + 8 + 16 * ib32 + 8 * (lpos & 1)));
    const int sh = 4 * (lpos >> 1);
    q0 = mr_pack4(kv[(q.x >> sh) & 0xF], kv[(q.x >> (8 + sh)) & 0xF], kv[(q.x >> (16 + sh)) & 0xF], kv[(q.x >> (24 + sh)) & 0xF]);
    q1 = mr_pack4(kv[(q.y >> sh) & 0xF], kv[(q.y >> (8 + sh)) & 0xF], kv[(q.y >> (16 + sh)) & 0xF], kv[(q.y >> (24 + sh)) & 0xF]);
}

// Q4_K: d, dmin, scales[12], qs[128]. 64-value sub-block j owns qs[32*j .. +32] (low nibbles values
// 0-31, high 32-63). For (ib32, lpos): j = ib32/2, the nibble is ib32's parity, the scale group ib32.
__device__ __forceinline__ void mr_q4k(const unsigned char* __restrict__ block,
                                       int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd   = __half2float(*((const __half*)(block + 0)));
    const float dmin = __half2float(*((const __half*)(block + 2)));
    unsigned char sc, mn;
    get_scale_min_k4(ib32, block + 4, &sc, &mn);
    d = dd * (float)sc;
    m = -dmin * (float)mn;
    const uint2 q = *((const uint2*)(block + 16 + 32 * (ib32 >> 1) + 8 * lpos));
    const int sh = 4 * (ib32 & 1);
    q0 = (int)((q.x >> sh) & 0x0F0F0F0Fu);
    q1 = (int)((q.y >> sh) & 0x0F0F0F0Fu);
}

// Q5_K: Q4_K plus qh[32]; value at offset l of sub-block j takes its fifth bit from qh[l], bit 2j
// for the low nibble and 2j+1 for the high one, which is bit ib32 here.
__device__ __forceinline__ void mr_q5k(const unsigned char* __restrict__ block,
                                       int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd   = __half2float(*((const __half*)(block + 0)));
    const float dmin = __half2float(*((const __half*)(block + 2)));
    unsigned char sc, mn;
    get_scale_min_k4(ib32, block + 4, &sc, &mn);
    d = dd * (float)sc;
    m = -dmin * (float)mn;
    const uint2 q = *((const uint2*)(block + 48 + 32 * (ib32 >> 1) + 8 * lpos));
    const uint2 h = *((const uint2*)(block + 16 + 8 * lpos));
    const int sh = 4 * (ib32 & 1);
    q0 = (int)(((q.x >> sh) & 0x0F0F0F0Fu) | (((h.x >> ib32) & 0x01010101u) << 4));
    q1 = (int)(((q.y >> sh) & 0x0F0F0F0Fu) | (((h.y >> ib32) & 0x01010101u) << 4));
}

// Q6_K: ql[128], qh[64], scales[16] (int8), d. For a 128-value chunk and l in [0,32): ql[l] low
// nibble -> value l, ql[l+32] low -> l+32, ql[l] high -> l+64, ql[l+32] high -> l+96, with qh[l]
// bits 0-1, 2-3, 4-5, 6-7 on top, minus 32. (ib32, lpos) is chunk ib32/4, quarter ib32%4, l = lpos*8+t.
__device__ __forceinline__ void mr_q6k(const unsigned char* __restrict__ block,
                                       int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const int chunk = ib32 >> 2;
    const int quarter = ib32 & 3;
    const float dd = __half2float(*((const __half*)(block + 208)));
    const signed char* sc = (const signed char*)(block + 192) + 8 * chunk;
    d = dd * (float)sc[(lpos >> 1) + 2 * quarter];
    m = 0.0f;
    // The 210-byte stride only guarantees 2-byte alignment, so the bytes come in as u16s.
    const unsigned short* qlp = (const unsigned short*)(block + 64 * chunk + 32 * (quarter & 1) + 8 * lpos);
    const unsigned short* qhp = (const unsigned short*)(block + 128 + 32 * chunk + 8 * lpos);
    const unsigned int l0 = (unsigned int)qlp[0] | ((unsigned int)qlp[1] << 16);
    const unsigned int l1 = (unsigned int)qlp[2] | ((unsigned int)qlp[3] << 16);
    const unsigned int h0 = (unsigned int)qhp[0] | ((unsigned int)qhp[1] << 16);
    const unsigned int h1 = (unsigned int)qhp[2] | ((unsigned int)qhp[3] << 16);
    const int lsh = 4 * (quarter >> 1);
    const int hsh = 2 * quarter;
    const unsigned int u0 = ((l0 >> lsh) & 0x0F0F0F0Fu) | (((h0 >> hsh) & 0x03030303u) << 4);
    const unsigned int u1 = ((l1 >> lsh) & 0x0F0F0F0Fu) | (((h1 >> hsh) & 0x03030303u) << 4);
    q0 = (int)__vsub4(u0, 0x20202020u);
    q1 = (int)__vsub4(u1, 0x20202020u);
}

// Q2_K: scales[16], qs[64], d, dmin. See decode_q2k_8 for the index arithmetic.
__device__ __forceinline__ void mr_q2k(const unsigned char* __restrict__ block,
                                       int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const unsigned char* scales = block;
    const unsigned char* qs = block + 16;
    const float dd = __half2float(*((const __half*)(block + 80)));
    const float dmin = __half2float(*((const __half*)(block + 82)));
    const int base = (ib32 * 4 + lpos) * 8;
    const int n = base >> 7;
    const int idx = base & 127;
    const int j = idx >> 5;
    const int sub = (idx & 31) >> 4;
    const int l0 = idx & 15;
    const unsigned char sc = scales[8 * n + 2 * j + sub];
    d = dd * (float)(sc & 0xF);
    m = -dmin * (float)(sc >> 4);
    // qs + 32n + 16sub + l0 with l0 in {0, 8}: 8-byte aligned in a block whose stride (84) is 4-aligned.
    const unsigned int* qp = (const unsigned int*)(qs + 32 * n + 16 * sub + l0);
    const int shift = 2 * j;
    q0 = (int)((qp[0] >> shift) & 0x03030303u);
    q1 = (int)((qp[1] >> shift) & 0x03030303u);
}

// IQ2_XXS: d, qs[64]; each 32-value sub-block is two words: four grid indices, then scale + signs.
__device__ __forceinline__ void mr_iq2xxs(const unsigned char* __restrict__ block,
                                          const unsigned long long* __restrict__ grid,
                                          const unsigned char* __restrict__ ksigns,
                                          int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd = __half2float(*((const __half*)(block + 0)));
    const unsigned char* qs = block + 2;
    unsigned int aux1;
    memcpy(&aux1, qs + 8 * ib32 + 4, sizeof(unsigned int));
    d = dd * (0.5f + (float)(aux1 >> 28)) * 0.25f;
    m = 0.0f;
    const unsigned int signs = ksigns[(aux1 >> (7 * lpos)) & 127];
    const unsigned long long g = grid[qs[8 * ib32 + lpos]];
    q0 = mr_signed4((unsigned int)g, signs & 0xf);
    q1 = mr_signed4((unsigned int)(g >> 32), signs >> 4);
}

// IQ2_XS: d, qs[32 x u16] (9 index bits + 7 sign bits), scales[8].
__device__ __forceinline__ void mr_iq2xs(const unsigned char* __restrict__ block,
                                         const unsigned long long* __restrict__ grid,
                                         const unsigned char* __restrict__ ksigns,
                                         int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd = __half2float(*((const __half*)(block + 0)));
    const unsigned short* qs = (const unsigned short*)(block + 2);
    const unsigned char sc = (block + 2 + 64)[ib32];
    d = dd * (0.5f + (float)((lpos >> 1) ? (sc >> 4) : (sc & 0xf))) * 0.25f;
    m = 0.0f;
    const unsigned short q = qs[4 * ib32 + lpos];
    const unsigned long long g = grid[q & 511];
    const unsigned int signs = ksigns[q >> 9];
    q0 = mr_signed4((unsigned int)g, signs & 0xf);
    q1 = mr_signed4((unsigned int)(g >> 32), signs >> 4);
}

// IQ2_S: d, qs[32], signs[32], qh[8], scales[8]; the tenth index bit comes from qh.
__device__ __forceinline__ void mr_iq2s(const unsigned char* __restrict__ block,
                                        const unsigned long long* __restrict__ grid,
                                        int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd = __half2float(*((const __half*)(block + 0)));
    const unsigned char* qs = block + 2;
    const unsigned char* signs = qs + 32;
    const unsigned char* qh = block + 2 + 64;
    const unsigned char sc = (block + 2 + 64 + 8)[ib32];
    d = dd * (0.5f + (float)((lpos >> 1) ? (sc >> 4) : (sc & 0xf))) * 0.25f;
    m = 0.0f;
    const int gi = qs[4 * ib32 + lpos] | (((int)qh[ib32] << (8 - 2 * lpos)) & 0x300);
    const unsigned long long g = grid[gi];
    const unsigned int sg = signs[4 * ib32 + lpos];
    q0 = mr_signed4((unsigned int)g, sg & 0xf);
    q1 = mr_signed4((unsigned int)(g >> 32), sg >> 4);
}

// IQ1_M: qs[32], qh[16], sc[4 x u16]; signed grid values plus a per-group +/- delta (folded into m).
__device__ __forceinline__ void mr_iq1m(const unsigned char* __restrict__ block, const uint64_t* __restrict__ grid,
                                        int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const unsigned char* qs = block + 0;
    const unsigned char* qh = block + 32;
    const unsigned short* sc = (const unsigned short*)(block + 48);
    const unsigned short scale_u16 =
        (unsigned short)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float dd = __half2float(*((const __half*)&scale_u16));
    const unsigned short scw = sc[ib32 / 2];
    const int sh = 6 * (ib32 % 2);
    d = (lpos < 2) ? dd * (float)(2 * ((scw >> (sh + 0)) & 0x7) + 1)
                   : dd * (float)(2 * ((scw >> (sh + 3)) & 0x7) + 1);
    const unsigned char* q = qs + 4 * ib32;
    const int hb = qh[2 * ib32 + (lpos >> 1)];
    const int idx = q[lpos] | (((lpos & 1) ? (hb << 4) : (hb << 8)) & 0x700);
    m = d * (((lpos & 1) ? (hb & 0x80) : (hb & 0x08)) ? -SHAINET_IQ1S_DELTA_F : SHAINET_IQ1S_DELTA_F);
    const uint64_t g = grid[idx];
    q0 = (int)(unsigned int)g;
    q1 = (int)(unsigned int)(g >> 32);
}

// IQ1_S: d, qs[32], qh[8 x u16] holding scale, high index bits and the delta sign.
__device__ __forceinline__ void mr_iq1s(const unsigned char* __restrict__ block, const uint64_t* __restrict__ grid,
                                        int ib32, int lpos, int& q0, int& q1, float& d, float& m) {
    const float dd = __half2float(*((const __half*)(block + 0)));
    const unsigned char* qs = block + 2;
    const unsigned short h = ((const unsigned short*)(block + 2 + 32))[ib32];
    d = dd * (float)(2 * ((h >> 12) & 7) + 1);
    m = d * ((h & 0x8000) ? -SHAINET_IQ1S_DELTA_F : SHAINET_IQ1S_DELTA_F);
    const uint64_t g = grid[qs[4 * ib32 + lpos] | (((h >> (3 * lpos)) & 7) << 8)];
    q0 = (int)(unsigned int)g;
    q1 = (int)(unsigned int)(g >> 32);
}

// ---------------------------------------------------------------------------------------------
// Kernel: persistent grid, one warp per COLS adjacent output columns, NR x COLS accumulators.
// xq [NR, K] int8, xs [NR, K/32] scales.
// ---------------------------------------------------------------------------------------------

#define SHAINET_GEMV_MR(NAME, BYTES, HAS_M, STAGE, DECODE)                                        \
template <int NR, int COLS>                                                                       \
__global__ void __launch_bounds__(256)                                                            \
gemv_mr_##NAME##_kernel(const signed char* __restrict__ xq, const float* __restrict__ xs,         \
                        const unsigned char* __restrict__ w, float* __restrict__ y, int N, int K) { \
    STAGE                                                                                         \
    const int lane = threadIdx.x & 31;                                                            \
    const int wpb = blockDim.x >> 5;                                                              \
    const int nblocks = K / GGUF_QK_K;                                                            \
    const int nb32 = K >> 5;                                                                      \
    const long row_bytes = (long)nblocks * (BYTES);                                               \
    const int ib32 = lane >> 2;                                                                   \
    const int lpos = lane & 3;                                                                    \
    const signed char* xl = xq + ib32 * 32 + lpos * 8;                                            \
    const int groups = (N + COLS - 1) / COLS;                                                     \
    for (int g = blockIdx.x * wpb + (threadIdx.x >> 5); g < groups; g += gridDim.x * wpb) {       \
        const int n0 = g * COLS;                                                                  \
        float acc[COLS][NR];                                                                      \
        _Pragma("unroll")                                                                          \
        for (int c = 0; c < COLS; ++c)                                                            \
            _Pragma("unroll")                                                                      \
            for (int r = 0; r < NR; ++r) acc[c][r] = 0.0f;                                        \
        for (int b = 0; b < nblocks; ++b) {                                                       \
            int x0[NR], x1[NR];                                                                   \
            float sx[NR], sq[NR];                                                                 \
            _Pragma("unroll")                                                                      \
            for (int r = 0; r < NR; ++r) {                                                        \
                const int2 v = *((const int2*)(xl + (long)r * K + b * GGUF_QK_K));                \
                x0[r] = v.x;                                                                      \
                x1[r] = v.y;                                                                      \
                sx[r] = xs[(long)r * nb32 + b * 8 + ib32];                                        \
                if (HAS_M) sq[r] = sx[r] * (float)__dp4a(0x01010101, x1[r], __dp4a(0x01010101, x0[r], 0)); \
            }                                                                                     \
            _Pragma("unroll")                                                                      \
            for (int c = 0; c < COLS; ++c) {                                                      \
                if (COLS > 1 && n0 + c >= N) break;                                               \
                const unsigned char* block = w + (long)(n0 + c) * row_bytes + (long)b * (BYTES);  \
                int q0, q1;                                                                       \
                float d, m;                                                                       \
                DECODE                                                                            \
                _Pragma("unroll")                                                                  \
                for (int r = 0; r < NR; ++r) {                                                    \
                    const int dot = __dp4a(q1, x1[r], __dp4a(q0, x0[r], 0));                      \
                    acc[c][r] += d * sx[r] * (float)dot;                                          \
                    if (HAS_M) acc[c][r] += m * sq[r];                                            \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        _Pragma("unroll")                                                                          \
        for (int c = 0; c < COLS; ++c) {                                                          \
            _Pragma("unroll")                                                                      \
            for (int r = 0; r < NR; ++r) {                                                        \
                float a = acc[c][r];                                                              \
                _Pragma("unroll")                                                                  \
                for (int s = 16; s > 0; s >>= 1) a += __shfl_down_sync(0xFFFFFFFF, a, s);         \
                if (lane == 0 && n0 + c < N) y[(long)r * N + n0 + c] = a;                         \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
}

#define STAGE_MR_IQ3S                                                                             \
    __shared__ unsigned int s_g3[512];                                                            \
    for (int i = threadIdx.x; i < 512; i += blockDim.x) s_g3[i] = d_iq3s_grid[i];                 \
    __syncthreads();

#define STAGE_MR_IQ3XXS                                                                           \
    __shared__ unsigned int s_g3[256];                                                            \
    __shared__ unsigned char s_k3[128];                                                           \
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_g3[i] = d_iq3xxs_grid[i];               \
    for (int i = threadIdx.x; i < 128; i += blockDim.x) s_k3[i] = d_ksigns_iq2xs[i];              \
    __syncthreads();

#define STAGE_MR_IQ4                                                                              \
    __shared__ signed char s_kv[16];                                                              \
    if (threadIdx.x < 16) s_kv[threadIdx.x] = kvalues_iq4nl_dev[threadIdx.x];                     \
    __syncthreads();

#define STAGE_MR_IQ2S                                                                             \
    __shared__ unsigned long long s_g[1024];                                                      \
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s_g[i] = d_iq2s_grid[i];                 \
    __syncthreads();

SHAINET_GEMV_MR(iq3s, 110, false, STAGE_MR_IQ3S, mr_iq3s(block, s_g3, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(iq3xxs, 98, false, STAGE_MR_IQ3XXS, mr_iq3xxs(block, s_g3, s_k3, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(iq4xs, 136, false, STAGE_MR_IQ4, mr_iq4xs(block, s_kv, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(q4k, 144, true, STAGE_NONE, mr_q4k(block, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(q5k, 176, true, STAGE_NONE, mr_q5k(block, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(q6k, 210, false, STAGE_NONE, mr_q6k(block, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(q2k, 84, true, STAGE_NONE, mr_q2k(block, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(iq2xxs, 66, false, STAGE_G256_SIGNS, mr_iq2xxs(block, s_g, s_ks, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(iq2xs, 74, false, STAGE_G512_SIGNS, mr_iq2xs(block, s_g, s_ks, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(iq2s, 82, false, STAGE_MR_IQ2S, mr_iq2s(block, s_g, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(iq1m, 56, true, STAGE_G2048_IQ1, mr_iq1m(block, s_g, ib32, lpos, q0, q1, d, m);)
SHAINET_GEMV_MR(iq1s, 50, true, STAGE_G2048_IQ1, mr_iq1s(block, s_g, ib32, lpos, q0, q1, d, m);)

// Columns per warp for an NR-row call. 0 = the default; gemv_mr_set_cols overrides it (benchmarks).
static int g_gemv_mr_cols = 0;

// Measured on the 27B's real weights (6 shapes, IQ3_S/IQ3_XXS/IQ4_XS/Q4_K/Q6_K): one column per warp
// is fastest up to 5 rows (4 rows cost 1.15x one fp32 row), and 2 columns win from 6.
static int gemv_mr_default_cols(int nr) { return nr <= 5 ? 1 : 2; }

#define GEMV_MR_WARPS 8

#define GEMV_MR_CASE(NAME)                                                                        \
    template <int NR, int COLS>                                                                   \
    static void gemv_mr_launch2_##NAME(const signed char* xq, const float* xs, const unsigned char* w, \
                                       float* y, int N, int K) {                                  \
        const int groups = (N + COLS - 1) / COLS;                                                 \
        int blocks = (groups + GEMV_MR_WARPS - 1) / GEMV_MR_WARPS;                                \
        const int cap = iq_gemv_grid(1 << 30) * 2;                                                \
        if (blocks > cap) blocks = cap;                                                           \
        gemv_mr_##NAME##_kernel<NR, COLS><<<blocks, GEMV_MR_WARPS * 32>>>(xq, xs, w, y, N, K);    \
    }                                                                                             \
    template <int NR>                                                                             \
    static void gemv_mr_launch_##NAME(const signed char* xq, const float* xs, const unsigned char* w, \
                                      float* y, int N, int K) {                                   \
        const int cols = g_gemv_mr_cols > 0 ? g_gemv_mr_cols : gemv_mr_default_cols(NR);          \
        if (cols >= 4) gemv_mr_launch2_##NAME<NR, 4>(xq, xs, w, y, N, K);                         \
        else if (cols == 2) gemv_mr_launch2_##NAME<NR, 2>(xq, xs, w, y, N, K);                    \
        else gemv_mr_launch2_##NAME<NR, 1>(xq, xs, w, y, N, K);                                   \
    }                                                                                             \
    static bool gemv_mr_dispatch_##NAME(const signed char* xq, const float* xs, const unsigned char* w, \
                                        float* y, int M, int N, int K) {                          \
        switch (M) {                                                                              \
        case 1: gemv_mr_launch_##NAME<1>(xq, xs, w, y, N, K); return true;                        \
        case 2: gemv_mr_launch_##NAME<2>(xq, xs, w, y, N, K); return true;                        \
        case 3: gemv_mr_launch_##NAME<3>(xq, xs, w, y, N, K); return true;                        \
        case 4: gemv_mr_launch_##NAME<4>(xq, xs, w, y, N, K); return true;                        \
        case 5: gemv_mr_launch_##NAME<5>(xq, xs, w, y, N, K); return true;                        \
        case 6: gemv_mr_launch_##NAME<6>(xq, xs, w, y, N, K); return true;                        \
        case 7: gemv_mr_launch_##NAME<7>(xq, xs, w, y, N, K); return true;                        \
        case 8: gemv_mr_launch_##NAME<8>(xq, xs, w, y, N, K); return true;                        \
        default: return false;                                                                    \
        }                                                                                         \
    }

GEMV_MR_CASE(iq3s)
GEMV_MR_CASE(iq3xxs)
GEMV_MR_CASE(iq4xs)
GEMV_MR_CASE(q4k)
GEMV_MR_CASE(q5k)
GEMV_MR_CASE(q6k)
GEMV_MR_CASE(q2k)
GEMV_MR_CASE(iq2xxs)
GEMV_MR_CASE(iq2xs)
GEMV_MR_CASE(iq2s)
GEMV_MR_CASE(iq1m)
GEMV_MR_CASE(iq1s)

// One warp per (row, 32-value block): scale = max|x| / 127, q = round(x / scale).
__global__ void gemv_mr_quantize_kernel(const float* __restrict__ x, signed char* __restrict__ xq,
                                        float* __restrict__ xs, int M, int K) {
    const long wid = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = K >> 5;
    if (wid >= (long)M * nb) return;
    const int r = (int)(wid / nb), b = (int)(wid - (long)r * nb);
    const float v = x[(long)r * K + b * 32 + lane];
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
    const float scale = a / 127.0f;
    xq[(long)r * K + b * 32 + lane] = (signed char)(scale > 0.0f ? __float2int_rn(v / scale) : 0);
    if (lane == 0) xs[(long)r * nb + b] = scale;
}

// Grow-only scratch for the quantized activations.
static signed char* g_mr_xq = nullptr;
static float* g_mr_xs = nullptr;
static long g_mr_cap = 0;
// The last input quantized, so projections that share an input (q/k/v/gate, gate/up) quantize it
// once. Only trusted inside gemv_mr_share_begin/end: activation buffers are recycled, so a matching
// pointer alone does not prove the contents did not change.
static const float* g_mr_last_x = nullptr;
static int g_mr_last_m = 0, g_mr_last_k = 0;
static bool g_mr_sharing = false;

} // extern "C++"

// Benchmark hook: force the columns per warp (1, 2 or 4); 0 restores the default.
void gemv_mr_set_cols(int cols) { g_gemv_mr_cols = cols; }

// Within begin/end, consecutive gemv_mr calls on the same input pointer and shape reuse its int8
// quantization.
void gemv_mr_share_begin() { g_mr_sharing = true; g_mr_last_x = nullptr; }
void gemv_mr_share_end() { g_mr_sharing = false; g_mr_last_x = nullptr; }

// Exported: ggml type id, then the GEMV arguments. Returns 0 when the type or row count is not
// covered (or scratch cannot be allocated), so the caller falls back to its batched path.
int gemv_mr(int ggml_type, const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    if (M < 1 || M > GEMV_MR_MAX_ROWS || K % GGUF_QK_K != 0) return 0;
    switch (ggml_type) {
    case 10: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21: case 22: case 23: case 29: break;
    default: return 0;
    }
    const long need = (long)M * K;
    if (need > g_mr_cap) {
        if (g_mr_xq) cudaFree(g_mr_xq);
        if (g_mr_xs) cudaFree(g_mr_xs);
        g_mr_xq = nullptr;
        g_mr_xs = nullptr;
        long cap = (long)GEMV_MR_MAX_ROWS * K;
        if (cap < need) cap = need;
        if (cudaMalloc(&g_mr_xq, cap) != cudaSuccess || cudaMalloc(&g_mr_xs, cap / 32 * sizeof(float)) != cudaSuccess) {
            g_mr_cap = 0;
            g_mr_xq = nullptr;
            g_mr_xs = nullptr;
            cudaGetLastError();
            return 0;
        }
        g_mr_cap = cap;
        g_mr_last_x = nullptr;
    }
    if (!(g_mr_sharing && g_mr_last_x == x && g_mr_last_m == M && g_mr_last_k == K)) {
        const long warps = (long)M * (K / 32);
        gemv_mr_quantize_kernel<<<(unsigned int)((warps * 32 + 255) / 256), 256>>>(x, g_mr_xq, g_mr_xs, M, K);
        g_mr_last_x = x;
        g_mr_last_m = M;
        g_mr_last_k = K;
    }
    const signed char* xq = g_mr_xq;
    const float* xs = g_mr_xs;
    bool ok = false;
    switch (ggml_type) {
    case 10: ok = gemv_mr_dispatch_q2k(xq, xs, w, y, M, N, K); break;
    case 12: ok = gemv_mr_dispatch_q4k(xq, xs, w, y, M, N, K); break;
    case 13: ok = gemv_mr_dispatch_q5k(xq, xs, w, y, M, N, K); break;
    case 14: ok = gemv_mr_dispatch_q6k(xq, xs, w, y, M, N, K); break;
    case 16: ok = gemv_mr_dispatch_iq2xxs(xq, xs, w, y, M, N, K); break;
    case 17: ok = gemv_mr_dispatch_iq2xs(xq, xs, w, y, M, N, K); break;
    case 18: ok = gemv_mr_dispatch_iq3xxs(xq, xs, w, y, M, N, K); break;
    case 19: ok = gemv_mr_dispatch_iq1s(xq, xs, w, y, M, N, K); break;
    case 21: ok = gemv_mr_dispatch_iq3s(xq, xs, w, y, M, N, K); break;
    case 22: ok = gemv_mr_dispatch_iq2s(xq, xs, w, y, M, N, K); break;
    case 23: ok = gemv_mr_dispatch_iq4xs(xq, xs, w, y, M, N, K); break;
    case 29: ok = gemv_mr_dispatch_iq1m(xq, xs, w, y, M, N, K); break;
    }
    if (!ok) return 0;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemv_mr: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

#endif // SHAINET_GEMV_MULTIROW_CUH
