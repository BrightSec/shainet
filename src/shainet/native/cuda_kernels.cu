#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include "iq_tables.cuh"  // i-quant codebook grids in constant memory

// Device kernels
// Simple row-wise softmax kernel. This version runs one thread per row and
// performs the computation sequentially. It uses the row maximum for numerical
// stability.
__global__ void softmax_rows_kernel(float* out, const float* in, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = in + row * cols;
    float* row_out = out + row * cols;

    // Find the maximum value for numerical stability
    float max_val = row_in[0];
    for (int j = 1; j < cols; ++j) {
        float v = row_in[j];
        if (v > max_val) max_val = v;
    }

    // Compute exponentials and their sum
    float sum = 0.0;
    for (int j = 0; j < cols; ++j) {
        float e = expf(row_in[j] - max_val);
        row_out[j] = e;
        sum += e;
    }

    // Normalize
    for (int j = 0; j < cols; ++j) {
        row_out[j] /= sum;
    }
}

__global__ void relu_backward_kernel(float* output, const float* input, const float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = input[idx] > 0.0 ? grad[idx] : 0.0;
}

// Host wrapper functions
extern "C" {
void softmax_rows(float* out, const float* in, int rows, int cols) {
    softmax_rows_kernel<<<rows, 1>>>(out, in, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax_rows: %s\n", cudaGetErrorString(err));
    }
}

void relu_backward(float* output, const float* input, const float* grad, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    relu_backward_kernel<<<blocks, threads_per_block>>>(output, input, grad, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in relu_backward: %s\n", cudaGetErrorString(err));
    }
}

__global__ void dropout_kernel(float* out, const float* in, int rows, int cols, double drop_p, unsigned long long seed) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const float *row_in = in + row * cols;
    float *row_out = out + row * cols;
    if(drop_p >= 1.0) {
        for(int j=0;j<cols;++j) row_out[j] = 0.0;
        return;
    }
    if(drop_p <= 0.0) {
        for(int j=0;j<cols;++j) row_out[j] = row_in[j];
        return;
    }
    curandState state;
    curand_init(seed + row, 0, 0, &state);
    float scale = 1.0 / (1.0 - drop_p);
    for(int j=0;j<cols;++j){
        float r = curand_uniform(&state);
        row_out[j] = r < drop_p ? 0.0 : row_in[j] * scale;
    }
}

void dropout(float* out, const float* in, int rows, int cols, double drop_p, unsigned long long seed) {
    dropout_kernel<<<rows, 1>>>(out, in, rows, cols, drop_p, seed);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in dropout: %s\n", cudaGetErrorString(err));
    }
}

// One block per row, threads cooperating across the row. It was <<<rows, 1>>> with a
// serial column loop, which made it useless for batching a d_model-wide gather: one
// thread copying 2048 floats per row.
__global__ void gather_rows_kernel(float* out, const float* in, const int* ids, int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    int id = ids[row];
    const float *row_in = in + (long)id * cols;
    float *row_out = out + (long)row * cols;
    for(int j = threadIdx.x; j < cols; j += blockDim.x){
        row_out[j] = row_in[j];
    }
}

// No cudaDeviceSynchronize: it is ordered on the default stream against whatever
// consumes the gathered batch, and the blocking sync per call was costing a device
// round trip on every gather.
void gather_rows(float* out, const float* in, const int* ids, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    int threads = cols < 256 ? 32 : 256;
    gather_rows_kernel<<<rows, threads>>>(out, in, ids, rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gather_rows: %s\n", cudaGetErrorString(err));
    }
}

__global__ void row_mean_var_kernel(const float* in, float* mean, float* var,
                                    int rows, int cols) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const float *row_in = in + row * cols;
    float sum = 0.0;
    float sq_sum = 0.0;
    for(int j=0;j<cols;++j){
        float v = row_in[j];
        sum += v;
        sq_sum += v*v;
    }
    float m = sum / cols;
    mean[row] = m;
    var[row] = sq_sum / cols - m*m;
}

void row_mean_var(const float* in, float* mean, float* var, int rows, int cols) {
    row_mean_var_kernel<<<rows, 1>>>(in, mean, var, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void apply_layer_norm_kernel(float* out, const float* in,
                                        const float* mean, const float* var,
                                        int rows, int cols, double epsilon) {
    int row = blockIdx.x;
    if(row >= rows) return;
    const float *row_in = in + row * cols;
    float *row_out = out + row * cols;
    float m = mean[row];
    float denom = sqrtf(var[row] + epsilon);
    for(int j=0;j<cols;++j){
        row_out[j] = (row_in[j] - m) / denom;
    }
}

void apply_layer_norm(float* out, const float* in,
                      const float* mean, const float* var,
                      int rows, int cols, double epsilon) {
    apply_layer_norm_kernel<<<rows, 1>>>(out, in, mean, var, rows, cols, epsilon);
    cudaDeviceSynchronize();
}

__global__ void slice_cols_kernel(float* out, const float* in, int rows, int src_cols, int start, int len){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row >= rows) return;
    for(; col < len; col += blockDim.x){
        out[row * len + col] = in[row * src_cols + start + col];
    }
}

void slice_cols(float* out, const float* in, int rows, int src_cols, int start, int len){
    int threads = len < 1024 ? len : 1024;
    slice_cols_kernel<<<rows, threads>>>(out, in, rows, src_cols, start, len);
    cudaDeviceSynchronize();
}

__global__ void set_cols_kernel(float* out, const float* in, int rows, int dst_cols, int start, int len){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(row >= rows) return;
    for(; col < len; col += blockDim.x){
        out[row * dst_cols + start + col] = in[row * len + col];
    }
}

void set_cols(float* out, const float* in, int rows, int dst_cols, int start, int len){
    int threads = len < 1024 ? len : 1024;
    set_cols_kernel<<<rows, threads>>>(out, in, rows, dst_cols, start, len);
    cudaDeviceSynchronize();
}

__global__ void count_token_pairs_kernel(const int* a, const int* b, const int* freq,
                                         int pair_count, int vocab_size, int* counts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pair_count) return;
    int offset = a[idx] * vocab_size + b[idx];
    atomicAdd(&counts[offset], freq[idx]);
}

void count_token_pairs(const int* a, const int* b, const int* freq,
                       int pair_count, int vocab_size, int* counts){
    int blocks = (pair_count + 255) / 256;
    count_token_pairs_kernel<<<blocks, 256>>>(a, b, freq, pair_count, vocab_size, counts);
    cudaDeviceSynchronize();
}

__global__ void layer_norm_backward_kernel(float* d_x, float* d_gamma, float* d_beta,
                                           const float* d_out, const float* x,
                                           const float* gamma, const float* mean,
                                           const float* var, const float* norm,
                                           int rows, int cols, double epsilon) {
    int row = blockIdx.x;
    if(row >= rows) return;

    const float *x_row = x + row * cols;
    const float *dout_row = d_out + row * cols;
    const float *norm_row = norm + row * cols;
    float *dx_row = d_x + row * cols;

    float m = mean[row];
    float v = var[row];
    float denom = sqrtf(v + epsilon);
    float inv = 1.0 / denom;
    float col_f = (float)cols;

    // Compute sum_dout_gamma and sum_dout_gamma_norm
    float sum_dout_gamma = 0.0;
    float sum_dout_gamma_norm = 0.0;
    for(int j = 0; j < cols; ++j) {
        float doutg = dout_row[j] * gamma[j];
        sum_dout_gamma += doutg;
        sum_dout_gamma_norm += doutg * (x_row[j] - m);

        // Accumulate gradients for gamma and beta
        atomicAdd(&d_gamma[j], dout_row[j] * norm_row[j]);
        atomicAdd(&d_beta[j], dout_row[j]);
    }

    // Compute d_x
    for(int j = 0; j < cols; ++j) {
        float xm = x_row[j] - m;
        float doutg = dout_row[j] * gamma[j];
        dx_row[j] = inv * (doutg - sum_dout_gamma/col_f - xm * inv*inv / col_f * sum_dout_gamma_norm);
    }
}

void layer_norm_backward(float* d_x, float* d_gamma, float* d_beta,
                         const float* d_out, const float* x,
                         const float* gamma, const float* mean,
                         const float* var, const float* norm,
                         int rows, int cols, double epsilon) {
    layer_norm_backward_kernel<<<rows, 1>>>(d_x, d_gamma, d_beta, d_out, x,
                                            gamma, mean, var, norm,
                                            rows, cols, epsilon);
    cudaDeviceSynchronize();
}

__global__ void sum_cols_kernel(float* out, const float* in, int rows, int cols) {
    int col = blockIdx.x;
    if(col >= cols) return;

    float sum = 0.0;
    for(int i = 0; i < rows; ++i) {
        sum += in[i * cols + col];
    }
    out[col] = sum;
}

void sum_cols(float* out, const float* in, int rows, int cols) {
    sum_cols_kernel<<<cols, 1>>>(out, in, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void mul_row_vector_kernel(float* matrix, const float* vec, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int col = idx % cols;
    matrix[idx] *= vec[col];
}

void mul_row_vector(float* matrix, const float* vec, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    mul_row_vector_kernel<<<blocks, threads_per_block>>>(matrix, vec, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in mul_row_vector: %s\n", cudaGetErrorString(err));
    }
}

__global__ void transpose_kernel(float* out, const float* in, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int row = idx / cols;
    int col = idx % cols;

    // Transpose: out[col][row] = in[row][col]
    // In row-major: out[col * rows + row] = in[row * cols + col]
    out[col * rows + row] = in[row * cols + col];
}

void transpose(float* out, const float* in, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    transpose_kernel<<<blocks, threads_per_block>>>(out, in, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in transpose: %s\n", cudaGetErrorString(err));
    }
}

__global__ void sigmoid_forward_kernel(float* activations, float* derivatives, const float* linear, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = linear[idx];
    // Sigmoid: 1 / (1 + expf(-x))
    float exp_neg_val = expf(-val);
    float sigmoid_val = 1.0 / (1.0 + exp_neg_val);
    
    activations[idx] = sigmoid_val;
    // Sigmoid derivative: σ(x) * (1 - σ(x))
    derivatives[idx] = sigmoid_val * (1.0 - sigmoid_val);
}

void sigmoid_forward(float* activations, float* derivatives, const float* linear, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    sigmoid_forward_kernel<<<blocks, threads_per_block>>>(activations, derivatives, linear, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in sigmoid_forward: %s\n", cudaGetErrorString(err));
    }
}

// hidden = silu(gate) * up, the SwiGLU activation, fused so the FFN's gate and up
// projections never have to come back to the host to be combined. Uses expf
// rather than __expf: the fast intrinsic drifts far enough from the CPU path to
// break device/host parity, and this kernel is memory bound anyway.
__global__ void swiglu_forward_kernel(float* hidden, const float* gate, const float* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float g = gate[idx];
    hidden[idx] = (g / (1.0f + expf(-g))) * up[idx];
}

// Deliberately does NOT cudaDeviceSynchronize: this sits between two GEMV launches
// on the default stream, so stream ordering already guarantees the gate/up writes
// are visible. Syncing here would reintroduce the pipeline stall that keeping
// activations on the device exists to remove.
void swiglu_forward(float* hidden, const float* gate, const float* up, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    swiglu_forward_kernel<<<blocks, threads_per_block>>>(hidden, gate, up, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in swiglu_forward: %s\n", cudaGetErrorString(err));
    }
}

// out = x / sqrt(mean(x^2) + eps) * gamma, one BLOCK per row so the sum of squares
// is a shared-memory reduction rather than a host loop.
//
// Until this existed there was no RMSNorm kernel at all, and the CudaMatrix path
// read the row back to the host, normalised it there and pushed it back, which made
// a device-resident block chain impossible: every norm broke the chain.
//
// Accumulates in float, not double. The host path sums in Float64, so results are
// close but not bit-identical; the specs assert parity to a tolerance and the real
// check is that greedy decoding still picks the same tokens.
__global__ void rms_norm_forward_kernel(float* out, const float* x, const float* gamma,
                                        int rows, int cols, float eps) {
    // One block per row. At decode that is ONE block for the whole norm, so the kernel is pure
    // latency: the original scalar loads and an eight-step halving reduction with a __syncthreads
    // each took ~17 us for a 5120-wide row, 128 times a token. float4 loads (when the row allows),
    // up to 1024 threads, and a warp-shuffle reduction bring it to a few microseconds.
    __shared__ float warp_sums[32];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* xr = x + (size_t)row * (size_t)cols;
    float* orow = out + (size_t)row * (size_t)cols;
    const bool vec = (cols & 3) == 0;

    float local = 0.0f;
    if (vec) {
        const float4* x4 = (const float4*)xr;
        for (int j = threadIdx.x; j < (cols >> 2); j += blockDim.x) {
            float4 v = x4[j];
            local += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int j = threadIdx.x; j < cols; j += blockDim.x) {
            float v = xr[j];
            local += v * v;
        }
    }
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) local += __shfl_down_sync(0xFFFFFFFF, local, s);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = local;
    __syncthreads();
    if (warp == 0) {
        const int nwarps = (blockDim.x + 31) >> 5;
        float t = lane < nwarps ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) t += __shfl_down_sync(0xFFFFFFFF, t, s);
        if (lane == 0) warp_sums[0] = t;
    }
    __syncthreads();

    float inv = 1.0f / sqrtf(warp_sums[0] / (float)cols + eps);
    if (vec) {
        const float4* x4 = (const float4*)xr;
        const float4* g4 = (const float4*)gamma;
        float4* o4 = (float4*)orow;
        for (int j = threadIdx.x; j < (cols >> 2); j += blockDim.x) {
            float4 v = x4[j], g = g4[j];
            o4[j] = make_float4(v.x * inv * g.x, v.y * inv * g.y, v.z * inv * g.z, v.w * inv * g.w);
        }
    } else {
        for (int j = threadIdx.x; j < cols; j += blockDim.x) {
            orow[j] = xr[j] * inv * gamma[j];
        }
    }
}

// dst[idx[r]] += w[r] * src[r], the routing weight applied as the batch is returned
// to token order.
//
// No atomics: a token selects a given expert at most once, so the rows of ONE launch
// never collide, and separate launches are ordered on the default stream.
__global__ void scatter_add_rows_kernel(float* __restrict__ dst,
                                        const float* __restrict__ src,
                                        const int* __restrict__ idx,
                                        const float* __restrict__ w,
                                        int n, int cols) {
    int r = blockIdx.x;
    if (r >= n) return;
    const float* s = src + (long)r * cols;
    float* d = dst + (long)idx[r] * cols;
    float wr = w[r];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) d[c] += wr * s[c];
}

// No cudaDeviceSynchronize, same reasoning as swiglu_forward: ordered on the default
// stream between the launches that produce and consume the batch.
void scatter_add_rows(float* dst, const float* src, const int* idx,
                      const float* w, int n, int cols) {
    if (n <= 0 || cols <= 0) return;
    int threads = cols < 256 ? 32 : 256;
    scatter_add_rows_kernel<<<n, threads>>>(dst, src, idx, w, n, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in scatter_add_rows: %s\n", cudaGetErrorString(err));
    }
}

// No cudaDeviceSynchronize, same reasoning as swiglu_forward: it is ordered on the
// default stream between the launches that produce and consume the row.
void rms_norm_forward(float* out, const float* x, const float* gamma,
                      int rows, int cols, float eps) {
    // Wide blocks only when a row is wide enough to use them: with many rows (prefill) the grid
    // already fills the card, and with one row (decode) the threads are the only parallelism.
    int threads_per_block = (rows <= 8 && cols >= 4096) ? 1024 : 256;

    rms_norm_forward_kernel<<<rows, threads_per_block>>>(out, x, gamma, rows, cols, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rms_norm_forward: %s\n", cudaGetErrorString(err));
    }
}

// dst = dst + src, elementwise, for the block's residual adds. cuBLAS axpy could do
// this, but it needs a handle per call and the residual is on the hot path.
__global__ void add_inplace_kernel(float* dst, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] += src[idx];
}

void add_inplace(float* dst, const float* src, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    add_inplace_kernel<<<blocks, threads_per_block>>>(dst, src, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in add_inplace: %s\n", cudaGetErrorString(err));
    }
}

// Rotary position embedding, HF half-split layout, applied in place to a single
// token's row of `heads` heads each `head_dim` wide.
//
// Until this existed RoPE ran on the host, which is the whole reason q/k/v had to be
// read back from the device every layer. inv_freq is precomputed once per block and
// kept on the device, so a decode step passes only the position.
//
// One block per head, half the head_dim worth of threads doing the pair rotation.
// rot_dim is the number of leading dimensions that are rotated. Qwen3.5 sets
// partial_rotary_factor 0.25 on a head_dim of 256, so only the first 64 are rotated and the
// remaining 192 pass through untouched. rot_dim == head_dim is the ordinary full-rotary case.
__global__ void rope_forward_kernel(float* x, const float* inv_freq, int pos,
                                    int heads, int head_dim, int rot_dim) {
    int head = blockIdx.x;
    if (head >= heads) return;
    int half = rot_dim / 2;
    float* row = x + (size_t)head * (size_t)head_dim;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = (float)pos * inv_freq[i];
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = row[i];
        float x1 = row[i + half];
        row[i] = x0 * c - x1 * s;
        row[i + half] = x1 * c + x0 * s;
    }
}

void rope_forward(float* x, const float* inv_freq, int pos, int heads, int head_dim, int rot_dim) {
    int threads = 128;
    if (rot_dim <= 0 || rot_dim > head_dim) rot_dim = head_dim;
    rope_forward_kernel<<<heads, threads>>>(x, inv_freq, pos, heads, head_dim, rot_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rope_forward: %s\n", cudaGetErrorString(err));
    }
}

// Qwen3 QK-norm: RMSNorm applied independently to each head's slice, over head_dim,
// in place. Same shape of reduction as rms_norm_forward but per head rather than per
// row, and gamma is shared across heads.
__global__ void head_rmsnorm_kernel(float* x, const float* gamma, int heads,
                                    int head_dim, float eps) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    if (head >= heads) return;
    float* row = x + (size_t)head * (size_t)head_dim;

    float local = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float v = row[j];
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv = 1.0f / sqrtf(sdata[0] / (float)head_dim + eps);
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        row[j] = row[j] * inv * gamma[j];
    }
}

// --- Multi-row variants for device-resident PREFILL -------------------------------
//
// The single-row kernels above serve decode, where there is one token and one
// position. Prefill has `rows` tokens at consecutive positions, laid out
// [rows, heads * head_dim] row-major, so these take a row dimension and derive each
// token's position from base_pos. The math is identical per row, which is what makes
// parity with the host path assertable.

__global__ void rope_forward_rows_kernel(float* x, const float* inv_freq, int base_pos,
                                        int rows, int heads, int head_dim, int rot_dim) {
    int head = blockIdx.x;
    int r = blockIdx.y;
    if (head >= heads || r >= rows) return;
    int half = rot_dim / 2;
    int stride = heads * head_dim;
    float* row = x + (size_t)r * (size_t)stride + (size_t)head * (size_t)head_dim;
    float pos = (float)(base_pos + r);

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = pos * inv_freq[i];
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = row[i];
        float x1 = row[i + half];
        row[i] = x0 * c - x1 * s;
        row[i + half] = x1 * c + x0 * s;
    }
}

void rope_forward_rows(float* x, const float* inv_freq, int base_pos,
                       int rows, int heads, int head_dim, int rot_dim) {
    if (rows <= 0 || heads <= 0) return;
    int threads = 128;
    if (rot_dim <= 0 || rot_dim > head_dim) rot_dim = head_dim;
    dim3 grid(heads, rows);
    rope_forward_rows_kernel<<<grid, threads>>>(x, inv_freq, base_pos, rows, heads, head_dim, rot_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rope_forward_rows: %s\n", cudaGetErrorString(err));
    }
}

__global__ void head_rmsnorm_rows_kernel(float* x, const float* gamma, int rows,
                                        int heads, int head_dim, float eps) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int r = blockIdx.y;
    if (head >= heads || r >= rows) return;
    int stride = heads * head_dim;
    float* row = x + (size_t)r * (size_t)stride + (size_t)head * (size_t)head_dim;

    float local = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float v = row[j];
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv = 1.0f / sqrtf(sdata[0] / (float)head_dim + eps);
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        row[j] = row[j] * inv * gamma[j];
    }
}

void head_rmsnorm_rows(float* x, const float* gamma, int rows, int heads,
                       int head_dim, float eps) {
    if (rows <= 0 || heads <= 0) return;
    int threads = 128; // power of two: the halving reduction above requires it
    size_t shmem = threads * sizeof(float);
    dim3 grid(heads, rows);
    head_rmsnorm_rows_kernel<<<grid, threads, shmem>>>(x, gamma, rows, heads, head_dim, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in head_rmsnorm_rows: %s\n", cudaGetErrorString(err));
    }
}

// x[r, c] += bias[c]: the projection biases are per output column, broadcast over
// tokens. Qwen2 has them, LLaMA does not, so this is a no-op path for some models.
__global__ void add_bias_rows_kernel(float* x, const float* bias, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float* row = x + (size_t)r * (size_t)cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) row[c] += bias[c];
}

void add_bias_rows(float* x, const float* bias, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    int threads = cols < 256 ? 32 : 256;
    add_bias_rows_kernel<<<rows, threads>>>(x, bias, rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in add_bias_rows: %s\n", cudaGetErrorString(err));
    }
}

// Repack token-major [rows, kv_heads * head_dim] into the kv-head-major layout the
// existing KV append expects: dst[kv_h][t][d]. This is what lets the projections stay
// on the device while append_kv and attend_kv keep working unchanged, instead of the
// host rebuilding the same blob from its mirror every chunk.
__global__ void pack_kv_heads_kernel(float* dst, const float* src,
                                    int rows, int kv_heads, int head_dim) {
    int kv_h = blockIdx.x;
    int r = blockIdx.y;
    if (kv_h >= kv_heads || r >= rows) return;
    int stride = kv_heads * head_dim;
    const float* s = src + (size_t)r * (size_t)stride + (size_t)kv_h * (size_t)head_dim;
    float* d = dst + (size_t)kv_h * (size_t)rows * (size_t)head_dim + (size_t)r * (size_t)head_dim;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) d[j] = s[j];
}

void pack_kv_heads(float* dst, const float* src, int rows, int kv_heads, int head_dim) {
    if (rows <= 0 || kv_heads <= 0) return;
    int threads = head_dim < 256 ? 32 : 256;
    dim3 grid(kv_heads, rows);
    pack_kv_heads_kernel<<<grid, threads>>>(dst, src, rows, kv_heads, head_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in pack_kv_heads: %s\n", cudaGetErrorString(err));
    }
}

void head_rmsnorm(float* x, const float* gamma, int heads, int head_dim, float eps) {
    int threads = 128;
    size_t shmem = threads * sizeof(float);
    head_rmsnorm_kernel<<<heads, threads, shmem>>>(x, gamma, heads, head_dim, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in head_rmsnorm: %s\n", cudaGetErrorString(err));
    }
}

__global__ void apply_gradient_kernel(float* local_grad, const float* grad, const float* derivatives, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    local_grad[idx] = grad[idx] * derivatives[idx];
}

void apply_gradient(float* local_grad, const float* grad, const float* derivatives, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    apply_gradient_kernel<<<blocks, threads_per_block>>>(local_grad, grad, derivatives, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in apply_gradient: %s\n", cudaGetErrorString(err));
    }
}

__global__ void accumulate_bias_grad_kernel(float* bias_grad, const float* local_grad, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float sum = 0.0;
    for (int row = 0; row < rows; row++) {
        sum += local_grad[row * cols + col];
    }
    atomicAdd(&bias_grad[col], sum);
}

void accumulate_bias_grad(float* bias_grad, const float* local_grad, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (cols + threads_per_block - 1) / threads_per_block;

    accumulate_bias_grad_kernel<<<blocks, threads_per_block>>>(bias_grad, local_grad, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in accumulate_bias_grad: %s\n", cudaGetErrorString(err));
    }
}

__global__ void row_sum_kernel(float* dst, const float* src, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float sum = 0.0;
    for (int row = 0; row < rows; ++row) {
        sum += src[row * cols + col];
    }
    atomicAdd(&dst[col], sum);
}

void row_sum(float* dst, const float* src, int rows, int cols) {
    int threads_per_block = 256;
    int blocks = (cols + threads_per_block - 1) / threads_per_block;

    row_sum_kernel<<<blocks, threads_per_block>>>(dst, src, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in row_sum: %s\n", cudaGetErrorString(err));
    }
}

__global__ void zero_matrix_kernel(float* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    matrix[idx] = 0.0;
}

void zero_matrix(float* matrix, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    zero_matrix_kernel<<<blocks, threads_per_block>>>(matrix, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in zero_matrix: %s\n", cudaGetErrorString(err));
    }
}

__global__ void fill_matrix_kernel(float* matrix, double value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    matrix[idx] = value;
}

void fill_matrix(float* matrix, double value, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    fill_matrix_kernel<<<blocks, threads_per_block>>>(matrix, value, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in fill_matrix: %s\n", cudaGetErrorString(err));
    }
}

__global__ void element_div_kernel(float* out, const float* a, const float* b, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;

    float denom = b[idx];
    out[idx] = denom == 0.0 ? 0.0 : a[idx] / denom;
}

void element_div(float* out, const float* a, const float* b, int size){
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    element_div_kernel<<<blocks, threads_per_block>>>(out, a, b, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in element_div: %s\n", cudaGetErrorString(err));
    }
}

__global__ void softmax_backward_kernel(float* output, const float* grad, const float* softmax_out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* grad_row = grad + row * cols;
    const float* softmax_row = softmax_out + row * cols;
    float* output_row = output + row * cols;
    
    // Compute sum of softmax * grad for this row
    float sum = 0.0;
    for (int j = 0; j < cols; j++) {
        sum += softmax_row[j] * grad_row[j];
    }
    
    // Compute softmax backward: softmax * (grad - sum)
    for (int j = 0; j < cols; j++) {
        output_row[j] = softmax_row[j] * (grad_row[j] - sum);
    }
}

void softmax_backward(float* output, const float* grad, const float* softmax_out, int rows, int cols) {
    softmax_backward_kernel<<<rows, 1>>>(output, grad, softmax_out, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax_backward: %s\n", cudaGetErrorString(err));
    }
}


__global__ void element_log_kernel(float* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = in[idx];
    out[idx] = logf(val);
}

void element_log(float* out, const float* in, int size) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    element_log_kernel<<<blocks, threads_per_block>>>(out, in, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in element_log: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cross_entropy_loss_gradient_kernel(const float* pred, const float* target, float* grad, float* loss, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float p = pred[idx];
    float t = target[idx];
    grad[idx] = p - t;

    float contrib = -t * logf(fmaxf(p, 1e-15f));
    atomicAdd(loss, contrib);
}

__global__ void softmax_cross_entropy_label_kernel(const float* pred, const int* labels,
                                                    float* grad, float* loss,
                                                    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_pred = pred + row * cols;
    float* row_grad = grad + row * cols;

    // Find maximum value in the row for numerical stability
    float max_val = row_pred[0];
    for (int j = 1; j < cols; ++j) {
        float v = row_pred[j];
        if (v > max_val) max_val = v;
    }

    // Compute exponentials and their sum
    float sum = 0.0;
    for (int j = 0; j < cols; ++j) {
        float e = expf(row_pred[j] - max_val);
        row_grad[j] = e;
        sum += e;
    }

    // Normalize to obtain probabilities
    for (int j = 0; j < cols; ++j) {
        row_grad[j] /= sum;
    }

    int label = labels[row];
    if (label >= 0 && label < cols) {
        float p = row_grad[label];
        row_grad[label] = p - 1.0;
        float contrib = -logf(fmaxf(p, 1e-15f));
        atomicAdd(loss, contrib);
    }
}

void cross_entropy_loss_gradient(float* pred, float* target,
                                 float* grad, float* loss,
                                 int rows, int cols) {
    int total = rows * cols;
    cudaMemset(loss, 0, sizeof(float));
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cross_entropy_loss_gradient_kernel<<<blocks, threads>>>(pred, target, grad, loss, total);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in cross_entropy_loss_gradient: %s\n", cudaGetErrorString(err));

    }
}

void softmax_cross_entropy_label(float* pred, const int* labels,
                                 float* grad, float* loss,
                                 int rows, int cols) {
    cudaMemset(loss, 0, sizeof(float));
    softmax_cross_entropy_label_kernel<<<rows, 1>>>(pred, labels, grad, loss, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in softmax_cross_entropy_label: %s\n", cudaGetErrorString(err));
    }
}

// ---- Q8_0-style quantized matmul: y[M,N] = x[M,K] * dequant(W) ----
// W is quantized weights laid out row-major [N, K] (out-major) as int8 (q),
// with one fp32 scale per BLOCK (=32) contiguous K elements per output column,
// scales laid out [N, ceil(K/BLOCK)]. This reproduces row-major fp32 GEMM
// semantics result[m,n] = sum_k x[m,k] * (q[n,k] * scale[n, k/BLOCK]).
// One CUDA block computes one (m,n) output via threaded reduction over K.
#define Q8_BLK 32
__global__ void gemm_q8_f32_kernel(const float* __restrict__ x,
                                   const signed char* __restrict__ q,
                                   const float* __restrict__ scales,
                                   float* __restrict__ y,
                                   int M, int N, int K) {
    int n = blockIdx.x; // output column (0..N)
    int m = blockIdx.y; // activation row (0..M)
    if (n >= N || m >= M) return;

    int nblocks = (K + Q8_BLK - 1) / Q8_BLK;
    const float* xrow = x + (long)m * K;
    const signed char* qrow = q + (long)n * K;
    const float* srow = scales + (long)n * nblocks;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Each thread strides over the full K dimension so all threads do work
    // (the previous one-thread-per-32-block scheme left most threads idle for
    // small K). Scale is per-32-element block; srow is tiny and L1/L2 cached.
    float partial = 0.0f;
    for (int k = tid; k < K; k += nthreads) {
        partial += (float)qrow[k] * xrow[k] * srow[k >> 5];
    }

    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(long)m * N + n] = sdata[0];
}

void gemm_q8_f32(const float* x, const signed char* q, const float* scales,
                 float* y, int M, int N, int K) {
    int threads = 256;
    dim3 grid(N, M);
    size_t shmem = threads * sizeof(float);
    gemm_q8_f32_kernel<<<grid, threads, shmem>>>(x, q, scales, y, M, N, K);
    // Avoid a full device sync on every projection/lm_head call. Callers read
    // results back via a default-stream D2H memcpy, which is ordered after this
    // kernel and provides the needed synchronization. Just surface launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemm_q8_f32: %s\n", cudaGetErrorString(err));
    }
}

// ---- Q4_0-style quantized GEMM (4-bit symmetric, block=32) ----
// Mirrors gemm_q8_f32 but weights are 4-bit. Two nibbles are packed per byte
// along K: byte q[n, k/2] holds k in the low nibble and k+1 in the high nibble,
// each stored as (value + 8), so value = nibble - 8. The host quantizer produces
// symmetric values in [-7, 7] (stored nibbles 1..15); the kernel decodes the
// full [-8, 7] nibble range so any unused/padding nibble is handled safely.
// One fp32 scale per BLOCK(=32) contiguous K elements per output column, scales
// laid out [N, ceil(K/32)]. Reproduces row-major fp32 GEMM semantics:
//   result[m,n] = sum_k x[m,k] * (value(n,k) * scale[n, k/32]).
#define Q4_BLK 32
// Blocks sharing one fp32 super-block scale. Each block's own scale is a byte
// relative to it: effective = d[block / Q4_SUPER] * sub[block] / 255. Must match
// SHAInet::Q4CudaMatrix::SUPER.
#define Q4_SUPER 8
__global__ void gemm_q4_f32_kernel(const float* __restrict__ x,
                                   const unsigned char* __restrict__ q,
                                   const float* __restrict__ d,
                                   const unsigned char* __restrict__ sub,
                                   float* __restrict__ y,
                                   int M, int N, int K) {
    int n = blockIdx.x; // output column (0..N)
    int m = blockIdx.y; // activation row (0..M)
    if (n >= N || m >= M) return;

    int nblocks = (K + Q4_BLK - 1) / Q4_BLK;
    int nsupers = (nblocks + Q4_SUPER - 1) / Q4_SUPER;
    int kbytes = (K + 1) >> 1; // packed bytes per output column
    const float* xrow = x + (long)m * K;
    const unsigned char* qrow = q + (long)n * kbytes;
    const float* drow = d + (long)n * nsupers;
    const unsigned char* subrow = sub + (long)n * nblocks;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Each thread strides over the full K dimension, unpacking its nibble. The
    // effective block scale is d[block / Q4_SUPER] * sub[block] / 255.
    float partial = 0.0f;
    for (int k = tid; k < K; k += nthreads) {
        unsigned char byte = qrow[k >> 1];
        int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
        int b = k >> 5;
        float eff = drow[b / Q4_SUPER] * (float)subrow[b] * (1.0f / 255.0f);
        partial += (float)(nib - 8) * xrow[k] * eff;
    }

    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[(long)m * N + n] = sdata[0];
}

// Output columns handled per block, one warp each. Raising this increases how
// many columns share a single staged activation tile.
#define Q4_COLS_PER_BLOCK 8
// Activation elements staged in shared memory per iteration. 1024 = 32 scale
// blocks = exactly one block per warp lane, so no lane is idle on a full tile.
#define Q4_TILE 1024

// Vectorized Q4 GEMM.
//
// Improves on gemm_q4_f32_kernel in four ways:
//   * one WARP per output column instead of one block, so the reduction is a
//     __shfl_down_sync chain rather than an 8-step shared-memory tree with a
//     __syncthreads between every step;
//   * Q4_COLS_PER_BLOCK columns share one staged activation tile, so x is read
//     from global memory once per block instead of once per output column;
//   * weights move as uint4 (16 bytes = 32 nibbles = exactly one scale block),
//     so each lane issues one 16-byte load where the scalar kernel issued 32
//     single-byte loads, and consecutive lanes cover 512 contiguous bytes;
//   * the per-lane unpack loop is fully unrolled over the 16 bytes.
//
// Requires K % Q4_BLK == 0. That also makes every packed column row a multiple
// of 16 bytes (kbytes = K/2), which is what makes the uint4 loads legal; the
// caller falls back to the scalar kernel otherwise.
__global__ void gemm_q4_f32_vec_kernel(const float* __restrict__ x,
                                       const unsigned char* __restrict__ q,
                                       const float* __restrict__ d,
                                       const unsigned char* __restrict__ sub,
                                       float* __restrict__ y,
                                       int M, int N, int K) {
    int m = blockIdx.y;
    if (m >= M) return; // uniform across the block, so no divergent barrier

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * Q4_COLS_PER_BLOCK + warp;

    int nblocks = K >> 5; // exact, K % 32 == 0
    int nsupers = (nblocks + Q4_SUPER - 1) / Q4_SUPER;
    int kbytes = K >> 1;
    const float* xrow = x + (long)m * K;
    const unsigned char* qrow = q + (long)n * kbytes;
    const float* drow = d + (long)n * nsupers;
    const unsigned char* subrow = sub + (long)n * nblocks;

    extern __shared__ float xs[]; // Q4_TILE floats

    float acc = 0.0f;
    for (int ktile = 0; ktile < K; ktile += Q4_TILE) {
        int tile = min(Q4_TILE, K - ktile);
        // Whole block cooperates on the load, then every warp reads it back.
        for (int i = threadIdx.x; i < tile; i += blockDim.x) xs[i] = xrow[ktile + i];
        __syncthreads();

        if (n < N && lane * Q4_BLK < tile) {
            int kb = (ktile >> 5) + lane; // this lane's scale block
            uint4 packed = *reinterpret_cast<const uint4*>(qrow + (long)kb * 16);
            const unsigned char* b = reinterpret_cast<const unsigned char*>(&packed);
            const float* xb = xs + lane * Q4_BLK;
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                unsigned char byte = b[j];
                sum += (float)((int)(byte & 0x0F) - 8) * xb[2 * j];
                sum += (float)((int)(byte >> 4) - 8) * xb[2 * j + 1];
            }
            acc += sum * (drow[kb / Q4_SUPER] * (float)subrow[kb] * (1.0f / 255.0f));
        }
        __syncthreads(); // tile is reused next iteration
    }

    // Warp reduction. Every lane reaches this, including lanes whose column is
    // out of range, so the shuffle mask stays full.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0 && n < N) y[(long)m * N + n] = acc;
}

// Register-tiled Q4 GEMM. Addresses what the current kernel is actually limited by, which
// is NOT what it looks like from the source.
//
// Bisection on the vectorized kernel: separately removing the weight load, the activation
// reads, the nibble decode, and even the whole shared-memory staging each changed the time
// by under 3%. Nothing in the inner loop dominates. What does is the shape of the work: at
// the expert projection (M=128, K=2048, N=768) it launches 12288 blocks x 256 threads for
// only 201M MACs, so each thread performs 64 MACs and then pays two __syncthreads and a
// five-step warp-shuffle reduction. Overhead per thread is comparable to its useful work,
// and it reached 1004 GFLOP/s, about 3.5% of this card's fp32 peak.
//
// So this kernel gives each thread a TILE of outputs it owns across the whole K:
//   - 8192 MACs per thread instead of 64
//   - no cross-lane reduction at all, since a thread accumulates its own outputs
//   - the Q4 weights are dequantized into shared memory ONCE per block per k-tile,
//     instead of once per row-block
//
// Deliberately NOT the llama.cpp approach of int8 DP4A with quantized activations: that
// accelerates the decode and the MAC, which the bisection above shows are already free
// here, and it would cost activation precision for nothing.
#define MMQ_BM 32 // rows per block
#define MMQ_BN 32 // cols per block
#define MMQ_TM 2  // rows per thread
#define MMQ_TN 2  // cols per thread
// k per iteration is Q4_BLK: one k-tile is exactly one quantization block, so a column's
// scale is loaded once per tile rather than recomputed per element.

__global__ void gemm_q4_f32_tiled_kernel(const float* __restrict__ x,
                                         const unsigned char* __restrict__ q,
                                         const float* __restrict__ d,
                                         const unsigned char* __restrict__ sub,
                                         float* __restrict__ y,
                                         int M, int N, int K) {
    __shared__ float xs[MMQ_BM][Q4_BLK];
    __shared__ float ws[Q4_BLK][MMQ_BN];

    const int m0 = blockIdx.y * MMQ_BM;
    const int n0 = blockIdx.x * MMQ_BN;
    const int tid = threadIdx.x;

    // Thread's output tile: MMQ_TM rows x MMQ_TN cols.
    const int tm = (tid / (MMQ_BN / MMQ_TN)) * MMQ_TM;
    const int tn = (tid % (MMQ_BN / MMQ_TN)) * MMQ_TN;

    const int nblocks = K >> 5; // K % 32 == 0, checked by the caller
    const int nsupers = (nblocks + Q4_SUPER - 1) / Q4_SUPER;
    const int kbytes = K >> 1;

    float acc[MMQ_TM][MMQ_TN];
#pragma unroll
    for (int i = 0; i < MMQ_TM; ++i)
#pragma unroll
        for (int j = 0; j < MMQ_TN; ++j) acc[i][j] = 0.0f;

    for (int kb = 0; kb < nblocks; ++kb) {
        const int k0 = kb * Q4_BLK;

        // Stage the activation tile: MMQ_BM x 32 floats.
        for (int idx = tid; idx < MMQ_BM * Q4_BLK; idx += blockDim.x) {
            const int r = idx / Q4_BLK;
            const int c = idx - r * Q4_BLK;
            const int gm = m0 + r;
            xs[r][c] = (gm < M) ? x[(long)gm * K + k0 + c] : 0.0f;
        }

        // Dequantize this k-block for all MMQ_BN columns into shared memory. Each column's
        // 32 values are 16 packed bytes; four threads share a column, eight values each.
        for (int idx = tid; idx < MMQ_BN * 4; idx += blockDim.x) {
            const int col = idx & (MMQ_BN - 1);
            const int part = idx / MMQ_BN; // 0..3
            const int gn = n0 + col;
            if (gn < N) {
                const unsigned char* qrow = q + (long)gn * kbytes + (long)kb * 16;
                const float scale = d[(long)gn * nsupers + kb / Q4_SUPER] *
                                    (float)sub[(long)gn * nblocks + kb] * (1.0f / 255.0f);
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned char byte = qrow[part * 4 + b];
                    const int kk = (part * 4 + b) * 2;
                    ws[kk][col] = (float)((int)(byte & 0x0F) - 8) * scale;
                    ws[kk + 1][col] = (float)((int)(byte >> 4) - 8) * scale;
                }
            } else {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int kk = (part * 4 + b) * 2;
                    ws[kk][col] = 0.0f;
                    ws[kk + 1][col] = 0.0f;
                }
            }
        }
        __syncthreads();

        // The point: 128 MACs per thread per k-tile, from registers, no reduction.
#pragma unroll
        for (int kk = 0; kk < Q4_BLK; ++kk) {
            float a[MMQ_TM];
            float b[MMQ_TN];
#pragma unroll
            for (int i = 0; i < MMQ_TM; ++i) a[i] = xs[tm + i][kk];
#pragma unroll
            for (int j = 0; j < MMQ_TN; ++j) b[j] = ws[kk][tn + j];
#pragma unroll
            for (int i = 0; i < MMQ_TM; ++i)
#pragma unroll
                for (int j = 0; j < MMQ_TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MMQ_TM; ++i) {
        const int gm = m0 + tm + i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < MMQ_TN; ++j) {
            const int gn = n0 + tn + j;
            if (gn < N) y[(long)gm * N + gn] = acc[i][j];
        }
    }
}

// Force the one-row-per-block kernel at M > 1, so the register-tiled kernel can be A/B'd
// end to end on a real model. Read once, on first use.
static int q4_no_tile_forced = -1;
static inline bool q4_no_tile() {
    if (q4_no_tile_forced < 0) {
        const char* e = getenv("SHAINET_Q4_NO_TILE");
        q4_no_tile_forced = (e && e[0] == '1') ? 1 : 0;
    }
    return q4_no_tile_forced == 1;
}

// Force the scalar Q4 kernel even on shapes the vectorized one supports. Exists
// so the two kernels can be A/B'd end to end on a real model, where every shape
// otherwise takes the vectorized path. Read once, on first use.
static int q4_scalar_forced = -1;
static inline bool q4_force_scalar() {
    if (q4_scalar_forced < 0) {
        const char* e = getenv("SHAINET_Q4_SCALAR");
        q4_scalar_forced = (e && e[0] == '1') ? 1 : 0;
    }
    return q4_scalar_forced == 1;
}

void gemm_q4_f32(const float* x, const unsigned char* q, const float* d,
                 const unsigned char* sub, float* y, int M, int N, int K) {
    if (K % Q4_BLK == 0 && !q4_force_scalar()) {
        // Multi-row shapes go to the register-tiled kernel; M == 1 (decode) keeps the
        // GEMV-shaped kernel, where there is no output tile to amortize over.
        // SHAINET_Q4_NO_TILE=1 forces the old path so the two can be A/B'd end to end.
        if (M > 1 && !q4_no_tile()) {
            dim3 grid((N + MMQ_BN - 1) / MMQ_BN, (M + MMQ_BM - 1) / MMQ_BM);
            int threads = (MMQ_BM / MMQ_TM) * (MMQ_BN / MMQ_TN);
            gemm_q4_f32_tiled_kernel<<<grid, threads>>>(x, q, d, sub, y, M, N, K);
        } else {
            int threads = Q4_COLS_PER_BLOCK * 32;
            dim3 grid((N + Q4_COLS_PER_BLOCK - 1) / Q4_COLS_PER_BLOCK, M);
            size_t shmem = Q4_TILE * sizeof(float);
            gemm_q4_f32_vec_kernel<<<grid, threads, shmem>>>(x, q, d, sub, y, M, N, K);
        }
    } else {
        // Packed rows are not 16-byte aligned for this K, so uint4 loads would
        // be illegal. Correctness first.
        int threads = 256;
        dim3 grid(N, M);
        size_t shmem = threads * sizeof(float);
        gemm_q4_f32_kernel<<<grid, threads, shmem>>>(x, q, d, sub, y, M, N, K);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemm_q4_f32: %s\n", cudaGetErrorString(err));
    }
}

// ---- KV-cache attention (LLaMA decode/prefill) ----
// Device cache layout: [num_kv_heads, capacity, head_dim] for both K and V.
// Templates cannot carry C linkage, so the generic kernels live in a nested
// extern "C++" block; only the concrete wrappers below stay callable by name.
extern "C++" {

// Storage traits for the KV cache element type. The cache may be kept in fp16
// to halve its VRAM footprint (it is the dominant consumer at long context);
// all arithmetic still happens in fp32 after conversion, so only the stored
// precision changes.
template <typename KV> struct KVStore;
template <> struct KVStore<float> {
    __device__ static inline float load(const float* p, long i) { return p[i]; }
    __device__ static inline void store(float* p, long i, float v) { p[i] = v; }
};
template <> struct KVStore<__half> {
    __device__ static inline float load(const __half* p, long i) { return __half2float(p[i]); }
    __device__ static inline void store(__half* p, long i, float v) { p[i] = __float2half(v); }
};
// fp8 e4m3, unscaled (vLLM's default fp8 KV mode). Half the bytes of fp16, which at long context is
// the difference between ~2 GB and ~4 GB of VRAM for a 16-attention-layer 27B at 64K -- about eight
// more layers resident. The store saturates at +-448 rather than producing inf, and K arrives here
// after q/k-norm and RoPE, so its range sits comfortably inside that.
template <> struct KVStore<__nv_fp8_e4m3> {
    __device__ static inline float load(const __nv_fp8_e4m3* p, long i) { return float(p[i]); }
    __device__ static inline void store(__nv_fp8_e4m3* p, long i, float v) { p[i] = __nv_fp8_e4m3(v); }
};

// `staging` holds the newly appended rows, already RoPE'd (K) on the host:
//   [num_kv_heads * new_tokens * head_dim] K chunks (kv_head-major, then
//   token-major — i.e. each kv_head's tail is contiguous), followed by the
//   same layout for V. Scatter them into the cache at position start_pos.
// Staging is always fp32; the cache element type is the template parameter, so
// the fp16 cache is written through a conversion here.
template <typename KV>
__global__ void kv_cache_append_kernel_t(const float* __restrict__ staging,
                                         KV* __restrict__ kc,
                                         KV* __restrict__ vc,
                                         int new_tokens, int start_pos,
                                         int num_kv_heads, int head_dim,
                                         int capacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = new_tokens * head_dim;       // floats per kv_head per tensor
    int total = num_kv_heads * chunk;        // floats per tensor (K or V)
    if (idx >= 2 * total) return;
    int is_v = idx >= total;
    int r = is_v ? idx - total : idx;
    int kv_h = r / chunk;
    int rem = r - kv_h * chunk;              // t * head_dim + d
    long dst = ((long)kv_h * capacity + start_pos) * head_dim + rem;
    if (is_v) KVStore<KV>::store(vc, dst, staging[idx]);
    else      KVStore<KV>::store(kc, dst, staging[idx]);
}

} // extern "C++"

void kv_cache_append_f32(const float* staging, float* kc, float* vc,
                         int new_tokens, int start_pos, int num_kv_heads,
                         int head_dim, int capacity) {
    int total = 2 * num_kv_heads * new_tokens * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_cache_append_kernel_t<float><<<blocks, threads>>>(staging, kc, vc, new_tokens,
                                                         start_pos, num_kv_heads,
                                                         head_dim, capacity);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in kv_cache_append_f32: %s\n", cudaGetErrorString(err));
    }
}

// fp16 cache variant. `kc`/`vc` are __half buffers passed from Crystal as
// UInt16 pointers (Crystal has no native half type); staging stays fp32.
void kv_cache_append_f16(const float* staging, unsigned short* kc, unsigned short* vc,
                         int new_tokens, int start_pos, int num_kv_heads,
                         int head_dim, int capacity) {
    int total = 2 * num_kv_heads * new_tokens * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_cache_append_kernel_t<__half><<<blocks, threads>>>(staging,
                                                          reinterpret_cast<__half*>(kc),
                                                          reinterpret_cast<__half*>(vc),
                                                          new_tokens, start_pos,
                                                          num_kv_heads, head_dim,
                                                          capacity);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in kv_cache_append_f16: %s\n", cudaGetErrorString(err));
    }
}

// fp8 (e4m3) cache variant. `kc`/`vc` travel from Crystal as byte pointers.
void kv_cache_append_fp8(const float* staging, unsigned char* kc, unsigned char* vc,
                         int new_tokens, int start_pos, int num_kv_heads,
                         int head_dim, int capacity) {
    int total = 2 * num_kv_heads * new_tokens * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_cache_append_kernel_t<__nv_fp8_e4m3><<<blocks, threads>>>(staging,
                                                                 reinterpret_cast<__nv_fp8_e4m3*>(kc),
                                                                 reinterpret_cast<__nv_fp8_e4m3*>(vc),
                                                                 new_tokens, start_pos,
                                                                 num_kv_heads, head_dim,
                                                                 capacity);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in kv_cache_append_fp8: %s\n", cudaGetErrorString(err));
    }
}

extern "C++" {

// Causal attention over the device KV cache. One block per (head, token):
//   scores[j] = scale * (q_h · K[kv_h][j])  for j < visible = start_pos+i+1
//   p = softmax(scores); out[h][d] = sum_j p[j] * V[kv_h][j][d]
// q/out are [new_tokens, num_heads*head_dim] row-major (q already RoPE'd).
// ws is a global scratch of at least num_heads*new_tokens*total_len floats.
// GQA: query head h reads kv head h / heads_per_kv.
// KV is the cache storage type (float or __half); loads convert to fp32 so the
// dot product, softmax and weighted sum are identical in both instantiations.
// ---------------------------------------------------------------------------
// Split-KV ("flash-decoding") attention for the single-token decode step.
//
// The kernel below this one parallelizes over (head, token) only. At decode that is ONE token, so
// the grid is num_heads blocks -- 24 for this model -- and with 128 threads that occupies 3072 of a
// 4090 Laptop's ~117k thread slots, about 2.6% of the card, while each block serially reduces over
// the entire KV length. It also stages the score row in GLOBAL memory (`ws`) and walks it three
// times: write scores, read/exponentiate/write, read for the V weighting. Both costs grow with
// context, which is why decode fell from ~8 tok/s to 3.7 tok/s at 14k tokens.
//
// Flash-decoding adds the KV length as a third grid dimension: each block handles a SLICE of the
// keys, keeps its slice's scores in shared memory, and emits a partial output plus the two softmax
// statistics (running max and sum). A second kernel combines the partials. Because softmax is
// associative under the standard max-rescale, the result is exact, not an approximation.
//
// Two consequences beyond parallelism: the global `ws` score buffer disappears entirely (it was
// sized num_heads * chunk * total_len, the single largest allocation in the process and the one that
// OOM'd a 16 GB card at 9.5k tokens), and the scores are never round-tripped through HBM.
#define ATTN_SPLIT_SCORES 512   // scores held in shared memory per split; 2 KB at fp32

template <typename KV>
__global__ void attention_split_kv_kernel(const float* __restrict__ q,
                                          const KV* __restrict__ kc,
                                          const KV* __restrict__ vc,
                                          float* __restrict__ part_out,
                                          float* __restrict__ part_max,
                                          float* __restrict__ part_sum,
                                          int new_tokens, int start_pos,
                                          int num_heads, int heads_per_kv,
                                          int head_dim, int capacity, float scale,
                                          int num_splits) {
    const int h = blockIdx.x;
    const int i = blockIdx.y;
    const int sp = blockIdx.z;
    if (h >= num_heads || i >= new_tokens || sp >= num_splits) return;

    const int kv_h = h / heads_per_kv;
    const int visible = start_pos + i + 1;
    const int d_model = num_heads * head_dim;

    const float* qv = q + (long)i * d_model + (long)h * head_dim;
    const KV* kh = kc + (long)kv_h * capacity * head_dim;
    const KV* vh = vc + (long)kv_h * capacity * head_dim;

    // This split's key range.
    const int j0 = sp * ATTN_SPLIT_SCORES;
    const int j1 = min(j0 + ATTN_SPLIT_SCORES, visible);

    const long pidx = ((long)h * new_tokens + i) * num_splits + sp;

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    extern __shared__ float smem[];
    float* q_s = smem;                        // head_dim
    float* sc = smem + head_dim;              // ATTN_SPLIT_SCORES
    float* red = sc + ATTN_SPLIT_SCORES;      // nt

    // An empty split contributes nothing. Identity for the combine: sum 0, max -inf.
    if (j0 >= j1) {
        if (tid == 0) { part_max[pidx] = -INFINITY; part_sum[pidx] = 0.0f; }
        for (int d = tid; d < head_dim; d += nt) part_out[pidx * head_dim + d] = 0.0f;
        return;
    }

    for (int d = tid; d < head_dim; d += nt) q_s[d] = qv[d];
    __syncthreads();

    // Scores for this slice, into SHARED memory. Max tracked for a stable softmax.
    float lmax = -INFINITY;
    for (int j = j0 + tid; j < j1; j += nt) {
        const KV* krow = kh + (long)j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_s[d] * KVStore<KV>::load(krow, d);
        dot *= scale;
        sc[j - j0] = dot;
        if (dot > lmax) lmax = dot;
    }
    red[tid] = lmax;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float gmax = red[0];
    __syncthreads();

    float lsum = 0.0f;
    for (int j = j0 + tid; j < j1; j += nt) {
        float e = expf(sc[j - j0] - gmax);
        sc[j - j0] = e;
        lsum += e;
    }
    red[tid] = lsum;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float ssum = red[0];
    __syncthreads();

    // UNNORMALIZED weighted V for this slice. Normalization happens in the combine, once, using the
    // slice sums -- dividing here would make the partials incombinable.
    for (int d = tid; d < head_dim; d += nt) {
        float acc = 0.0f;
        for (int j = j0; j < j1; ++j) {
            acc += sc[j - j0] * KVStore<KV>::load(vh, (long)j * head_dim + d);
        }
        part_out[pidx * head_dim + d] = acc;
    }
    if (tid == 0) { part_max[pidx] = gmax; part_sum[pidx] = ssum; }
}

// Combine the per-split partials. Each split reported its own max, so rescale every slice onto the
// global max before summing -- the standard log-sum-exp merge, which makes the split exact.
__global__ void attention_split_combine_kernel(const float* __restrict__ part_out,
                                               const float* __restrict__ part_max,
                                               const float* __restrict__ part_sum,
                                               float* __restrict__ out,
                                               int new_tokens, int num_heads, int head_dim,
                                               int num_splits) {
    const int h = blockIdx.x;
    const int i = blockIdx.y;
    if (h >= num_heads || i >= new_tokens) return;

    const long base = ((long)h * new_tokens + i) * num_splits;

    float gmax = -INFINITY;
    for (int s = 0; s < num_splits; ++s) gmax = fmaxf(gmax, part_max[base + s]);
    if (!isfinite(gmax)) gmax = 0.0f;   // every split empty; leaves a zero row rather than NaNs

    float denom = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        const float m = part_max[base + s];
        if (!isfinite(m)) continue;
        denom += part_sum[base + s] * expf(m - gmax);
    }
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;

    const int d_model = num_heads * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            const float m = part_max[base + s];
            if (!isfinite(m)) continue;
            acc += part_out[(base + s) * head_dim + d] * expf(m - gmax);
        }
        out[(long)i * d_model + (long)h * head_dim + d] = acc * inv;
    }
}

template <typename KV>
__global__ void attention_kv_kernel_t(const float* __restrict__ q,
                                      const KV* __restrict__ kc,
                                      const KV* __restrict__ vc,
                                      float* __restrict__ out,
                                      float* __restrict__ ws,
                                      int new_tokens, int start_pos,
                                      int num_heads, int heads_per_kv,
                                      int head_dim, int capacity, float scale) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= num_heads || i >= new_tokens) return;

    int kv_h = h / heads_per_kv;
    int visible = start_pos + i + 1;
    int total_len = start_pos + new_tokens;
    int d_model = num_heads * head_dim;

    const float* qv = q + (long)i * d_model + (long)h * head_dim;
    const KV* kh = kc + (long)kv_h * capacity * head_dim;
    const KV* vh = vc + (long)kv_h * capacity * head_dim;
    float* wsrow = ws + ((long)h * new_tokens + i) * total_len;

    int tid = threadIdx.x;
    int nt = blockDim.x;
    extern __shared__ float smem[];
    float* q_s = smem;            // head_dim
    float* red = smem + head_dim; // nt

    for (int d = tid; d < head_dim; d += nt) q_s[d] = qv[d];
    __syncthreads();

    // Pass 1: scores into ws, track max for stable softmax.
    float lmax = -INFINITY;
    for (int j = tid; j < visible; j += nt) {
        const KV* krow = kh + (long)j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_s[d] * KVStore<KV>::load(krow, d);
        dot *= scale;
        wsrow[j] = dot;
        if (dot > lmax) lmax = dot;
    }
    red[tid] = lmax;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float gmax = red[0];
    __syncthreads();

    // Pass 2: exponentiate + sum.
    float lsum = 0.0f;
    for (int j = tid; j < visible; j += nt) {
        float e = expf(wsrow[j] - gmax);
        wsrow[j] = e;
        lsum += e;
    }
    red[tid] = lsum;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / red[0];

    // Pass 3: weighted sum of V rows. Threads cover head_dim, so adjacent
    // threads read adjacent V elements (coalesced per j).
    for (int d = tid; d < head_dim; d += nt) {
        float acc = 0.0f;
        for (int j = 0; j < visible; ++j) acc += wsrow[j] * KVStore<KV>::load(vh, (long)j * head_dim + d);
        out[(long)i * d_model + (long)h * head_dim + d] = acc * inv_sum;
    }
}

} // extern "C++"

// How many splits a given context length uses, and how many floats of partials that needs. The
// caller sizes its buffer from this rather than duplicating the arithmetic.
int attention_split_count(int total_len) {
    if (total_len <= 0) return 0;
    return (total_len + ATTN_SPLIT_SCORES - 1) / ATTN_SPLIT_SCORES;
}

int attention_split_ws_floats(int new_tokens, int num_heads, int head_dim, int total_len) {
    int splits = attention_split_count(total_len);
    if (splits <= 0) return 0;
    // partial outputs + one max and one sum per split
    return num_heads * new_tokens * splits * (head_dim + 2);
}

// Split-KV decode attention. `ws` holds the partials: [heads, tokens, splits, head_dim] outputs,
// then the per-split maxima, then the per-split sums.
//
// extern "C++" because a template cannot carry C linkage, and this sits inside the file's extern "C"
// block alongside the two exported entry points that instantiate it.
extern "C++" {

// ---------------------------------------------------------------------------------------------
// Single-token decode attention, grouped by KV head.
//
// attention_split_kv_kernel runs a block per (query head, split): every one of the 6 query heads that
// share a KV head re-reads that head's keys and values, and each thread walks one key row a byte at a
// time with a 256-byte stride between neighbouring threads. At 55K tokens of context that made decode
// 111 ms per token against 58 ms at 3.5K, where reading the whole fp8 cache once costs ~5 ms.
//
// Here a block owns one KV head and one 512-key split and serves ALL its query heads: K and V are read
// once, coalesced. Scores: a warp per key, each lane 8 dims (one 8-byte fp8 / 16-byte fp16 load) with
// the query heads' matching dims in registers, reduced by shuffles. Softmax statistics: a warp per
// head over the split. P*V: each thread owns output dims. The partials land in exactly the layout of
// attention_split_kv_kernel, so attention_split_combine_kernel and the workspace sizing are shared.
#define DGQA_MAX_HPK 8
#define DGQA_THREADS 256

__device__ __forceinline__ void kv_load8f(const __half* p, float* f) {
    const uint4 u = *((const uint4*)p);
    const __half2* h = (const __half2*)&u;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
__device__ __forceinline__ void kv_load8f(const __nv_fp8_e4m3* p, float* f) {
    const uint2 raw = *((const uint2*)p);
    const unsigned int w[2] = {raw.x, raw.y};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)((w[i >> 1] >> (16 * (i & 1))) & 0xFFFF), __NV_E4M3);
        const float2 t = __half22float2(*((const __half2*)&hr));
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

template <typename KV, int D>
__global__ void __launch_bounds__(DGQA_THREADS) attention_decode_gqa_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc, const KV* __restrict__ vc,
    float* __restrict__ part_out, float* __restrict__ part_max, float* __restrict__ part_sum,
    int start_pos, int num_heads, int hpk, int capacity, float scale, int num_splits) {
    __shared__ float sc[DGQA_MAX_HPK][ATTN_SPLIT_SCORES];
    __shared__ float red[DGQA_THREADS / D][DGQA_MAX_HPK][D];
    const int kvh = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int visible = start_pos + 1;
    const int j0 = sp * ATTN_SPLIT_SCORES;
    const int j1 = min(j0 + ATTN_SPLIT_SCORES, visible);
    const int h0 = kvh * hpk;
    const KV* kh = kc + (long)kvh * capacity * D;
    const KV* vh = vc + (long)kvh * capacity * D;

    if (j0 >= j1) {   // empty split: the combine's identity (max -inf, sum 0, zero partial)
        for (int i = tid; i < hpk * D; i += DGQA_THREADS) {
            const int hl = i / D, d = i - hl * D;
            const long pidx = (long)(h0 + hl) * num_splits + sp;
            part_out[pidx * D + d] = 0.0f;
            if (d == 0) { part_max[pidx] = -INFINITY; part_sum[pidx] = 0.0f; }
        }
        return;
    }

    // Scores. Lane owns dims [lane*8, lane*8+8) for D=256, [lane*4, ...) pairs for D=128 (handled by
    // giving each lane D/32 dims through 8-wide loads over half the lanes' worth of rows).
    constexpr int DPL = D / 32;                 // dims per lane: 8 (D=256) or 4 (D=128)
    float qr[DGQA_MAX_HPK][DPL];
    #pragma unroll
    for (int hl = 0; hl < DGQA_MAX_HPK; ++hl)
        #pragma unroll
        for (int e = 0; e < DPL; ++e)
            qr[hl][e] = (hl < hpk) ? q[(long)(h0 + hl) * D + lane * DPL + e] : 0.0f;

    for (int j = j0 + warp; j < j1; j += DGQA_THREADS / 32) {
        float kf[8];
        if (DPL == 8) {
            kv_load8f(kh + (long)j * D + lane * 8, kf);
        } else {
            #pragma unroll
            for (int e = 0; e < DPL; ++e) kf[e] = KVStore<KV>::load(kh, (long)j * D + lane * DPL + e);
        }
        float s[DGQA_MAX_HPK];
        #pragma unroll
        for (int hl = 0; hl < DGQA_MAX_HPK; ++hl) {
            float a = 0.0f;
            #pragma unroll
            for (int e = 0; e < DPL; ++e) a += qr[hl][e] * kf[e];
            s[hl] = a;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            #pragma unroll
            for (int hl = 0; hl < DGQA_MAX_HPK; ++hl) s[hl] += __shfl_xor_sync(0xFFFFFFFF, s[hl], o);
        if (lane == 0)
            #pragma unroll
            for (int hl = 0; hl < DGQA_MAX_HPK; ++hl)
                if (hl < hpk) sc[hl][j - j0] = s[hl] * scale;
    }
    __syncthreads();

    // Per-head max and sum over the split; p written back in place.
    const int nk = j1 - j0;
    for (int hl = warp; hl < hpk; hl += DGQA_THREADS / 32) {
        float m = -INFINITY;
        for (int i = lane; i < nk; i += 32) m = fmaxf(m, sc[hl][i]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
        float sum = 0.0f;
        for (int i = lane; i < nk; i += 32) {
            const float e = expf(sc[hl][i] - m);
            sc[hl][i] = e;
            sum += e;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, o);
        if (lane == 0) {
            const long pidx = (long)(h0 + hl) * num_splits + sp;
            part_max[pidx] = m;
            part_sum[pidx] = sum;
        }
    }
    __syncthreads();

    // P*V: thread owns dim d; with D=128 two thread groups split the keys and are summed after.
    constexpr int G = DGQA_THREADS / D;
    const int d = tid % D, grp = tid / D;
    float acc[DGQA_MAX_HPK];
    #pragma unroll
    for (int hl = 0; hl < DGQA_MAX_HPK; ++hl) acc[hl] = 0.0f;
    #pragma unroll 4
    for (int i = grp; i < nk; i += G) {
        const float v = KVStore<KV>::load(vh, (long)(j0 + i) * D + d);
        #pragma unroll
        for (int hl = 0; hl < DGQA_MAX_HPK; ++hl)
            if (hl < hpk) acc[hl] += sc[hl][i] * v;
    }
    if (G > 1) {
        #pragma unroll
        for (int hl = 0; hl < DGQA_MAX_HPK; ++hl) red[grp][hl][d] = acc[hl];
        __syncthreads();
        if (grp == 0)
            #pragma unroll
            for (int hl = 0; hl < DGQA_MAX_HPK; ++hl)
                for (int g2 = 1; g2 < G; ++g2) acc[hl] += red[g2][hl][d];
    }
    if (grp == 0)
        for (int hl = 0; hl < hpk; ++hl)
            part_out[((long)(h0 + hl) * num_splits + sp) * D + d] = acc[hl];
}

// 1 when the grouped kernel took the call (one token, head_dim 128/256, at most 8 query heads per KV
// head, a half or fp8 cache); 0 leaves it to attention_split_kv_kernel.
template <typename KV>
static int attention_decode_gqa_launch(const float* q, const KV* kc, const KV* vc,
                                       float* part_out, float* part_max, float* part_sum,
                                       int start_pos, int num_heads, int heads_per_kv, int head_dim,
                                       int capacity, float scale, int splits) {
    static const bool enabled = [] {
        const char* e = getenv("SHAINET_DECODE_GQA_ATTN");
        return !(e && e[0] == '0');
    }();
    if (!enabled || heads_per_kv > DGQA_MAX_HPK || num_heads % heads_per_kv != 0) return 0;
    dim3 grid(num_heads / heads_per_kv, splits);
    switch (head_dim) {
    case 256:
        attention_decode_gqa_kernel<KV, 256><<<grid, DGQA_THREADS>>>(q, kc, vc, part_out, part_max, part_sum,
            start_pos, num_heads, heads_per_kv, capacity, scale, splits);
        return 1;
    case 128:
        attention_decode_gqa_kernel<KV, 128><<<grid, DGQA_THREADS>>>(q, kc, vc, part_out, part_max, part_sum,
            start_pos, num_heads, heads_per_kv, capacity, scale, splits);
        return 1;
    default:
        return 0;
    }
}
template <>
int attention_decode_gqa_launch<float>(const float*, const float*, const float*, float*, float*, float*,
                                       int, int, int, int, int, float, int) {
    return 0;   // the f32 cache keeps the per-head kernel
}

template <typename KV>
static void attention_split_kv_launch(const float* q, const KV* kc, const KV* vc,
                                      float* out, float* ws, int new_tokens, int start_pos,
                                      int num_heads, int heads_per_kv, int head_dim,
                                      int capacity, float scale) {
    const int total_len = start_pos + new_tokens;
    const int splits = attention_split_count(total_len);
    if (splits <= 0) return;

    const long n_part = (long)num_heads * new_tokens * splits;
    float* part_out = ws;
    float* part_max = ws + n_part * head_dim;
    float* part_sum = part_max + n_part;

    const int threads = 128;
    const bool grouped = new_tokens == 1 &&
        attention_decode_gqa_launch<KV>(q, kc, vc, part_out, part_max, part_sum, start_pos, num_heads,
                                        heads_per_kv, head_dim, capacity, scale, splits);
    if (!grouped) {
        dim3 grid(num_heads, new_tokens, splits);
        size_t shmem = (head_dim + ATTN_SPLIT_SCORES + threads) * sizeof(float);
        attention_split_kv_kernel<KV><<<grid, threads, shmem>>>(q, kc, vc,
                                                                part_out, part_max, part_sum,
                                                                new_tokens, start_pos, num_heads,
                                                                heads_per_kv, head_dim, capacity,
                                                                scale, splits);
    }
    dim3 cgrid(num_heads, new_tokens);
    attention_split_combine_kernel<<<cgrid, threads>>>(part_out, part_max, part_sum, out,
                                                       new_tokens, num_heads, head_dim, splits);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_split_kv: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C++"

// ---------------------------------------------------------------------------------------------
// Prefill attention: tiled, flash-attention style.
//
// The split-KV kernel above is built for DECODE -- one query row, the key axis split across blocks.
// Prefill reused it with one block per (head, query token, split), so every query re-read its keys
// from global memory -- with 24 query heads over 4 KV heads, a 256-token chunk read each K/V row
// 6 x 256 = 1536 times -- and the dot products ran one thread per key with row-strided loads.
// Measured on a 3.5K-token prefill of a 27B: 19 ms per call, 5.0 s of a 22.6 s prefill.
//
// Here a block owns 16 queries of one head and walks the keys in tiles of 16, staging Q, K and V in
// shared memory. Scores stay in shared memory, softmax is the online max-rescale (exact, as in the
// split combine), and causal masking is per (query, key). Each K/V row is read from global memory
// once per 16 queries instead of once per query.
//
// Thread map: 256 threads as 16 rows x 16 lanes. Lane c of row r computes score (r, c) of a tile,
// then accumulates output dims c, c+16, ... of query r, so a row's softmax statistics are a
// 16-lane shuffle reduction and the PV reads of V are conflict-free.
#define PFA_BQ 16
#define PFA_BK 16
#define PFA_MAX_DPL 16   // head_dim / 16 output dims per thread; head_dim <= 256

extern "C++" {
template <typename KV>
__global__ void attention_prefill_kernel(const float* __restrict__ q,
                                         const KV* __restrict__ kc,
                                         const KV* __restrict__ vc,
                                         float* __restrict__ out,
                                         int new_tokens, int start_pos,
                                         int num_heads, int heads_per_kv,
                                         int head_dim, int capacity, float scale) {
    extern __shared__ float pfa_smem[];
    const int hd = head_dim;
    const int kstride = hd + 1;                 // pad: 16 K rows at the same d hit 16 banks
    float* Qs = pfa_smem;                       // [BQ][hd]
    float* Ks = Qs + PFA_BQ * hd;               // [BK][hd+1]
    float* Vs = Ks + PFA_BK * kstride;          // [BK][hd]
    float* Ps = Vs + PFA_BK * hd;               // [BQ][BK]

    const int t = threadIdx.x;
    const int r = t >> 4;                       // query row within the tile
    const int c = t & 15;                       // key column / output-dim lane
    const int q0 = blockIdx.x * PFA_BQ;
    const int h = blockIdx.y;
    const int kv_h = h / heads_per_kv;
    const int d_model = num_heads * hd;
    const int dpl = hd >> 4;
    const int rows = min(PFA_BQ, new_tokens - q0);

    const KV* kh = kc + (long)kv_h * capacity * hd;
    const KV* vh = vc + (long)kv_h * capacity * hd;

    for (int idx = t; idx < PFA_BQ * hd; idx += blockDim.x) {
        const int qi = idx / hd, d = idx - qi * hd;
        Qs[idx] = (qi < rows) ? q[(long)(q0 + qi) * d_model + (long)h * hd + d] : 0.0f;
    }

    const bool row_ok = r < rows;
    const int my_pos = start_pos + q0 + r;      // last key this query may see
    const int kend = start_pos + q0 + rows;     // keys any query of the tile may see

    float o[PFA_MAX_DPL];
    #pragma unroll
    for (int m = 0; m < PFA_MAX_DPL; ++m) o[m] = 0.0f;
    float m_i = -INFINITY, l_i = 0.0f;

    for (int j0 = 0; j0 < kend; j0 += PFA_BK) {
        __syncthreads();   // the previous tile's PV reads are done before K/V are overwritten
        for (int idx = t; idx < PFA_BK * hd; idx += blockDim.x) {
            const int kj = idx / hd, d = idx - kj * hd;
            const int j = j0 + kj;
            const bool ok = j < kend;
            Ks[kj * kstride + d] = ok ? KVStore<KV>::load(kh, (long)j * hd + d) : 0.0f;
            Vs[kj * hd + d] = ok ? KVStore<KV>::load(vh, (long)j * hd + d) : 0.0f;
        }
        __syncthreads();

        const int j = j0 + c;
        float s = -INFINITY;
        if (row_ok && j <= my_pos && j < kend) {
            const float* qr = Qs + r * hd;
            const float* kr = Ks + c * kstride;
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d) dot += qr[d] * kr[d];
            s = dot * scale;
        }
        float tmax = s;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFF, tmax, off, 16));
        const float m_new = fmaxf(m_i, tmax);
        float p = 0.0f, alpha = 1.0f;
        if (m_new != -INFINITY) {
            p = (s == -INFINITY) ? 0.0f : expf(s - m_new);
            alpha = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
        }
        float tsum = p;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) tsum += __shfl_xor_sync(0xFFFFFFFF, tsum, off, 16);
        l_i = l_i * alpha + tsum;
        m_i = m_new;
        Ps[r * PFA_BK + c] = p;
        #pragma unroll
        for (int m = 0; m < PFA_MAX_DPL; ++m) o[m] *= alpha;
        __syncthreads();

        #pragma unroll 4
        for (int kj = 0; kj < PFA_BK; ++kj) {
            const float pk = Ps[r * PFA_BK + kj];
            const float* vr = Vs + kj * hd + c;
            #pragma unroll
            for (int m = 0; m < PFA_MAX_DPL; ++m)
                if (m < dpl) o[m] += pk * vr[m * 16];
        }
    }

    if (row_ok) {
        const float inv = 1.0f / l_i;
        float* orow = out + (long)(q0 + r) * d_model + (long)h * hd + c;
        #pragma unroll
        for (int m = 0; m < PFA_MAX_DPL; ++m)
            if (m < dpl) orow[m * 16] = o[m] * inv;
    }
}

// Returns 1 when it ran, 0 when this shape or device cannot use it (head_dim not a multiple of 16 or
// above 256, or the >48 KB shared-memory opt-in refused), so the caller keeps the split kernel.
template <typename KV>
static int attention_prefill_launch(const float* q, const KV* kc, const KV* vc, float* out,
                                    int new_tokens, int start_pos, int num_heads, int heads_per_kv,
                                    int head_dim, int capacity, float scale) {
    if (new_tokens <= 0 || head_dim <= 0 || (head_dim & 15) || head_dim > 16 * PFA_MAX_DPL) return 0;
    const size_t smem = (size_t)(PFA_BQ * head_dim + PFA_BK * (head_dim + 1) + PFA_BK * head_dim +
                                 PFA_BQ * PFA_BK) * sizeof(float);
    static size_t granted = 0;   // per instantiation: the largest opt-in already accepted
    if (smem > granted) {
        if (cudaFuncSetAttribute(attention_prefill_kernel<KV>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        granted = smem;
    }
    dim3 grid((new_tokens + PFA_BQ - 1) / PFA_BQ, num_heads);
    attention_prefill_kernel<KV><<<grid, 256, smem>>>(q, kc, vc, out, new_tokens, start_pos,
                                                      num_heads, heads_per_kv, head_dim,
                                                      capacity, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_prefill: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}
} // extern "C++"

int attention_prefill_f32(const float* q, const float* kc, const float* vc, float* out,
                          int new_tokens, int start_pos, int num_heads, int heads_per_kv,
                          int head_dim, int capacity, float scale) {
    return attention_prefill_launch<float>(q, kc, vc, out, new_tokens, start_pos, num_heads,
                                           heads_per_kv, head_dim, capacity, scale);
}

// ---------------------------------------------------------------------------------------------
// Tensor-core prefill attention (WMMA, fp16 inputs, fp32 accumulation).
//
// The tiled kernel above does its dot products on CUDA cores out of shared memory, which is fine at
// a few thousand tokens and dominant at tens of thousands: a 55K-token prefill spent ~540 s of 797 s
// in it, because per-query work grows with context. This one does Q*K^T and P*V as 16x16x16 MMAs.
//
// A block owns 64 queries of one head (4 warps x 16 rows) and walks the keys in tiles of 32, with Q,
// K and V staged in shared memory as half. fp8 and fp16 caches convert to half EXACTLY, so the only
// rounding added is Q and P to half -- the same 10-bit mantissa TF32 matmuls already use. f32 caches
// keep the CUDA-core kernel.
//
// Two passes over the keys rather than one online pass. WMMA accumulator fragments have an opaque
// element-to-row mapping, so the per-row rescale an online softmax needs cannot be applied to the O
// accumulators. Pass 1 computes each row's exact max and sum (scalars, which CAN be rescaled);
// pass 2 recomputes the scores and accumulates the already-normalized P*V. Q*K^T runs twice, 1.5x
// the FLOPs of one pass, all of it on tensor cores.
#define PFW_BQ 64
#define PFW_BK 32
#define PFW_WARPS 4
#define PFW_SPAD 4      // float row pad for the score tile (WMMA float ldm must be a multiple of 4)
#define PFW_HPAD 8      // half row pad (ldm multiple of 8, and breaks bank alignment of rows)

extern "C++" {
// Eight consecutive cache elements as eight halves packed in a uint4. `p` is 8-element aligned.
__device__ __forceinline__ uint4 kv_to_half8(const __half* p) {
    return *((const uint4*)p);
}
__device__ __forceinline__ uint4 kv_to_half8(const __nv_fp8_e4m3* p) {
    const uint2 raw = *((const uint2*)p);
    uint4 r;
    __half2_raw h;
    h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(raw.x & 0xFFFF), __NV_E4M3);
    r.x = (unsigned int)h.x | ((unsigned int)h.y << 16);
    h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(raw.x >> 16), __NV_E4M3);
    r.y = (unsigned int)h.x | ((unsigned int)h.y << 16);
    h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(raw.y & 0xFFFF), __NV_E4M3);
    r.z = (unsigned int)h.x | ((unsigned int)h.y << 16);
    h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(raw.y >> 16), __NV_E4M3);
    r.w = (unsigned int)h.x | ((unsigned int)h.y << 16);
    return r;
}

template <typename KV, int D>
__global__ void attention_prefill_wmma_kernel(const float* __restrict__ q,
                                              const KV* __restrict__ kc,
                                              const KV* __restrict__ vc,
                                              float* __restrict__ out,
                                              int new_tokens, int start_pos,
                                              int num_heads, int heads_per_kv,
                                              int capacity, float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(128) unsigned char pfw_raw[];
    constexpr int HLD = D + PFW_HPAD;                 // half row stride of Q/K/V tiles
    constexpr int SLD = PFW_BK + PFW_SPAD;            // float row stride of a warp's score tile
    constexpr int PLD = PFW_BK + PFW_HPAD;            // half row stride of a warp's P tile
    __half* Qs = (__half*)pfw_raw;                    // [BQ][HLD]
    __half* Ks = Qs + PFW_BQ * HLD;                   // [BK][HLD]
    __half* Vs = Ks + PFW_BK * HLD;                   // [BK][HLD]
    float* Sall = (float*)(Vs + PFW_BK * HLD);        // [WARPS][16][SLD]
    __half* Pall = (__half*)(Sall + PFW_WARPS * 16 * SLD); // [WARPS][16][PLD]

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q0 = blockIdx.x * PFW_BQ;
    const int h = blockIdx.y;
    const int kv_h = h / heads_per_kv;
    const int d_model = num_heads * D;
    const int rows = min(PFW_BQ, new_tokens - q0);
    const int kend = start_pos + q0 + rows;           // keys any query of the block may see
    float* Ss = Sall + warp * 16 * SLD;
    __half* Ps = Pall + warp * 16 * PLD;

    const KV* kh = kc + (long)kv_h * capacity * D;
    const KV* vh = vc + (long)kv_h * capacity * D;

    for (int idx = tid; idx < PFW_BQ * D; idx += blockDim.x) {
        const int qi = idx / D, d = idx - qi * D;
        const float v = (qi < rows) ? q[(long)(q0 + qi) * d_model + (long)h * D + d] : 0.0f;
        Qs[qi * HLD + d] = __float2half(v);
    }

    // Lane pair (2r, 2r+1) owns score row r of this warp's 16, each lane 16 of the 32 columns.
    const int sr = lane >> 1;
    const int sc0 = (lane & 1) * 16;
    const int row_q = q0 + warp * 16 + sr;            // this lane's query index in the chunk
    const bool row_ok = row_q < q0 + rows;
    const int my_pos = start_pos + row_q;             // last key it may see
    float m_i = -INFINITY, l_i = 0.0f;

    // S = Q_w (16 x D) * K^T (D x 32) into this warp's score tile.
    auto scores = [&]() {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s0, s1;
        wmma::fill_fragment(s0, 0.0f);
        wmma::fill_fragment(s1, 0.0f);
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b0, b1;
            wmma::load_matrix_sync(a, Qs + (warp * 16) * HLD + kk * 16, HLD);
            wmma::load_matrix_sync(b0, Ks + 0 * HLD + kk * 16, HLD);
            wmma::load_matrix_sync(b1, Ks + 16 * HLD + kk * 16, HLD);
            wmma::mma_sync(s0, a, b0, s0);
            wmma::mma_sync(s1, a, b1, s1);
        }
        wmma::store_matrix_sync(Ss, s0, SLD, wmma::mem_row_major);
        wmma::store_matrix_sync(Ss + 16, s1, SLD, wmma::mem_row_major);
        __syncwarp();
    };

    // Eight cache elements per thread per step, converted to half8 and stored as one 16-byte write.
    // Element-wise loads made this the kernel's bottleneck: one-byte fp8 loads, 64 per thread per
    // tile, for a tile that is read twice (K) and once (V) per block.
    auto load_kv_tile = [&](int j0, bool with_v) {
        constexpr int V8 = D / 8;
        for (int idx = tid; idx < PFW_BK * V8; idx += blockDim.x) {
            const int kj = idx / V8, d = (idx - kj * V8) * 8;
            const int j = j0 + kj;
            uint4* kdst = (uint4*)(Ks + kj * HLD + d);
            uint4* vdst = (uint4*)(Vs + kj * HLD + d);
            if (j < kend) {
                *kdst = kv_to_half8(kh + (long)j * D + d);
                if (with_v) *vdst = kv_to_half8(vh + (long)j * D + d);
            } else {
                *kdst = make_uint4(0, 0, 0, 0);
                if (with_v) *vdst = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // Pass 1: exact row max and sum.
    for (int j0 = 0; j0 < kend; j0 += PFW_BK) {
        __syncthreads();
        load_kv_tile(j0, false);
        __syncthreads();
        scores();
        float tmax = -INFINITY;
        float sv[16];
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            const int j = j0 + sc0 + c;
            float s = -INFINITY;
            if (row_ok && j <= my_pos && j < kend) s = Ss[sr * SLD + sc0 + c] * scale;
            sv[c] = s;
            tmax = fmaxf(tmax, s);
        }
        tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFF, tmax, 1));
        const float m_new = fmaxf(m_i, tmax);
        // The shuffle stays OUTSIDE any row-dependent branch: rows past the end of the chunk have
        // m_new == -inf, and a full-mask shuffle that some lanes skip deadlocks the warp.
        float ts = 0.0f;
        if (m_new != -INFINITY) {
            #pragma unroll
            for (int c = 0; c < 16; ++c) ts += (sv[c] == -INFINITY) ? 0.0f : expf(sv[c] - m_new);
        }
        ts += __shfl_xor_sync(0xFFFFFFFF, ts, 1);
        if (m_new != -INFINITY) {
            l_i = l_i * ((m_i == -INFINITY) ? 0.0f : expf(m_i - m_new)) + ts;
            m_i = m_new;
        }
    }
    const float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;

    // Pass 2: O = sum over tiles of softmax(S) * V, with the final statistics.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[D / 16];
    #pragma unroll
    for (int f = 0; f < D / 16; ++f) wmma::fill_fragment(o[f], 0.0f);

    for (int j0 = 0; j0 < kend; j0 += PFW_BK) {
        __syncthreads();
        load_kv_tile(j0, true);
        __syncthreads();
        scores();
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            const int j = j0 + sc0 + c;
            float p = 0.0f;
            if (row_ok && j <= my_pos && j < kend)
                p = expf(Ss[sr * SLD + sc0 + c] * scale - m_i) * inv_l;
            Ps[sr * PLD + sc0 + c] = __float2half(p);
        }
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < PFW_BK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
            wmma::load_matrix_sync(pa, Ps + kk * 16, PLD);
            #pragma unroll
            for (int f = 0; f < D / 16; ++f) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
                wmma::load_matrix_sync(vb, Vs + (kk * 16) * HLD + f * 16, HLD);
                wmma::mma_sync(o[f], pa, vb, o[f]);
            }
        }
    }

    // Stage each 16x16 output fragment through the warp's score tile and write the valid rows.
    const int wrow0 = q0 + warp * 16;
    #pragma unroll
    for (int f = 0; f < D / 16; ++f) {
        __syncwarp();
        wmma::store_matrix_sync(Ss, o[f], SLD, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = e >> 4, c = e & 15;
            if (wrow0 + r < q0 + rows)
                out[(long)(wrow0 + r) * d_model + (long)h * D + f * 16 + c] = Ss[r * SLD + c];
        }
    }
#endif
}

template <typename KV, int D>
static int attention_prefill_wmma_launch_d(const float* q, const KV* kc, const KV* vc, float* out,
                                           int new_tokens, int start_pos, int num_heads,
                                           int heads_per_kv, int capacity, float scale) {
    constexpr int HLD = D + PFW_HPAD;
    const size_t smem = (size_t)(PFW_BQ + 2 * PFW_BK) * HLD * sizeof(__half) +
                        (size_t)PFW_WARPS * 16 * (PFW_BK + PFW_SPAD) * sizeof(float) +
                        (size_t)PFW_WARPS * 16 * (PFW_BK + PFW_HPAD) * sizeof(__half);
    static int state = 0;   // 0 untested, 1 usable, -1 refused
    if (state == 0) {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        state = (major >= 7 &&
                 cudaFuncSetAttribute(attention_prefill_wmma_kernel<KV, D>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)smem) == cudaSuccess) ? 1 : -1;
        cudaGetLastError();
    }
    if (state < 0) return 0;
    dim3 grid((new_tokens + PFW_BQ - 1) / PFW_BQ, num_heads);
    attention_prefill_wmma_kernel<KV, D><<<grid, PFW_WARPS * 32, smem>>>(
        q, kc, vc, out, new_tokens, start_pos, num_heads, heads_per_kv, capacity, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_prefill_wmma: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

// 1 when the tensor-core kernel ran; 0 hands the call to the CUDA-core tiled kernel.
template <typename KV>
static int attention_prefill_wmma_launch(const float* q, const KV* kc, const KV* vc, float* out,
                                         int new_tokens, int start_pos, int num_heads,
                                         int heads_per_kv, int head_dim, int capacity, float scale) {
    static const bool enabled = [] {
        const char* e = getenv("SHAINET_WMMA_PREFILL_ATTN");
        return !(e && e[0] == '0');
    }();
    if (!enabled || new_tokens <= 0) return 0;
    switch (head_dim) {
    case 128: return attention_prefill_wmma_launch_d<KV, 128>(q, kc, vc, out, new_tokens, start_pos,
                                                              num_heads, heads_per_kv, capacity, scale);
    case 256: return attention_prefill_wmma_launch_d<KV, 256>(q, kc, vc, out, new_tokens, start_pos,
                                                              num_heads, heads_per_kv, capacity, scale);
    default: return 0;
    }
}

// ---------------------------------------------------------------------------------------------
// FlashAttention-2-style prefill attention on mma.sync (fp16 inputs, fp32 accumulation).
//
// The WMMA kernel above runs TWO passes over the keys because WMMA's accumulator layout is opaque and
// an online softmax cannot rescale its output rows. mma.sync m16n8k16 has a documented layout -- a
// thread holds rows g and g+8 of every accumulator tile -- so here the running max and sum are kept
// per row, the output accumulators are rescaled in registers, and P goes straight from the score
// accumulators into the A operand of the P*V product. One pass: Q*K^T once instead of twice.
//
// Block: 64 queries of one head (4 warps x 16 rows); keys in tiles of 32. Q, K and V are staged as
// half with rows padded by 8 so ldmatrix hits distinct banks; Q and K feed ldmatrix, V feeds
// ldmatrix.trans (P*V needs V as a column-major operand). fp8 and fp16 caches convert to half exactly.
#define FA2_BQ 64
#define FA2_BK 32

__device__ __forceinline__ void fa2_ldsm_x4(unsigned int* r, const __half* p) {
    const unsigned int a = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fa2_ldsm_x4_t(unsigned int* r, const __half* p) {
    const unsigned int a = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fa2_mma(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ unsigned int fa2_pack(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *((unsigned int*)&h);
}

template <typename KV, int D>
__global__ void __launch_bounds__(128) attention_prefill_fa2_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc, const KV* __restrict__ vc,
    float* __restrict__ out, int new_tokens, int start_pos, int num_heads, int heads_per_kv,
    int capacity, float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(128) unsigned char fa2_raw[];
    constexpr int LD = D + 8;
    constexpr int NT_O = D / 8;                 // output n8 tiles per warp row block
    __half* Qs = (__half*)fa2_raw;              // [BQ][LD]
    __half* Ks = Qs + FA2_BQ * LD;              // [BK][LD]
    __half* Vs = Ks + FA2_BK * LD;              // [BK][LD]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int q0 = blockIdx.x * FA2_BQ, h = blockIdx.y;
    const int kv_h = h / heads_per_kv;
    const int d_model = num_heads * D;
    const int rows = min(FA2_BQ, new_tokens - q0);
    const int kend = start_pos + q0 + rows;
    const KV* kh = kc + (long)kv_h * capacity * D;
    const KV* vh = vc + (long)kv_h * capacity * D;

    for (int idx = tid; idx < FA2_BQ * D; idx += 128) {
        const int qi = idx / D, d = idx - qi * D;
        const float v = (qi < rows) ? q[(long)(q0 + qi) * d_model + (long)h * D + d] : 0.0f;
        Qs[qi * LD + d] = __float2half(v);
    }

    const int r0 = warp * 16 + g;               // this thread's rows within the block: r0, r0 + 8
    const int pos0 = start_pos + q0 + r0, pos1 = pos0 + 8;
    const bool ok0 = r0 < rows, ok1 = r0 + 8 < rows;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    float o[NT_O][4];
    #pragma unroll
    for (int i = 0; i < NT_O; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

    for (int j0 = 0; j0 < kend; j0 += FA2_BK) {
        __syncthreads();
        constexpr int V8 = D / 8;
        for (int idx = tid; idx < FA2_BK * V8; idx += 128) {
            const int kj = idx / V8, d = (idx - kj * V8) * 8;
            const int j = j0 + kj;
            uint4* kd = (uint4*)(Ks + kj * LD + d);
            uint4* vd = (uint4*)(Vs + kj * LD + d);
            if (j < kend) {
                *kd = kv_to_half8(kh + (long)j * D + d);
                *vd = kv_to_half8(vh + (long)j * D + d);
            } else {
                *kd = make_uint4(0, 0, 0, 0);
                *vd = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // S = Q_w K^T: 16 x 32 per warp, four n8 tiles.
        float sacc[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) sacc[i][0] = sacc[i][1] = sacc[i][2] = sacc[i][3] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            unsigned int a[4];
            fa2_ldsm_x4(a, Qs + (warp * 16 + (lane & 15)) * LD + kk * 16 + (lane >> 4) * 8);
            #pragma unroll
            for (int np = 0; np < 2; ++np) {      // key tiles (2np, 2np+1)
                unsigned int b[4];
                fa2_ldsm_x4(b, Ks + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + kk * 16 + ((lane >> 3) & 1) * 8);
                fa2_mma(sacc[2 * np], a, b[0], b[1]);
                fa2_mma(sacc[2 * np + 1], a, b[2], b[3]);
            }
        }

        // Online softmax on rows r0 (c0, c1) and r0 + 8 (c2, c3).
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int j = j0 + nt * 8 + 2 * t + e;
                const bool v0 = ok0 && j <= pos0 && j < kend, v1 = ok1 && j <= pos1 && j < kend;
                sacc[nt][e] = v0 ? sacc[nt][e] * scale : -INFINITY;
                sacc[nt][2 + e] = v1 ? sacc[nt][2 + e] * scale : -INFINITY;
                mx0 = fmaxf(mx0, sacc[nt][e]);
                mx1 = fmaxf(mx1, sacc[nt][2 + e]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = (mn0 == -INFINITY || m0 == -INFINITY) ? (mn0 == -INFINITY ? 1.0f : 0.0f) : expf(m0 - mn0);
        const float al1 = (mn1 == -INFINITY || m1 == -INFINITY) ? (mn1 == -INFINITY ? 1.0f : 0.0f) : expf(m1 - mn1);
        m0 = mn0;
        m1 = mn1;
        float ps0 = 0.0f, ps1 = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float p0 = (sacc[nt][e] == -INFINITY) ? 0.0f : expf(sacc[nt][e] - mn0);
                const float p1 = (sacc[nt][2 + e] == -INFINITY) ? 0.0f : expf(sacc[nt][2 + e] - mn1);
                sacc[nt][e] = p0;
                sacc[nt][2 + e] = p1;
                ps0 += p0;
                ps1 += p1;
            }
        }
        l0 = l0 * al0 + ps0;
        l1 = l1 * al1 + ps1;
        #pragma unroll
        for (int i = 0; i < NT_O; ++i) {
            o[i][0] *= al0; o[i][1] *= al0;
            o[i][2] *= al1; o[i][3] *= al1;
        }

        // O += P (16 x 32) * V (32 x D). P's accumulator tiles pair into the A operand directly.
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            unsigned int pa[4];
            pa[0] = fa2_pack(sacc[2 * ks][0], sacc[2 * ks][1]);
            pa[1] = fa2_pack(sacc[2 * ks][2], sacc[2 * ks][3]);
            pa[2] = fa2_pack(sacc[2 * ks + 1][0], sacc[2 * ks + 1][1]);
            pa[3] = fa2_pack(sacc[2 * ks + 1][2], sacc[2 * ks + 1][3]);
            #pragma unroll
            for (int np = 0; np < NT_O / 2; ++np) {   // dim tiles (2np, 2np+1)
                unsigned int b[4];
                fa2_ldsm_x4_t(b, Vs + (ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + np * 16 + (lane >> 4) * 8);
                fa2_mma(o[2 * np], pa, b[0], b[1]);
                fa2_mma(o[2 * np + 1], pa, b[2], b[3]);
            }
        }
    }

    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 1);
    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 2);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 1);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 2);
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    #pragma unroll
    for (int i = 0; i < NT_O; ++i) {
        const int d = i * 8 + 2 * t;
        if (ok0) {
            float* orow = out + (long)(q0 + r0) * d_model + (long)h * D + d;
            orow[0] = o[i][0] * inv0;
            orow[1] = o[i][1] * inv0;
        }
        if (ok1) {
            float* orow = out + (long)(q0 + r0 + 8) * d_model + (long)h * D + d;
            orow[0] = o[i][2] * inv1;
            orow[1] = o[i][3] * inv1;
        }
    }
#endif
}

template <typename KV, int D>
static int attention_prefill_fa2_launch_d(const float* q, const KV* kc, const KV* vc, float* out,
                                          int new_tokens, int start_pos, int num_heads,
                                          int heads_per_kv, int capacity, float scale) {
    const size_t smem = (size_t)(FA2_BQ + 2 * FA2_BK) * (D + 8) * sizeof(__half);
    static int state = 0;
    if (state == 0) {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        state = (major >= 8 &&
                 cudaFuncSetAttribute(attention_prefill_fa2_kernel<KV, D>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess) ? 1 : -1;
        cudaGetLastError();
    }
    if (state < 0) return 0;
    dim3 grid((new_tokens + FA2_BQ - 1) / FA2_BQ, num_heads);
    attention_prefill_fa2_kernel<KV, D><<<grid, 128, smem>>>(q, kc, vc, out, new_tokens, start_pos,
                                                             num_heads, heads_per_kv, capacity, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_prefill_fa2: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

// 1 when the FA2 kernel ran (sm_80+, head_dim 128/256, half or fp8 cache); 0 falls back.
template <typename KV>
static int attention_prefill_fa2_launch(const float* q, const KV* kc, const KV* vc, float* out,
                                        int new_tokens, int start_pos, int num_heads,
                                        int heads_per_kv, int head_dim, int capacity, float scale) {
    static const bool enabled = [] {
        const char* e = getenv("SHAINET_FA2_PREFILL_ATTN");
        return !(e && e[0] == '0');
    }();
    if (!enabled || new_tokens <= 0) return 0;
    switch (head_dim) {
    case 128: return attention_prefill_fa2_launch_d<KV, 128>(q, kc, vc, out, new_tokens, start_pos,
                                                             num_heads, heads_per_kv, capacity, scale);
    case 256: return attention_prefill_fa2_launch_d<KV, 256>(q, kc, vc, out, new_tokens, start_pos,
                                                             num_heads, heads_per_kv, capacity, scale);
    default: return 0;
    }
}
} // extern "C++"

int attention_prefill_f16(const float* q, const unsigned short* kc, const unsigned short* vc,
                          float* out, int new_tokens, int start_pos, int num_heads,
                          int heads_per_kv, int head_dim, int capacity, float scale) {
    if (attention_prefill_fa2_launch<__half>(q, reinterpret_cast<const __half*>(kc),
                                             reinterpret_cast<const __half*>(vc), out, new_tokens,
                                             start_pos, num_heads, heads_per_kv, head_dim,
                                             capacity, scale)) return 1;
    if (attention_prefill_wmma_launch<__half>(q, reinterpret_cast<const __half*>(kc),
                                              reinterpret_cast<const __half*>(vc), out, new_tokens,
                                              start_pos, num_heads, heads_per_kv, head_dim,
                                              capacity, scale)) return 1;
    return attention_prefill_launch<__half>(q, reinterpret_cast<const __half*>(kc),
                                            reinterpret_cast<const __half*>(vc), out, new_tokens,
                                            start_pos, num_heads, heads_per_kv, head_dim,
                                            capacity, scale);
}

int attention_prefill_fp8(const float* q, const unsigned char* kc, const unsigned char* vc,
                          float* out, int new_tokens, int start_pos, int num_heads,
                          int heads_per_kv, int head_dim, int capacity, float scale) {
    if (attention_prefill_fa2_launch<__nv_fp8_e4m3>(q, reinterpret_cast<const __nv_fp8_e4m3*>(kc),
                                                    reinterpret_cast<const __nv_fp8_e4m3*>(vc), out,
                                                    new_tokens, start_pos, num_heads, heads_per_kv,
                                                    head_dim, capacity, scale)) return 1;
    if (attention_prefill_wmma_launch<__nv_fp8_e4m3>(q, reinterpret_cast<const __nv_fp8_e4m3*>(kc),
                                                     reinterpret_cast<const __nv_fp8_e4m3*>(vc), out,
                                                     new_tokens, start_pos, num_heads, heads_per_kv,
                                                     head_dim, capacity, scale)) return 1;
    return attention_prefill_launch<__nv_fp8_e4m3>(q, reinterpret_cast<const __nv_fp8_e4m3*>(kc),
                                                   reinterpret_cast<const __nv_fp8_e4m3*>(vc), out,
                                                   new_tokens, start_pos, num_heads, heads_per_kv,
                                                   head_dim, capacity, scale);
}

void attention_split_kv_f32(const float* q, const float* kc, const float* vc,
                            float* out, float* ws, int new_tokens, int start_pos,
                            int num_heads, int heads_per_kv, int head_dim,
                            int capacity, float scale) {
    attention_split_kv_launch<float>(q, kc, vc, out, ws, new_tokens, start_pos,
                                     num_heads, heads_per_kv, head_dim, capacity, scale);
}

void attention_split_kv_f16(const float* q, const unsigned short* kc, const unsigned short* vc,
                            float* out, float* ws, int new_tokens, int start_pos,
                            int num_heads, int heads_per_kv, int head_dim,
                            int capacity, float scale) {
    attention_split_kv_launch<__half>(q, reinterpret_cast<const __half*>(kc),
                                      reinterpret_cast<const __half*>(vc),
                                      out, ws, new_tokens, start_pos,
                                      num_heads, heads_per_kv, head_dim, capacity, scale);
}

void attention_split_kv_fp8(const float* q, const unsigned char* kc, const unsigned char* vc,
                            float* out, float* ws, int new_tokens, int start_pos,
                            int num_heads, int heads_per_kv, int head_dim,
                            int capacity, float scale) {
    attention_split_kv_launch<__nv_fp8_e4m3>(q, reinterpret_cast<const __nv_fp8_e4m3*>(kc),
                                             reinterpret_cast<const __nv_fp8_e4m3*>(vc),
                                             out, ws, new_tokens, start_pos,
                                             num_heads, heads_per_kv, head_dim, capacity, scale);
}

void attention_kv_f32(const float* q, const float* kc, const float* vc,
                      float* out, float* ws, int new_tokens, int start_pos,
                      int num_heads, int heads_per_kv, int head_dim,
                      int capacity, float scale) {
    int threads = 128;
    dim3 grid(num_heads, new_tokens);
    size_t shmem = (head_dim + threads) * sizeof(float);
    attention_kv_kernel_t<float><<<grid, threads, shmem>>>(q, kc, vc, out, ws,
                                                           new_tokens, start_pos,
                                                           num_heads, heads_per_kv,
                                                           head_dim, capacity, scale);
    // No device sync: callers read `out` back via a default-stream D2H memcpy
    // which is ordered after this kernel. Just surface launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_kv_f32: %s\n", cudaGetErrorString(err));
    }
}

// fp16 cache variant. Q, out and ws stay fp32; only the cache is half.
void attention_kv_f16(const float* q, const unsigned short* kc, const unsigned short* vc,
                      float* out, float* ws, int new_tokens, int start_pos,
                      int num_heads, int heads_per_kv, int head_dim,
                      int capacity, float scale) {
    int threads = 128;
    dim3 grid(num_heads, new_tokens);
    size_t shmem = (head_dim + threads) * sizeof(float);
    attention_kv_kernel_t<__half><<<grid, threads, shmem>>>(q,
                                                            reinterpret_cast<const __half*>(kc),
                                                            reinterpret_cast<const __half*>(vc),
                                                            out, ws,
                                                            new_tokens, start_pos,
                                                            num_heads, heads_per_kv,
                                                            head_dim, capacity, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_kv_f16: %s\n", cudaGetErrorString(err));
    }
}

void attention_kv_fp8(const float* q, const unsigned char* kc, const unsigned char* vc,
                      float* out, float* ws, int new_tokens, int start_pos,
                      int num_heads, int heads_per_kv, int head_dim,
                      int capacity, float scale) {
    int threads = 128;
    dim3 grid(num_heads, new_tokens);
    size_t shmem = (head_dim + threads) * sizeof(float);
    attention_kv_kernel_t<__nv_fp8_e4m3><<<grid, threads, shmem>>>(q,
                                                                   reinterpret_cast<const __nv_fp8_e4m3*>(kc),
                                                                   reinterpret_cast<const __nv_fp8_e4m3*>(vc),
                                                                   out, ws,
                                                                   new_tokens, start_pos,
                                                                   num_heads, heads_per_kv,
                                                                   head_dim, capacity, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in attention_kv_fp8: %s\n", cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// Gated delta rule (Qwen3.5 linear attention), whole sequence in ONE launch.
//
// One CUDA block per VALUE head, with that head's [dk, dv] recurrent state resident in shared
// memory for the entire sequence, and the token loop INSIDE the kernel. That is the point: the
// recurrence is sequential, so a kernel per step would pay a launch per token per layer (29k
// launches for a 1216-token prefill over 24 layers), and the host implementation it replaces spent
// 4.4 s per layer -- 114 s for a prefill -- in scalar matrix products.
//
// The state is dk*dv floats = 64 KB at Qwen3.5's 128x128, which exceeds the 48 KB default limit,
// so the launcher opts in to the larger dynamic allocation (sm_80+ allows ~100 KB). Occupancy is
// one block per SM by construction; with only nv=32 blocks against 76 SMs that costs nothing.
//
// L2 normalization of q and k, and the 1/sqrt(dk) scale on q, are done HERE rather than on the
// host, which removes the per-head slicing and copying that cost as much as the arithmetic.
//
// q and k are indexed by KEY head. Which key head a value head reads depends on the weight
// layout, so the caller passes k_head_tiled: 0 selects grouped indexing (kh = h / heads_per_k),
// which the SafeTensors layout uses and which matches HF's repeat_interleave; 1 selects tiled
// indexing (kh = h % nk), which the GGUF layout uses because llama.cpp widens q/k with
// ggml_repeat, and ggml_repeat tiles rather than interleaves.
__global__ void gated_delta_rule_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ alpha, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ out,
    int seq, int nv, int nk, int dk, int dv, int heads_per_k, float q_scale, int k_head_tiled)
{
    extern __shared__ float sh[];
    const int h = blockIdx.x;
    if (h >= nv) return;
    const int kh = k_head_tiled ? (h % nk) : (h / heads_per_k);
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    float* S   = sh;                 // [dk, dv], row-major in dk
    float* qs  = S + (size_t)dk * dv;
    float* ks  = qs + dk;
    float* kvm = ks + dk;            // [dv] readout of the decayed state
    float* red = kvm + dv;           // [nthr] reduction scratch

    for (int idx = tid; idx < dk * dv; idx += nthr) {
        S[idx] = state[(size_t)h * dk * dv + idx];
    }
    __syncthreads();

    const size_t kstride = (size_t)nk * dk;
    const size_t vstride = (size_t)nv * dv;

    for (int t = 0; t < seq; ++t) {
        // Load this position's q/k for the shared key head.
        for (int i = tid; i < dk; i += nthr) {
            qs[i] = q[(size_t)t * kstride + (size_t)kh * dk + i];
            ks[i] = k[(size_t)t * kstride + (size_t)kh * dk + i];
        }
        __syncthreads();

        // L2 norms, one block reduction each.
        float lq = 0.0f, lk = 0.0f;
        for (int i = tid; i < dk; i += nthr) { lq += qs[i] * qs[i]; lk += ks[i] * ks[i]; }
        red[tid] = lq;
        __syncthreads();
        for (int off = nthr / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        const float nq = rsqrtf(fmaxf(red[0], 1e-12f));
        __syncthreads();
        red[tid] = lk;
        __syncthreads();
        for (int off = nthr / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        const float nk_inv = rsqrtf(fmaxf(red[0], 1e-12f));
        __syncthreads();

        for (int i = tid; i < dk; i += nthr) {
            qs[i] = qs[i] * nq * q_scale;
            ks[i] = ks[i] * nk_inv;
        }
        const float a = alpha[(size_t)t * nv + h];
        const float b = beta[(size_t)t * nv + h];
        __syncthreads();

        // Decay the state, then read it out with k. Order matters: the reference computes kv_mem
        // from the ALREADY DECAYED state.
        for (int idx = tid; idx < dk * dv; idx += nthr) S[idx] *= a;
        __syncthreads();

        for (int c = tid; c < dv; c += nthr) {
            float acc = 0.0f;
            for (int i = 0; i < dk; ++i) acc += S[(size_t)i * dv + c] * ks[i];
            kvm[c] = acc;
        }
        __syncthreads();

        // S += k (x) beta*(v - kv_mem), then out = q^T S.
        for (int c = tid; c < dv; c += nthr) {
            const float delta = (v[(size_t)t * vstride + (size_t)h * dv + c] - kvm[c]) * b;
            for (int i = 0; i < dk; ++i) S[(size_t)i * dv + c] += ks[i] * delta;
        }
        __syncthreads();

        for (int c = tid; c < dv; c += nthr) {
            float acc = 0.0f;
            for (int i = 0; i < dk; ++i) acc += S[(size_t)i * dv + c] * qs[i];
            out[(size_t)t * vstride + (size_t)h * dv + c] = acc;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < dk * dv; idx += nthr) {
        state[(size_t)h * dk * dv + idx] = S[idx];
    }
}

// The same recurrence, column-parallel with the state in REGISTERS.
//
// The kernel above runs one block per value head -- 48 blocks on a 76-SM card -- and walks its 64 KB
// shared-memory state four times per token, with about twenty __syncthreads and two halving
// reductions in between. Measured: 2.6 ms per 256-token chunk, ~21% of a prefill, and ~40 us per
// layer per decode token.
//
// But the state's value columns never interact: column c is decayed, read with k, corrected by
// v[c], and read with q, all on its own. So a block here owns 32 columns of one head (grid nv x
// dv/32), and each column is split across 4 lanes of the same warp, each holding dk/4 rows of it
// in registers. Readouts reduce with two shuffles; the only block-wide step is staging q and k,
// double-buffered so it costs one __syncthreads per token. Norms are recomputed per warp, which is
// cheaper than synchronizing to share them.
//
// Mathematically identical to the kernel above; summation order differs, so results agree to fp32
// rounding rather than bit for bit.
#define GDR_COLS 32         // value columns per block: 4 warps x 8 columns
#define GDR_GROUPS 4        // lanes (row groups) per column
#define GDR_MAX_ROWS 32     // rows per lane, so dk <= 128

__global__ void gated_delta_rule_cols_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ alpha, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ out,
    int seq, int nv, int nk, int dk, int dv, int heads_per_k, float q_scale, int k_head_tiled)
{
    // Padded by one per group so the four groups' lanes read four different banks.
    __shared__ float qs[2][GDR_GROUPS * (GDR_MAX_ROWS + 1)];
    __shared__ float ks[2][GDR_GROUPS * (GDR_MAX_ROWS + 1)];
    const int h = blockIdx.x;
    const int kh = k_head_tiled ? (h % nk) : (h / heads_per_k);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane & 3;
    const int c = blockIdx.y * GDR_COLS + warp * 8 + (lane >> 2);
    const int rpg = dk / GDR_GROUPS;
    const int pad = rpg + 1;

    const size_t sbase = (size_t)h * dk * dv;
    float S[GDR_MAX_ROWS];
    #pragma unroll
    for (int r = 0; r < GDR_MAX_ROWS; ++r)
        S[r] = (r < rpg) ? state[sbase + (size_t)(g * rpg + r) * dv + c] : 0.0f;

    const size_t kstride = (size_t)nk * dk;
    const size_t vstride = (size_t)nv * dv;

    for (int t = 0; t < seq; ++t) {
        const int buf = t & 1;
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            const int gi = i / rpg, ri = i - gi * rpg;
            qs[buf][gi * pad + ri] = q[(size_t)t * kstride + (size_t)kh * dk + i];
            ks[buf][gi * pad + ri] = k[(size_t)t * kstride + (size_t)kh * dk + i];
        }
        __syncthreads();

        float lq = 0.0f, lk = 0.0f;
        for (int i = lane; i < dk; i += 32) {
            const int gi = i / rpg, ri = i - gi * rpg;
            const float a = qs[buf][gi * pad + ri], b = ks[buf][gi * pad + ri];
            lq += a * a;
            lk += b * b;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            lq += __shfl_xor_sync(0xFFFFFFFF, lq, o);
            lk += __shfl_xor_sync(0xFFFFFFFF, lk, o);
        }
        const float nq = rsqrtf(fmaxf(lq, 1e-12f)) * q_scale;
        const float nkk = rsqrtf(fmaxf(lk, 1e-12f));
        const float a = alpha[(size_t)t * nv + h];
        const float b = beta[(size_t)t * nv + h];
        const float* qg = &qs[buf][g * pad];
        const float* kg = &ks[buf][g * pad];

        // Decay, then read out with the normalized k (the reference reads the DECAYED state).
        float kv = 0.0f;
        #pragma unroll
        for (int r = 0; r < GDR_MAX_ROWS; ++r) {
            if (r < rpg) {
                S[r] *= a;
                kv += S[r] * kg[r];
            }
        }
        kv *= nkk;
        kv += __shfl_xor_sync(0xFFFFFFFF, kv, 1);
        kv += __shfl_xor_sync(0xFFFFFFFF, kv, 2);

        // S += k (x) beta*(v - kv_mem), then out = q^T S.
        const float delta = (v[(size_t)t * vstride + (size_t)h * dv + c] - kv) * b * nkk;
        float o = 0.0f;
        #pragma unroll
        for (int r = 0; r < GDR_MAX_ROWS; ++r) {
            if (r < rpg) {
                S[r] += kg[r] * delta;
                o += S[r] * qg[r];
            }
        }
        o *= nq;
        o += __shfl_xor_sync(0xFFFFFFFF, o, 1);
        o += __shfl_xor_sync(0xFFFFFFFF, o, 2);
        if (g == 0) out[(size_t)t * vstride + (size_t)h * dv + c] = o;
    }

    #pragma unroll
    for (int r = 0; r < GDR_MAX_ROWS; ++r)
        if (r < rpg) state[sbase + (size_t)(g * rpg + r) * dv + c] = S[r];
}

// 1 when the column-parallel kernel ran; 0 for shapes it does not cover (dv not a multiple of 32,
// dk not a multiple of 4 or above 128) or when SHAINET_GDR_COLS=0, leaving the per-head kernel.
static int gated_delta_rule_cols(const float* q, const float* k, const float* v,
                                 const float* alpha, const float* beta,
                                 float* state, float* out,
                                 int seq, int nv, int nk, int dk, int dv, int heads_per_k,
                                 float q_scale, int k_head_tiled) {
    static const bool enabled = [] {
        const char* e = getenv("SHAINET_GDR_COLS");
        return !(e && e[0] == '0');
    }();
    if (!enabled || dv % GDR_COLS != 0 || dk % GDR_GROUPS != 0 || dk / GDR_GROUPS > GDR_MAX_ROWS) return 0;
    dim3 grid(nv, dv / GDR_COLS);
    gated_delta_rule_cols_kernel<<<grid, 128>>>(q, k, v, alpha, beta, state, out, seq, nv, nk, dk,
                                                dv, heads_per_k, q_scale, k_head_tiled);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gated_delta_rule_cols: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

void gated_delta_rule(const float* q, const float* k, const float* v,
                      const float* alpha, const float* beta,
                      float* state, float* out,
                      int seq, int nv, int nk, int dk, int dv, int heads_per_k, float q_scale,
                      int k_head_tiled) {
    if (seq <= 0 || nv <= 0) return;
    if (gated_delta_rule_cols(q, k, v, alpha, beta, state, out, seq, nv, nk, dk, dv, heads_per_k,
                              q_scale, k_head_tiled)) return;
    int threads = 128;
    if (dv < threads) threads = dv < 32 ? 32 : dv;
    // Round to a power of two: the reduction loop halves nthr.
    int p = 32;
    while (p * 2 <= threads) p *= 2;
    threads = p;

    size_t shmem = ((size_t)dk * dv + 2 * (size_t)dk + (size_t)dv + (size_t)threads) * sizeof(float);
    // 64 KB of state exceeds the 48 KB default cap, so opt in explicitly. Without this the launch
    // fails with invalid argument rather than falling back.
    cudaFuncSetAttribute(gated_delta_rule_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    gated_delta_rule_kernel<<<nv, threads, shmem>>>(q, k, v, alpha, beta, state, out,
                                                    seq, nv, nk, dk, dv, heads_per_k, q_scale,
                                                    k_head_tiled);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gated_delta_rule: %s (shmem=%zu threads=%d)\n",
               cudaGetErrorString(err), shmem, threads);
    }
}

// out[i] *= sigmoid(gate[i]). Qwen3.5's attention output gate.
//
// SIGMOID, not SiLU: HF applies `attn_output * torch.sigmoid(gate)`, and SiLU carries an extra
// factor of the unbounded pre-activation. Exists so a gated attention layer can keep the
// device prefill path -- applying the gate on the host instead meant declining that path, which
// measured 259.9 s per layer at a 1216-token prefill against 0.384 s on the device.
__global__ void mul_sigmoid_kernel(float* out, const float* gate, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = out[i] * (1.0f / (1.0f + __expf(-gate[i])));
    }
}

void mul_sigmoid(float* out, const float* gate, int size) {
    if (size <= 0) return;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_sigmoid_kernel<<<blocks, threads>>>(out, gate, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in mul_sigmoid: %s\n", cudaGetErrorString(err));
    }
}

// ---- GGUF k-quant GEMV: Q4_K and Q6_K ----
// These operate on llama.cpp's quantization block layouts directly, so GGUF
// tensors can be loaded and used without transcoding to our own Q4 format.
//
// Q4_K block (144 bytes, QK_K=256 values, 8 groups of 32):
//   fp16 d, fp16 dmin, uint8 scales[12], uint8 qs[128]
//   Asymmetric: val = d * group_scale * nibble - dmin * group_min
//
// Q6_K block (210 bytes, QK_K=256 values, 16 groups of 16):
//   uint8 ql[128], uint8 qh[64], int8 scales[16], fp16 d
//   Symmetric: val = d * scale * (6-bit value - 32)

#define GGUF_QK_K 256
#define GGUF_Q4K_BLOCK_BYTES 144
#define GGUF_Q6K_BLOCK_BYTES 210

// Unpack 6-bit scale and min from the 12-byte packed scales array.
__device__ __forceinline__ void get_scale_min_k4(int j, const unsigned char* scales,
                                                  unsigned char* sc, unsigned char* m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4);
    }
}

// Q4_K GEMV: y[m, n] = x[m, K] * dequant(W_q4k[n, K])
//
// One WARP per output row, four warps per block, and each qs byte read EXACTLY ONCE.
//
// The previous shape gave lane l the eight values [l*8, l*8+8), which meant lanes 0-3 read bytes
// 0-31 for their low nibbles and lanes 4-7 read the SAME bytes for the high nibbles. Every byte was
// fetched twice, so DRAM traffic was double the useful bytes and the kernel measured 170 GB/s of a
// 576 GB/s card while actually moving ~340.
//
// Now lane l owns bytes [l*4, l*4+4) -- 32 lanes x 4 = the whole 128-byte qs -- and produces both
// nibbles of each: byte b of sub-block j64 = b/32 at offset off = b%32 carries value j64*64 + off in
// its low nibble and j64*64 + off + 32 in its high. Those two values sit in scale groups j64*2 and
// j64*2 + 1, both fixed for the lane, so it unpacks two 6-bit scales per superblock and no more.
// ── IQ4_XS ────────────────────────────────────────────────────────────────────
//
// 136-byte block over 256 values: d (fp16), scales_h (u16), scales_l[4], qs[128].
// Eight sub-blocks of 32. Each sub-block's 6-bit scale is split across scales_l (low 4 bits) and
// scales_h (high 2 bits), biased by -32. Values are 4-bit indices into a 16-entry non-linear table.
//
// The value layout is the same shape as Q4_K's: within a sub-block, the low nibbles of qs[0..15]
// give values 0..15 and the HIGH nibbles of those same bytes give 16..31. So the byte-once warp
// assignment developed for Q4_K carries over directly -- a lane owns bytes and emits both nibbles,
// rather than owning values and reading each byte twice.
__device__ __constant__ int8_t kvalues_iq4nl_dev[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// The same sixteen values held in two registers instead of constant memory.
//
// Constant memory is built to broadcast ONE address to a whole warp; these lookups are indexed by
// the weight nibble, so all 32 lanes want different entries and the access serializes. Sixteen
// int8 fit in two 64-bit literals, making the lookup a shift and a sign-extend with no memory
// traffic at all. Measured: IQ4_XS GEMV 62.5 GB/s before, against 446 for the k-quant kernels.
__device__ __forceinline__ float iq4nl_val(int idx) {
    const unsigned long long lo = 0xF6EADDCFBFAD9881ULL;  // entries 0..7
    const unsigned long long hi = 0x7159453526190D01ULL;  // entries 8..15
    const unsigned long long p = (idx < 8) ? lo : hi;
    return (float)(signed char)((p >> (8 * (idx & 7))) & 0xFFULL);
}

#define GGUF_IQ4XS_BLOCK_BYTES 136

// ── IQ3_XXS and IQ3_S ─────────────────────────────────────────────────────────
//
// Both decode through a codebook: a grid index selects four packed int8 values, and a sign selector
// flips them. One 256-value block therefore splits into 32 groups of 8 values (two grid entries and
// one sign byte each), which maps exactly onto a warp -- lane l owns group l, so each lane reads its
// own two grid indices and its own sign byte, with no duplicated reads and no cross-lane sharing
// until the final shuffle reduction.
//
// Unlike Q4_K there is no nibble-splitting subtlety here; the bytes are already one-per-purpose.
#define GGUF_IQ3XXS_BLOCK_BYTES 98
#define GGUF_IQ3S_BLOCK_BYTES 110

// Output stores for the row-dequant kernels, which are templated on the scratch type: fp32 for the
// TF32 GEMM path, half for the fp16 tensor-core path (half the bytes written and read back).
extern "C++" {
__device__ __forceinline__ void dq_store1(float* p, float v) { *p = v; }
__device__ __forceinline__ void dq_store1(__half* p, float v) { *p = __float2half(v); }
// Four values; p is 4-element aligned.
__device__ __forceinline__ void dq_store4(float* p, float a, float b, float c, float d) {
    *((float4*)p) = make_float4(a, b, c, d);
}
__device__ __forceinline__ void dq_store4(__half* p, float a, float b, float c, float d) {
    __half2 lo = __floats2half2_rn(a, b), hi = __floats2half2_rn(c, d);
    uint2 u;
    u.x = *((unsigned int*)&lo);
    u.y = *((unsigned int*)&hi);
    *((uint2*)p) = u;
}
// Eight values; p is 8-element aligned.
template <typename OUT>
__device__ __forceinline__ void dq_store8(OUT* p, const float* v) {
    dq_store4(p, v[0], v[1], v[2], v[3]);
    dq_store4(p + 4, v[4], v[5], v[6], v[7]);
}
} // extern "C++"

#include "iq_lowbit_kernels.cuh"

// Shared shape of the IQ3 GEMVs below (and of the low-bit GEMVs in iq_lowbit_kernels.cuh).
//
// One warp per output row as before, but the grid is PERSISTENT: about six 16-warp blocks per SM,
// and each warp walks rows with a grid stride. The codebook is staged into shared memory once per
// resident block rather than once per four rows -- at 4 rows per block that staging was ~23% extra
// L2 traffic on a K=5120 projection, plus a __syncthreads per four rows.
//
// The inner loop also loads wider (the lane's two grid indices as one u16, its eight activations as
// two float4) and is unrolled so several blocks' loads are in flight per lane. Each iteration is a
// short chain of dependent loads -- weight bytes, then the codebook entry they index -- so the
// rolled loop left the kernel latency-bound. Measured at the 27B's real shapes: IQ3_S 147 -> 350-400
// GB/s, IQ3_XXS 120-129 -> 310-350 GB/s, i.e. 2.2-2.8x, results equal to fp32 reassociation.
#define IQ_GEMV_WARPS 16
#define IQ_GEMV_BLOCKS_PER_SM 6

static int iq_gemv_grid(int rows) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 1;
    }
    int blocks = sms * IQ_GEMV_BLOCKS_PER_SM;
    int need = (rows + IQ_GEMV_WARPS - 1) / IQ_GEMV_WARPS;
    return blocks < need ? blocks : need;
}

// Grid for the persistent i-quant row-dequant kernels: 256 threads (8 warps) per block, one warp per
// 256-value block of weight, a few blocks per SM so each stages its codebook once.
static int iq_dequant_grid(long items) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 1;
    }
    long need = (items + 7) / 8;
    long blocks = (long)sms * 8;
    return (int)(blocks < need ? blocks : (need > 0 ? need : 1));
}

// Sum of eight codebook values with their signs applied, against eight activations. `g1`/`g2` each
// pack four unsigned magnitudes; bit j of `sg` negates value j.
__device__ __forceinline__ float iq3_dot8(unsigned int g1, unsigned int g2, unsigned int sg,
                                          float4 xa, float4 xb) {
    float s = 0.0f;
    s += (float)((g1      ) & 0xff) * ((sg & 0x01) ? -xa.x : xa.x);
    s += (float)((g1 >>  8) & 0xff) * ((sg & 0x02) ? -xa.y : xa.y);
    s += (float)((g1 >> 16) & 0xff) * ((sg & 0x04) ? -xa.z : xa.z);
    s += (float)((g1 >> 24)       ) * ((sg & 0x08) ? -xa.w : xa.w);
    s += (float)((g2      ) & 0xff) * ((sg & 0x10) ? -xb.x : xb.x);
    s += (float)((g2 >>  8) & 0xff) * ((sg & 0x20) ? -xb.y : xb.y);
    s += (float)((g2 >> 16) & 0xff) * ((sg & 0x40) ? -xb.z : xb.z);
    s += (float)((g2 >> 24)       ) * ((sg & 0x80) ? -xb.w : xb.w);
    return s;
}

__global__ void gemv_iq3xxs_kernel(const float* __restrict__ x,
                                   const unsigned char* __restrict__ w,
                                   float* __restrict__ y,
                                   int M, int N, int K) {
    // Codebook staged in SHARED memory, not read from constant memory.
    //
    // Constant memory serves a warp best when all 32 lanes read the same address; a codebook lookup
    // indexed by packed weight bits is the opposite, so every lane wants a different entry and the
    // access serializes. Shared memory is banked and satisfies a divergent read per bank per cycle.
    __shared__ unsigned int s_grid[256];
    __shared__ unsigned char s_ksigns[128];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_grid[i] = d_iq3xxs_grid[i];
    for (int i = threadIdx.x; i < 128; i += blockDim.x) s_ksigns[i] = d_ksigns_iq2xs[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int m = blockIdx.y;
    if (m >= M) return;

    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_IQ3XXS_BLOCK_BYTES;
    const float* xrow = x + (long)m * K;

    // Lane l handles group l of 32: sub-block ib32 = l/4, position within it lpos = l%4.
    const int ib32 = lane >> 2;
    const int lpos = lane & 3;

    for (int n = blockIdx.x * wpb + (threadIdx.x >> 5); n < N; n += gridDim.x * wpb) {
        const unsigned char* wrow = w + (long)n * row_bytes;
        float acc = 0.0f;
        #pragma unroll 4
        for (int b = 0; b < nblocks; ++b) {
            const unsigned char* block = wrow + (long)b * GGUF_IQ3XXS_BLOCK_BYTES;
            const float d = __half2float(*((const __half*)(block + 0)));
            const unsigned char* qs = block + 2;
            // The 98-byte block stride only guarantees 2-byte alignment, so the scales-and-signs
            // word is assembled from two u16 loads.
            const unsigned short* sp = (const unsigned short*)(qs + 64 + 4 * ib32);
            const unsigned int aux32 = (unsigned int)sp[0] | ((unsigned int)sp[1] << 16);
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            const unsigned int sg = s_ksigns[(aux32 >> (7 * lpos)) & 127];
            const unsigned short q2 = *((const unsigned short*)(qs + 8 * ib32 + 2 * lpos));
            const float4* xp = (const float4*)(xrow + b * GGUF_QK_K + ib32 * 32 + lpos * 8);
            acc += db * iq3_dot8(s_grid[q2 & 0xff], s_grid[q2 >> 8], sg, xp[0], xp[1]);
        }
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, s);
        if (lane == 0) y[(long)m * N + n] = acc;
    }
}

__global__ void gemv_iq3s_kernel(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    // 2 KB codebook staged in shared memory -- see gemv_iq3xxs_kernel for why constant memory is
    // the wrong home for a lookup whose index differs per lane.
    __shared__ unsigned int s_grid[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) s_grid[i] = d_iq3s_grid[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int m = blockIdx.y;
    if (m >= M) return;

    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_IQ3S_BLOCK_BYTES;
    const float* xrow = x + (long)m * K;

    const int ib32 = lane >> 2;   // which 32-value sub-block
    const int lpos = lane & 3;    // group of 8 within it

    for (int n = blockIdx.x * wpb + (threadIdx.x >> 5); n < N; n += gridDim.x * wpb) {
        const unsigned char* wrow = w + (long)n * row_bytes;
        float acc = 0.0f;
        #pragma unroll 4
        for (int b = 0; b < nblocks; ++b) {
            const unsigned char* block = wrow + (long)b * GGUF_IQ3S_BLOCK_BYTES;
            const float d = __half2float(*((const __half*)(block + 0)));
            const unsigned char* qs = block + 2;
            const unsigned char* qh = block + 2 + 64;
            const unsigned char* signs = block + 2 + 64 + 8;
            const unsigned char* scales = block + 2 + 64 + 8 + 32;

            // The reference walks ib32 in steps of two, taking the low nibble of scales[ib32/2] for
            // the even sub-block and the high nibble for the odd one, and reads qh[ib32] for its
            // ninth index bit. Expressed per-lane that is just: pick the nibble by parity.
            const int ls = (scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xf;
            const float db = d * (1 + 2 * ls);
            const unsigned int sg = signs[4 * ib32 + lpos];
            const unsigned int qhb = qh[ib32];
            const unsigned short q2 = *((const unsigned short*)(qs + 8 * ib32 + 2 * lpos));
            const int i1 = (q2 & 0xff) | ((qhb << (8 - 2 * lpos)) & 256);
            const int i2 = (q2 >> 8) | ((qhb << (7 - 2 * lpos)) & 256);
            const float4* xp = (const float4*)(xrow + b * GGUF_QK_K + ib32 * 32 + lpos * 8);
            acc += db * iq3_dot8(s_grid[i1], s_grid[i2], sg, xp[0], xp[1]);
        }
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, s);
        if (lane == 0) y[(long)m * N + n] = acc;
    }
}

// Persistent grid with a grid-stride over rows and an unrolled block loop, like the IQ3 kernels
// (see iq_gemv_grid): more loads in flight per lane for a loop whose body is short.
__global__ void gemv_iq4xs_kernel(const float* __restrict__ x,
                                  const unsigned char* __restrict__ w,
                                  float* __restrict__ y,
                                  int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int m = blockIdx.y;
    if (m >= M) return;

    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_IQ4XS_BLOCK_BYTES;
    const float* xrow = x + (long)m * K;

    // 32 lanes x 4 bytes covers the whole 128-byte qs, each byte read exactly once.
    const int b0 = lane * 4;       // first qs byte this lane owns
    const int ib = b0 >> 4;        // sub-block (16 bytes of qs per 32-value sub-block)
    const int off = b0 & 15;       // byte offset inside the sub-block
    const int vlo = ib * 32 + off; // first low-nibble value
    const int vhi = vlo + 16;      // matching high-nibble value

    for (int n = blockIdx.x * wpb + (threadIdx.x >> 5); n < N; n += gridDim.x * wpb) {
        const unsigned char* wrow = w + (long)n * row_bytes;
        float acc = 0.0f;
        #pragma unroll 4
        for (int b = 0; b < nblocks; ++b) {
            const unsigned char* block = wrow + (long)b * GGUF_IQ4XS_BLOCK_BYTES;
            const float d = __half2float(*((const __half*)(block + 0)));
            const unsigned short scales_h = *((const unsigned short*)(block + 2));
            const unsigned char* scales_l = block + 4;

            const int ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);

            const uchar4 q = *((const uchar4*)(block + 8 + b0));
            const float4 xlo = *((const float4*)(xrow + b * GGUF_QK_K + vlo));
            const float4 xhi = *((const float4*)(xrow + b * GGUF_QK_K + vhi));

            float s = iq4nl_val(q.x & 0xF) * xlo.x;
            s += iq4nl_val(q.y & 0xF) * xlo.y;
            s += iq4nl_val(q.z & 0xF) * xlo.z;
            s += iq4nl_val(q.w & 0xF) * xlo.w;
            s += iq4nl_val(q.x >> 4) * xhi.x;
            s += iq4nl_val(q.y >> 4) * xhi.y;
            s += iq4nl_val(q.z >> 4) * xhi.z;
            s += iq4nl_val(q.w >> 4) * xhi.w;
            acc += dl * s;
        }
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, s);
        if (lane == 0) y[(long)m * N + n] = acc;
    }
}

// Dequantize IQ4_XS rows to fp32 for the batched GEMM path.
extern "C++" {
template <typename OUT>
__global__ void dequant_iq4xs_rows_kernel(const unsigned char* __restrict__ w,
                                          OUT* __restrict__ out,
                                          int row0, int rows, int K) {
    // Persistent, one warp per 256-value block (see iq_dequant_grid). A lane owns four qs bytes,
    // i.e. four low-nibble values and the four high-nibble values 16 further on, as in the GEMV.
    const int lane = threadIdx.x & 31;
    const int b0 = lane * 4;
    const int ib = b0 >> 4;
    const int vlo = ib * 32 + (b0 & 15);
    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_IQ4XS_BLOCK_BYTES;
    const long items = (long)rows * nblocks;
    const long nwarps = (long)gridDim.x * (blockDim.x >> 5);
    for (long it = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); it < items; it += nwarps) {
        const int r = (int)(it / nblocks);
        const int b = (int)(it - (long)r * nblocks);
        const unsigned char* block = w + (long)(row0 + r) * row_bytes + (long)b * GGUF_IQ4XS_BLOCK_BYTES;
        const float d = __half2float(*((const __half*)(block + 0)));
        const unsigned short scales_h = *((const unsigned short*)(block + 2));
        const unsigned char* scales_l = block + 4;
        const int ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const uchar4 q = *((const uchar4*)(block + 8 + b0));
        OUT* orow = out + (long)r * K + b * GGUF_QK_K;
        dq_store4(orow + vlo, dl * iq4nl_val(q.x & 0xF), dl * iq4nl_val(q.y & 0xF),
                                               dl * iq4nl_val(q.z & 0xF), dl * iq4nl_val(q.w & 0xF));
        dq_store4(orow + vlo + 16, dl * iq4nl_val(q.x >> 4), dl * iq4nl_val(q.y >> 4),
                                                    dl * iq4nl_val(q.z >> 4), dl * iq4nl_val(q.w >> 4));
    }
}
} // extern "C++"

// Row-chunk dequant for the batched path, mirroring the two IQ3 GEMV kernels.
//
// Prefill multiplies with cuBLAS rather than a GEMV, which needs the weight as fp32, so every type
// that has a GEMV also needs one of these or a long prompt dies where a one-token prompt works. A
// 256-value block is 32 groups of 8, so one thread per group covers a block exactly -- the same
// decomposition the GEMV uses, just writing values out instead of accumulating a dot product.
extern "C++" {
template <typename OUT>
__global__ void dequant_iq3xxs_rows_kernel(const unsigned char* __restrict__ w,
                                          OUT* __restrict__ out,
                                          int row0, int rows, int K) {
    // Persistent, one warp per 256-value block, codebook staged once per resident block. The
    // previous shape launched 256 threads per (row, block) of which only lanes 0-31 worked.
    __shared__ unsigned int s_grid[256];
    __shared__ unsigned char s_ksigns[128];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_grid[i] = d_iq3xxs_grid[i];
    for (int i = threadIdx.x; i < 128; i += blockDim.x) s_ksigns[i] = d_ksigns_iq2xs[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int ib32 = lane >> 2;
    const int lpos = lane & 3;
    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_IQ3XXS_BLOCK_BYTES;
    const long items = (long)rows * nblocks;
    const long nwarps = (long)gridDim.x * (blockDim.x >> 5);
    for (long it = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); it < items; it += nwarps) {
        const int r = (int)(it / nblocks);
        const int b = (int)(it - (long)r * nblocks);
        const unsigned char* block = w + (long)(row0 + r) * row_bytes + (long)b * GGUF_IQ3XXS_BLOCK_BYTES;
        const float d = __half2float(*((const __half*)(block + 0)));
        const unsigned char* qs = block + 2;
        const unsigned short* sp = (const unsigned short*)(qs + 64 + 4 * ib32);
        const unsigned int aux32 = (unsigned int)sp[0] | ((unsigned int)sp[1] << 16);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        const unsigned int sg = s_ksigns[(aux32 >> (7 * lpos)) & 127];
        const unsigned short q2 = *((const unsigned short*)(qs + 8 * ib32 + 2 * lpos));
        const unsigned int g1 = s_grid[q2 & 0xff], g2 = s_grid[q2 >> 8];
        OUT* op = (out + (long)r * K + b * GGUF_QK_K + ib32 * 32 + lpos * 8);
        dq_store4(op, db * (float)((g1      ) & 0xff) * ((sg & 0x01) ? -1.0f : 1.0f),
                            db * (float)((g1 >>  8) & 0xff) * ((sg & 0x02) ? -1.0f : 1.0f),
                            db * (float)((g1 >> 16) & 0xff) * ((sg & 0x04) ? -1.0f : 1.0f),
                            db * (float)((g1 >> 24)       ) * ((sg & 0x08) ? -1.0f : 1.0f));
        dq_store4(op + 4, db * (float)((g2      ) & 0xff) * ((sg & 0x10) ? -1.0f : 1.0f),
                            db * (float)((g2 >>  8) & 0xff) * ((sg & 0x20) ? -1.0f : 1.0f),
                            db * (float)((g2 >> 16) & 0xff) * ((sg & 0x40) ? -1.0f : 1.0f),
                            db * (float)((g2 >> 24)       ) * ((sg & 0x80) ? -1.0f : 1.0f));
    }
}
} // extern "C++"

extern "C++" {
template <typename OUT>
__global__ void dequant_iq3s_rows_kernel(const unsigned char* __restrict__ w,
                                          OUT* __restrict__ out,
                                        int row0, int rows, int K) {
    // Same persistent shape as dequant_iq3xxs_rows_kernel.
    __shared__ unsigned int s_grid[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) s_grid[i] = d_iq3s_grid[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int ib32 = lane >> 2;
    const int lpos = lane & 3;
    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_IQ3S_BLOCK_BYTES;
    const long items = (long)rows * nblocks;
    const long nwarps = (long)gridDim.x * (blockDim.x >> 5);
    for (long it = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); it < items; it += nwarps) {
        const int r = (int)(it / nblocks);
        const int b = (int)(it - (long)r * nblocks);
        const unsigned char* block = w + (long)(row0 + r) * row_bytes + (long)b * GGUF_IQ3S_BLOCK_BYTES;
        const float d = __half2float(*((const __half*)(block + 0)));
        const unsigned char* qs = block + 2;
        const unsigned char* qh = block + 2 + 64;
        const unsigned char* signs = block + 2 + 64 + 8;
        const unsigned char* scales = block + 2 + 64 + 8 + 32;
        const int ls = (scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xf;
        const float db = d * (1 + 2 * ls);
        const unsigned int sg = signs[4 * ib32 + lpos];
        const unsigned int qhb = qh[ib32];
        const unsigned short q2 = *((const unsigned short*)(qs + 8 * ib32 + 2 * lpos));
        const unsigned int g1 = s_grid[(q2 & 0xff) | ((qhb << (8 - 2 * lpos)) & 256)];
        const unsigned int g2 = s_grid[(q2 >> 8) | ((qhb << (7 - 2 * lpos)) & 256)];
        OUT* op = (out + (long)r * K + b * GGUF_QK_K + ib32 * 32 + lpos * 8);
        dq_store4(op, db * (float)((g1      ) & 0xff) * ((sg & 0x01) ? -1.0f : 1.0f),
                            db * (float)((g1 >>  8) & 0xff) * ((sg & 0x02) ? -1.0f : 1.0f),
                            db * (float)((g1 >> 16) & 0xff) * ((sg & 0x04) ? -1.0f : 1.0f),
                            db * (float)((g1 >> 24)       ) * ((sg & 0x08) ? -1.0f : 1.0f));
        dq_store4(op + 4, db * (float)((g2      ) & 0xff) * ((sg & 0x10) ? -1.0f : 1.0f),
                            db * (float)((g2 >>  8) & 0xff) * ((sg & 0x20) ? -1.0f : 1.0f),
                            db * (float)((g2 >> 16) & 0xff) * ((sg & 0x40) ? -1.0f : 1.0f),
                            db * (float)((g2 >> 24)       ) * ((sg & 0x80) ? -1.0f : 1.0f));
    }
}
} // extern "C++"

__global__ void gemv_q4k_kernel(const float* __restrict__ x,
                                const unsigned char* __restrict__ w,
                                float* __restrict__ y,
                                int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_Q4K_BLOCK_BYTES;
    const unsigned char* wrow = w + (long)n * row_bytes;
    const float* xrow = x + (long)m * K;

    const int b0 = lane * 4;          // first qs byte this lane owns
    const int j64 = b0 >> 5;          // 64-value sub-block
    const int off = b0 & 31;          // offset inside it
    const int glo = j64 * 2;          // scale group of the low-nibble values
    const int ghi = glo + 1;          // and of the high-nibble values
    const int vlo = j64 * 64 + off;   // first low-nibble value index
    const int vhi = vlo + 32;         // first high-nibble value index

    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const unsigned char* block = wrow + (long)b * GGUF_Q4K_BLOCK_BYTES;
        const float d    = __half2float(*((const __half*)(block + 0)));
        const float dmin = __half2float(*((const __half*)(block + 2)));

        unsigned char sc_l, mn_l, sc_h, mn_h;
        get_scale_min_k4(glo, block + 4, &sc_l, &mn_l);
        get_scale_min_k4(ghi, block + 4, &sc_h, &mn_h);
        const float dl = d * (float)sc_l, ml = dmin * (float)mn_l;
        const float dh = d * (float)sc_h, mh = dmin * (float)mn_h;

        // One 32-bit load for the lane's four bytes.
        const uchar4 q = *((const uchar4*)(block + 16 + b0));
        const float* xl = xrow + b * GGUF_QK_K + vlo;
        const float* xh = xrow + b * GGUF_QK_K + vhi;
        const float4 xlo = *((const float4*)xl);
        const float4 xhi = *((const float4*)xh);

        acc += (dl * (float)(q.x & 0xF) - ml) * xlo.x;
        acc += (dl * (float)(q.y & 0xF) - ml) * xlo.y;
        acc += (dl * (float)(q.z & 0xF) - ml) * xlo.z;
        acc += (dl * (float)(q.w & 0xF) - ml) * xlo.w;
        acc += (dh * (float)(q.x >> 4) - mh) * xhi.x;
        acc += (dh * (float)(q.y >> 4) - mh) * xhi.y;
        acc += (dh * (float)(q.z >> 4) - mh) * xhi.z;
        acc += (dh * (float)(q.w >> 4) - mh) * xhi.w;
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, s);
    if (lane == 0) y[(long)m * N + n] = acc;
}

// Q5_K block (176 bytes, 256 values): fp16 d, fp16 dmin, scales[12] (Q4_K packing), qh[32], qs[128].
// Identical to Q4_K plus a fifth bit per value: for qs byte b in 64-value sub-block j at offset l,
// the low nibble's high bit is qh[l] bit 2j and the high nibble's is bit 2j+1. Some i-quant GGUFs
// store only output.weight this way, so the LM head needs it.
#define GGUF_Q5K_BLOCK_BYTES 176

// Q5_K GEMV with Q4_K's lane assignment: a lane owns four consecutive qs bytes (eight values) and the
// four matching qh bytes, so every byte is read exactly once per row.
__global__ void gemv_q5k_kernel(const float* __restrict__ x,
                                const unsigned char* __restrict__ w,
                                float* __restrict__ y,
                                int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_Q5K_BLOCK_BYTES;
    const unsigned char* wrow = w + (long)n * row_bytes;
    const float* xrow = x + (long)m * K;

    const int b0 = lane * 4;
    const int j64 = b0 >> 5;
    const int off = b0 & 31;
    const int glo = j64 * 2;
    const int ghi = glo + 1;
    const int vlo = j64 * 64 + off;
    const int vhi = vlo + 32;
    const unsigned int ulo = 1u << (2 * j64);
    const unsigned int uhi = 2u << (2 * j64);

    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const unsigned char* block = wrow + (long)b * GGUF_Q5K_BLOCK_BYTES;
        const float d    = __half2float(*((const __half*)(block + 0)));
        const float dmin = __half2float(*((const __half*)(block + 2)));

        unsigned char sc_l, mn_l, sc_h, mn_h;
        get_scale_min_k4(glo, block + 4, &sc_l, &mn_l);
        get_scale_min_k4(ghi, block + 4, &sc_h, &mn_h);
        const float dl = d * (float)sc_l, ml = dmin * (float)mn_l;
        const float dh = d * (float)sc_h, mh = dmin * (float)mn_h;

        // 48 + b0 and 16 + off are multiples of 4, so both 32-bit loads are aligned.
        const uchar4 q = *((const uchar4*)(block + 48 + b0));
        const uchar4 h = *((const uchar4*)(block + 16 + off));
        const float4 xlo = *((const float4*)(xrow + b * GGUF_QK_K + vlo));
        const float4 xhi = *((const float4*)(xrow + b * GGUF_QK_K + vhi));

        acc += (dl * (float)((q.x & 0xF) + ((h.x & ulo) ? 16 : 0)) - ml) * xlo.x;
        acc += (dl * (float)((q.y & 0xF) + ((h.y & ulo) ? 16 : 0)) - ml) * xlo.y;
        acc += (dl * (float)((q.z & 0xF) + ((h.z & ulo) ? 16 : 0)) - ml) * xlo.z;
        acc += (dl * (float)((q.w & 0xF) + ((h.w & ulo) ? 16 : 0)) - ml) * xlo.w;
        acc += (dh * (float)((q.x >> 4) + ((h.x & uhi) ? 16 : 0)) - mh) * xhi.x;
        acc += (dh * (float)((q.y >> 4) + ((h.y & uhi) ? 16 : 0)) - mh) * xhi.y;
        acc += (dh * (float)((q.z >> 4) + ((h.z & uhi) ? 16 : 0)) - mh) * xhi.z;
        acc += (dh * (float)((q.w >> 4) + ((h.w & uhi) ? 16 : 0)) - mh) * xhi.w;
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, s);
    if (lane == 0) y[(long)m * N + n] = acc;
}

// Q6_K GEMV, warp per output row, each byte read EXACTLY ONCE. Same reasoning as Q4_K above.
//
// Q6_K packs three bytes into four values: for a chunk of 128 and an offset l in [0,32),
// ql[l], ql[l+32] and qh[l] together give values l, l+32, l+64, l+96 of that chunk. So a lane takes
// one l and does both chunks -- six bytes, eight values, nothing fetched twice. All eight of its
// 6-bit scale indices are fixed by l, so the scale lookups leave the inner loop as well.
__global__ void gemv_q6k_kernel(const float* __restrict__ x,
                                const unsigned char* __restrict__ w,
                                float* __restrict__ y,
                                int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nblocks = K / GGUF_QK_K;
    const long row_bytes = (long)nblocks * GGUF_Q6K_BLOCK_BYTES;
    const unsigned char* wrow = w + (long)n * row_bytes;
    const float* xrow = x + (long)m * K;

    const int l = lane;          // offset within the 32-wide group
    const int is = l >> 4;       // 16-value scale group, fixed for this lane

    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        const unsigned char* block = wrow + (long)b * GGUF_Q6K_BLOCK_BYTES;
        const float d = __half2float(*((const __half*)(block + 208)));
        const float* xb = xrow + b * GGUF_QK_K;

        #pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
            const unsigned char* ql_c = block + chunk * 64;
            const unsigned char* qh_c = block + 128 + chunk * 32;
            const signed char*   sc_c = (const signed char*)(block + 192) + chunk * 8;

            const unsigned char a = ql_c[l];       // low nibble -> value l, high -> value l+64
            const unsigned char c = ql_c[l + 32];  // low nibble -> value l+32, high -> l+96
            const unsigned char h = qh_c[l];       // the four 2-bit high parts
            const float* xc = xb + chunk * 128;

            const int q0 = (int)((a & 0xF) | (((h >> 0) & 3) << 4)) - 32;
            const int q1 = (int)((c & 0xF) | (((h >> 2) & 3) << 4)) - 32;
            const int q2 = (int)((a >> 4)  | (((h >> 4) & 3) << 4)) - 32;
            const int q3 = (int)((c >> 4)  | (((h >> 6) & 3) << 4)) - 32;

            acc += d * (float)sc_c[is + 0] * (float)q0 * xc[l];
            acc += d * (float)sc_c[is + 2] * (float)q1 * xc[l + 32];
            acc += d * (float)sc_c[is + 4] * (float)q2 * xc[l + 64];
            acc += d * (float)sc_c[is + 6] * (float)q3 * xc[l + 96];
        }
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, s);
    if (lane == 0) y[(long)m * N + n] = acc;
}

// ---- Gated DeltaNet: gates and short causal convolution on the device ----
//
// These two are the last elementwise stages the mixer ran on the host. With them the whole mixer
// stays device-resident, which removes a device->host sync after every projection: at 288 matmuls
// per decode token those syncs measured 64 ms, more than the matmul kernels themselves.

// alpha[t,h] = exp(-exp(a_log[h]) * softplus(a_proj[t,h] + dt_bias[h]))
// beta [t,h] = sigmoid(b_proj[t,h])
//
// Both ranges are structural rather than clamped: alpha lands in (0,1) for any real a_log and any
// dt > 0, which is what keeps the recurrent state bounded over a long context.
__global__ void gdn_gates_kernel(float* __restrict__ alpha, float* __restrict__ beta,
                                 const float* __restrict__ a_proj,
                                 const float* __restrict__ b_proj,
                                 const float* __restrict__ a_log,
                                 const float* __restrict__ dt_bias,
                                 int seq, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq * heads) return;
    int h = idx % heads;

    float x = a_proj[idx] + dt_bias[h];
    // log1p(exp(x)), guarded: above ~20 the function is x to fp32 precision and exp overflows.
    float sp = (x > 20.0f) ? x : log1pf(__expf(x));
    alpha[idx] = __expf(-__expf(a_log[h]) * sp);
    beta[idx] = 1.0f / (1.0f + __expf(-b_proj[idx]));
}

// Short causal depthwise convolution over the sequence, per channel.
//
// dst[t,c] = sum_j w[c,j] * x[t-j, c], reading the carried state for t-j < 0. w[c,0] is the current
// position, matching ShortConv's tap order (the loader reverses GGUF's order once at load time).
// The state is updated to the last kernel-1 positions so a decode step continues the sequence.
// Causal conv + SiLU for q, k and v in ONE launch.
//
// The decode path ran nine kernels for this stage: three short_conv (each of which is itself two
// launches, conv then state roll) plus three mul_sigmoid to apply SiLU in place. That is nine
// launches per GDN layer, and with 48 GDN layers it is ~430 of the ~1280 launches a token issues.
// Batch=1 decode is latency-bound, not arithmetic-bound, so those launches are the cost.
//
// Two things happen here. SiLU is folded into the convolution's write, which is exact -- the same
// x*sigmoid(x) applied to the same value, just before it leaves a register instead of after a round
// trip through HBM. And all three tensors share one grid: a thread's channel index selects its
// tensor from the cumulative channel offsets, so q (k_dim), k (k_dim) and v (v_dim) are convolved
// together. Nine launches become two.
__global__ void short_conv_silu3_kernel(float* __restrict__ d0, const float* __restrict__ s0,
                                        const float* __restrict__ st0, const float* __restrict__ w0,
                                        float* __restrict__ d1, const float* __restrict__ s1,
                                        const float* __restrict__ st1, const float* __restrict__ w1,
                                        float* __restrict__ d2, const float* __restrict__ s2,
                                        const float* __restrict__ st2, const float* __restrict__ w2,
                                        int seq, int ch0, int ch1, int ch2, int kernel,
                                        int apply_silu) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ch0 + ch1 + ch2) return;

    float* dst;
    const float* src;
    const float* state;
    const float* w;
    int channels, c;
    if (g < ch0) {
        dst = d0; src = s0; state = st0; w = w0; channels = ch0; c = g;
    } else if (g < ch0 + ch1) {
        dst = d1; src = s1; state = st1; w = w1; channels = ch1; c = g - ch0;
    } else {
        dst = d2; src = s2; state = st2; w = w2; channels = ch2; c = g - ch0 - ch1;
    }

    for (int t = 0; t < seq; ++t) {
        float acc = 0.0f;
        for (int j = 0; j < kernel; ++j) {
            int back = t - j;
            float v;
            if (back >= 0) {
                v = src[(long)back * channels + c];
            } else {
                int si = kernel - 1 + back;          // back is negative; -1 -> last state column
                v = (si >= 0) ? state[(long)c * (kernel - 1) + si] : 0.0f;
            }
            acc += w[(long)c * kernel + j] * v;
        }
        // SiLU folded in. Written in exactly the form mul_sigmoid uses --
        // x * (1/(1+__expf(-x))), same intrinsic, same order of operations -- so fusing it does not
        // move the result. A reciprocal-then-multiply and a plain divide round differently, and the
        // fast intrinsic __expf differs from expf, so this is not a place to simplify the algebra.
        if (apply_silu) acc = acc * (1.0f / (1.0f + __expf(-acc)));
        dst[(long)t * channels + c] = acc;
    }
}

// The matching state roll for all three tensors, one launch. Split from the convolution for the same
// reason the single-tensor version is: the reads above must not race these writes.
__global__ void short_conv_state3_kernel(float* __restrict__ st0, const float* __restrict__ s0,
                                         float* __restrict__ st1, const float* __restrict__ s1,
                                         float* __restrict__ st2, const float* __restrict__ s2,
                                         int seq, int ch0, int ch1, int ch2, int kernel) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ch0 + ch1 + ch2) return;

    float* state;
    const float* src;
    int channels, c;
    if (g < ch0) {
        state = st0; src = s0; channels = ch0; c = g;
    } else if (g < ch0 + ch1) {
        state = st1; src = s1; channels = ch1; c = g - ch0;
    } else {
        state = st2; src = s2; channels = ch2; c = g - ch0 - ch1;
    }

    int keep = kernel - 1;
    float tmp[8]; // kernel is 4 for every shipped Gated DeltaNet; 8 is slack
    if (keep > 8) return;
    for (int i = 0; i < keep; ++i) {
        int s = seq - keep + i;
        // A call shorter than the window keeps part of the OLD state, otherwise history is lost --
        // which is every decode step, where seq is 1 and the window is 4. Staged through tmp so
        // these reads are not clobbered by the writes below.
        tmp[i] = (s >= 0) ? src[(long)s * channels + c] : state[(long)c * keep + (keep + s)];
    }
    for (int i = 0; i < keep; ++i) state[(long)c * keep + i] = tmp[i];
}

__global__ void short_conv_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                  const float* __restrict__ state,
                                  const float* __restrict__ w,
                                  int seq, int channels, int kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    for (int t = 0; t < seq; ++t) {
        float acc = 0.0f;
        for (int j = 0; j < kernel; ++j) {
            int back = t - j;
            float v;
            if (back >= 0) {
                v = src[(long)back * channels + c];
            } else {
                int si = kernel - 1 + back;          // back is negative; -1 -> last state column
                v = (si >= 0) ? state[(long)c * (kernel - 1) + si] : 0.0f;
            }
            acc += w[(long)c * kernel + j] * v;
        }
        dst[(long)t * channels + c] = acc;
    }
}

// Roll the conv state forward to the last kernel-1 positions of this call. Split from the
// convolution so the reads above are never racing these writes.
__global__ void short_conv_state_kernel(float* __restrict__ state, const float* __restrict__ src,
                                        int seq, int channels, int kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    int keep = kernel - 1;
    float tmp[8]; // kernel is 4 for every shipped Gated DeltaNet; 8 is slack
    if (keep > 8) return;
    for (int i = 0; i < keep; ++i) {
        int s = seq - keep + i;
        // A call shorter than the window keeps part of the old state, otherwise history is lost.
        tmp[i] = (s >= 0) ? src[(long)s * channels + c] : state[(long)c * keep + (keep + s)];
    }
    for (int i = 0; i < keep; ++i) state[(long)c * keep + i] = tmp[i];
}

void gdn_gates(float* alpha, float* beta, const float* a_proj, const float* b_proj,
               const float* a_log, const float* dt_bias, int seq, int heads) {
    int n = seq * heads;
    if (n <= 0) return;
    int threads = 256;
    gdn_gates_kernel<<<(n + threads - 1) / threads, threads>>>(
        alpha, beta, a_proj, b_proj, a_log, dt_bias, seq, heads);
}

// Activations for the fp16 tensor-core prefill GEMM.
//
// fp16 keeps TF32's 10-bit mantissa but only a 5-bit exponent, so a row is scaled by a power of two
// that brings its largest magnitude into [2^13, 2^14) before rounding: no overflow however large the
// activations get, no subnormals however small, and -- being a power of two -- the scale itself adds
// no rounding. `inv_scale[r]` undoes it on the GEMM output (scale_rows below).
__global__ void rows_to_f16_scaled_kernel(const float* __restrict__ x, __half* __restrict__ xh,
                                          float* __restrict__ inv_scale, int K) {
    __shared__ float warp_max[32];
    const int r = blockIdx.x;
    const float* xr = x + (long)r * K;
    float m = 0.0f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) m = fmaxf(m, fabsf(xr[j]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? warp_max[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xFFFFFFFF, t, o));
        if (threadIdx.x == 0) warp_max[0] = t;
    }
    __syncthreads();
    const float mx = warp_max[0];
    // exponent e with mx in [2^(e-1), 2^e); scale = 2^(14 - e). A zero or non-finite row keeps 1.
    float scale = 1.0f;
    if (mx > 0.0f && isfinite(mx)) {
        int e;
        frexpf(mx, &e);
        scale = ldexpf(1.0f, 14 - e);
    }
    if (threadIdx.x == 0) inv_scale[r] = 1.0f / scale;
    __half* hr = xh + (long)r * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x) hr[j] = __float2half(xr[j] * scale);
}

void rows_to_f16_scaled(const float* x, unsigned short* xh, float* inv_scale, int M, int K) {
    if (M <= 0 || K <= 0) return;
    rows_to_f16_scaled_kernel<<<M, 256>>>(x, (__half*)xh, inv_scale, K);
}

// One power-of-two scale for the whole activation matrix, applied by cuBLAS as alpha.
//
// The per-row variant above needs scale_rows over the GEMM output afterwards: ~115 us per call at a
// 1024-token chunk, 0.23 s of a 3.5K-token prefill, spent rewriting the whole output. One scale for
// the matrix needs no pass at all -- cuBLAS applies 1/scale as alpha, read from device memory so
// nothing waits on the host. fp16 spans ~2.7e8 between the scaled maximum (2^14) and its subnormal
// edge, far more than activations in one chunk spread, so a shared scale costs no precision.
//
// `scales` receives {1/scale, 0}: alpha and beta for the device-pointer-mode GEMM.
__global__ void absmax_bits_kernel(const float* __restrict__ x, long n, unsigned int* __restrict__ out) {
    float m = 0.0f;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(x[i]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    // Non-negative floats order like their bit patterns, so an integer atomicMax is a float max.
    if ((threadIdx.x & 31) == 0 && isfinite(m)) atomicMax(out, __float_as_uint(m));
}

__global__ void to_f16_global_scaled_kernel(const float* __restrict__ x, __half* __restrict__ xh, long n,
                                            const unsigned int* __restrict__ maxbits,
                                            float* __restrict__ scales) {
    const float mx = __uint_as_float(*maxbits);
    float scale = 1.0f;
    if (mx > 0.0f) {
        int e;
        frexpf(mx, &e);
        scale = ldexpf(1.0f, 14 - e);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scales[0] = 1.0f / scale;
        scales[1] = 0.0f;
    }
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        xh[i] = __float2half(x[i] * scale);
}

// `work` is one uint of scratch for the reduction.
void matrix_to_f16_scaled(const float* x, unsigned short* xh, float* scales, unsigned int* work, int M, int K) {
    const long n = (long)M * K;
    if (n <= 0) return;
    cudaMemsetAsync(work, 0, sizeof(unsigned int));
    long blocks = (n + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    absmax_bits_kernel<<<(unsigned int)blocks, 256>>>(x, n, work);
    to_f16_global_scaled_kernel<<<(unsigned int)blocks, 256>>>(x, (__half*)xh, n, work, scales);
}

// y[r, :] *= s[r] for a row-major [M, N].
__global__ void scale_rows_kernel(float* __restrict__ y, const float* __restrict__ s, int N) {
    const int r = blockIdx.y;
    const float f = s[r];
    float* yr = y + (long)r * N;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += gridDim.x * blockDim.x) yr[j] *= f;
}

void scale_rows(float* y, const float* s, int M, int N) {
    if (M <= 0 || N <= 0) return;
    int bx = (N + 255) / 256;
    if (bx > 64) bx = 64;
    scale_rows_kernel<<<dim3(bx, M), 256>>>(y, s, N);
}

// Decode-time alpha/beta projections of a Gated DeltaNet layer, both in one pass.
//
//   ap[h] = sum_k wa[k, h] * x[k],  bp[h] = sum_k wb[k, h] * x[k]
//
// with the weights row-major [d_model, heads] as loaded. These were two cuBLAS sgemm calls, which at
// one token and 48 heads land on a gemv kernel that measured ~31 us each -- ~32 GB/s on a 1 MB
// weight, 96 calls and ~3 ms per token across the stack. Here K is split into chunks, one block per
// (chunk, alpha|beta), with heads across the threads so each k reads one contiguous row of the
// weight. The chunk partials are then summed in a fixed order by a second tiny kernel: no atomics,
// so the result is identical run to run.
#define GDN_AB_KCHUNK 128

__global__ void gdn_ab_proj_partial_kernel(const float* __restrict__ x,
                                           const float* __restrict__ wa,
                                           const float* __restrict__ wb,
                                           float* __restrict__ partial,
                                           int d_model, int heads) {
    const int c = blockIdx.x;
    const int z = blockIdx.y;
    const float* W = z ? wb : wa;
    const int k0 = c * GDN_AB_KCHUNK;
    const int k1 = min(k0 + GDN_AB_KCHUNK, d_model);
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = k0; k < k1; ++k) acc += W[(long)k * heads + h] * x[k];
        partial[((long)z * gridDim.x + c) * heads + h] = acc;
    }
}

__global__ void gdn_ab_proj_reduce_kernel(const float* __restrict__ partial,
                                          float* __restrict__ ap, float* __restrict__ bp,
                                          int nch, int heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * heads) return;
    const int z = i / heads;
    const int h = i - z * heads;
    float acc = 0.0f;
    for (int c = 0; c < nch; ++c) acc += partial[((long)z * nch + c) * heads + h];
    (z ? bp : ap)[h] = acc;
}

// Returns 0 when it cannot run (its scratch could not be allocated), so the caller keeps cuBLAS.
int gdn_ab_proj(const float* x, const float* wa, const float* wb, float* ap, float* bp,
                int d_model, int heads) {
    static float* partial = nullptr;
    static size_t partial_floats = 0;
    if (d_model <= 0 || heads <= 0) return 0;
    const int nch = (d_model + GDN_AB_KCHUNK - 1) / GDN_AB_KCHUNK;
    const size_t need = (size_t)2 * nch * heads;
    if (need > partial_floats) {
        if (partial) cudaFree(partial);
        partial = nullptr;
        partial_floats = 0;
        if (cudaMalloc(&partial, need * sizeof(float)) != cudaSuccess) return 0;
        partial_floats = need;
    }
    const int threads = heads <= 32 ? 32 : (heads <= 64 ? 64 : 128);
    gdn_ab_proj_partial_kernel<<<dim3(nch, 2), threads>>>(x, wa, wb, partial, d_model, heads);
    gdn_ab_proj_reduce_kernel<<<(2 * heads + 127) / 128, 128>>>(partial, ap, bp, nch, heads);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gdn_ab_proj: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

// Fused causal conv (+ optional SiLU) for q, k and v in two launches instead of nine.
void short_conv_silu3(float* d0, const float* s0, float* st0, const float* w0,
                      float* d1, const float* s1, float* st1, const float* w1,
                      float* d2, const float* s2, float* st2, const float* w2,
                      int seq, int ch0, int ch1, int ch2, int kernel, int apply_silu) {
    if (seq <= 0 || kernel <= 0) return;
    int total = ch0 + ch1 + ch2;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    short_conv_silu3_kernel<<<blocks, threads>>>(d0, s0, st0, w0,
                                                d1, s1, st1, w1,
                                                d2, s2, st2, w2,
                                                seq, ch0, ch1, ch2, kernel, apply_silu);
    if (kernel > 1) {
        short_conv_state3_kernel<<<blocks, threads>>>(st0, s0, st1, s1, st2, s2,
                                                      seq, ch0, ch1, ch2, kernel);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in short_conv_silu3: %s\n", cudaGetErrorString(err));
    }
}

void short_conv(float* dst, const float* src, float* state, const float* w,
                int seq, int channels, int kernel) {
    if (seq <= 0 || channels <= 0 || kernel <= 0) return;
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;
    short_conv_kernel<<<blocks, threads>>>(dst, src, state, w, seq, channels, kernel);
    if (kernel > 1) {
        short_conv_state_kernel<<<blocks, threads>>>(state, src, seq, channels, kernel);
    }
}

// C entry points

// ---- Dequantize a row range to fp32, so cuBLAS can do the GEMM ----
//
// The GEMV kernels above give each (output, token) pair its own block, so every block
// re-dequantizes the whole weight row and NOTHING is reused across tokens. Measured on
// blk.0.ffn_up.weight (Q4_K, 5120x17408, 47.8 MB): 0.54 ms per token flat from M=1 to M=256, i.e.
// the 47.8 MB weight is re-streamed once per token -- 54 GB of traffic for a 1200-token prefill,
// per matmul, per layer.
//
// For M=1 that is optimal (the weight is read exactly once and the result is one vector), so
// decode keeps using the GEMV. For M>1 the weight should be read once and reused across all M,
// which is what a GEMM does. Rather than hand-roll a tiled k-quant GEMM, dequantize a chunk of
// rows into an fp32 scratch buffer and hand it to cuBLAS, which is already tuned for this shape.
//
// Output is [n_rows, K] row-major, matching the GGUF row order, so the caller can treat it as a
// column-major [K, n_rows] operand and reach the whole result with one cublasSgemm.

extern "C++" {
template <typename OUT>
__global__ void dequant_q4k_rows_kernel(const unsigned char* __restrict__ w,
                                          OUT* __restrict__ out,
                                        int row0, int n_rows, int K) {
    int r = blockIdx.x;
    if (r >= n_rows) return;

    int nblocks = K / GGUF_QK_K;
    long row_bytes = (long)nblocks * GGUF_Q4K_BLOCK_BYTES;
    const unsigned char* wrow = w + (long)(row0 + r) * row_bytes;
    OUT* orow = out + (long)r * K;

    for (int blk = blockIdx.y; blk < nblocks; blk += gridDim.y) {
        const unsigned char* block = wrow + (long)blk * GGUF_Q4K_BLOCK_BYTES;
        float d    = __half2float(*((const __half*)(block + 0)));
        float dmin = __half2float(*((const __half*)(block + 2)));
        const unsigned char* scales = block + 4;
        const unsigned char* qs     = block + 16;
        int base_k = blk * GGUF_QK_K;

        for (int k = threadIdx.x; k < GGUF_QK_K; k += blockDim.x) {
            unsigned char sc, mn;
            get_scale_min_k4(k / 32, scales, &sc, &mn);
            int block_of_64 = k / 64;
            int offset_in_64 = k % 64;
            int nibble_val = (offset_in_64 < 32)
                ? (qs[block_of_64 * 32 + offset_in_64] & 0xF)
                : (qs[block_of_64 * 32 + (offset_in_64 - 32)] >> 4);
            dq_store1(orow + base_k + k, d * (float)sc * (float)nibble_val - dmin * (float)mn);
        }
    }
}
} // extern "C++"

extern "C++" {
template <typename OUT>
__global__ void dequant_q5k_rows_kernel(const unsigned char* __restrict__ w,
                                          OUT* __restrict__ out,
                                        int row0, int n_rows, int K) {
    int r = blockIdx.x;
    if (r >= n_rows) return;

    int nblocks = K / GGUF_QK_K;
    long row_bytes = (long)nblocks * GGUF_Q5K_BLOCK_BYTES;
    const unsigned char* wrow = w + (long)(row0 + r) * row_bytes;
    OUT* orow = out + (long)r * K;

    for (int blk = blockIdx.y; blk < nblocks; blk += gridDim.y) {
        const unsigned char* block = wrow + (long)blk * GGUF_Q5K_BLOCK_BYTES;
        float d    = __half2float(*((const __half*)(block + 0)));
        float dmin = __half2float(*((const __half*)(block + 2)));
        const unsigned char* scales = block + 4;
        const unsigned char* qh     = block + 16;
        const unsigned char* qs     = block + 48;
        int base_k = blk * GGUF_QK_K;

        for (int k = threadIdx.x; k < GGUF_QK_K; k += blockDim.x) {
            unsigned char sc, mn;
            get_scale_min_k4(k / 32, scales, &sc, &mn);
            int j = k / 64;
            int o = k % 64;
            int l = o & 31;
            int hi = o >> 5; // 0 = low nibble, 1 = high nibble
            int nib = hi ? (qs[j * 32 + l] >> 4) : (qs[j * 32 + l] & 0xF);
            int bit = (qh[l] >> (2 * j + hi)) & 1;
            dq_store1(orow + base_k + k, d * (float)sc * (float)(nib + 16 * bit) - dmin * (float)mn);
        }
    }
}
} // extern "C++"

extern "C++" {
template <typename OUT>
__global__ void dequant_q6k_rows_kernel(const unsigned char* __restrict__ w,
                                          OUT* __restrict__ out,
                                        int row0, int n_rows, int K) {
    int r = blockIdx.x;
    if (r >= n_rows) return;

    int nblocks = K / GGUF_QK_K;
    long row_bytes = (long)nblocks * GGUF_Q6K_BLOCK_BYTES;
    const unsigned char* wrow = w + (long)(row0 + r) * row_bytes;
    OUT* orow = out + (long)r * K;

    for (int blk = blockIdx.y; blk < nblocks; blk += gridDim.y) {
        const unsigned char* block = wrow + (long)blk * GGUF_Q6K_BLOCK_BYTES;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* sc   = (const signed char*)(block + 192);
        float d = __half2float(*((const __half*)(block + 208)));
        int base_k = blk * GGUF_QK_K;

        for (int k = threadIdx.x; k < GGUF_QK_K; k += blockDim.x) {
            int chunk = k / 128;
            int local = k % 128;
            const unsigned char* ql_c = ql + chunk * 64;
            const unsigned char* qh_c = qh + chunk * 32;
            const signed char* sc_c   = sc + chunk * 8;
            int sub = local / 32;
            int l   = local % 32;
            int qval;
            switch (sub) {
                case 0: qval = (int)((ql_c[l]       & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32; break;
                case 1: qval = (int)((ql_c[l + 32]  & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32; break;
                case 2: qval = (int)((ql_c[l]       >> 4)  | (((qh_c[l] >> 4) & 3) << 4)) - 32; break;
                default: qval = (int)((ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4)) - 32; break;
            }
            dq_store1(orow + base_k + k, d * (float)sc_c[(l / 16) + sub * 2] * (float)qval);
        }
    }
}
} // extern "C++"

void dequant_iq4xs_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq4xs_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_iq4xs_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq4xs_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void dequant_iq3xxs_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq3xxs_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_iq3xxs_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq3xxs_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void dequant_iq3s_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq3s_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_iq3s_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq3s_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_q2k_lb(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    if (N <= 0 || M <= 0 || K <= 0) return;
    dim3 grid(iq_gemv_grid(N), M);
    gemv_q2k_lb_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
}

void dequant_q2k_lb_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_q2k_lb_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_q2k_lb_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_q2k_lb_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_iq2xxs(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    if (N <= 0 || M <= 0 || K <= 0) return;
    dim3 grid(iq_gemv_grid(N), M);
    gemv_iq2xxs_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
}

void dequant_iq2xxs_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq2xxs_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_iq2xxs_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq2xxs_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_iq2xs(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    if (N <= 0 || M <= 0 || K <= 0) return;
    dim3 grid(iq_gemv_grid(N), M);
    gemv_iq2xs_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
}

void dequant_iq2xs_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq2xs_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_iq2xs_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq2xs_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_iq2s(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    if (N <= 0 || M <= 0 || K <= 0) return;
    dim3 grid(iq_gemv_grid(N), M);
    gemv_iq2s_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
}

void dequant_iq2s_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq2s_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_iq2s_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq2s_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_iq1m(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    if (N <= 0 || M <= 0 || K <= 0) return;
    dim3 grid(iq_gemv_grid(N), M);
    gemv_iq1m_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
}

void dequant_iq1m_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq1m_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_iq1m_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq1m_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_iq1s(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    if (N <= 0 || M <= 0 || K <= 0) return;
    dim3 grid(iq_gemv_grid(N), M);
    gemv_iq1s_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
}

void dequant_iq1s_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq1s_rows_kernel<float><<<iq_dequant_grid(items), 256>>>(w, out, row0, n_rows, K);
}

void dequant_iq1s_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    long items = (long)n_rows * (K / GGUF_QK_K);
    dequant_iq1s_rows_kernel<__half><<<iq_dequant_grid(items), 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_iq3xxs(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    dim3 grid(iq_gemv_grid(N), M);
    gemv_iq3xxs_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error in gemv_iq3xxs: %s\n", cudaGetErrorString(err));
}

void gemv_iq3s(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    dim3 grid(iq_gemv_grid(N), M);
    gemv_iq3s_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error in gemv_iq3s: %s\n", cudaGetErrorString(err));
}

void gemv_iq4xs(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    dim3 grid(iq_gemv_grid(N), M);
    gemv_iq4xs_kernel<<<grid, IQ_GEMV_WARPS * 32>>>(x, w, y, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemv_iq4xs: %s\n", cudaGetErrorString(err));
    }
}

void dequant_q4k_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    int nblocks = K / GGUF_QK_K;
    int by = nblocks < 32 ? nblocks : 32;
    dim3 grid(n_rows, by);
    dequant_q4k_rows_kernel<float><<<grid, 256>>>(w, out, row0, n_rows, K);
}

void dequant_q4k_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    int nblocks = K / GGUF_QK_K;
    int by = nblocks < 32 ? nblocks : 32;
    dim3 grid(n_rows, by);
    dequant_q4k_rows_kernel<__half><<<grid, 256>>>(w, (__half*)out, row0, n_rows, K);
}

void dequant_q5k_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    int nblocks = K / GGUF_QK_K;
    int by = nblocks < 32 ? nblocks : 32;
    dim3 grid(n_rows, by);
    dequant_q5k_rows_kernel<float><<<grid, 256>>>(w, out, row0, n_rows, K);
}

void dequant_q5k_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    int nblocks = K / GGUF_QK_K;
    int by = nblocks < 32 ? nblocks : 32;
    dim3 grid(n_rows, by);
    dequant_q5k_rows_kernel<__half><<<grid, 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_q5k(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    const int warps = 4;
    const int threads = warps * 32;
    dim3 grid((N + warps - 1) / warps, M);
    gemv_q5k_kernel<<<grid, threads>>>(x, w, y, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemv_q5k: %s\n", cudaGetErrorString(err));
    }
}

void dequant_q6k_rows(const unsigned char* w, float* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    int nblocks = K / GGUF_QK_K;
    int by = nblocks < 32 ? nblocks : 32;
    dim3 grid(n_rows, by);
    dequant_q6k_rows_kernel<float><<<grid, 256>>>(w, out, row0, n_rows, K);
}

void dequant_q6k_rows_f16(const unsigned char* w, unsigned short* out, int row0, int n_rows, int K) {
    if (n_rows <= 0 || K <= 0) return;
    int nblocks = K / GGUF_QK_K;
    int by = nblocks < 32 ? nblocks : 32;
    dim3 grid(n_rows, by);
    dequant_q6k_rows_kernel<__half><<<grid, 256>>>(w, (__half*)out, row0, n_rows, K);
}

void gemv_q4k(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    // One warp per output row, 4 warps per block. No shared memory: the reduction is a shuffle.
    const int warps = 4;
    const int threads = warps * 32;
    dim3 grid((N + warps - 1) / warps, M);
    gemv_q4k_kernel<<<grid, threads>>>(x, w, y, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemv_q4k: %s\n", cudaGetErrorString(err));
    }
}

void gemv_q6k(const float* x, const unsigned char* w, float* y, int M, int N, int K) {
    // One warp per output row, 4 warps per block. No shared memory: the reduction is a shuffle.
    const int warps = 4;
    const int threads = warps * 32;
    dim3 grid((N + warps - 1) / warps, M);
    gemv_q6k_kernel<<<grid, threads>>>(x, w, y, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in gemv_q6k: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"

extern "C++" {
// ---------------------------------------------------------------------------------------------
// int8 MMQ: prefill matmul straight from the quantized weights on int8 tensor cores.
//
// y[m, n] = sum_k x[m, k] * W[n, k], with x quantized per 32-value block to int8 (scale xs, and the
// block's float sum xsum for types whose weights carry an offset) and W decoded tile by tile from its
// GGUF blocks into int8 values with one float scale (and offset) per 32 -- never materialized in
// global memory. Each 32-wide k step is one mma.sync m16n8k32 s8 producing an exact int32 dot,
// scaled into fp32 accumulators:
//
//   acc += dot_int(xq, wq) * xs * ws  (+ wo * xsum)
//
// This is llama.cpp's MMQ scheme. It replaces a dequant pass per 256-token chunk plus an fp16 GEMM
// with one kernel at int8 tensor rate.
//
// ---------------------------------------------------------------------------------------------
#define MMQ_TYPE_Q4K 0
#define MMQ_TYPE_IQ4XS 1
#define MMQ_TYPE_IQ3S 2
#define MMQ_TYPE_IQ3XXS 3

// One warp per (row, 32-value block): scale = max|x| / 127, q = round(x / scale), plus the float sum.
__global__ void mmq_quantize_x_kernel(const float* __restrict__ x, signed char* __restrict__ xq,
                                      float* __restrict__ xs, float* __restrict__ xsum,
                                      int M, int K) {
    const long wid = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = K >> 5;
    if (wid >= (long)M * nb) return;
    const int r = (int)(wid / nb), b = (int)(wid - (long)r * nb);
    const float v = x[(long)r * K + b * 32 + lane];
    float a = fabsf(v), s = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
        s += __shfl_xor_sync(0xFFFFFFFF, s, o);
    }
    const float scale = a / 127.0f;
    const int q = scale > 0.0f ? __float2int_rn(v / scale) : 0;
    xq[(long)r * K + b * 32 + lane] = (signed char)q;
    if (lane == 0) {
        xs[(long)r * nb + b] = scale;
        xsum[(long)r * nb + b] = s;
    }
}

// mmq_quantize_x for the down projection's input, computing it on the way: v = silu(gate) * up in
// exactly swiglu_forward_kernel's arithmetic. The [M, ff] hidden activation is never written or read
// back -- at a 1024-token chunk of the 27B that is 71 MB each way, per layer.
__global__ void mmq_quantize_swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                           signed char* __restrict__ xq, float* __restrict__ xs,
                                           float* __restrict__ xsum, int M, int K) {
    const long wid = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = K >> 5;
    if (wid >= (long)M * nb) return;
    const int r = (int)(wid / nb), b = (int)(wid - (long)r * nb);
    const long i = (long)r * K + b * 32 + lane;
    const float g = gate[i];
    const float v = (g / (1.0f + expf(-g))) * up[i];
    float a = fabsf(v), s = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
        s += __shfl_xor_sync(0xFFFFFFFF, s, o);
    }
    const float scale = a / 127.0f;
    const int q = scale > 0.0f ? __float2int_rn(v / scale) : 0;
    xq[i] = (signed char)q;
    if (lane == 0) {
        xs[(long)r * nb + b] = scale;
        xsum[(long)r * nb + b] = s;
    }
}

__device__ __forceinline__ unsigned int mmq_pack4(int a, int b, int c, int d) {
    return (unsigned int)(a & 0xFF) | ((unsigned int)(b & 0xFF) << 8) |
           ((unsigned int)(c & 0xFF) << 16) | ((unsigned int)(d & 0xFF) << 24);
}

// Decode one 256-value GGUF block of one weight row into `wq` (256 int8), `ws` (8 scales) and `wo`
// (8 offsets), one warp, lane-parallel. The lane mappings are the GEMV kernels'.
template <int TYPE>
__device__ __forceinline__ void mmq_decode_block(const unsigned char* __restrict__ block, int lane,
                                                 signed char* wq, float* ws, float* wo,
                                                 const unsigned int* s_grid,
                                                 const unsigned char* s_ksigns) {
    if (TYPE == MMQ_TYPE_Q4K) {
        const float d = __half2float(*((const __half*)(block + 0)));
        const float dmin = __half2float(*((const __half*)(block + 2)));
        const int b0 = lane * 4, j64 = b0 >> 5, off = b0 & 31;
        const uchar4 q = *((const uchar4*)(block + 16 + b0));
        *((unsigned int*)(wq + j64 * 64 + off)) = mmq_pack4(q.x & 0xF, q.y & 0xF, q.z & 0xF, q.w & 0xF);
        *((unsigned int*)(wq + j64 * 64 + off + 32)) = mmq_pack4(q.x >> 4, q.y >> 4, q.z >> 4, q.w >> 4);
        if (off == 0) {
            unsigned char sc, mn;
            get_scale_min_k4(2 * j64, block + 4, &sc, &mn);
            ws[2 * j64] = d * (float)sc;
            wo[2 * j64] = -dmin * (float)mn;
            get_scale_min_k4(2 * j64 + 1, block + 4, &sc, &mn);
            ws[2 * j64 + 1] = d * (float)sc;
            wo[2 * j64 + 1] = -dmin * (float)mn;
        }
    } else if (TYPE == MMQ_TYPE_IQ4XS) {
        const float d = __half2float(*((const __half*)(block + 0)));
        const unsigned short scales_h = *((const unsigned short*)(block + 2));
        const unsigned char* scales_l = block + 4;
        const int b0 = lane * 4, ib = b0 >> 4, off = b0 & 15;
        const uchar4 q = *((const uchar4*)(block + 8 + b0));
        *((unsigned int*)(wq + ib * 32 + off)) =
            mmq_pack4((int)iq4nl_val(q.x & 0xF), (int)iq4nl_val(q.y & 0xF),
                      (int)iq4nl_val(q.z & 0xF), (int)iq4nl_val(q.w & 0xF));
        *((unsigned int*)(wq + ib * 32 + off + 16)) =
            mmq_pack4((int)iq4nl_val(q.x >> 4), (int)iq4nl_val(q.y >> 4),
                      (int)iq4nl_val(q.z >> 4), (int)iq4nl_val(q.w >> 4));
        if (off == 0) {
            const int ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
            ws[ib] = d * (float)(ls - 32);
            wo[ib] = 0.0f;
        }
    } else {
        const int ib32 = lane >> 2, lpos = lane & 3;
        const float d = __half2float(*((const __half*)(block + 0)));
        const unsigned char* qs = block + 2;
        unsigned int g1, g2, sg;
        float db;
        if (TYPE == MMQ_TYPE_IQ3S) {
            const unsigned char* qh = block + 2 + 64;
            const unsigned char* signs = block + 2 + 64 + 8;
            const unsigned char* scales = block + 2 + 64 + 8 + 32;
            const int ls = (scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xf;
            db = d * (float)(1 + 2 * ls);
            sg = signs[4 * ib32 + lpos];
            const unsigned int qhb = qh[ib32];
            const unsigned short q2 = *((const unsigned short*)(qs + 8 * ib32 + 2 * lpos));
            g1 = s_grid[(q2 & 0xff) | ((qhb << (8 - 2 * lpos)) & 256)];
            g2 = s_grid[(q2 >> 8) | ((qhb << (7 - 2 * lpos)) & 256)];
        } else {
            const unsigned short* sp = (const unsigned short*)(qs + 64 + 4 * ib32);
            const unsigned int aux32 = (unsigned int)sp[0] | ((unsigned int)sp[1] << 16);
            db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            sg = s_ksigns[(aux32 >> (7 * lpos)) & 127];
            const unsigned short q2 = *((const unsigned short*)(qs + 8 * ib32 + 2 * lpos));
            g1 = s_grid[q2 & 0xff];
            g2 = s_grid[q2 >> 8];
        }
        // Grid bytes are magnitudes (< 128); bit j of sg negates value j.
        int v[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m1 = (g1 >> (8 * j)) & 0xff, m2 = (g2 >> (8 * j)) & 0xff;
            v[j] = (sg & (1u << j)) ? -m1 : m1;
            v[j + 4] = (sg & (1u << (j + 4))) ? -m2 : m2;
        }
        unsigned int* dst = (unsigned int*)(wq + ib32 * 32 + lpos * 8);
        dst[0] = mmq_pack4(v[0], v[1], v[2], v[3]);
        dst[1] = mmq_pack4(v[4], v[5], v[6], v[7]);
        if (lpos == 0) {
            ws[ib32] = db;
            wo[ib32] = 0.0f;
        }
    }
}

__device__ __forceinline__ void mmq_mma(int* c, const unsigned int* a, const unsigned int* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// ---------------------------------------------------------------------------------------------
// Structure: decode once, then a pipelined int8 tensor-core GEMM with per-32 scales.
//
// The single-kernel version decoded the weight tile inside the GEMM, once per 64-token block, with
// scattered small global loads and no pipelining: decode alone cost ~1 ms of a 1.7 ms 17408x5120x256
// product, slower than the dequant + fp16 cuBLAS path it was meant to replace. So the work splits:
//
//   1. mmq_decode_rows: each 256-value GGUF block of a weight slice -> 256 int8 + 8 scales (+ 8
//      offsets for Q4_K), one warp per block, streaming. One byte per weight, half the fp16 scratch.
//   2. mmq_gemm: y[m, n] = sum over 32-blocks of dot_int8 * xs[m] * ws[n] (+ wo[n] * xsum[m]),
//      128x128 tiles, 8 warps of 64x32, K in 64-byte stages through a 3-deep cp.async pipeline.
// ---------------------------------------------------------------------------------------------
#define MMQ2_BK 64
#define MMQ2_LD 80          // 64 bytes + 16 pad: fragment loads hit 32 distinct banks

template <int TYPE>
__global__ void mmq_decode_rows_kernel(const unsigned char* __restrict__ w, signed char* __restrict__ wq,
                                       float* __restrict__ ws, float* __restrict__ wo,
                                       int row0, int rows, int K, int block_bytes) {
    __shared__ unsigned int s_grid[512];
    __shared__ unsigned char s_ksigns[128];
    if (TYPE == MMQ_TYPE_IQ3S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) s_grid[i] = d_iq3s_grid[i];
    } else if (TYPE == MMQ_TYPE_IQ3XXS) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) s_grid[i] = d_iq3xxs_grid[i];
        for (int i = threadIdx.x; i < 128; i += blockDim.x) s_ksigns[i] = d_ksigns_iq2xs[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int nblocks = K / 256;
    const long row_bytes = (long)nblocks * block_bytes;
    const long items = (long)rows * nblocks;
    const long nwarps = (long)gridDim.x * (blockDim.x >> 5);
    for (long it = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); it < items; it += nwarps) {
        const int r = (int)(it / nblocks), b = (int)(it - (long)r * nblocks);
        mmq_decode_block<TYPE>(w + (long)(row0 + r) * row_bytes + (long)b * block_bytes, lane,
                               wq + (long)r * K + b * 256, ws + (long)r * (K / 32) + b * 8,
                               wo + (long)r * (K / 32) + b * 8, s_grid, s_ksigns);
    }
}

__device__ __forceinline__ void mmq2_cp16(void* smem, const void* gmem, bool ok) {
    const unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void mmq2_cp4(void* smem, const void* gmem, bool ok) {
    const unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
    const int n = ok ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void mmq2_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void mmq2_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// y[m, n0 + n] for a slice of `rows` weight rows starting at output column n0, ldy = full N.
template <bool OFFSET, int BM, int BN, int WM_, int WN_, int ST>
__global__ void __launch_bounds__(WM_ * WN_ * 32) mmq_gemm_kernel(const signed char* __restrict__ xq,
                                                       const float* __restrict__ xs,
                                                       const float* __restrict__ xsum,
                                                       const signed char* __restrict__ wq,
                                                       const float* __restrict__ ws,
                                                       const float* __restrict__ wo,
                                                       float* __restrict__ y,
                                                       int M, int rows, int K, int ldy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char mmq2_smem[];
    signed char* Xs = (signed char*)mmq2_smem;                          // [S][BM][LD]
    signed char* Ws = Xs + ST * BM * MMQ2_LD;             // [S][BN][LD]
    float* sxs = (float*)(Ws + ST * BN * MMQ2_LD);        // [S][BM][2]
    float* sws = sxs + ST * BM * 2;                        // [S][BN][2]
    float* sxm = sws + ST * BN * 2;                        // [S][BM][2]
    float* swo = sxm + ST * BM * 2;                        // [S][BN][2]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    constexpr int WTM = BM / WM_, WTN = BN / WN_, MT = WTM / 16, NT = WTN / 8, NTHR = WM_ * WN_ * 32;
    const int wm = (warp % WM_) * WTM, wn = (warp / WM_) * WTN;
    const int nb32 = K >> 5;
    const int ktiles = K / MMQ2_BK;

    auto load_stage = [&](int stage, int kt) {
        const int k0 = kt * MMQ2_BK;
        for (int c = tid; c < BM * 4; c += NTHR) {
            const int r = c >> 2, q = c & 3;
            mmq2_cp16(Xs + (stage * BM + r) * MMQ2_LD + q * 16,
                      xq + (long)(m0 + r < M ? m0 + r : 0) * K + k0 + q * 16, m0 + r < M);
        }
        for (int c = tid; c < BN * 4; c += NTHR) {
            const int r = c >> 2, q = c & 3;
            mmq2_cp16(Ws + (stage * BN + r) * MMQ2_LD + q * 16,
                      wq + (long)(n0 + r < rows ? n0 + r : 0) * K + k0 + q * 16, n0 + r < rows);
        }
        for (int c = tid; c < BM * 2; c += NTHR) {
            const int r = c >> 1, s = c & 1, kb = (k0 >> 5) + s;
            const bool okm = m0 + r < M;
            mmq2_cp4(sxs + (stage * BM + r) * 2 + s, xs + (long)(okm ? m0 + r : 0) * nb32 + kb, okm);
            if (OFFSET) mmq2_cp4(sxm + (stage * BM + r) * 2 + s, xsum + (long)(okm ? m0 + r : 0) * nb32 + kb, okm);
        }
        for (int c = tid; c < BN * 2; c += NTHR) {
            const int r = c >> 1, s = c & 1, kb = (k0 >> 5) + s;
            const bool okn = n0 + r < rows;
            mmq2_cp4(sws + (stage * BN + r) * 2 + s, ws + (long)(okn ? n0 + r : 0) * nb32 + kb, okn);
            if (OFFSET) mmq2_cp4(swo + (stage * BN + r) * 2 + s, wo + (long)(okn ? n0 + r : 0) * nb32 + kb, okn);
        }
    };

    float acc[MT][NT][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

    #pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < ktiles) load_stage(s, s);
        mmq2_commit();
    }

    for (int kt = 0; kt < ktiles; ++kt) {
        mmq2_wait<ST - 2>();
        __syncthreads();
        const int nxt = kt + ST - 1;
        if (nxt < ktiles) load_stage(nxt % ST, nxt);
        mmq2_commit();

        const int st = kt % ST;
        const signed char* X = Xs + st * BM * MMQ2_LD;
        const signed char* Wt = Ws + st * BN * MMQ2_LD;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int kofs = s * 32 + t * 4;
            unsigned int b[NT][2];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const signed char* wr = Wt + (wn + nt * 8 + g) * MMQ2_LD + kofs;
                b[nt][0] = *((const unsigned int*)(wr));
                b[nt][1] = *((const unsigned int*)(wr + 16));
            }
            float w0[NT], w1[NT], o0[NT], o1[NT];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int wr0 = wn + nt * 8 + t * 2;
                w0[nt] = sws[(st * BN + wr0) * 2 + s];
                w1[nt] = sws[(st * BN + wr0 + 1) * 2 + s];
                if (OFFSET) {
                    o0[nt] = swo[(st * BN + wr0) * 2 + s];
                    o1[nt] = swo[(st * BN + wr0 + 1) * 2 + s];
                }
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const signed char* xr = X + (wm + mt * 16 + g) * MMQ2_LD + kofs;
                unsigned int a[4];
                a[0] = *((const unsigned int*)(xr));
                a[1] = *((const unsigned int*)(xr + 8 * MMQ2_LD));
                a[2] = *((const unsigned int*)(xr + 16));
                a[3] = *((const unsigned int*)(xr + 8 * MMQ2_LD + 16));
                const int tr0 = wm + mt * 16 + g;
                const float x0 = sxs[(st * BM + tr0) * 2 + s];
                const float x1 = sxs[(st * BM + tr0 + 8) * 2 + s];
                float xm0 = 0.0f, xm1 = 0.0f;
                if (OFFSET) {
                    xm0 = sxm[(st * BM + tr0) * 2 + s];
                    xm1 = sxm[(st * BM + tr0 + 8) * 2 + s];
                }
                // All four MMAs first, scaling after: each result is otherwise consumed the moment
                // it is produced, which exposes the full MMA latency with only 8 warps per SM.
                int cc[NT][4];
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    cc[nt][0] = cc[nt][1] = cc[nt][2] = cc[nt][3] = 0;
                    mmq_mma(cc[nt], a, b[nt]);
                }
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int* c = cc[nt];
                    acc[mt][nt][0] += (float)c[0] * (x0 * w0[nt]);
                    acc[mt][nt][1] += (float)c[1] * (x0 * w1[nt]);
                    acc[mt][nt][2] += (float)c[2] * (x1 * w0[nt]);
                    acc[mt][nt][3] += (float)c[3] * (x1 * w1[nt]);
                    if (OFFSET) {
                        acc[mt][nt][0] += o0[nt] * xm0;
                        acc[mt][nt][1] += o1[nt] * xm0;
                        acc[mt][nt][2] += o0[nt] * xm1;
                        acc[mt][nt][3] += o1[nt] * xm1;
                    }
                }
            }
        }
    }
    mmq2_wait<0>();

    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int tok = m0 + wm + mt * 16 + g;
            const int row = n0 + wn + nt * 8 + t * 2;
            if (tok < M) {
                if (row < rows) y[(long)tok * ldy + row] = acc[mt][nt][0];
                if (row + 1 < rows) y[(long)tok * ldy + row + 1] = acc[mt][nt][1];
            }
            if (tok + 8 < M) {
                if (row < rows) y[(long)(tok + 8) * ldy + row] = acc[mt][nt][2];
                if (row + 1 < rows) y[(long)(tok + 8) * ldy + row + 1] = acc[mt][nt][3];
            }
        }
#endif
}

template <int BM, int BN, int ST> static size_t mmq2_smem_bytes() {
    return (size_t)ST * (BM + BN) * MMQ2_LD + (size_t)ST * (BM + BN) * 2 * 2 * sizeof(float);
}

} // extern "C++"

extern "C" {
int mmq_available() {
    static int avail = -1;
    if (avail < 0) {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        avail = major >= 8 ? 1 : 0;   // mma.sync m16n8k32 s8 is sm_80+
    }
    return avail;
}

void mmq_quantize_x(const float* x, signed char* xq, float* xs, float* xsum, int M, int K) {
    const long warps = (long)M * (K / 32);
    mmq_quantize_x_kernel<<<(unsigned int)((warps * 32 + 255) / 256), 256>>>(x, xq, xs, xsum, M, K);
}

void mmq_quantize_swiglu(const float* gate, const float* up, signed char* xq, float* xs, float* xsum,
                         int M, int K) {
    const long warps = (long)M * (K / 32);
    mmq_quantize_swiglu_kernel<<<(unsigned int)((warps * 32 + 255) / 256), 256>>>(gate, up, xq, xs, xsum, M, K);
}

// `ggml_type` is the GGUF type id. Returns 0 for a type this path does not decode.
int mmq_decode_rows(int ggml_type, const unsigned char* w, signed char* wq, float* ws, float* wo,
                    int row0, int rows, int K) {
    const long items = (long)rows * (K / 256);
    const int grid = iq_dequant_grid(items);
    switch (ggml_type) {
    case 12: mmq_decode_rows_kernel<MMQ_TYPE_Q4K><<<grid, 256>>>(w, wq, ws, wo, row0, rows, K, 144); return 1;
    case 23: mmq_decode_rows_kernel<MMQ_TYPE_IQ4XS><<<grid, 256>>>(w, wq, ws, wo, row0, rows, K, 136); return 1;
    case 21: mmq_decode_rows_kernel<MMQ_TYPE_IQ3S><<<grid, 256>>>(w, wq, ws, wo, row0, rows, K, 110); return 1;
    case 18: mmq_decode_rows_kernel<MMQ_TYPE_IQ3XXS><<<grid, 256>>>(w, wq, ws, wo, row0, rows, K, 98); return 1;
    default: return 0;
    }
}

extern "C++" {
template <int BM, int BN, int WM_, int WN_, int ST>
static int mmq_gemm_launch(int offset, const signed char* xq, const float* xs, const float* xsum,
                           const signed char* wq, const float* ws, const float* wo, float* y,
                           int M, int rows, int K, int ldy) {
    static int ready = 0;
    const size_t smem = mmq2_smem_bytes<BM, BN, ST>();
    if (ready == 0) {
        const bool ok =
            cudaFuncSetAttribute(mmq_gemm_kernel<true, BM, BN, WM_, WN_, ST>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess &&
            cudaFuncSetAttribute(mmq_gemm_kernel<false, BM, BN, WM_, WN_, ST>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess;
        cudaGetLastError();
        ready = ok ? 1 : -1;
    }
    if (ready < 0) return 0;
    dim3 grid((rows + BN - 1) / BN, (M + BM - 1) / BM);
    if (offset)
        mmq_gemm_kernel<true, BM, BN, WM_, WN_, ST><<<grid, WM_ * WN_ * 32, smem>>>(xq, xs, xsum, wq, ws, wo, y, M, rows, K, ldy);
    else
        mmq_gemm_kernel<false, BM, BN, WM_, WN_, ST><<<grid, WM_ * WN_ * 32, smem>>>(xq, xs, xsum, wq, ws, wo, y, M, rows, K, ldy);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in mmq_gemm: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}
} // extern "C++"

// y[m, n] for `rows` weight rows into columns starting at y (leading dimension ldy). `offset` adds the
// per-block weight offsets (Q4_K). Returns 0 when the shared-memory opt-in was refused.
int mmq_gemm(int offset, const signed char* xq, const float* xs, const float* xsum,
             const signed char* wq, const float* ws, const float* wo, float* y,
             int M, int rows, int K, int ldy) {
    static int tile = -1;
    if (tile < 0) {
        const char* e = getenv("SHAINET_MMQ_TILE");
        tile = e ? atoi(e) : 0;
    }
    switch (tile) {
    case 1: return mmq_gemm_launch<64, 128, 2, 4, 3>(offset, xq, xs, xsum, wq, ws, wo, y, M, rows, K, ldy);
    case 2: return mmq_gemm_launch<64, 64, 2, 2, 3>(offset, xq, xs, xsum, wq, ws, wo, y, M, rows, K, ldy);
    case 3: return mmq_gemm_launch<128, 64, 4, 2, 3>(offset, xq, xs, xsum, wq, ws, wo, y, M, rows, K, ldy);
    case 4: return mmq_gemm_launch<64, 128, 1, 4, 3>(offset, xq, xs, xsum, wq, ws, wo, y, M, rows, K, ldy);
    default: return mmq_gemm_launch<128, 128, 2, 4, 2>(offset, xq, xs, xsum, wq, ws, wo, y, M, rows, K, ldy);
    }
}
} // extern "C"
