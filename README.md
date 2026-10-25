## SHAInet - A neural network in pure [Crystal](https://crystal-lang.org/)

SHAInet (Super Human Artificial Intelligence Network) is a neural network library written in pure [Crystal](https://crystal-lang.org/). Originally created for biologically inspired neural network research, it has evolved into a general-purpose library for training and running neural networks, with a focus on simplicity and ease of use.

---

## Features

- CPU and GPU (CUDA) support
- Multiple layer types and activation functions
- Various training algorithms (SGD, Adam, iRprop+, etc.)
- Streaming data support for large datasets
- HuggingFace model import via SafeTensors (no Python required)
- LLM inference: GPT-2, LLaMA, Mistral, Qwen2, Qwen3, Qwen3-MoE, and Qwen3.5
  (Qwen3.5's Gated DeltaNet hybrid stack runs on the GPU; a 27B k-quant GGUF fits
  a 16 GB card with the trunk placed automatically -- see below)
- Direct GGUF loading (Ollama blobs or `.gguf` files) with weights dequantized
  inside the GEMV/GEMM kernels, and automatic layer placement -- no environment
  variables required. Q4_K/Q6_K plus the i-quants IQ4_XS, IQ3_S and IQ3_XXS run on
  dedicated kernels; the lower-bit types (IQ2_S/XS/XXS, Q2_K, IQ1_M) load too, so
  mixed-precision files like ISTA-DASLab's GSQ-RCO builds work as shipped
- KV-cache decoding, Q8/Q4 weight quantization, and MoE expert offload
  (run large Mixture-of-Experts models on small GPUs)

---

## Installation

Add to your `shard.yml`:

```yaml
dependencies:
  shainet:
    github: BrightSec/shainet
```

### GPU Acceleration (Optional)

- Install the CUDA Toolkit and ensure `libcudart.so` and `libcublas.so` are in your `LD_LIBRARY_PATH`.
- SHAInet will auto-detect CUDA and use GPU acceleration if available.
- For cuDNN support, ensure `libcudnn.so` is also in your `LD_LIBRARY_PATH`.
- Compile the project with `-Denable_cuda`

Check CUDA availability:

```crystal
require "shainet"
puts "CUDA available: #{SHAInet::CUDA.available?}"
puts "CUDA version: #{SHAInet::CUDA.version || "unknown"}"
```

#### Optimized GPU Setup

For best performance (especially with transformers):

```bash
git clone https://github.com/BrightSec/shainet.git
cd shainet
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
make test
```

To build kernels manually:

```bash
./build_cuda_kernels.sh
```

#### RTX 30/40 Series (Ampere/Ada) Note

These GPUs use TF32 tensor cores by default for FP32 matrix multiply, which
reduces mantissa precision from 23 to 10 bits. For LLM inference this can cause
non-deterministic token generation. SHAInet sets
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` automatically, but for full
FP32 precision also set:

```bash
export NVIDIA_TF32_OVERRIDE=0
```

### Device management

Layers such as `LayerNorm` allocate workspace matrices on the first forward pass
and reuse them across iterations. Call `to_gpu!` or `to_cpu!` only when
switching devices. Repeated calls without a device change keep the existing
workspaces to avoid unnecessary allocations.

---

## Usage

See `examples/` for more.

### XOR Example

```crystal
require "shainet"

data = [
  [[0.0, 0.0], [0.0]],
  [[1.0, 0.0], [1.0]],
  [[0.0, 1.0], [1.0]],
  [[1.0, 1.0], [0.0]],
]

net = SHAInet::Network.new
net.add_layer(:input, 2)
net.add_layer(:hidden, 2)
net.add_layer(:output, 1)
net.fully_connect

net.train(data: data,
  training_type: :sgdm,
  cost_function: :mse,
  epochs: 5000,
  log_each: 1000)

puts net.run([0, 1])
```

### Load a GGUF model (Ollama blob or `.gguf` file)

Point `HFLoader.load` at an Ollama model name, a blob, or a `.gguf` file. Q4_K and
Q6_K weights are kept quantized and dequantized inside the GEMV/GEMM kernels, and
layer placement is automatic — **no environment variables are needed**:

```crystal
net = SHAInet::HFLoader.load("qwen3.5:27b")          # Ollama model name
net = SHAInet::HFLoader.load("/path/to/model.gguf")  # or a file
net.use_kv_cache = true
tokenizer = SHAInet::GGUF.extract_tokenizer("qwen3.5:27b")
```

The loader sums the real GGUF tensor sizes, measures free VRAM, and places as many
layers on the card as fit, leaving the rest resident in host RAM. On a 16 GB card a
27B Q4_K/Q6_K model places 52 of 64 layers plus `lm_head`.

A mixed-precision i-quant build of the same 27B model fits **entirely** on the same
card -- ISTA-DASLab's `Qwen3.8-27B-GSQ-RCO-IQ3_S` (11.8 GB) places all 64 layers with
nothing left on the host. That matters more than the file size suggests: eliminating
host-resident layers removes the per-token PCIe streaming that otherwise dominates
decode. Measured on one 16 GB card, same prompt and same greedy output: **85 ms/token
all-resident against 127 ms** for the Q4_K build at 52 of 64 layers.

Files like these assign a quantization type per tensor rather than one type for the
whole model, so a single file mixes eight of them. Those carrying most of the weight
(IQ4_XS, IQ3_S, IQ3_XXS) have dedicated GEMV kernels; the low-bit tail is converted
once at load time.

Weights left on the host are handled differently depending on batch size, because
the two regimes are bound by different resources. Generating one token is
memory-bound — the CPU streams those weights at full DDR5 bandwidth, so it keeps
up. Prefill at hundreds of rows is compute-bound instead, where a CPU is about two
orders of magnitude off the GPU, so past 32 rows the quantized weight is uploaded
to the card and the matmul runs there. This mirrors llama.cpp's `offload_op`
scheduler hook and uses its 32-row threshold; override with
`SHAINET_HOST_OFFLOAD_MIN_ROWS`, or disable with `SHAINET_HOST_OFFLOAD_OP=0`.

For full fp32 determinism SHAInet sets `CUBLAS_PEDANTIC_MATH` on its cuBLAS
handles, so TF32 tensor cores are not used for SGEMM and `NVIDIA_TF32_OVERRIDE=0`
is not required. Set `SHAINET_CUBLAS_TF32=1` to allow TF32 back.

### Load a HuggingFace Model (SafeTensors)

Load models directly from HuggingFace SafeTensors — no Python, no PyTorch, just
pure Crystal binary parsing. `HFLoader.load` auto-detects the architecture from
the model's `config.json`:

```crystal
require "shainet"

# Auto-detects the architecture (gpt2 / llama / mistral / qwen2 / qwen3 / qwen3_moe / qwen3_5)
net = SHAInet::HFLoader.load("/path/to/model-dir")

# Optionally quantize weights to int8 at load time (Q8):
net = SHAInet::HFLoader.load("/path/to/model-dir", quantize: true, bits: 8)
```

Supported architectures: **GPT-2, LLaMA, Mistral, Qwen2, Qwen3, Qwen3-MoE, Qwen3.5**.
Supported tensor dtypes: F16, BF16, F32, F64.

For a full chat loop (tokenizer, KV-cache decoding, sampling) see
`examples/llama_chat.cr`; for a tool-using coding agent built on `Network#run`
see `examples/agent.cr`.

### Iris Classification

```crystal
data = SHAInet::Data.new_with_csv_input_target("iris.csv", 0..3, 4)
train, test = data.split(0.67)

iris = SHAInet::Network.new
iris.add_layer(:input, 4)
iris.add_layer(:hidden, 5)
iris.add_layer(:output, 3)
iris.fully_connect

iris.train_batch(
  data: train,
  training_type: :adam,
  cost_function: :mse,
  epochs: 2000,
  log_each: 100)

puts iris.test(test)
```

### Streaming Data

Efficiently train on large datasets:

```crystal
stream = SHAInet::StreamingData.new(
  "data.txt",
  shuffle: true,
  chunk_size: 1024,
  gpu_batches: true)

net = SHAInet::Network.new
net.add_layer(:input, 2, :memory, SHAInet.sigmoid)
net.add_layer(:hidden, 3, :memory, SHAInet.sigmoid)
net.add_layer(:output, 1, :memory, SHAInet.sigmoid)
net.fully_connect

net.train(
  data: stream,
  training_type: :sgdm,
  epochs: 5000,
  mini_batch_size: 2,
  log_each: 1000)
```

---

## Advanced

- Run a real LLaMA model: `crystal run examples/llama_chat.cr -Denable_cuda`
  (auto-downloads Llama-3.2-1B-Instruct, chats with KV cache + GPU).
- Quantized inference (Q8_0): call `net.quantize!` after loading to run with
  int8 weights + per-32-block fp32 scales (dequant-in-kernel GEMV). Cuts weight
  VRAM ~4x (1B model: ~5GB fp32 → ~1.3GB) and speeds up memory-bound decode.
  `llama_chat.cr` quantizes by default on GPU; set `SHAINET_FP32=1` to keep fp32.
  Benchmark/eval both paths with `examples/q8_eval.cr`.
- 4-bit quantization (Q4) + MoE expert offload: run large Mixture-of-Experts
  models on small GPUs by keeping experts in host RAM (pinned) and streaming
  them to the GPU on demand, backed by a hot-expert LRU cache. Controlled via
  environment variables:
  - `SHAINET_Q4=1` — 4-bit weight quantization
  - `SHAINET_MOE_OFFLOAD=1` — offload MoE experts to host RAM
  - `SHAINET_DENSE_OFFLOAD=1` — offload the **dense** weights too (attention
    projections, dense FFN, lm_head). Requires `SHAINET_Q4=1`. This is what takes
    dense models off the VRAM budget, so a far larger dense model fits. Unlike
    experts, dense weights are touched on every token, so there is no sparsity to
    amortize the transfer: expect slower decode. **Pair it with
    `SHAINET_EXPERT_CACHE_MB`** — left unbounded the hot cache promotes the dense
    weights straight back onto the card and you can end up using more VRAM than
    not offloading (measured on Qwen3-0.6B: 586 MB with the default budget vs
    483 MB not offloading vs 228 MB with the cache disabled, at 51 ms/step
    resident and 80 ms/step streamed, with identical greedy output throughout).
  - `SHAINET_EXPERT_CACHE_MB=<N>` — VRAM budget for the hot-weight cache (`0` disables)

  For example, Qwen3-Coder-30B-A3B (30B params, ~3B active) runs on a 16 GB GPU.
- Tool-using coding agent: `examples/agent.cr` is a small CLI coding agent
  (file tools + shell, context management, streaming UI) built entirely on
  `Network#run`. Run it against a local instruct model, e.g.
  `SHAINET_Q4=1 SHAINET_MOE_OFFLOAD=1 crystal run examples/agent.cr -Denable_cuda -- /path/to/model-dir`.
- OpenAI-compatible API server: `examples/openai_server.cr` serves a loaded
  model over a subset of the OpenAI REST API (`POST /v1/chat/completions` with
  streaming SSE, `GET /v1/models`), so existing OpenAI clients (the `openai`
  Python package, `curl`, LangChain, …) can use it as a drop-in local endpoint.
  Built entirely on `Network#run`. Example:
  `crystal run examples/openai_server.cr -Denable_cuda -- /path/to/model-dir 8080`.
  Binds to `127.0.0.1` with no authentication by default — set `SHAINET_API_KEY`
  to require an `Authorization: Bearer <key>` header before exposing it.
- See `examples/babylm_transformer.cr` for training a transformer language model.
- Use `SHAInet::SafeTensors::File` to read any `.safetensors` file directly.

### SafeTensors API

```crystal
# Low-level tensor access
sf = SHAInet::SafeTensors::File.new("model.safetensors")
sf.tensor_names          # => ["transformer.wte.weight", ...]
info = sf.tensors["transformer.wte.weight"]
info.dtype               # => F32
info.shape               # => [50257, 768]

matrix = sf.read_matrix("transformer.wte.weight")  # => SimpleMatrix
data = sf.read_f64("transformer.h.0.ln_1.weight")  # => Array(Float64)
sf.close
```

### Autograd

```crystal
a = SHAInet::SimpleMatrix.tensor(1, 2)
a[0, 0] = SHAInet::Autograd::Tensor.new(2.0)
a[0, 1] = SHAInet::Autograd::Tensor.new(3.0)

w = SHAInet::SimpleMatrix.tensor(2, 1)
w[0, 0] = SHAInet::Autograd::Tensor.new(4.0)
w[1, 0] = SHAInet::Autograd::Tensor.new(5.0)

out = a * w
out[0, 0].as(SHAInet::Autograd::Tensor).backward
```

## Contributing

1. Fork [https://github.com/BrightSec/shainet](https://github.com/BrightSec/shainet)
2. Create a feature branch
3. Commit and push your changes
4. Open a Pull Request

---

## Contributors

- [ArtLinkov](https://github.com/ArtLinkov) - creator, maintainer
- [bararchy](https://github.com/bararchy) - creator, maintainer
- [drujensen](https://github.com/drujensen) - contributor
- [hugoabonizio](https://github.com/hugoabonizio) - contributor
- [Rémy Marronnier](https://github.com/rmarronnier) - contributor
- [psikoz](https://github.com/psikoz) - logo design

---
