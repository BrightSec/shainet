require "log"

{% if flag?(:enable_cuda) %}
  require "./shainet/cuda"
  require "./shainet/cudnn"
{% else %}
  require "./shainet/cuda_stub"
{% end %}

require "./shainet/profile"
require "./shainet/autograd/tensor"
require "./shainet/autograd/grad_matrix"
require "./shainet/basic/exceptions"
require "./shainet/basic/matrix_layer"
require "./shainet/basic/network_run"
require "./shainet/basic/network_setup"
require "./shainet/concurrency/concurrent_tokenizer"
require "./shainet/data/data"
require "./shainet/data/json_data"
require "./shainet/data/streaming_data"
require "./shainet/data/test_data"
require "./shainet/data/training_data"
require "./shainet/math/batch_processor"
require "./shainet/math/cuda_matrix"
require "./shainet/math/cuda_matrix_ext"
require "./shainet/math/quantized_cuda_matrix"
require "./shainet/math/q4_cuda_matrix"
require "./shainet/math/q4_host_matrix"
require "./shainet/math/cuda_tensor_matrix"
require "./shainet/math/functions"
require "./shainet/math/gpu_memory"
require "./shainet/math/random_normal"
require "./shainet/math/simple_matrix"
require "./shainet/math/tensor_matrix"
require "./shainet/safetensors"
require "./shainet/gguf"
require "./shainet/gguf_loader"
require "./shainet/cpu_kernels"
require "./shainet/ollama_resolve"
require "./shainet/math/gguf_matrix"
require "./shainet/math/gguf_host_matrix"
require "./shainet/hf_loader"
require "./shainet/text/bpe_tokenizer"
require "./shainet/text/embedding_layer"
require "./shainet/text/sampler"
require "./shainet/text/tokenizer"
require "./shainet/transformer/dropout"
require "./shainet/transformer/ext"
require "./shainet/transformer/layer_norm"
require "./shainet/transformer/quantized_projection"
require "./shainet/transformer/llama_block"
require "./shainet/transformer/multi_head_attention"
require "./shainet/transformer/positional_encoding"
require "./shainet/transformer/positionwise_ff"
require "./shainet/transformer/rms_norm"
require "./shainet/transformer/rope"
require "./shainet/transformer/short_conv"
require "./shainet/transformer/gated_deltanet"
require "./shainet/transformer/gated_deltanet_block"
require "./shainet/transformer/mtp_head"
require "./shainet/transformer/swiglu_ff"
require "./shainet/transformer/moe_ff"
require "./shainet/transformer/transformer_block"
require "./shainet/version"

module SHAInet
  Log = ::Log.for(self)
  alias GenNum = Float64 | Int32 | Int64 | Float32

  lvl = {
    "info"  => ::Log::Severity::Info,
    "debug" => ::Log::Severity::Debug,
    "warn"  => ::Log::Severity::Warn,
    "error" => ::Log::Severity::Error,
    "fatal" => ::Log::Severity::Fatal,
    "trace" => ::Log::Severity::Trace,
  }

  log_level = (ENV["LOG_LEVEL"]? || "info")
  iobackend = ::Log::IOBackend.new(io: STDOUT, dispatcher: ::Log::DispatchMode::Sync)
  ::Log.setup(lvl[log_level.downcase], backend: iobackend)
end
