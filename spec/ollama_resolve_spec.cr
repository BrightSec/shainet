require "./spec_helper"

describe SHAInet::OllamaResolve do
  it "recognizes official and namespaced model references" do
    SHAInet::OllamaResolve.ollama_name?("qwen3.8:27b").should be_true
    SHAInet::OllamaResolve.ollama_name?("incu6us/qwen3.8-gsq:27b-iq3_s").should be_true
    SHAInet::OllamaResolve.ollama_name?("incu6us/qwen3.8-gsq").should be_true
  end

  it "leaves files, GGUF paths and existing directories alone" do
    SHAInet::OllamaResolve.ollama_name?("model.gguf").should be_false
    SHAInet::OllamaResolve.ollama_name?("/home/u/models/Qwen3.8-27B").should be_false
    SHAInet::OllamaResolve.ollama_name?("a/b/c").should be_false
    # Shaped exactly like namespace/model, but it is a directory on disk.
    SHAInet::OllamaResolve.ollama_name?("spec/linear_data").should be_false
  end
end
