require "json"

module SHAInet
  module OllamaResolve
    OLLAMA_MODELS = ::File.join(Path.home.to_s, ".ollama", "models")

    # Accepts both official models ("qwen3.8:27b", stored under library/) and namespaced community
    # ones ("incu6us/qwen3.8-gsq:27b-iq3_s", stored under the user's namespace).
    def self.resolve(name : String) : String?
      parts = name.split(":", 2)
      model = parts[0]
      tag = parts[1]? || "latest"
      repo = model.includes?("/") ? model : "library/#{model}"

      manifest_path = ::File.join(OLLAMA_MODELS, "manifests", "registry.ollama.ai", repo, tag)
      return unless ::File.exists?(manifest_path)

      manifest = JSON.parse(::File.read(manifest_path))
      layers = manifest["layers"]?.try(&.as_a)
      return unless layers

      model_layer = layers.find { |l| l["mediaType"]?.try(&.as_s) == "application/vnd.ollama.image.model" }
      return unless model_layer

      digest = model_layer["digest"]?.try(&.as_s)
      return unless digest

      blob_path = ::File.join(OLLAMA_MODELS, "blobs", digest.gsub(":", "-"))
      ::File.exists?(blob_path) ? blob_path : nil
    end

    # An Ollama reference is "[namespace/]model[:tag]". A path that exists on disk always wins, so a
    # relative "models/foo" directory is never mistaken for a namespaced model.
    def self.ollama_name?(s : String) : Bool
      return false if s.ends_with?(".gguf") || ::File.exists?(s)
      s.matches?(/\A[\w.\-]+(\/[\w.\-]+)?(:[\w.\-]+)?\z/)
    end
  end
end
