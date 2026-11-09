module SHAInet
  # GGUF file format parser (v3).
  #
  # GGUF is a self-contained model format: weights, architecture metadata, and
  # tokenizer all in one file. Designed for llama.cpp/Ollama inference.
  #
  # Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
  module GGUF
    MAGIC = 0x46554747_u32 # "GGUF" in little-endian

    # GGML tensor data types.
    enum GGMLType : UInt32
      F32     =  0
      F16     =  1
      Q4_0    =  2
      Q4_1    =  3
      Q5_0    =  6
      Q5_1    =  7
      Q8_0    =  8
      Q8_1    =  9
      Q2_K    = 10
      Q3_K    = 11
      Q4_K    = 12
      Q5_K    = 13
      Q6_K    = 14
      Q8_K    = 15
      IQ2_XXS = 16
      IQ2_XS  = 17
      IQ3_XXS = 18
      IQ1_S   = 19
      IQ4_NL  = 20
      IQ3_S   = 21
      IQ2_S   = 22
      IQ4_XS  = 23
      I8      = 24
      I16     = 25
      I32     = 26
      I64     = 27
      F64     = 28
      IQ1_M   = 29
      BF16    = 30
      TQ1_0   = 34
      TQ2_0   = 35
    end

    # Bytes per block and values per block for supported quant types.
    BLOCK_SIZE = {
      GGMLType::F32  => {4, 1},
      GGMLType::F16  => {2, 1},
      GGMLType::BF16 => {2, 1},
      GGMLType::Q4_0 => {18, 32},
      GGMLType::Q4_1 => {20, 32},
      GGMLType::Q5_0 => {22, 32},
      GGMLType::Q5_1 => {24, 32},
      GGMLType::Q8_0 => {34, 32},
      GGMLType::Q8_1 => {36, 32},
      GGMLType::Q2_K => {84, 256},
      GGMLType::Q3_K => {110, 256},
      GGMLType::Q4_K => {144, 256},
      GGMLType::Q5_K => {176, 256},
      GGMLType::Q6_K => {210, 256},
      GGMLType::Q8_K => {292, 256},
      # i-quants. Sizes from ggml-common.h's block structs:
      #   IQ2_XXS  d + qs[QK_K/8] u16                          = 2 + 32      = 66
      #   IQ2_XS   d + qs[QK_K/8] u16 + scales[QK_K/32]        = 2 + 32 + 8  = 74
      #   IQ2_S    d + qs[QK_K/4] + qh[QK_K/32] + scales[..]   = 2 + 64+8+8  = 82
      #   IQ3_XXS  d + qs[3*QK_K/8]                            = 2 + 96      = 98
      #   IQ3_S    d + qs[QK_K/4] + qh[QK_K/32] + signs[QK_K/8] + scales[QK_K/64]
      #                                                        = 2+64+8+32+4 = 110
      #   IQ1_S    d + qs[QK_K/8] + qh[QK_K/32] u16            = 2 + 32 + 16 = 50
      #   IQ1_M    qs[QK_K/8] + qh[QK_K/16] + scales[QK_K/32]  = 32 + 16 + 8 = 56 (no d: scales carry it)
      #   IQ4_NL   d + qs[QK4_NL/2]                            = 2 + 16      = 18
      #   IQ4_XS   d + scales_h u16 + scales_l[QK_K/64] + qs[QK_K/2]
      #                                                        = 2+2+4+128   = 136
      GGMLType::IQ2_XXS => {66, 256},
      GGMLType::IQ2_XS  => {74, 256},
      GGMLType::IQ2_S   => {82, 256},
      GGMLType::IQ3_XXS => {98, 256},
      GGMLType::IQ3_S   => {110, 256},
      GGMLType::IQ1_S   => {50, 256},
      GGMLType::IQ1_M   => {56, 256},
      GGMLType::IQ4_NL  => {18, 32},
      GGMLType::IQ4_XS  => {136, 256},
    }

    # Metadata value types.
    enum MetaType : UInt32
      UINT8   =  0
      INT8    =  1
      UINT16  =  2
      INT16   =  3
      UINT32  =  4
      INT32   =  5
      FLOAT32 =  6
      BOOL    =  7
      STRING  =  8
      ARRAY   =  9
      UINT64  = 10
      INT64   = 11
      FLOAT64 = 12
    end

    # A value from the metadata key-value store.
    alias MetaValue = UInt8 | Int8 | UInt16 | Int16 | UInt32 | Int32 | Float32 |
                      Bool | String | Array(MetaValue) | UInt64 | Int64 | Float64

    # A tensor descriptor (name, shape, type, byte offset relative to tensor_data).
    record TensorInfo, name : String, shape : Array(UInt64), type : GGMLType, offset : UInt64 do
      def element_count : UInt64
        shape.reduce(1_u64) { |a, b| a * b }
      end

      def byte_size : UInt64
        bs, vs = BLOCK_SIZE[type]? || raise "unsupported GGML type #{type} for tensor #{name}"
        blocks = (element_count + vs.to_u64 - 1) // vs.to_u64
        blocks * bs.to_u64
      end
    end

    # Parsed GGUF file. Metadata is accessible as a Hash; tensor data is read on
    # demand from the IO (the file must stay open).
    #
    # When opened with mmap: true (the default), the entire file is memory-mapped
    # and tensor data is accessed via direct pointer arithmetic into the mapping.
    # This is how llama.cpp loads GGUF: instant load, zero-copy for CPU layers,
    # and the OS handles paging. GPU layers are uploaded with cudaMemcpy directly
    # from the mmap'd region.
    class File
      getter version : UInt32
      getter metadata : Hash(String, MetaValue)
      getter tensors : Hash(String, TensorInfo)
      getter alignment : UInt32

      # The absolute byte offset where tensor data begins.
      getter tensor_data_offset : UInt64

      # mmap state
      getter mmap_ptr : Pointer(UInt8)? = nil
      @mmap_size : UInt64 = 0
      @io : ::IO::FileDescriptor

      # Parse position within the mmap'd buffer (used during initialization).
      @parse_pos : UInt64 = 0

      def initialize(@io : ::IO::FileDescriptor)
        @metadata = Hash(String, MetaValue).new
        @tensors = Hash(String, TensorInfo).new
        @version = 0_u32
        @alignment = 32_u32
        @tensor_data_offset = 0_u64
      end

      # Parse the GGUF header and metadata. Called after mmap is set up so the
      # entire parse runs on the mmap'd buffer (pointer arithmetic, no IO calls).
      # This is what makes the 500K tokenizer strings fast.
      protected def set_mmap(ptr : Pointer(UInt8), size : UInt64)
        @mmap_ptr = ptr
        @mmap_size = size
      end

      protected def parse!
        magic = parse_u32
        raise "not a GGUF file (magic #{magic.to_s(16)})" unless magic == MAGIC
        @version = parse_u32
        raise "unsupported GGUF version #{@version} (expected 2 or 3)" unless @version >= 2

        tensor_count = parse_u64
        kv_count = parse_u64

        kv_count.times { parse_kv }
        @alignment = (metadata["general.alignment"]?.try(&.as(UInt32)) || 32_u32)

        tensor_count.times { parse_tensor_info }
        @tensor_data_offset = align(@parse_pos)
      end

      def self.open(path : String, mmap : Bool = true) : File
        io = ::File.open(path, "r")
        f = new(io)
        size = ::File.size(path).to_u64
        if mmap
          fd = io.fd
          ptr = LibC.mmap(Pointer(Void).null, size, LibC::PROT_READ, LibC::MAP_PRIVATE, fd, 0_i64)
          if ptr != LibC::MAP_FAILED
            LibC.madvise(ptr, size, LibC::POSIX_MADV_SEQUENTIAL)
            f.set_mmap(ptr.as(Pointer(UInt8)), size)
          end
        end
        f.parse!
        f
      end

      def mmap? : Bool
        !@mmap_ptr.nil?
      end

      # Get a direct pointer to a tensor's data in the mmap'd region.
      # Returns nil if not mmap'd.
      def tensor_ptr(info : TensorInfo) : Pointer(UInt8)?
        if ptr = @mmap_ptr
          offset = @tensor_data_offset + info.offset
          raise "tensor offset #{offset} + size #{info.byte_size} exceeds mmap size #{@mmap_size}" if offset + info.byte_size > @mmap_size
          ptr + offset
        end
      end

      def close
        if ptr = @mmap_ptr
          LibC.munmap(ptr.as(Pointer(Void)), @mmap_size)
          @mmap_ptr = nil
        end
        @io.close
      end

      # Read raw tensor bytes into a pre-allocated Slice.
      def read_tensor_data(info : TensorInfo, dst : Slice(UInt8))
        @io.seek((@tensor_data_offset + info.offset).to_i64)
        @io.read_fully(dst)
      end

      # Read raw tensor bytes into a Pointer.
      def read_tensor_raw(info : TensorInfo) : Pointer(UInt8)
        size = info.byte_size
        ptr = Pointer(UInt8).malloc(size)
        @io.seek((@tensor_data_offset + info.offset).to_i64)
        # Chunked read for large tensors (IO#read_fully overflow at > 2 GB).
        read = 0_u64
        while read < size
          n = Math.min(size - read, 1_073_741_824_u64).to_i32
          @io.read_fully(Slice.new(ptr + read, n))
          read += n
        end
        ptr
      end

      # Convenience: get a metadata string or nil.
      def meta_string(key : String) : String?
        metadata[key]?.try(&.as(String))
      end

      def meta_u32(key : String) : UInt32?
        v = metadata[key]?
        case v
        when UInt32 then v
        when UInt64 then v.to_u32
        when Int32  then v.to_u32
        end
      end

      def meta_u64(key : String) : UInt64?
        v = metadata[key]?
        case v
        when UInt64 then v
        when UInt32 then v.to_u64
        when Int32  then v.to_u64
        end
      end

      def meta_f32(key : String) : Float32?
        v = metadata[key]?
        case v
        when Float32 then v
        when Float64 then v.to_f32
        end
      end

      # --- mmap-based parsers (used during initialization) ---
      # These read from the mmap'd buffer at @parse_pos, advancing it.
      # All header/metadata parsing uses these, so the 500K tokenizer
      # strings are read via pointer arithmetic, not IO syscalls.

      private def mmap_ptr_bang! : Pointer(UInt8)
        @mmap_ptr || raise "GGUF: mmap not available for parsing"
      end

      private def parse_u8 : UInt8
        ptr = mmap_ptr_bang!
        v = ptr[@parse_pos]
        @parse_pos += 1
        v
      end

      private def parse_i8 : Int8
        parse_u8.to_i8!
      end

      private def parse_u16 : UInt16
        ptr = mmap_ptr_bang!
        v = (ptr + @parse_pos).as(Pointer(UInt16)).value
        @parse_pos += 2
        v
      end

      private def parse_i16 : Int16
        parse_u16.to_i16!
      end

      private def parse_u32 : UInt32
        ptr = mmap_ptr_bang!
        v = (ptr + @parse_pos).as(Pointer(UInt32)).value
        @parse_pos += 4
        v
      end

      private def parse_i32 : Int32
        parse_u32.to_i32!
      end

      private def parse_u64 : UInt64
        ptr = mmap_ptr_bang!
        v = (ptr + @parse_pos).as(Pointer(UInt64)).value
        @parse_pos += 8
        v
      end

      private def parse_i64 : Int64
        parse_u64.to_i64!
      end

      private def parse_f32 : Float32
        ptr = mmap_ptr_bang!
        v = (ptr + @parse_pos).as(Pointer(Float32)).value
        @parse_pos += 4
        v
      end

      private def parse_f64 : Float64
        ptr = mmap_ptr_bang!
        v = (ptr + @parse_pos).as(Pointer(Float64)).value
        @parse_pos += 8
        v
      end

      private def parse_bool : Bool
        parse_u8 != 0
      end

      private def parse_string : String
        len = parse_u64
        ptr = mmap_ptr_bang!
        s = String.new(Slice.new(ptr + @parse_pos, len.to_i32))
        @parse_pos += len
        s
      end

      private def parse_value(type : MetaType) : MetaValue
        case type
        when .uint8?   then parse_u8
        when .int8?    then parse_i8
        when .uint16?  then parse_u16
        when .int16?   then parse_i16
        when .uint32?  then parse_u32
        when .int32?   then parse_i32
        when .float32? then parse_f32
        when .bool?    then parse_bool
        when .string?  then parse_string
        when .uint64?  then parse_u64
        when .int64?   then parse_i64
        when .float64? then parse_f64
        when .array?
          elem_type = MetaType.new(parse_u32)
          len = parse_u64
          arr = Array(MetaValue).new(len.to_i32)
          len.times { arr << parse_value(elem_type) }
          arr
        else
          raise "unknown GGUF metadata type #{type.value}"
        end
      end

      private def parse_kv
        key = parse_string
        vtype = MetaType.new(parse_u32)
        # Skip large tokenizer arrays (248K strings each) -- they take 2+ min to
        # allocate as Crystal objects and the agent uses tokenizer.json instead.
        if key.starts_with?("tokenizer.ggml.") && vtype.array?
          skip_value(vtype)
          return
        end
        @metadata[key] = parse_value(vtype)
      end

      # Advance parse_pos past a value without allocating Crystal objects.
      private def skip_value(type : MetaType)
        case type
        when .uint8?, .int8?, .bool?      then @parse_pos += 1
        when .uint16?, .int16?            then @parse_pos += 2
        when .uint32?, .int32?, .float32? then @parse_pos += 4
        when .uint64?, .int64?, .float64? then @parse_pos += 8
        when .string?
          len = parse_u64
          @parse_pos += len
        when .array?
          elem_type = MetaType.new(parse_u32)
          len = parse_u64
          len.times { skip_value(elem_type) }
        end
      end

      private def parse_tensor_info
        name = parse_string
        ndim = parse_u32
        shape = Array(UInt64).new(ndim.to_i32) { parse_u64 }
        type = GGMLType.new(parse_u32)
        offset = parse_u64
        @tensors[name] = TensorInfo.new(name, shape, type, offset)
      end

      # --- private readers (IO fallback, used by read_tensor_data) ---

      private def align(pos : UInt64) : UInt64
        a = @alignment.to_u64
        pos + (a - (pos % a)) % a
      end

      private def read_u8 : UInt8
        @io.read_byte.not_nil!
      end

      private def read_i8 : Int8
        read_u8.to_i8!
      end

      private def read_u16 : UInt16
        buf = uninitialized UInt8[2]
        @io.read_fully(buf.to_slice)
        IO::ByteFormat::LittleEndian.decode(UInt16, buf.to_slice)
      end

      private def read_i16 : Int16
        read_u16.to_i16!
      end

      private def read_u32 : UInt32
        buf = uninitialized UInt8[4]
        @io.read_fully(buf.to_slice)
        IO::ByteFormat::LittleEndian.decode(UInt32, buf.to_slice)
      end

      private def read_i32 : Int32
        read_u32.to_i32!
      end

      private def read_u64 : UInt64
        buf = uninitialized UInt8[8]
        @io.read_fully(buf.to_slice)
        IO::ByteFormat::LittleEndian.decode(UInt64, buf.to_slice)
      end

      private def read_i64 : Int64
        read_u64.to_i64!
      end

      private def read_f32 : Float32
        buf = uninitialized UInt8[4]
        @io.read_fully(buf.to_slice)
        IO::ByteFormat::LittleEndian.decode(Float32, buf.to_slice)
      end

      private def read_f64 : Float64
        buf = uninitialized UInt8[8]
        @io.read_fully(buf.to_slice)
        IO::ByteFormat::LittleEndian.decode(Float64, buf.to_slice)
      end

      private def read_bool : Bool
        read_u8 != 0
      end

      private def read_string : String
        len = read_u64
        buf = Bytes.new(len.to_i32)
        @io.read_fully(buf)
        String.new(buf)
      end

      private def read_value(type : MetaType) : MetaValue
        case type
        when .uint8?   then read_u8
        when .int8?    then read_i8
        when .uint16?  then read_u16
        when .int16?   then read_i16
        when .uint32?  then read_u32
        when .int32?   then read_i32
        when .float32? then read_f32
        when .bool?    then read_bool
        when .string?  then read_string
        when .uint64?  then read_u64
        when .int64?   then read_i64
        when .float64? then read_f64
        when .array?
          elem_type = MetaType.new(read_u32)
          len = read_u64
          arr = Array(MetaValue).new(len.to_i32)
          len.times { arr << read_value(elem_type) }
          arr
        else
          raise "unknown GGUF metadata type #{type.value}"
        end
      end

      private def read_kv
        key = read_string
        vtype = MetaType.new(read_u32)
        @metadata[key] = read_value(vtype)
      end

      private def read_tensor_info
        name = read_string
        ndim = read_u32
        shape = Array(UInt64).new(ndim.to_i32) { read_u64 }
        type = GGMLType.new(read_u32)
        offset = read_u64
        @tensors[name] = TensorInfo.new(name, shape, type, offset)
      end
    end
  end
end

module SHAInet
  module GGUF
    # Extract a BPE tokenizer from a GGUF file's metadata.
    # This re-parses ONLY the tokenizer KV pairs (tokens + merges) from the
    # mmap'd buffer, building the BPETokenizer directly. Faster than parsing
    # all metadata because it skips non-tokenizer keys.
    def self.extract_tokenizer(path : String) : BPETokenizer
      gf = File.open(path, mmap: true)
      begin
        tok = BPETokenizer.new
        tok.hf_mode = true

        # Re-parse the KV section to extract tokenizer data.
        # We need: tokenizer.ggml.tokens, tokenizer.ggml.merges
        ptr = gf.mmap_ptr.not_nil!
        pos = 0_u64
        _magic = (ptr + pos).as(Pointer(UInt32)).value; pos += 4
        _version = (ptr + pos).as(Pointer(UInt32)).value; pos += 4
        _tensor_count = (ptr + pos).as(Pointer(UInt64)).value; pos += 8
        kv_count = (ptr + pos).as(Pointer(UInt64)).value; pos += 8

        types = [] of Tuple(Int32, Int32)
        kv_count.times do
          # Read key string
          key_len = (ptr + pos).as(Pointer(UInt64)).value; pos += 8
          key = String.new(Slice.new(ptr + pos, key_len.to_i32)); pos += key_len
          vtype = (ptr + pos).as(Pointer(UInt32)).value; pos += 4

          if key == "tokenizer.ggml.tokens" && vtype == 9 # ARRAY
            _elem_type = (ptr + pos).as(Pointer(UInt32)).value; pos += 4
            arr_len = (ptr + pos).as(Pointer(UInt64)).value; pos += 8
            max_id = 0
            arr_len.times do |i|
              slen = (ptr + pos).as(Pointer(UInt64)).value; pos += 8
              token = String.new(Slice.new(ptr + pos, slen.to_i32)); pos += slen
              tok.vocab[token] = i.to_i32
              max_id = i.to_i32 if i.to_i32 > max_id
            end
            tok.inv_vocab.concat(Array(String).new(max_id + 1, ""))
            tok.vocab.each { |t, id| tok.inv_vocab[id] = t }
          elsif key == "tokenizer.ggml.token_type" && vtype == 9
            elem_type = (ptr + pos).as(Pointer(UInt32)).value; pos += 4
            arr_len = (ptr + pos).as(Pointer(UInt64)).value; pos += 8
            if elem_type == 5 # int32
              arr_len.times do |i|
                types << {i.to_i32, (ptr + pos).as(Pointer(Int32)).value}
                pos += 4
              end
            else
              arr_len.times { skip_gguf_value(ptr, pointerof(pos), elem_type) }
            end
          elsif key == "tokenizer.ggml.pre" && vtype == 8
            slen = (ptr + pos).as(Pointer(UInt64)).value; pos += 8
            tok.pre_tokenizer = String.new(Slice.new(ptr + pos, slen.to_i32)); pos += slen
          elsif key == "tokenizer.ggml.merges" && vtype == 9
            _elem_type = (ptr + pos).as(Pointer(UInt32)).value; pos += 4
            arr_len = (ptr + pos).as(Pointer(UInt64)).value; pos += 8
            arr_len.times do |rank|
              slen = (ptr + pos).as(Pointer(UInt64)).value; pos += 8
              merge_str = String.new(Slice.new(ptr + pos, slen.to_i32)); pos += slen
              parts = merge_str.split(' ', 2)
              next unless parts.size == 2
              pair = {parts[0], parts[1]}
              merged = parts[0] + parts[1]
              tok.merges << pair
              tok.merges_map[pair] = merged
              tok.merges_rank[pair] = rank.to_i32
            end
          else
            # Skip this value
            skip_gguf_value(ptr, pointerof(pos), vtype)
          end
        end

        # Type 4 = USER_DEFINED (llama.cpp's LLAMA_TOKEN_TYPE_USER_DEFINED). Resolved after the loop
        # because the type array may come before the token strings.
        types.each do |(id, ty)|
          next unless ty == 4
          if (t = tok.inv_vocab[id]?) && !t.empty?
            tok.add_literal_token(t, id)
          end
        end
        tok
      ensure
        gf.close
      end
    end

    # Skip a GGUF metadata value by advancing pos past it.
    private def self.skip_gguf_value(ptr : Pointer(UInt8), pos : Pointer(UInt64), vtype : UInt32)
      case vtype
      when 0, 1, 7    then pos.value += 1 # u8, i8, bool
      when 2, 3       then pos.value += 2 # u16, i16
      when 4, 5, 6    then pos.value += 4 # u32, i32, f32
      when 10, 11, 12 then pos.value += 8 # u64, i64, f64
      when 8                              # string
        len = (ptr + pos.value).as(Pointer(UInt64)).value; pos.value += 8
        pos.value += len
      when 9 # array
        elem_type = (ptr + pos.value).as(Pointer(UInt32)).value; pos.value += 4
        arr_len = (ptr + pos.value).as(Pointer(UInt64)).value; pos.value += 8
        arr_len.times { skip_gguf_value(ptr, pos, elem_type) }
      end
    end
  end
end
