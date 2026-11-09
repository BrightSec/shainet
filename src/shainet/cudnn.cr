require "./cuda"

# cuDNN bindings for high-performance deep learning operations
@[Link("cudnn")]
lib LibCUDNN
  alias CudnnHandle = Void*
  alias CudnnTensorDescriptor = Void*
  alias CudnnFilterDescriptor = Void*
  alias CudnnConvolutionDescriptor = Void*
  alias CudnnActivationDescriptor = Void*
  alias CudnnPoolingDescriptor = Void*
  alias CudnnDropoutDescriptor = Void*
  alias CudnnRNNDescriptor = Void*
  alias CudnnOpTensorDescriptor = Void*
  alias CudnnAttentionDescriptor = Void*

  # Status codes
  enum CudnnStatus
    CUDNN_STATUS_SUCCESS                      =  0
    CUDNN_STATUS_NOT_INITIALIZED              =  1
    CUDNN_STATUS_ALLOC_FAILED                 =  2
    CUDNN_STATUS_BAD_PARAM                    =  3
    CUDNN_STATUS_INTERNAL_ERROR               =  4
    CUDNN_STATUS_INVALID_VALUE                =  5
    CUDNN_STATUS_ARCH_MISMATCH                =  6
    CUDNN_STATUS_MAPPING_ERROR                =  7
    CUDNN_STATUS_EXECUTION_FAILED             =  8
    CUDNN_STATUS_NOT_SUPPORTED                =  9
    CUDNN_STATUS_LICENSE_ERROR                = 10
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11
    CUDNN_STATUS_RUNTIME_IN_PROGRESS          = 12
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13
    CUDNN_STATUS_VERSION_MISMATCH             = 14
  end

  # Data types
  enum CudnnDataType
    CUDNN_DATA_FLOAT    =  0
    CUDNN_DATA_DOUBLE   =  1
    CUDNN_DATA_HALF     =  2
    CUDNN_DATA_INT8     =  3
    CUDNN_DATA_INT32    =  4
    CUDNN_DATA_INT8x4   =  5
    CUDNN_DATA_UINT8    =  6
    CUDNN_DATA_UINT8x4  =  7
    CUDNN_DATA_INT8x32  =  8
    CUDNN_DATA_BFLOAT16 =  9
    CUDNN_DATA_INT64    = 10
  end

  # Tensor formats
  enum CudnnTensorFormat
    CUDNN_TENSOR_NCHW        = 0
    CUDNN_TENSOR_NHWC        = 1
    CUDNN_TENSOR_NCHW_VECT_C = 2
  end

  # Activation modes
  enum CudnnActivationMode
    CUDNN_ACTIVATION_SIGMOID      = 0
    CUDNN_ACTIVATION_RELU         = 1
    CUDNN_ACTIVATION_TANH         = 2
    CUDNN_ACTIVATION_CLIPPED_RELU = 3
    CUDNN_ACTIVATION_ELU          = 4
    CUDNN_ACTIVATION_IDENTITY     = 5
  end

  # Math types
  enum CudnnMathType
    CUDNN_DEFAULT_MATH                    = 0
    CUDNN_TENSOR_OP_MATH                  = 1
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2
    CUDNN_FMA_MATH                        = 3
  end

  # Softmax algorithms
  enum CudnnSoftmaxAlgorithm
    CUDNN_SOFTMAX_FAST     = 0
    CUDNN_SOFTMAX_ACCURATE = 1
    CUDNN_SOFTMAX_LOG      = 2
  end

  # Softmax mode
  enum CudnnSoftmaxMode
    CUDNN_SOFTMAX_MODE_INSTANCE = 0
    CUDNN_SOFTMAX_MODE_CHANNEL  = 1
  end

  # OpTensor operations
  enum CudnnOpTensorOp
    CUDNN_OP_TENSOR_ADD  = 0
    CUDNN_OP_TENSOR_MUL  = 1
    CUDNN_OP_TENSOR_MIN  = 2
    CUDNN_OP_TENSOR_MAX  = 3
    CUDNN_OP_TENSOR_SQRT = 4
    CUDNN_OP_TENSOR_NOT  = 5
  end

  # Core functions
  fun cudnnCreate(handle : CudnnHandle*) : CudnnStatus
  fun cudnnDestroy(handle : CudnnHandle) : CudnnStatus
  fun cudnnGetVersion : LibC::SizeT
  fun cudnnGetErrorString(status : CudnnStatus) : LibC::Char*

  # Tensor descriptor functions
  fun cudnnCreateTensorDescriptor(tensorDesc : CudnnTensorDescriptor*) : CudnnStatus
  fun cudnnDestroyTensorDescriptor(tensorDesc : CudnnTensorDescriptor) : CudnnStatus
  fun cudnnSetTensorNdDescriptor(tensorDesc : CudnnTensorDescriptor, dataType : CudnnDataType,
                                 nbDims : LibC::Int, dimA : LibC::Int*, strideA : LibC::Int*) : CudnnStatus

  # Activation functions
  fun cudnnCreateActivationDescriptor(activationDesc : CudnnActivationDescriptor*) : CudnnStatus
  fun cudnnDestroyActivationDescriptor(activationDesc : CudnnActivationDescriptor) : CudnnStatus
  fun cudnnSetActivationDescriptor(activationDesc : CudnnActivationDescriptor, mode : CudnnActivationMode,
                                   reluNanOpt : LibC::Int, coef : LibC::Double) : CudnnStatus
  fun cudnnActivationForward(handle : CudnnHandle, activationDesc : CudnnActivationDescriptor,
                             alpha : Void*, xDesc : CudnnTensorDescriptor, x : Void*,
                             beta : Void*, yDesc : CudnnTensorDescriptor, y : Void*) : CudnnStatus
  fun cudnnActivationBackward(handle : CudnnHandle, activationDesc : CudnnActivationDescriptor,
                              alpha : Void*, yDesc : CudnnTensorDescriptor, y : Void*,
                              dyDesc : CudnnTensorDescriptor, dy : Void*,
                              xDesc : CudnnTensorDescriptor, x : Void*,
                              beta : Void*, dxDesc : CudnnTensorDescriptor, dx : Void*) : CudnnStatus

  # Softmax functions
  fun cudnnSoftmaxForward(handle : CudnnHandle, algorithm : CudnnSoftmaxAlgorithm, mode : CudnnSoftmaxMode,
                          alpha : Void*, xDesc : CudnnTensorDescriptor, x : Void*,
                          beta : Void*, yDesc : CudnnTensorDescriptor, y : Void*) : CudnnStatus
  fun cudnnSoftmaxBackward(handle : CudnnHandle, algorithm : CudnnSoftmaxAlgorithm, mode : CudnnSoftmaxMode,
                           alpha : Void*, yDesc : CudnnTensorDescriptor, y : Void*,
                           dyDesc : CudnnTensorDescriptor, dy : Void*,
                           beta : Void*, dxDesc : CudnnTensorDescriptor, dx : Void*) : CudnnStatus

  # Dropout functions
  fun cudnnCreateDropoutDescriptor(dropoutDesc : CudnnDropoutDescriptor*) : CudnnStatus
  fun cudnnDestroyDropoutDescriptor(dropoutDesc : CudnnDropoutDescriptor) : CudnnStatus
  fun cudnnDropoutGetStatesSize(handle : CudnnHandle, sizeInBytes : LibC::SizeT*) : CudnnStatus
  fun cudnnSetDropoutDescriptor(dropoutDesc : CudnnDropoutDescriptor, handle : CudnnHandle, dropout : LibC::Float,
                                states : Void*, stateSizeInBytes : LibC::SizeT, seed : LibC::ULongLong) : CudnnStatus
  fun cudnnDropoutForward(handle : CudnnHandle, dropoutDesc : CudnnDropoutDescriptor,
                          xDesc : CudnnTensorDescriptor, x : Void*,
                          yDesc : CudnnTensorDescriptor, y : Void*,
                          reserveSpace : Void*, reserveSpaceSizeInBytes : LibC::SizeT) : CudnnStatus
  fun cudnnDropoutBackward(handle : CudnnHandle, dropoutDesc : CudnnDropoutDescriptor,
                           dyDesc : CudnnTensorDescriptor, dy : Void*,
                           dxDesc : CudnnTensorDescriptor, dx : Void*,
                           reserveSpace : Void*, reserveSpaceSizeInBytes : LibC::SizeT) : CudnnStatus

  # OpTensor functions (for element-wise operations)
  fun cudnnCreateOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor*) : CudnnStatus
  fun cudnnDestroyOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor) : CudnnStatus
  fun cudnnSetOpTensorDescriptor(opTensorDesc : CudnnOpTensorDescriptor, opTensorOp : CudnnOpTensorOp,
                                 opTensorCompType : CudnnDataType, opTensorNanOpt : LibC::Int) : CudnnStatus
  fun cudnnOpTensor(handle : CudnnHandle, opTensorDesc : CudnnOpTensorDescriptor,
                    alpha1 : Void*, aDesc : CudnnTensorDescriptor, a : Void*,
                    alpha2 : Void*, bDesc : CudnnTensorDescriptor, b : Void*,
                    beta : Void*, cDesc : CudnnTensorDescriptor, c : Void*) : CudnnStatus

  # AddTensor (for bias addition)
  fun cudnnAddTensor(handle : CudnnHandle,
                     alpha : Void*, aDesc : CudnnTensorDescriptor, a : Void*,
                     beta : Void*, cDesc : CudnnTensorDescriptor, c : Void*) : CudnnStatus
end

module SHAInet
  module CUDNN
    extend self

    @@handle : LibCUDNN::CudnnHandle?
    @@available : Bool?

    class CudnnError < Exception
      def initialize(@status : LibCUDNN::CudnnStatus)
        super("cuDNN error: #{String.new(LibCUDNN.cudnnGetErrorString(@status))}")
      end
    end

    def available?
      @@available ||= begin
        return false unless CUDA.available?

        # Try to create and destroy a cuDNN handle
        result = LibCUDNN.cudnnCreate(out handle)
        if result == LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
          LibCUDNN.cudnnDestroy(handle)
          Log.info { "cuDNN available, version: #{LibCUDNN.cudnnGetVersion}" }
          true
        else
          Log.info { "cuDNN not available: #{String.new(LibCUDNN.cudnnGetErrorString(result))}" }
          false
        end
      rescue ex
        Log.debug { "cuDNN availability check failed: #{ex}" }
        false
      end
    end

    def handle
      @@handle ||= begin
        raise "cuDNN not available" unless available?

        result = LibCUDNN.cudnnCreate(out h)
        if result != LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
          raise CudnnError.new(result)
        end
        h
      end
    end

    def check_status(status : LibCUDNN::CudnnStatus)
      unless status == LibCUDNN::CudnnStatus::CUDNN_STATUS_SUCCESS
        raise CudnnError.new(status)
      end
    end

    # Cleanup
    def cleanup
      if h = @@handle
        LibCUDNN.cudnnDestroy(h)
        @@handle = nil
      end
    end

    # Helper function to create 2D tensor descriptors for matrices
    private def self.create_tensor_descriptor_2d(rows : Int32, cols : Int32)
      # Create the descriptor first
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out desc))

      # Set up 4D tensor descriptor: [batch, channels, height, width] = [rows, cols, 1, 1]
      dims = [rows, cols, 1, 1]
      strides = [cols, 1, cols, cols] # Row-major ordering
      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        desc,
        LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT,
        4,
        dims.to_unsafe,
        strides.to_unsafe
      ))

      desc
    end

    # High-level operations

    # Optimized ReLU forward pass
    def self.relu_forward(input : CudaMatrix, output : CudaMatrix)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Set up 4D tensor descriptors: [batch, channels, height, width] = [rows, cols, 1, 1]
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out output_desc))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        input_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT, 4,
        dims.to_unsafe, strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        output_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT, 4,
        dims.to_unsafe, strides.to_unsafe))

      # Create activation descriptor for ReLU
      CUDNN.check_status(LibCUDNN.cudnnCreateActivationDescriptor(out activation_desc))
      CUDNN.check_status(LibCUDNN.cudnnSetActivationDescriptor(
        activation_desc,
        LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_RELU,
        0,  # reluNanOpt
        0.0 # coef
      ))

      alpha = 1.0_f32
      beta = 0.0_f32

      input.sync_to_device!("cudnn_relu_input") unless input.device_dirty?

      # Get device pointers and ensure they're not nil
      input_ptr = input.device_ptr
      output_ptr = output.device_ptr
      raise "Device pointers are nil" if input_ptr.nil? || output_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnActivationForward(
        CUDNN.handle,
        activation_desc,
        pointerof(alpha),
        input_desc,
        input_ptr.as(Pointer(Void)),
        pointerof(beta),
        output_desc,
        output_ptr.as(Pointer(Void))
      ))

      output.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyActivationDescriptor(activation_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
    end

    # Optimized ReLU backward pass
    def self.relu_backward(input : CudaMatrix, grad_output : CudaMatrix, grad_input : CudaMatrix)
      raise "Matrices must have same dimensions" unless input.rows == grad_output.rows && input.rows == grad_input.rows

      # Set up 4D tensor descriptors
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out grad_desc))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        input_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT, 4,
        dims.to_unsafe, strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        grad_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT, 4,
        dims.to_unsafe, strides.to_unsafe))

      # Create activation descriptor for ReLU
      CUDNN.check_status(LibCUDNN.cudnnCreateActivationDescriptor(out activation_desc))
      CUDNN.check_status(LibCUDNN.cudnnSetActivationDescriptor(
        activation_desc,
        LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_RELU,
        0,  # reluNanOpt
        0.0 # coef
      ))

      alpha = 1.0_f32
      beta = 0.0_f32

      input.sync_to_device!("cudnn_relu_backward_input") unless input.device_dirty?
      grad_output.sync_to_device!("cudnn_relu_backward_grad") unless grad_output.device_dirty?

      # Get device pointers and ensure they're not nil
      input_ptr = input.device_ptr
      grad_output_ptr = grad_output.device_ptr
      grad_input_ptr = grad_input.device_ptr
      raise "Device pointers are nil" if input_ptr.nil? || grad_output_ptr.nil? || grad_input_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnActivationBackward(
        CUDNN.handle,
        activation_desc,
        pointerof(alpha),
        input_desc,
        input_ptr.as(Pointer(Void)),
        grad_desc,
        grad_output_ptr.as(Pointer(Void)),
        input_desc,
        input_ptr.as(Pointer(Void)),
        pointerof(beta),
        grad_desc,
        grad_input_ptr.as(Pointer(Void))
      ))

      grad_input.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyActivationDescriptor(activation_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(grad_desc)
    end

    # Optimized bias addition
    def self.add_bias!(matrix : CudaMatrix, bias : CudaMatrix)
      raise "Bias must be a row vector" unless bias.rows == 1 && bias.cols == matrix.cols

      # For bias addition, cuDNN expects the bias to be a 4D tensor with shape [1, C, 1, 1]
      # and the matrix to be [N, C, H, W]. For 2D matrices, we treat them as [N, C, 1, 1]

      # Matrix descriptor: treat as [batch_size, channels, 1, 1]
      matrix_dims = [matrix.rows, matrix.cols, 1, 1]
      matrix_strides = [matrix.cols, 1, 1, 1]

      # Bias descriptor: [1, channels, 1, 1]
      bias_dims = [1, bias.cols, 1, 1]
      bias_strides = [bias.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out matrix_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out bias_desc))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        matrix_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT, 4,
        matrix_dims.to_unsafe, matrix_strides.to_unsafe))

      CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
        bias_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT, 4,
        bias_dims.to_unsafe, bias_strides.to_unsafe))

      alpha = 1.0_f32
      beta = 1.0_f32 # Add to existing values

      matrix.sync_to_device!("cudnn_bias_matrix") unless matrix.device_dirty?
      bias.sync_to_device!("cudnn_bias_vector") unless bias.device_dirty?

      # Get device pointers and ensure they're not nil
      bias_ptr = bias.device_ptr
      matrix_ptr = matrix.device_ptr
      raise "Device pointers are nil" if bias_ptr.nil? || matrix_ptr.nil?

      CUDNN.check_status(LibCUDNN.cudnnAddTensor(
        CUDNN.handle,
        pointerof(alpha),
        bias_desc,
        bias_ptr.as(Pointer(Void)),
        pointerof(beta),
        matrix_desc,
        matrix_ptr.as(Pointer(Void))
      ))

      matrix.mark_device_dirty!

      # Clean up
      LibCUDNN.cudnnDestroyTensorDescriptor(matrix_desc)
      LibCUDNN.cudnnDestroyTensorDescriptor(bias_desc)
    end

    # Optimized softmax (for attention) with proper descriptor management
    def self.softmax_rows(input : CudaMatrix, output : CudaMatrix)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Set up 4D tensor descriptors: [batch, channels, height, width] = [rows, cols, 1, 1]
      dims = [input.rows, input.cols, 1, 1]
      strides = [input.cols, 1, 1, 1]

      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out input_desc))
      CUDNN.check_status(LibCUDNN.cudnnCreateTensorDescriptor(out output_desc))

      begin
        CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
          input_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT, 4,
          dims.to_unsafe, strides.to_unsafe))

        CUDNN.check_status(LibCUDNN.cudnnSetTensorNdDescriptor(
          output_desc, LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT, 4,
          dims.to_unsafe, strides.to_unsafe))

        alpha = 1.0_f32
        beta = 0.0_f32

        input.sync_to_device!("cudnn_softmax_input") unless input.device_dirty?

        # Get device pointers and ensure they're not nil
        input_ptr = input.device_ptr
        output_ptr = output.device_ptr
        raise "Device pointers are nil" if input_ptr.nil? || output_ptr.nil?

        CUDNN.check_status(LibCUDNN.cudnnSoftmaxForward(
          CUDNN.handle,
          LibCUDNN::CudnnSoftmaxAlgorithm::CUDNN_SOFTMAX_ACCURATE,
          LibCUDNN::CudnnSoftmaxMode::CUDNN_SOFTMAX_MODE_INSTANCE,
          pointerof(alpha),
          input_desc,
          input_ptr.as(Pointer(Void)),
          pointerof(beta),
          output_desc,
          output_ptr.as(Pointer(Void))
        ))

        output.mark_device_dirty!
      ensure
        # Clean up descriptors
        LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
        LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
      end
    end

    # Optimized element-wise addition using cuDNN (fallback to cuBLAS if OpTensor not available)
    def self.element_add!(result : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 1.0)
      # For now, fall back to cuBLAS GEAM since OpTensor may not be available in all cuDNN versions
      # This can be optimized later when OpTensor support is confirmed
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && result.rows == a.rows && result.cols == a.cols

      a.sync_to_device!("cudnn_element_add_a") unless a.device_dirty?
      b.sync_to_device!("cudnn_element_add_b") unless b.device_dirty?

      # Use cuBLAS GEAM for element-wise addition as fallback
      handle = CUDA.create_handle
      begin
        CUDA.geam(handle, a.device_ptr.not_nil!, b.device_ptr.not_nil!,
          result.device_ptr.not_nil!, a.rows, a.cols, alpha.to_f32, beta.to_f32)
      ensure
        CUDA.destroy_handle(handle)
      end

      result.mark_device_dirty!
    end

    # GPU-optimized softmax + cross-entropy loss and gradient computation
    def self.softmax_cross_entropy_loss_and_gradient(predicted : CudaMatrix, target : CudaMatrix,
                                                     loss : Float64*, grad_output : CudaMatrix)
      raise "Predicted and target must have same dimensions" unless predicted.rows == target.rows && predicted.cols == target.cols
      raise "Gradient output must have same dimensions as predicted" unless grad_output.rows == predicted.rows && grad_output.cols == predicted.cols

      # Compute softmax of logits into grad_output
      softmax_rows(predicted, grad_output)

      # Now compute cross-entropy on the probabilities in grad_output
      loss_f32 = 0.0_f32
      result = CUDA.cross_entropy_loss_gradient(
        grad_output.device_ptr.not_nil!,
        target.device_ptr.not_nil!,
        grad_output.device_ptr.not_nil!,
        pointerof(loss_f32),
        predicted.rows,
        predicted.cols
      )

      raise "CUDA softmax cross-entropy failed" if result != 0
      loss.value = loss_f32.to_f64

      grad_output.mark_device_dirty!
    end

    # GPU-optimized softmax + cross-entropy using label indices.
    # +labels+ should be a column vector (rows x 1) containing integer class indices.
    def self.softmax_cross_entropy_label_loss_and_gradient(predicted : CudaMatrix, labels : CudaMatrix,
                                                           loss : Float64*, grad_output : CudaMatrix)
      raise "Labels must have one column" unless labels.cols == 1
      raise "Label rows must match predictions" unless labels.rows == predicted.rows
      raise "Gradient output must have same dimensions as predicted" unless grad_output.rows == predicted.rows && grad_output.cols == predicted.cols

      # Do NOT compute softmax into grad_output before calling the kernel!
      # The kernel expects logits as input and writes softmax/grad to grad_output.

      # Pull labels to host as Int32 array
      labels.sync_from_device!("sm_xent_labels") if labels.device_dirty?
      label_ids = Array(Int32).new(labels.rows) do |i|
        labels.unsafe_get(i, 0).to_i
      end

      bytes = (label_ids.size * 4).to_u64
      labels_dev = Pointer(Int32).null
      CUDA.malloc(pointerof(labels_dev).as(Pointer(Pointer(Void))), bytes)
      begin
        CUDA.memcpy(labels_dev.as(Pointer(Void)), label_ids.to_unsafe.as(Pointer(Void)), bytes, CUDA::MemcpyKind::HostToDevice)

        # Compute cross-entropy using CUDA kernel
        loss_f32 = 0.0_f32
        result = CUDA.softmax_cross_entropy_label(
          grad_output.device_ptr.not_nil!,
          labels_dev,
          grad_output.device_ptr.not_nil!,
          pointerof(loss_f32),
          predicted.rows,
          predicted.cols
        )

        raise "CUDA softmax cross-entropy label failed" if result != 0
        loss.value = loss_f32.to_f64
      ensure
        CUDA.free(labels_dev.as(Pointer(Void))) unless labels_dev.null?
      end

      grad_output.mark_device_dirty!
    end

    # Element-wise operations using cuDNN OpTensor
    def self.element_multiply!(output : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 0.0)
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && output.rows == a.rows && output.cols == a.cols

      # Create OpTensor descriptor (`out` declares op_desc)
      CUDNN.check_status(LibCUDNN.cudnnCreateOpTensorDescriptor(out op_desc))

      begin
        # Set up for element-wise multiplication
        CUDNN.check_status(LibCUDNN.cudnnSetOpTensorDescriptor(
          op_desc,
          LibCUDNN::CudnnOpTensorOp::CUDNN_OP_TENSOR_MUL,
          LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT,
          0 # NaN propagation option
        ))

        # Create tensor descriptors
        a_desc = create_tensor_descriptor_2d(a.rows, a.cols)
        b_desc = create_tensor_descriptor_2d(b.rows, b.cols)
        c_desc = create_tensor_descriptor_2d(output.rows, output.cols)

        begin
          # Ensure matrices are on device
          a.sync_to_device!("cudnn_element_mul") unless a.device_dirty?
          b.sync_to_device!("cudnn_element_mul") unless b.device_dirty?
          output.sync_to_device!("cudnn_element_mul") unless output.device_dirty?

          alpha_val = alpha
          beta_val = beta

          CUDNN.check_status(LibCUDNN.cudnnOpTensor(
            CUDNN.handle,
            op_desc,
            pointerof(alpha_val).as(Pointer(Void)),
            a_desc,
            a.device_ptr.not_nil!.as(Pointer(Void)),
            pointerof(alpha_val).as(Pointer(Void)),
            b_desc,
            b.device_ptr.not_nil!.as(Pointer(Void)),
            pointerof(beta_val).as(Pointer(Void)),
            c_desc,
            output.device_ptr.not_nil!.as(Pointer(Void))
          ))

          output.mark_device_dirty!
        ensure
          LibCUDNN.cudnnDestroyTensorDescriptor(a_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(b_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(c_desc)
        end
      ensure
        LibCUDNN.cudnnDestroyOpTensorDescriptor(op_desc)
      end
    end

    def self.element_log!(output : CudaMatrix, input : CudaMatrix)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      if CUDA.kernels_available? && (out_ptr = output.device_ptr) && (in_ptr = input.device_ptr)
        begin
          input.sync_to_device!("cudnn_element_log_in") unless input.device_dirty?
          output.sync_to_device!("cudnn_element_log_out") unless output.device_dirty?
          size = input.rows * input.cols
          CUDA.element_log(out_ptr, in_ptr, size)
          output.mark_device_dirty!
          return
        rescue ex
          Log.error { "CUDA element_log kernel failed: #{ex}" }
        end
      end

      # CPU fallback
      input.sync_from_device!("element_log_fallback") if input.device_dirty?
      input.rows.times do |i|
        input.cols.times do |j|
          output.unsafe_set(i, j, Math.log(input.unsafe_get(i, j)))
        end
      end
      output.sync_to_device!("element_log_result") if CUDA.available?
    end

    # Element-wise subtraction using OpTensor
    def self.element_subtract!(output : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = -1.0)
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && output.rows == a.rows && output.cols == a.cols

      # Use element addition with negative beta to achieve subtraction
      element_addition!(output, a, b, alpha, beta)
    end

    # Element-wise addition using OpTensor (more general version)
    def self.element_addition!(output : CudaMatrix, a : CudaMatrix, b : CudaMatrix, alpha : Float64 = 1.0, beta : Float64 = 1.0)
      raise "Matrices must have same dimensions" unless a.rows == b.rows && a.cols == b.cols && output.rows == a.rows && output.cols == a.cols

      # Create OpTensor descriptor (`out` declares op_desc)
      CUDNN.check_status(LibCUDNN.cudnnCreateOpTensorDescriptor(out op_desc))

      begin
        # Set up for element-wise addition
        CUDNN.check_status(LibCUDNN.cudnnSetOpTensorDescriptor(
          op_desc,
          LibCUDNN::CudnnOpTensorOp::CUDNN_OP_TENSOR_ADD,
          LibCUDNN::CudnnDataType::CUDNN_DATA_FLOAT,
          0 # NaN propagation option
        ))

        # Create tensor descriptors
        a_desc = create_tensor_descriptor_2d(a.rows, a.cols)
        b_desc = create_tensor_descriptor_2d(b.rows, b.cols)
        c_desc = create_tensor_descriptor_2d(output.rows, output.cols)

        begin
          # Ensure matrices are on device
          a.sync_to_device!("cudnn_element_add") unless a.device_dirty?
          b.sync_to_device!("cudnn_element_add") unless b.device_dirty?
          output.sync_to_device!("cudnn_element_add") unless output.device_dirty?

          alpha_val = alpha
          beta_val = beta

          CUDNN.check_status(LibCUDNN.cudnnOpTensor(
            CUDNN.handle,
            op_desc,
            pointerof(alpha_val).as(Pointer(Void)),
            a_desc,
            a.device_ptr.not_nil!.as(Pointer(Void)),
            pointerof(beta_val).as(Pointer(Void)),
            b_desc,
            b.device_ptr.not_nil!.as(Pointer(Void)),
            pointerof(alpha_val).as(Pointer(Void)), # Use alpha again for output scaling
            c_desc,
            output.device_ptr.not_nil!.as(Pointer(Void))
          ))

          output.mark_device_dirty!
        ensure
          LibCUDNN.cudnnDestroyTensorDescriptor(a_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(b_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(c_desc)
        end
      ensure
        LibCUDNN.cudnnDestroyOpTensorDescriptor(op_desc)
      end
    end

    # Optimized sigmoid using cuDNN activation
    def self.sigmoid_forward!(output : CudaMatrix, input : CudaMatrix)
      activation_forward!(output, input, LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_SIGMOID)
    end

    # Optimized tanh using cuDNN activation
    def self.tanh_forward!(output : CudaMatrix, input : CudaMatrix)
      activation_forward!(output, input, LibCUDNN::CudnnActivationMode::CUDNN_ACTIVATION_TANH)
    end

    # Generic activation forward
    def self.activation_forward!(output : CudaMatrix, input : CudaMatrix, mode : LibCUDNN::CudnnActivationMode)
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Create activation descriptor
      activation_desc = uninitialized LibCUDNN::CudnnActivationDescriptor
      CUDNN.check_status(LibCUDNN.cudnnCreateActivationDescriptor(pointerof(activation_desc)))

      begin
        # Set activation mode (sigmoid, tanh, relu, etc.)
        CUDNN.check_status(LibCUDNN.cudnnSetActivationDescriptor(
          activation_desc,
          mode,
          0,  # reluNanOpt (not used for sigmoid/tanh)
          0.0 # coef (not used for sigmoid/tanh)
        ))

        # Create tensor descriptors
        input_desc = create_tensor_descriptor_2d(input.rows, input.cols)
        output_desc = create_tensor_descriptor_2d(output.rows, output.cols)

        begin
          # Ensure matrices are on device
          input.sync_to_device!("cudnn_activation") unless input.device_dirty?
          output.sync_to_device!("cudnn_activation") unless output.device_dirty?

          alpha = 1.0_f32
          beta = 0.0_f32 # Get device pointers and ensure they're valid
          input_ptr = input.device_ptr
          output_ptr = output.device_ptr
          raise "Invalid device pointers" unless input_ptr && output_ptr && !input_ptr.null? && !output_ptr.null?

          CUDNN.check_status(LibCUDNN.cudnnActivationForward(
            CUDNN.handle,
            activation_desc,
            pointerof(alpha).as(Pointer(Void)),
            input_desc,
            input_ptr.as(Pointer(Void)),
            pointerof(beta).as(Pointer(Void)),
            output_desc,
            output_ptr.as(Pointer(Void))
          ))

          output.mark_device_dirty!
        ensure
          LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
          LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
        end
      ensure
        LibCUDNN.cudnnDestroyActivationDescriptor(activation_desc)
      end
    end

    # Dropout forward (GPU-accelerated dropout with mask generation)
    def self.dropout_forward!(output : CudaMatrix, input : CudaMatrix, dropout_prob : Float64, seed : UInt64 = Random.rand(UInt64::MAX))
      raise "Input and output must have same dimensions" unless input.rows == output.rows && input.cols == output.cols

      # Get dropout states size
      states_size = uninitialized LibC::SizeT
      CUDNN.check_status(LibCUDNN.cudnnDropoutGetStatesSize(CUDNN.handle, pointerof(states_size)))

      # Allocate dropout states on GPU
      states_ptr = Pointer(Void).null
      CUDA.malloc(pointerof(states_ptr), states_size.to_u64)

      begin
        # Create dropout descriptor
        dropout_desc = uninitialized LibCUDNN::CudnnDropoutDescriptor
        CUDNN.check_status(LibCUDNN.cudnnCreateDropoutDescriptor(pointerof(dropout_desc)))

        begin
          # Set dropout parameters
          CUDNN.check_status(LibCUDNN.cudnnSetDropoutDescriptor(
            dropout_desc,
            CUDNN.handle,
            dropout_prob.to_f32,
            states_ptr,
            states_size,
            seed
          ))

          # Create tensor descriptors
          input_desc = create_tensor_descriptor_2d(input.rows, input.cols)
          output_desc = create_tensor_descriptor_2d(output.rows, output.cols)

          begin
            # Ensure matrices are on device
            input.sync_to_device!("cudnn_dropout") unless input.device_dirty?
            output.sync_to_device!("cudnn_dropout") unless output.device_dirty?

            # Reserve space for backward pass (can be null for forward-only)
            reserve_space_ptr = Pointer(Void).null
            reserve_space_size = 0_u64

            CUDNN.check_status(LibCUDNN.cudnnDropoutForward(
              CUDNN.handle,
              dropout_desc,
              input_desc,
              input.device_ptr.not_nil!.as(Pointer(Void)),
              output_desc,
              output.device_ptr.not_nil!.as(Pointer(Void)),
              reserve_space_ptr,
              reserve_space_size
            ))

            output.mark_device_dirty!
          ensure
            LibCUDNN.cudnnDestroyTensorDescriptor(input_desc)
            LibCUDNN.cudnnDestroyTensorDescriptor(output_desc)
          end
        ensure
          LibCUDNN.cudnnDestroyDropoutDescriptor(dropout_desc)
        end
      ensure
        CUDA.free(states_ptr) unless states_ptr.null?
      end
    end
  end
end
