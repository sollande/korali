#include "modules/neuralNetwork/layer/recurrent/gru/gru.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"

#ifdef _KORALI_USE_CUDNN
  #include "auxiliar/cudaUtils.hpp"
#endif

#ifdef _KORALI_USE_ONEDNN
  #include "auxiliar/dnnUtils.hpp"
using namespace dnnl;
#endif

#include <Eigen/Dense>
using namespace Eigen;

namespace korali
{
namespace neuralNetwork
{
namespace layer
{
namespace recurrent
{
;

void GRU::initialize()
{
  Recurrent::initialize();

  // Setting number of recurrent gates to 3 (GRU)
  _gateCount = 3;

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN") _rnnMode = CUDNN_GRU;
#endif
}

void GRU::createForwardPipeline()
{
  // Calling base layer function
  Layer::createForwardPipeline();

  // Checking Layer sizes
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Checking Layer sizes
    const memory::dim T = 1;                            // time steps
    const memory::dim N = _batchSize;                   // Batch size
    const memory::dim IC = _prevLayer->_outputChannels; // channels
    const memory::dim OC = _outputChannels;             // channels
    const memory::dim L = _depth;                       // layers
    const memory::dim D = 1;                            // directions

    // Creating descriptor for layer memory
    const memory::dims layerInputDims = {T, N, IC};
    auto layerInputMemDesc = memory::desc(layerInputDims, memory::data_type::f32, memory::format_tag::tnc);

    // Creating descriptor for layer memory
    const memory::dims layerOutputDims = {T, N, OC};
    auto layerOutputMemDesc = memory::desc(layerOutputDims, memory::data_type::f32, memory::format_tag::tnc);

    // Creating descriptor for the hidden state memory
    const memory::dims stateLayerDims = {L, D, N, OC};
    auto stateMemDesc = memory::desc(stateLayerDims, memory::data_type::f32, memory::format_tag::ldnc);

    // Creating descriptor for the weights memory
    auto weightsInputMemDesc = memory::desc(_weightsInputDims, memory::data_type::f32, memory::format_tag::ldigo);
    auto weightsRecurrentMemDesc = memory::desc(_weightsRecurrentDims, memory::data_type::f32, memory::format_tag::ldigo);
    // reorder(user_weights_layer_mem, lstm_weights_layer_mem).execute(engine_stream, user_weights_layer_mem, lstm_weights_layer_mem);

    // Creating memory for the hidden state
    _hiddenStateMem.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) _hiddenStateMem[i] = memory(stateMemDesc, _nn->_dnnlEngine);

    // Crating null hidden state mems for initial timestep
    _nullStateInputMem = memory(stateMemDesc, _nn->_dnnlEngine);
    _nullStateOutputMem = memory(stateMemDesc, _nn->_dnnlEngine);

    // Setting them to zero
    std::vector<float> nullState(L * D * N * OC, 0.0f);
    write_to_dnnl_memory(nullState.data(), _nullStateInputMem);
    write_to_dnnl_memory(nullState.data(), _nullStateOutputMem);

    // Creating descriptor for the GRU operation
    auto forwardGRUDesc = gru_forward::desc(
      _propKind,                                // aprop_kind
      rnn_direction::unidirectional_left2right, // direction
      layerInputMemDesc,                        // src_layer_desc
      stateMemDesc,                             // src_iter_desc
      weightsInputMemDesc,                      // weights_layer_desc
      weightsRecurrentMemDesc,                  // weights_iter_desc
      _biasMem.get_desc(),                      // bias_desc
      layerOutputMemDesc,                       // dst_layer_desc
      stateMemDesc                              // dst_iter_desc
    );

    // Create GRU primitive descriptor.
    dnnl::primitive_attr gruPrimitiveAttributes;
    _forwardGRUPrimitiveDesc = gru_forward::primitive_desc(forwardGRUDesc, gruPrimitiveAttributes, _nn->_dnnlEngine);

    // _optimizedWeightsLayerMem = _weightsLayerMem;
    // _optimizedWeightsRecurrentMem = _weightsRecurrentMem;
    // // REORDER weights to RNN cell suggested format: the data in case the weights memory layout to optimized layout
    // if (_forwardGRUPrimitiveDesc.weights_desc() != _weightsLayerMem.get_desc()) {
    //     optimizedWeightsLayerMem = memory(_forwardGRUPrimitiveDesc.weights_desc(), engine);
    //     reorder(_weightsLayerMem, optimizedWeightsLayerMem).execute(engine_stream, _weightsLayerMem, optimizedWeightsLayerMem);
    // }
    // if (_forwardGRUPrimitiveDesc.weights_iter_desc() != _weightsRecurrentMem.get_desc()) {
    //     optimizedWeightsRecurrentMem = memory(_forwardGRUPrimitiveDesc.weights_iter_desc(), engine);
    //     reorder(user_weights_iter_mem, lstm_weights_iter_mem).execute(engine_stream, user_weights_iter_mem, lstm_weights_iter_mem);
    // }

    // Create the primitive.
    _forwardGRUPrimitive = gru_forward(_forwardGRUPrimitiveDesc);

    // Now allocating workspace
    _workspaceMem.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++)
      _workspaceMem[i] = memory(_forwardGRUPrimitiveDesc.workspace_desc(), _nn->_dnnlEngine);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // Obtaining batch size
    const size_t L = _depth;
    const size_t N = _batchSize;
    const size_t IC = _prevLayer->_outputChannels;
    const size_t OC = _outputChannels;

    int dimA[3];
    int strideA[3];

    dimA[0] = L;  // Hidden Layer count
    dimA[1] = N;  // Minibatch size
    dimA[2] = OC; // Hidden Size

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    // Allocating hidden state descriptor
    cudnnErrCheck(cudnnCreateTensorDescriptor(&_hTensorDesc));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(_hTensorDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

    // Allocating hidden state tensors
    _hStateTensor.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) cudaErrCheck(cudaMalloc((void **)&_hStateTensor[i], L * N * OC * sizeof(float)));

    // Creating RNN data descriptors for input and output
    cudnnErrCheck(cudnnCreateRNNDataDescriptor(&_inputRNNDataDesc));
    cudnnErrCheck(cudnnCreateRNNDataDescriptor(&_outputRNNDataDesc));

    // Setting and copying sequence length array to device
    std::vector<int> seqLengthArray(N, 1);
    cudaErrCheck(cudaMalloc((void **)&_devSequenceLengths, N * sizeof(int)));
    cudaErrCheck(cudaMemcpy(_devSequenceLengths, seqLengthArray.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Setting intput/output RNN data descriptors
    cudnnErrCheck(cudnnSetRNNDataDescriptor(
      _inputRNNDataDesc,
      CUDNN_DATA_FLOAT,
      CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
      1, // Max Sequence Length
      N,
      IC,
      seqLengthArray.data(),
      NULL));

    cudnnErrCheck(cudnnSetRNNDataDescriptor(
      _outputRNNDataDesc,
      CUDNN_DATA_FLOAT,
      CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
      1, // Max Sequence Length
      N,
      OC,
      seqLengthArray.data(),
      NULL));

    // Now allocating workspace
    cudnnErrCheck(cudnnGetRNNTempSpaceSizes(
      _nn->_cuDNNHandle,
      _rnnDesc,
      _forwardMode,
      _inputRNNDataDesc,
      &_workSpaceSize,
      &_reserveSpaceSize));

    _workSpaceTensor.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++) cudaErrCheck(cudaMalloc((void **)&_workSpaceTensor[t], _workSpaceSize));

    _reserveSpaceTensor.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++) cudaErrCheck(cudaMalloc((void **)&_reserveSpaceTensor[t], _reserveSpaceSize));
  }
#endif
}

void GRU::createBackwardPipeline()
{
  // Calling base layer function
  Recurrent::createBackwardPipeline();

  // Checking Layer sizes
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);
  std::exception_ptr eptr;
try{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // ==================================================================================================================
    // Checking Layer sizes
    const memory::dim T = 1;                            // time steps
    const memory::dim N = _batchSize;                   // Batch size
    const memory::dim IC = _prevLayer->_outputChannels; // channels
    const memory::dim OC = _outputChannels;             // channels
    const memory::dim L = _depth;                       // layers
    const memory::dim D = 1;                            // directions
    // Creating memory for the hidden state
    _hiddenStateGradientMem.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) _hiddenStateGradientMem[i] = memory(_hiddenStateMem[i].get_desc(), _nn->_dnnlEngine);

    // Creating descriptor for layer memory
    const memory::dims layerInputDims = {T, N, IC};
    auto layerInputMemDesc = memory::desc(layerInputDims, memory::data_type::f32, memory::format_tag::tnc);

    // Creating descriptor for layer memory
    const memory::dims layerOutputDims = {T, N, OC};
    auto layerOutputMemDesc = memory::desc(layerOutputDims, memory::data_type::f32, memory::format_tag::tnc);

    // Creating descriptor for the hidden state memory
    const memory::dims stateLayerDims = {L, D, N, OC};
    auto stateMemDesc = memory::desc(stateLayerDims, memory::data_type::f32, memory::format_tag::ldnc);
    //
    // Creating descriptor for the weights memory
    const memory::dim G = _gateCount;                   // Gates
    memory::dims weightInputDims = {L, D, IC, G, OC};
    auto weightInputMemDesc = memory::desc(weightInputDims, memory::data_type::f32, memory::format_tag::any);

    memory::dims weightRecurrentDims = {L, D, OC, G, OC};
    auto weightRecurrentMemDesc = memory::desc(weightRecurrentDims, memory::data_type::f32, memory::format_tag::any);
    //

    // Creating descriptor for the GRU operation
    auto backwardGRUDesc = gru_backward::desc(
      prop_kind::backward,                      // aprop_kind
      rnn_direction::unidirectional_left2right, // direction
      layerInputMemDesc,                        // src_layer_desc
      stateMemDesc,                             // src_iter_desc
      weightInputMemDesc,                       // weights_layer_desc DIFFERENT
      weightRecurrentMemDesc,                   // weights_iter_desc DIFFERENT
      _biasMem.get_desc(),                      // bias_desc
      layerOutputMemDesc,                       // dst_layer_desc
      stateMemDesc,                             // dst_iter_desc
      layerInputMemDesc,                        // diff_src_layer_desc
      stateMemDesc,                             // diff_src_iter_desc
      weightInputMemDesc,                       // diff_weights_layer_desc DIFFERENT
      weightRecurrentMemDesc,                   // diff_weights_iter_desc DIFFERENT
      _biasGradientMem.get_desc(),              // diff_bias_desc
      layerOutputMemDesc,                       // diff_dst_layer_desc
      stateMemDesc                              // diff_dst_iter_desc
    );

    // Create GRU primitive descriptor.
    // _backwardGRUPrimitiveDesc = gru_backward::primitive_desc(backwardGRUDesc, _nn->_dnnlEngine);
    _backwardGRUPrimitiveDesc = gru_backward::primitive_desc(backwardGRUDesc, _nn->_dnnlEngine, _forwardGRUPrimitiveDesc);

    // create the primitive.
    _backwardGRUPrimitive = gru_backward(_backwardGRUPrimitiveDesc);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // Obtaining batch size
    size_t L = _depth;
    size_t N = _batchSize;
    size_t C = _outputChannels;

    // Allocating hidden state tensors
    _hGradientTensor.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) cudaErrCheck(cudaMalloc((void **)&_hGradientTensor[i], L * N * C * sizeof(float)));
  }
#endif
  } catch (...) {
    eptr = std::current_exception();
  }
  exceptionHandler(eptr);
}

void GRU::forwardData(const size_t t)
{
  // std::raise(SIGINT);
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Configuring forward arguments
    std::unordered_map<int, memory> forwardGRUArgs;
    forwardGRUArgs.insert({DNNL_ARG_SRC_LAYER, _prevLayer->_outputMem[t]});
    forwardGRUArgs.insert({DNNL_ARG_WEIGHTS_LAYER, _weightsLayerMem});
    forwardGRUArgs.insert({DNNL_ARG_WEIGHTS_ITER, _weightsRecurrentMem});
    forwardGRUArgs.insert({DNNL_ARG_BIAS, _biasMem});
    forwardGRUArgs.insert({DNNL_ARG_DST_LAYER, _outputMem[t]});
    forwardGRUArgs.insert({DNNL_ARG_SRC_ITER, t == 0 ? _nullStateInputMem : _hiddenStateMem[t - 1]}); // Input
    forwardGRUArgs.insert({DNNL_ARG_DST_ITER, _hiddenStateMem[t]});                                   // Output
    forwardGRUArgs.insert({DNNL_ARG_WORKSPACE, _workspaceMem[t]});
    // Primitive execution
    _forwardGRUPrimitive.execute(_nn->_dnnlStream, forwardGRUArgs);

  }
#ifdef DEBUG
  auto outVec = getOutput();
  // Check for non-finite values
  for(auto& batch : outVec){
    if(std::any_of(batch.begin(), batch.end(), [](const float v) { return !std::isfinite(v);}))
      KORALI_LOG_ERROR("[Layer %zu/Type %s/Time %zu] Non-finite value inside forward output values.", _index, _type.c_str(), t);
  }
#endif
#endif




#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    const size_t N = _batchSize;

    // Creating array of sequence lengths necessary for CuDNN
    std::vector<int> seqLengthArray(N, 1);

    cudnnErrCheck(cudnnRNNForward(
      _nn->_cuDNNHandle, // handle
      _rnnDesc,          // rnnDesc
      _forwardMode,
      _devSequenceLengths,                  // devSeqLengths
      _inputRNNDataDesc,                    // xDesc
      _prevLayer->_outputTensor[t],         // x
      _outputRNNDataDesc,                   // yDesc
      _outputTensor[t],                     // y
      _hTensorDesc,                         // hDesc
      t == 0 ? NULL : _hStateTensor[t - 1], // hx
      _hStateTensor[t],                     // hy
      NULL,                                 // cDesc -- Not necessary for GRU
      NULL,                                 // cx -- Not necessary for GRU
      NULL,                                 // cy -- Not necessary for GRU
      _weightsSize,
      _weightsTensor,
      _workSpaceSize,
      _workSpaceTensor[t],
      _reserveSpaceSize,
      _reserveSpaceTensor[t]));
  }
#endif
}

void GRU::backwardData(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

  std::exception_ptr eptr;
try{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Resetting current weight gradients to zero
    std::vector<float> nullWeightLayerGradients(_weightsInputCount, 0.0f);
    std::vector<float> nullWeightRecurrentGradients(_weightsRecurrentCount, 0.0f);
    std::vector<float> nullBiasGradients(_biasCount, 0.0f);
    write_to_dnnl_memory(nullWeightLayerGradients.data(), _weightsLayerGradientMem);
    write_to_dnnl_memory(nullWeightRecurrentGradients.data(), _weightsRecurrentGradientMem);
    write_to_dnnl_memory(nullBiasGradients.data(), _biasGradientMem);
    // =============================================
    // REORDER weights to RNN cell suggested format:
    // =============================================
    // SRC and DST Layer Weights
    auto weights_layer_bwd_memory = _weightsLayerMem;
    if (_backwardGRUPrimitiveDesc.weights_layer_desc() != _forwardGRUPrimitiveDesc.weights_layer_desc()) {
        weights_layer_bwd_memory = memory(_backwardGRUPrimitiveDesc.weights_layer_desc(), _nn->_dnnlEngine);
        reorder(_weightsLayerMem, weights_layer_bwd_memory).execute(_nn->_dnnlStream, _weightsLayerMem, weights_layer_bwd_memory);
    }
    // Iter Weights for src and dst
    auto weights_iter_bwd_memory = _weightsRecurrentMem;
    if (_backwardGRUPrimitiveDesc.weights_iter_desc() != _forwardGRUPrimitiveDesc.weights_iter_desc()) {
        weights_iter_bwd_memory = memory(_backwardGRUPrimitiveDesc.weights_iter_desc(), _nn->_dnnlEngine);
        reorder(_weightsRecurrentMem, weights_iter_bwd_memory).execute(_nn->_dnnlStream, _weightsRecurrentMem, weights_iter_bwd_memory);
    }
    // Diffs
    auto reorder_diff_weights_layer = false;
    auto diff_weights_layer_memory = _weightsLayerGradientMem;
    if (_backwardGRUPrimitiveDesc.diff_weights_layer_desc() != _weightsLayerGradientMem.get_desc()) {
        diff_weights_layer_memory = dnnl::memory(_backwardGRUPrimitiveDesc.diff_weights_layer_desc(), _nn->_dnnlEngine);
        reorder(_weightsLayerGradientMem, diff_weights_layer_memory).execute(_nn->_dnnlStream, _weightsLayerGradientMem, diff_weights_layer_memory);
        reorder_diff_weights_layer = true;
    }
    auto reorder_diff_weights_iter = false;
    auto diff_weights_iter_memory = _weightsRecurrentGradientMem;
    if (_backwardGRUPrimitiveDesc.diff_weights_iter_desc() != _weightsRecurrentGradientMem.get_desc()) {
        diff_weights_iter_memory = dnnl::memory(_backwardGRUPrimitiveDesc.diff_weights_iter_desc(), _nn->_dnnlEngine);
        reorder(_weightsRecurrentGradientMem, diff_weights_iter_memory).execute(_nn->_dnnlStream, _weightsRecurrentGradientMem, diff_weights_iter_memory);
        reorder_diff_weights_iter = true;
    }
    // Configuring backward arguments
    std::unordered_map<int, memory> backwardGRUArgs;
    backwardGRUArgs.insert({DNNL_ARG_SRC_LAYER, _prevLayer->_outputMem[t]});
    backwardGRUArgs.insert({DNNL_ARG_SRC_ITER, t == 0 ? _nullStateInputMem : _hiddenStateMem[t - 1]});
    backwardGRUArgs.insert({DNNL_ARG_BIAS, _biasMem});
    backwardGRUArgs.insert({DNNL_ARG_DST_LAYER, _outputMem[t]});
    backwardGRUArgs.insert({DNNL_ARG_DST_ITER, t == _nn->_timestepCount - 1 ? _nullStateInputMem : _hiddenStateMem[t]});
    // DONE: need differently reordered memory here
    backwardGRUArgs.insert({DNNL_ARG_WEIGHTS_LAYER, weights_layer_bwd_memory});
    backwardGRUArgs.insert({DNNL_ARG_WEIGHTS_ITER, weights_iter_bwd_memory});
    // DONE: need differently reordered memory here
    backwardGRUArgs.insert({DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer_memory });
    backwardGRUArgs.insert({DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter_memory});
    //
    backwardGRUArgs.insert({DNNL_ARG_DIFF_BIAS, _biasGradientMem});
    backwardGRUArgs.insert({DNNL_ARG_DIFF_SRC_LAYER, _prevLayer->_outputGradientMem[t]});
    backwardGRUArgs.insert({DNNL_ARG_DIFF_SRC_ITER, t == 0 ? _nullStateOutputMem : _hiddenStateGradientMem[t - 1]});
    backwardGRUArgs.insert({DNNL_ARG_DIFF_DST_LAYER, _outputGradientMem[t]});
    backwardGRUArgs.insert({DNNL_ARG_DIFF_DST_ITER, t == _nn->_timestepCount - 1 ? _nullStateInputMem : _hiddenStateGradientMem[t]});
    backwardGRUArgs.insert({DNNL_ARG_WORKSPACE, _workspaceMem[t]});
    _backwardGRUPrimitive.execute(_nn->_dnnlStream, backwardGRUArgs);

    if (reorder_diff_weights_layer) {
        reorder(diff_weights_layer_memory, _weightsLayerGradientMem).execute(_nn->_dnnlStream, diff_weights_layer_memory, _weightsLayerGradientMem);
        _weightsLayerGradientMem = diff_weights_layer_memory;
    }
    if (reorder_diff_weights_iter) {
        reorder(diff_weights_iter_memory, _weightsRecurrentGradientMem).execute(_nn->_dnnlStream, diff_weights_iter_memory, _weightsRecurrentGradientMem);
        _weightsRecurrentGradientMem = diff_weights_iter_memory;
    }

  }
#endif


#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudnnErrCheck(cudnnRNNBackwardData_v8(
      _nn->_cuDNNHandle,                                         // handle
      _rnnDesc,                                                  // rnnDesc
      _devSequenceLengths,                                       // devSeqLengths
      _outputRNNDataDesc,                                        // yDesc
      _outputTensor[t],                                          // y
      _outputGradientTensor[t],                                  // dy
      _inputRNNDataDesc,                                         // xDesc
      _prevLayer->_outputGradientTensor[t],                      // dx
      _hTensorDesc,                                              // hDesc
      t == 0 ? NULL : _hStateTensor[t - 1],                      // hx
      t == _nn->_timestepCount - 1 ? NULL : _hGradientTensor[t], // dhy
      t == 0 ? NULL : _hGradientTensor[t - 1],                   // dhx
      NULL,                                                      // cDesc -- Not necessary for GRU
      NULL,                                                      // cx -- Not necessary for GRU
      NULL,                                                      // dcy -- Not necessary for GRU
      NULL,                                                      // dcx -- Not necessary for GRU
      _weightsSize,
      _weightsTensor,
      _workSpaceSize,
      _workSpaceTensor[t],
      _reserveSpaceSize,
      _reserveSpaceTensor[t]));
  }
#endif
  } catch (...) {
    eptr = std::current_exception();
  }
  exceptionHandler(eptr);
}

void GRU::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Recurrent::setConfiguration(js);
 _type = "layer/recurrent/gru";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: gru: \n%s\n", js.dump(2).c_str());
} 

void GRU::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Recurrent::getConfiguration(js);
} 

void GRU::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Recurrent::applyModuleDefaults(js);
} 

void GRU::applyVariableDefaults() 
{

 Recurrent::applyVariableDefaults();
} 

;

} //recurrent
} //layer
} //neuralNetwork
} //korali
;
