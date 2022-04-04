import math
def make_autencoder_experiment(e, latentDim, img_height, img_width):
    """Configures one hidden layer autoencoder
    :param e: korali experiment
    :param latentDim: encoding dimension
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    # ===================== Encoder
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = latentDim
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/ReLU"
    ##  =================== Decoder
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = (
        img_height * img_width
    )
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Logistic"

def make_cnn_autencoder_experiment(e, latentDim, img_width, img_height, inputChannels = 1):
    """Configures one cnn autoencoder experiment
    Halfs the img_width and img_size till we reach the encoding dimension
    :param e: korali experiment
    :param latentDim: encoding dimension
    :param img_height: input/output image height
    :param img_width: input/output image height
    :param inputChannels: number of input channels i.e. RGB
    """
    assert img_width==2*img_height, "Image width should be twice the Image height"
    encodingLayers = int(math.log2(img_height)-1)
    encodingCNNLayers = encodingLayers-1
    kernelSizeConv = 13
    paddingConv = 6
    strideConv = 1
    # 20 output channels and for the last layer 2 output channels
    outputChannelsConv = (encodingLayers-1)*[20]
    outputChannelsConv.append(2)
    outputChannelsDeconv = (encodingLayers-1)*[20]
    outputChannelsDeconv.append(inputChannels)
    kernelSizePooling = 2
    paddingPooling = 6
    stridePooling = 2
    stepsPerCNNLayer  = 3
    totalCNNLayers = stepsPerCNNLayer*encodingCNNLayers
    ffnLayers = 2
    stepsPerFFLayer = 2
    totalEncodingLayers = totalCNNLayers+ffnLayers*stepsPerFFLayer
    stepsPerDeCNNLayer  = 3
    totalDeCNNLayers = stepsPerDeCNNLayer*encodingCNNLayers
    # Idea create a list of hl
    # ====================================================================================================
    # ENCODER ==========================================================
    # ====================================================================================================
    for idx, l in enumerate(range(0, totalCNNLayers, stepsPerCNNLayer), 1):
        ## Convolution ==========================================================
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Type"] = "Layer/Convolution"
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Image Height"]      = img_width
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Image Width"]       = img_height
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Padding Left"]      = paddingConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Padding Right"]     = paddingConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Padding Top"]       = paddingConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Padding Bottom"]    = paddingConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Kernel Height"]     = kernelSizeConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Kernel Width"]      = kernelSizeConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Vertical Stride"]   = strideConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Horizontal Stride"] = strideConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Output Channels"]   = img_width*img_height*outputChannelsConv[idx]
        ## Batch Normalization ==========================================================
        ## TODO
        ## Pooling ==========================================================
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Type"] = "Layer/Pooling"
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Function"]          = "Exclusive Average"
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Image Height"]      = img_width
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Image Width"]       = img_height
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Kernel Height"]     = kernelSizePooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Kernel Width"]      = kernelSizePooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Vertical Stride"]   = stridePooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Horizontal Stride"] = stridePooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Padding Left"]      = paddingPooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Padding Right"]     = paddingPooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Padding Top"]       = paddingPooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Padding Bottom"]    = paddingPooling
        img_height=img_height/2
        img_width=img_width/2
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Output Channels"]   = img_width*img_height*outputChannelsConv[idx]
        ## Activation ==========================================================
        e["Solver"]["Neural Network"]["Hidden Layers"][l+2]["Type"] = "Layer/Activation"
        e["Solver"]["Neural Network"]["Hidden Layers"][l+2]["Function"] = "Elementwise/ReLU"
    # ====================================================================================================
    # Linear Layerrs =============================
    e["Solver"]["Neural Network"]["Hidden Layers"][totalCNNLayers]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][totalCNNLayers]["Output Channels"] = img_height*img_width*latentDim
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][totalCNNLayers+1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][totalCNNLayers+1]["Function"] = "Elementwise/ReLU"
    # ====================================================================================================
    # Decoder 
    # Linear Layer 
    # ====================================================================================================
    e["Solver"]["Neural Network"]["Hidden Layers"][totalCNNLayers+2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][totalCNNLayers+2]["Output Channels"] = img_width*img_height*outputChannelsConv[-1]
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][totalCNNLayers+3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][totalCNNLayers+3]["Function"] = "Elementwise/ReLU"
    ## De-onvolution ==========================================================
    for idx, l in enumerate(range(totalEncodingLayers, totalEncodingLayers+totalDeCNNLayers, stepsPerDeCNNLayer), 1):
        ## De-pooling ==============================================================
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Type"] = "Layer/Deconvolution"
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Image Height"]      = img_width
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Image Width"]       = img_height
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Padding Left"]      = paddingPooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Padding Right"]     = paddingPooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Padding Top"]       = paddingPooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Padding Bottom"]    = paddingPooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Kernel Height"]     = kernelSizePooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Kernel Width"]      = kernelSizePooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Vertical Stride"]   = stridePooling
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Horizontal Stride"] = stridePooling
        img_height=img_height*2
        img_width=img_width*2
        e["Solver"]["Neural Network"]["Hidden Layers"][l]["Output Channels"]   = img_width*img_height*outputChannelsDeconv[idx]
        ## De-convolution ==============================================================
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Type"] = "Layer/Deconvolution"
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Image Height"]      = img_width
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Image Width"]       = img_height
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Padding Left"]      = paddingConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Padding Right"]     = paddingConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Padding Top"]       = paddingConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Padding Bottom"]    = paddingConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Kernel Height"]     = kernelSizeConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Kernel Width"]      = kernelSizeConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Vertical Stride"]   = strideConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Horizontal Stride"] = strideConv
        e["Solver"]["Neural Network"]["Hidden Layers"][l+1]["Output Channels"]   = img_width*img_height*outputChannelsDeconv[idx]
        ## Batch Normalization =========================================================
        ## TODO
        ## Activation ==================================================================
        e["Solver"]["Neural Network"]["Hidden Layers"][l+2]["Type"] = "Layer/Activation"
        e["Solver"]["Neural Network"]["Hidden Layers"][l+2]["Function"] = "Elementwise/ReLU"

