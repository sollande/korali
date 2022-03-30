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

def make_cnn_autencoder_experiment(e, latentDim, imgWidth, imgHeight, inputChannels = 1):
    """Configures one cnn autoencoder experiment
    :param e: korali experiment
    :param latentDim: encoding dimension
    :param img_height: input/output image height
    :param img_width: input/output image height
    :param inputChannels: number of input channels i.e. RGB
    TODO: fix
    """
    assert imgWidth==2*imgHeight, "Image width should be twice the Image height"
    encodingLayers = int(math.log2(imgHeight)-1)
    encodingCNNLayers = encodingLayers-1
    e["Solver"]["Type"] = "Learner/DeepSupervisor"
    e["Solver"]["Loss Function"] = "Mean Squared Error"
    e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
    e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
    hl = e["Solver"]["Neural Network"]["Hidden Layers"]
    kernelSizeConv = 13
    paddingConv = 6
    strideConv = 1
    # 20 output channels and for the last layer 2 output channels
    outputChannelsConv = (encodingLayers-1)*[20]
    outputChannelsConv.append(2)
    outputChannelsDeconv = (encodingLayers-1)*[20]
    outputChannelsDeconv.append(inputChannels)
    kernelSizeActivat = 2
    paddingActivat = 6
    strideActivat = 2
    stepsPerCNNLayer  = 3
    totalCNNLayers = stepsPerCNNLayer*encodingCNNLayers
    ffnLayers = 2
    stepsPerFFLayer = 2
    totalEncodingLayers = totalCNNLayers+ffnLayers*stepsPerFFLayer
    stepsPerDeCNNLayer  = 2
    totalDeCNNLayers = stepsPerDeCNNLayer*encodingCNNLayers
    # Idea create a list of hl
    # ==================================================================
    # ENCODER ==========================================================
    # ==================================================================
    for idx, l in enumerate(range(0, totalCNNLayers, stepsPerCNNLayer), 1):
        ## Convolution =======================
        hl[l]["Type"] = "Layer/Convolution"
        hl[l]["Image Height"]      = imgWidth
        hl[l]["Image Width"]       = imgHeight
        hl[l]["Padding Left"]      = paddingConv
        hl[l]["Padding Right"]     = paddingConv
        hl[l]["Padding Top"]       = paddingConv
        hl[l]["Padding Bottom"]    = paddingConv
        hl[l]["Kernel Height"]     = kernelSizeConv
        hl[l]["Kernel Width"]      = kernelSizeConv
        hl[l]["Vertical Stride"]   = strideConv
        hl[l]["Horizontal Stride"] = strideConv
        hl[l]["Output Channels"]   = imgWidth*imgHeight*outputChannelsConv[l]
        ## Batch Normalization ===============
        ## TODO
        imgHeight=imgHeight/(idx*2)
        imgWidth=imgWidth/(idx*2)
        ## Pooling ===========================
        hl[l+1]["Type"] = "Layer/Pooling"
        hl[l+1]["Function"]          = "Exclusive Average"
        hl[l+1]["Image Height"]      = imgWidth
        hl[l+1]["Image Width"]       = imgHeight
        hl[l+1]["Kernel Height"]     = kernelSizeActivat
        hl[l+1]["Kernel Width"]      = kernelSizeActivat
        hl[l+1]["Vertical Stride"]   = strideActivat
        hl[l+1]["Horizontal Stride"] = strideActivat
        hl[l+1]["Padding Left"]      = paddingActivat
        hl[l+1]["Padding Right"]     = paddingActivat
        hl[l+1]["Padding Top"]       = paddingActivat
        hl[l+1]["Padding Bottom"]    = paddingActivat
        hl[l+1]["Output Channels"]   = imgWidth*imgHeight*outputChannelsConv[l]
        ## Activation ========================
        hl[l+2]["Type"] = "Layer/Activation"
        hl[l+2]["Function"] = "Elementwise/ReLU"
    # Linear Layer =============================
    hl[totalCNNLayers]["Type"] = "Layer/Linear"
    hl[totalCNNLayers]["Output Channels"] = imgHeight*imgWidth*latentDim
    ## Activation ========================
    hl[totalCNNLayers+1]["Type"] = "Layer/Activation"
    hl[totalCNNLayers+1]["Function"] = "Elementwise/ReLU"
    # ==================================================================
    # Decoder ==========================================================
    # ==================================================================
    # Linear Layer ============================= [latentDim]->[2*4*8=64]
    hl[totalCNNLayers+2]["Type"] = "Layer/Linear"
    hl[totalCNNLayers+2]["Output Channels"] = imgWidth*imgHeight*outputChannelsConv[-1]
    ## Activation ========================
    hl[totalCNNLayers+3]["Type"] = "Layer/Activation"
    hl[totalCNNLayers+3]["Function"] = "Elementwise/ReLU"
    for idx, l in enumerate(range(totalEncodingLayers, totalEncodingLayers+totalDeCNNLayers, stepsPerDeCNNLayer), 1):
        imgHeight=imgHeight*(idx*2)
        imgWidth=imgWidth*(idx*2)
        ## De-convolution =======================
        hl[l]["Type"] = "Layer/Deconvolution"
        hl[l]["Image Height"]      = imgWidth
        hl[l]["Image Width"]       = imgHeight
        hl[l]["Padding Left"]      = paddingConv
        hl[l]["Padding Right"]     = paddingConv
        hl[l]["Padding Top"]       = paddingConv
        hl[l]["Padding Bottom"]    = paddingConv
        hl[l]["Kernel Height"]     = kernelSizeConv
        hl[l]["Kernel Width"]      = kernelSizeConv
        hl[l]["Vertical Stride"]   = strideConv
        hl[l]["Horizontal Stride"] = strideConv
        hl[l]["Output Channels"]   = imgWidth*imgHeight*outputChannelsDeconv[l]
        ## Batch Normalization ===============
        ## TODO
        ## Activation ========================
        hl[l+1]["Type"] = "Layer/Activation"
        hl[l+1]["Function"] = "Elementwise/ReLU"
