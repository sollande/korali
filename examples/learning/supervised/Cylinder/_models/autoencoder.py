import math

def configure_autencoder(e, img_width, img_height, channels, latentDim):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param latentDim: encoding dimension
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    input_size = output_size = img_width*img_height*channels
    img_height_red = img_height/2
    img_width_red = img_width/2
    # ===================== Input Layer
    e["Problem"]["Input"]["Size"] = input_size
    # ===================== Down Sampling
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Resampling Type"] = "Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Resampling"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Width"] = img_width
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Image Height"] = img_height
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Width"] = img_width_red
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Height"] = img_height_red
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = channels*img_height_red*img_width_red
    # ===================== Encoder
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = latentDim
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"] = "Elementwise/ReLU"
    ##  =================== Decoder
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"] = channels*img_height_red*img_width_red
    # ===================== Up Sampling
    e["Solver"]["Neural Network"]["Output Layer"]["Resampling Type"] = "Linear"
    e["Solver"]["Neural Network"]["Output Layer"]["Type"] = "Layer/Resampling"
    e["Solver"]["Neural Network"]["Output Layer"]["Image Width"] = img_width_red
    e["Solver"]["Neural Network"]["Output Layer"]["Image Height"] = img_height_red
    e["Solver"]["Neural Network"]["Output Layer"]["Output Width"] = img_width
    e["Solver"]["Neural Network"]["Output Layer"]["Output Height"] = img_height
    e["Solver"]["Neural Network"]["Output Layer"]["Output Channels"] = channels*img_width*img_height
    # Activation ========================
    e["Solver"]["Neural Network"]["Output Activation"] = "Elementwise/Logistic"
    e["Problem"]["Solution"]["Size"] = output_size
