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
