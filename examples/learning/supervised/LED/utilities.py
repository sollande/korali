#!/usr/bin/env ipython
def getSamePadding(stride, image_size, filter_size):
    # Input image (W_i,W_i)
    # Output image (W_o,W_o) with W_o = (W_i - F + 2P)/S + 1
    # W_i == W_o -> P = ((S-1)W + F - S)/2
    S = stride
    W = image_size  # width or height
    F = filter_size
    half_pad = int((S - 1) * W - S + F)
    if half_pad % 2 == 1:
        raise ValueError(
            "(S-1) * W  - S + F has to be divisible by two ({:}-1)*{:} - {:} + {:} = {:}"
            .format(S, W, S, F, half_pad))
    else:
        pad = int(half_pad / 2)
    if (pad > image_size / 2):
        raise ValueError(
            "Very large padding P={:}, compared to input width {:}. Reduce the strides."
            .format(pad, image_size))
    return pad

def printArchitecture(input, CNN_channels, CNN_output, kernelDim_CNN, kernelStrides_CNN, paddingCNN,\
                      pooling_channels, pooling_output, kernelDim_pooling, kernelStrides_pooling, padding_pooling,
                      activation, type):
        type = "["+type+"]"
        input_height = [v for k,v in input.items() if k=="height"]
        input_width = [v for k,v in input.items() if k=="width"]
        CNN_height = [[v for k,v in ele.items() if k=="height"][0] for ele in CNN_output]
        CNN_width = [[v for k,v in ele.items() if k=="width"][0] for ele in CNN_output]
        CNN_filter_height = [[v for k,v in ele.items() if k=="height"][0] for ele in kernelDim_CNN]
        CNN_filter_width = [[v for k,v in ele.items() if k=="width"][0] for ele in kernelDim_CNN]
        CNN_strides_height = [[v for k,v in ele.items() if k=="height"][0] for ele in kernelStrides_CNN]
        CNN_strides_width = [[v for k,v in ele.items() if k=="width"][0] for ele in kernelStrides_CNN]
        CNN_padding = [[v for k,v in ele.items() if k=="top"][0] for ele in paddingCNN]

        pooling_height = [[v for k,v in ele.items() if k=="height"][0] for ele in pooling_output]
        pooling_width = [[v for k,v in ele.items() if k=="width"][0] for ele in pooling_output]
        pooling_filter_height = [[v for k,v in ele.items() if k=="height"][0] for ele in kernelDim_pooling]
        pooling_filter_width = [[v for k,v in ele.items() if k=="width"][0] for ele in kernelDim_pooling]
        pooling_strides_height = [[v for k,v in ele.items() if k=="height"][0] for ele in kernelStrides_pooling]
        pooling_strides_width = [[v for k,v in ele.items() if k=="width"][0] for ele in kernelStrides_pooling]
        pooling_padding = [[v for k,v in ele.items() if k=="top"][0] for ele in padding_pooling]
        print("{:<10} -------------------".format(type))
        print("{:<10}CNN Kernel Stride Padding".format(type))
        print("{:<10}Pooling Type".format(type))
        print("{:<10}Pooling Kernel Stride Padding".format(type))
        print("{:<10}Output Dim".format(type))
        print("{:<10} -------------------".format(type))
        print("{:<10}{:<10} {:}x{:}".format(type,"Input", input_width, input_height))
        for cnn_c, cnn_w, cnn_h, cnn_f_w, cnn_f_h, cnn_s_w, cnn_s_h, cnn_p,\
            pooling_c, pooling_w, pooling_h, pooling_f_w, pooling_f_h, pooling_s_w, pooling_s_h, pooling_p,\
            a in \
        zip(CNN_channels, CNN_width, CNN_height,
            CNN_filter_width, CNN_filter_height, CNN_strides_width, CNN_strides_height, CNN_padding,
            pooling_channels, pooling_width, pooling_height,
            pooling_filter_width, pooling_filter_height, pooling_strides_width, pooling_strides_height, pooling_padding,
            activation):
            print("{:<10}{:<10} [{:<2} {:<3}x {:<3}]{:8}{:^}x{:^}{:5}{:<},{:<}{:5}{:<}".\
                  format(type, "CNN", cnn_c, cnn_w, cnn_h,"",cnn_f_w, cnn_f_h,"", cnn_s_w, cnn_s_h,"", cnn_p))
            print("{:<10}{:<10} [{:<2} {:<3}x {:<3}]{:8}{:^}x{:^}{:5}{:<},{:<}{:5}{:<}".\
                  format(type, "Pooling", pooling_c, pooling_w, pooling_h,"", pooling_f_w, pooling_f_h,"", pooling_s_w, pooling_s_h,"", pooling_p))
            print("{:<10}{:<10} {:}".format(type, "Activation", a))
        print("{:<10} -------------------".format(type))
