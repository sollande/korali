def print_header(text="", width=80, sep="="):
    if len(text) == 0:
        print(sep*width)
    else:
        txt_legnth = len(text)+2
        fill_width = int((width-txt_legnth)/2)
        print(sep*fill_width+" "+text+" "+sep*fill_width)

def print_args(d, heder_text = "Running with args", width=30, header_width=80, sep="="):
   print_header(heder_text)
   for key, value in d.items():
      # print('\t' * indent + str(key))
      # if isinstance(value, dict):
      #    pretty(value, indent+1)
      # else:
         # print('\t' * (indent+1) + str(value))
      out_string = '\t{:<{width}} {:<}'.format(key, value, width=width)
      print(out_string)
   print_header()

def get_output_dim(I, P1, P2, K, S):
    img = (I+P1+P2-K)
    if img % 2 == 1:
        raise ValueError(
            "(I+P1+P2-K) has to be divisible by K ({:}+{:}+{:}-{:})/{:}"
            .format(I, P1, P2, K, S))
    return int((I+P1+P2-K)/S+1)

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

