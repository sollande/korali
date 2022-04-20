import pickle
import os
import shutil
import argparse
from datetime import datetime

def initialize_constants():
    global SEQUENTIAL
    SEQUENTIAL = "Sequential"
    global DISTRIBUTED
    DISTRIBUTED = "Distributed"
    global CONCURRENT
    CONCURRENT = "Concurrent"
    global CNN_AUTOENCODER
    CNN_AUTOENCODER = 'cnn-autoencoder'
    global AUTOENCODER
    AUTOENCODER = 'autoencoder'
    global SILENT
    SILENT = "Silent"
    global NORMAL
    NORMAL = "Normal"
    global DETAILED
    DETAILED = "Detailed"
    global SCRATCH
    SCRATCH = os.environ['SCRATCH'] if "SCRATCH" in os.environ else False
    global HOME
    HOME = os.environ['HOME']
    global DATE_FORMAT
    DATE_FORMAT = "%d-%m-%y-%H:%M:%S"

def min_max_scalar(arr):
    """Scales data to [0, 1] range

    :param arr:
    :returns:

    """
    return (arr - arr.min()) / (arr.max() - arr.min())

def print_header(text="", width=80, sep="=", color=None):
    """Prints header with seperator

    :param text:
    :param width:
    :param sep:
    :returns:
    """
    if len(text) == 0:
        text = sep*width
        if color:
            text = color + text + bcolors.ENDC
        print(text)
    else:
        txt_legnth = len(text)+2
        fill_width = int((width-txt_legnth)/2)
        if color:
            text = color + text + bcolors.ENDC
        print(sep*fill_width+" "+text+" "+sep*fill_width)

def print_args(d, header_text = "Running with args", color=None, width=30, header_width=80, sep="="):
    """print args from args parser formated nicely

    :param d:
    :param heder_text:
    :param width:
    :param header_width:
    :param sep:
    :returns:
    """
    print_header(header_text, color=color)
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
    """TODO describe function

    :param stride:
    :param image_size:
    :param filter_size:
    :returns:

    """
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

def make_test_data(input_path, output_path, samples):
    """Helper function to create test file

    :param input_path:
    :param output_path:
    :param samples: number of samples to keep
    :returns:

    """
    with open(input_path, "rb") as file:
        data = pickle.load(file)
        assert samples < len(data["trajectories"])
        trajectories = data["trajectories"][:samples]
        del data
    with open(output_path, "wb") as file:
        pickle.dump(trajectories, file)


class bcolors:
    """Helper function to print colored output:
    Example: print(bcolors.WARNING + "Warning" + bcolors.ENDC)
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def exp_dir_str(args):
    """returns experiments output file path relative to the current script directory
    :param args: argument parser
    :returns: path
    """
    return os.path.join("_korali_result", args.model, f"lat{args.latent_dim}", args.output_dir_append)


def make_parser():
    """Create the argument parser:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--engine", help="NN backend to use", default="OneDNN", required=False
    )
    parser.add_argument(
        "--max-generations",
        help="Maximum Number of generations to run",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--optimizer",
        help="Optimizer to use for NN parameter updates",
        default="Adam",
        required=False,
    )
    parser.add_argument(
        "--data-path",
        help="Path to the training data",
        default="./_data/data.pickle",
        required=False,
    )
    parser.add_argument(
        "--test-path",
        help="Path to the test training data",
        default="./_data/test.pickle",
        required=False,
    )
    parser.add_argument(
        "--test-file",
        help="Filename for testing error",
        default="testing_error.txt",
        required=False,
    )
    parser.add_argument(
        "--train-split", help="If 0<--train-split<=1 fraction of training samples; \
        else number of training samples",
        default=6*128,
        required=False,
        type=float
    )
    parser.add_argument(
        "--learning-rate",
        help="Learning rate for the selected optimizer",
        default=0.0001,
        required=False,
        type=float
    )
    parser.add_argument(
        "--decay",
        help="Decay of the learning rate.",
        default=0.0001,
        required=False,
        type=float
    )
    parser.add_argument(
        "--training-batch-size",
        help="Batch size to use for training data; must divide the --train-split",
        type=int,
        default=128,
        required=False,
    )
    parser.add_argument(
        "--batch-concurrency",
        help="Batch Concurrency for the minibatches",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--output-dir-append",
        help="string that can be used add an outputfolder i.e. for a date",
        default="",
        required=False,
    )
    parser.add_argument("--epochs", help="Number of epochs", default=100, type=int, required=False)
    parser.add_argument(
        "--latent-dim",
        help="Latent dimension of the encoder",
        default=10,
        required=False,
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32, 36, 40, 64]
    )
    parser.add_argument(
        "--conduit",
        help="Conduit to use",
        choices=[SEQUENTIAL, CONCURRENT, DISTRIBUTED],
        default="Sequential",
        required=False,
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--file-output", action="store_true")
    parser.add_argument(
        "--frequency",
        help="Change output frequency [per generation]",
        default=10,
        type=int,
        required=False,
    )
    parser.add_argument('--model',
                        choices=[AUTOENCODER, CNN_AUTOENCODER],
                        help='Model to use.', default=AUTOENCODER)
    parser.add_argument('--verbosity',
                        choices=[SILENT, NORMAL, DETAILED],
                        help='Verbosity Level', default="Detailed")
    parser.add_argument(
        "--plot",
        help="Indicates whether to plot results after testing",
        default=False,
        required=False,
    )
    return parser
def get_newest_dir(dest, format):
    """Returns the newst dir path for a folder of dirs consisting of dates
    :param dest: folder of dires in with dates as names
    :returns: path

    """
    files = os.listdir(dest)
    dates = [datetime.strptime(f, format) for f in files]
    indices = [i[0] for i in sorted(enumerate(dates), key=lambda x:x[1])]
    return os.path.join(dest, files[indices[-1]])

def move_dir(src, dest):
    """Move all files from one directory to another

    :param src: source directory
    :param dest: destination directory

    """
    files = os.listdir(src)
    for f in files:
        shutil.move(os.path.join(src, f), os.path.join(dest, f))


def copy_dir(src, dest):
    """Copy all files from one directory to another

    :param src: source directory
    :param dest: destination directory

    """
    shutil.copytree(src, dest, dirs_exist_ok=True)


def mkdir_p(dir):
    """make a directory if it doesn't exist and create intermediates as well
    """
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
