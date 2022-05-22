"""
Includes lots of helper and utility functions.

Defines:
    - argparser
    - global constansts
    - print functinos
    - copy/move functions
    - ...
"""
import pickle
import os
import shutil
import argparse
from datetime import datetime
import h5py
from pathlib import Path
from torch.utils import data
import numpy as np


def initialize_constants():
    """Global shared variables."""
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
    global TEST1000
    TEST1000 = "test1000"
    global TEST128
    TEST128 = "test128"
    global RE1000
    RE1000 = "RE1000"
    global RE100
    RE100 = "RE100"
    global PATH_DATA_TEST_1000
    BASE_PROJECT_PATH = os.path.normpath("/project/s929/pollakg/cyl/_data")
    PATH_DATA_TEST_1000 = os.path.join(BASE_PROJECT_PATH, "data.pickle")
    global PATH_DATA_TEST_128
    PATH_DATA_TEST_128 = os.path.join(BASE_PROJECT_PATH, "test.pickle")
    global PATH_DATA_RE100
    PATH_DATA_RE100 = "/project/s929/pollakg/cyl/cylRe100HR/Data/"
    global PATH_DATA_RE1000
    PATH_DATA_RE1000 = "/project/s929/pollakg/cyl/cylRe1000HR/Data/test/"


def min_max_scalar(arr):
    """Scales data to [0, 1] range.

    :param arr:
    :returns:

    """
    return (arr - arr.min()) / (arr.max() - arr.min())


def print_header(text="", width=80, sep="=", color=None):
    """Print header with seperator.

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


def print_args(d, header_text="Running with args", color=None, width=30, header_width=80, sep="="):
    """Print args from args parser formated nicely.

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
        #     print('\t' * (indent+1) + str(value))
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
    """TODO describe function.

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
    """Help to create test data file.

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
    """Helper function to print colored output.

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
    """Return experiments output file path relative to the current script directory.

    :param args: argument parser
    :returns: path
    """
    return os.path.join("_korali_result", args.model, f"lat{args.latent_dim}", args.output_dir_append)


def make_parser():
    """Create the argument parser."""
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
        "--data-type",
        help="Type of data to use.",
        default=RE100,
        choices=[TEST128, TEST1000, RE100, RE1000],
        required=False,
    )
    parser.add_argument(
        "--data-path",
        help="Custom Path to data",
        default="",
        required=False,
    )
    parser.add_argument(
        "--result-file",
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
        default=4,
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


def get_prediction(X):
    """Return the output of the last timestep.

    :param X: list of samples X[sample][timesteps][features]
    :returns: list of X[time][-1][features]
    """
    return [x[-1] for x in X]


def get_minibatch(X, step, batchsize):
    """Return a minibatch from data.

    :param X: list of samples X[sample][timesteps][features]
    :param step: step of epoch
    :param: batchsize
    :returns: next minibatch
    """
    assert((step + 1) * batchsize <= len(X))
    return X[step * batchsize: (step + 1) * batchsize]


def get_newest_dir(dest, format):
    """Return the newst dir path for a folder of dirs consisting of dates.

    :param dest: folder of dires in with dates as names
    :returns: path

    """
    files = os.listdir(dest)
    dates = [datetime.strptime(f, format) for f in files]
    indices = [i[0] for i in sorted(enumerate(dates), key=lambda x:x[1])]
    return os.path.join(dest, files[indices[-1]])


def move_dir(src, dest):
    """Move all files from one directory to another.

    :param src: source directory
    :param dest: destination directory

    """
    files = os.listdir(src)
    for f in files:
        shutil.move(os.path.join(src, f), os.path.join(dest, f))


def copy_dir(src, dest):
    """Copy all files from one directory to another.

    :param src: source directory
    :param dest: destination directory

    """
    shutil.copytree(src, dest, dirs_exist_ok=True)


def mkdir_p(dir):
    """Make a directory if it doesn't exist and create intermediates as well."""
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


class PreProcessor():
    """Base Class for preporcessing.

    Attributes:
        self.args: is the type of data we want to load.
    """

    def __init__(self, args):
        self.args = args

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        pass


class DataLoader():
    """Container to load and hold Datasets.

    Attributes:
        data_type: is the type of data we want to load.
        X_test: test data
        y_test: testing labels
        X_train: training data
        y_train: training labels
        X_val: validation set
        y_val: validation labels
        training_batch_size: batch size of the training data
        train_split: train split for pickle data
        TIMEINDEX: timeindex if we only want one timepoint for training/testing.
        dim: shape of the features.
    """

    def __init__(self, args, TIMEINDEX=0):
        """Set attributes.

        :param args: argparser object.
                    Needs at least the data_type to be loaded.
        """
        self.data_type = args.data_type
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_split = args.train_split
        self.TIMEINDEX = TIMEINDEX
        self.dim = None
        self.training_batch_size = args.training_batch_size

    def load_data(self):
        """Load desired data.

        :returns: trajectories that we work with.

        """
        if self.data_type in (TEST128, TEST1000):
            self.load_pickle()
        elif self.data_type in (RE100, RE1000):
            self.load_hdf5()

    def add_time_dimension(self, X):
        """Add time dimension to list of sample.

        Useful for samples without time dimension.
        :param X: list of samples X[sample]
        :returns: list of X[time][sample]
        """
        return [[x] for x in X]

    def load_pickle(self):
        """Load pickle test data.

        Loads test trajectories and splits them into test and training sets.

        """
        if self.data_type == TEST128:
            # Load 128 test sample file
            with open(PATH_DATA_TEST_128, "rb") as file:
                trajectories = pickle.load(file)
        elif self.data_type == TEST1000:
            # Load 1000 sample file
            with open(PATH_DATA_TEST_1000, "rb") as file:
                data = pickle.load(file)
                trajectories = data["trajectories"]
                del data
        samples, img_height, img_width = np.shape(trajectories)
        # timesteps, channels, img_height, img_width
        """ flatten images:
            [samples][imd_height][img_width]=>[samples][img_height x img_width]
            64x32 => 204
        """
        trajectories = np.reshape(trajectories, (samples, -1))
        idx = np.random.permutation(samples)
        if self.train_split >= 1:
            nb_train_samples = int(self.train_split)
        else:
            nb_train_samples = int(samples * self.train_split)
        train_idx = idx[: nb_train_samples]
        X_train = trajectories[train_idx].tolist()
        X_test = trajectories[~train_idx].tolist()
        self.X_train = self.add_time_dimension(X_train)
        self.X_test = self.add_time_dimension(X_test)
        self.dim = (1, 1, img_height, img_width)

    def load_hdf5(self):
        """Load hdf5 data set and convert it to a list.

        size: [sample][time][features].
        """
        if self.data_type == RE100:
            self.X_test = HDF5Dataset(os.path.join(PATH_DATA_RE100, "test"))
            self.X_train = HDF5Dataset(os.path.join(PATH_DATA_RE100, "train"))
            self.X_val = HDF5Dataset(os.path.join(PATH_DATA_RE100, "val"))
            channels = self.X_train.channels
            img_width = self.X_train.img_width
            img_height = self.X_train.img_height
            if self.TIMEINDEX is not None:
                timesteps = 1
                self.X_test = self.X_test.tolist(self.TIMEINDEX)
                self.X_train = self.X_train.tolist(self.TIMEINDEX)
                self.X_val = self.X_val.tolist(self.TIMEINDEX)
            else:
                self.X_test = self.X_test.tolist()
                self.X_train = self.X_train.tolist()
                self.X_val = self.X_val.tolist()
                timesteps = self.X_train.num_timesteps

            assert self.X_test[0][0].shape == self.X_train[0][0].shape\
                and self.X_test[0][0].shape == self.X_val[0][0].shape,\
                "Dimensions of test, train and val data sets are non-matching"
            self.dim = (timesteps, channels, img_width, img_height)

        elif self.data_type == RE1000:
            pass

    def fetch_data(self):
        """Load desired data.

        Sets:
        - X_test
        - X_train
        - X_val (if available)
        """
        self.load_data()
        assert len(self.X_train) % self.training_batch_size == 0, \
            "Batch Size {} must divide the number of training samples {}"\
            .format(self.training_batch_size, len(self.X_train))

    def __len__(self):
        """Return the number of samples."""
        return self.samples

    @property
    def shape(self):
        """Return feature shape."""
        return self.dim


class HDF5Dataset(data.Dataset):
    """Container to load and hold HDF5 Datasets.

    Adapted from: Vlachas Pantelis, CSE-lab, ETH Zurich
    Attributes:
        seq_paths: is a list of size samples that holds for each sample:
        - gname_seq: the name of the current sample
        - timesteps: the idx names of the timesteps
        - idx: the index of the current sample
        - num_timesteps: the number of timesps
        - channels: number of channels i.e. rgb
        - img_width
        - img_height
        h5_file: holds h5py objet.
    """

    def __init__(self, file_path):
        """Load the first sorted <file>.h5 file for a given folder path.

        :param file_path: folder to path with <file>.h5 files.
        :returns:

        """
        super().__init__()

        self.seq_paths = []

        # Search for all h5 files
        p = Path(file_path)
        assert p.is_dir(), "Path to data files {:} is not found.".format(p)
        files = sorted(p.glob('*.h5'))
        if len(files) != 1:
            raise RuntimeError(
                '[utils_data] No or more than one hdf5 datasets found')
        file_path = files[0]
        self.h5_file = h5py.File(file_path, 'r')

        """ Adding the sequence paths """
        self._add_seq_paths()
        self.start = 0
        self.end = len(self)
        self.channels, self.img_width, self.img_height = np.shape(self[0][0])

    class HDF5DatasetWrapper():
        """Wrapper Class for better item access.

        Allows us to access items as data[sample][time]
        wher both sample and time are integers.
        Gets returned from __getitem__ of HDF5Dataset.
        """

        def __init__(self, idx, hdf5, seq_paths):
            """Constucts object.

            Attributes:
                :sample_idx: sample integer idx
                :gname_seq: sample name string
                :sample: sample of given gname_seq
                :hdf5: current dataset i.e. self.h5_file
                            of the outer class.
                :seq_paths: self.seq_path of the outer class.

            """
            self.seq_paths = seq_paths
            self.sample_idx = idx
            self.hdf5 = hdf5
            self.gname_seq = self.sample_name_from_idx(idx)
            self.sample = self.hdf5[self.gname_seq]
            self.start = 0
            self.end = len(self)

        def __getitem__(self, idx):
            """Get self.h5_file[sample][time].

            :param idx: time index
            :returns: self.h5_file[sample][time]["data"]

            """
            assert idx >= 0 and idx < len(self),\
                f"time index must be valid 0<{idx}<={len(self)} must be inside valid range"
            key_list = list(self.sample.keys())
            key_from_int = key_list[idx]
            return self.sample[key_from_int]["data"]

        def sample_name_from_idx(self, idx):
            """Return the name of the sampel with index idx.

            Each sample is stored as a hdf5 group.
            In order to access samples by index,
            we need to convert the index of a sample to the name of the group.

            """
            return self.seq_paths[idx]["gname_seq"]

        def __delete__(self):
            """deltes/closes the current h5 file."""
            self.hdf5.close()

        def __len__(self):
            """Return the number of timesteps."""
            return self.seq_paths[self.sample_idx]["num_timesteps"]

        def __iter__(self):
            """Implementd by next."""
            return self

        def __next__(self):
            """Implement iterator.

            :returns: the next element.
            """
            if self.start < self.end:
                ret = self[self.start]
                self.start += 1
                return ret
            else:
                raise StopIteration

    def __delete__(self):
        """deltes/closes the current h5 file."""
        self.h5_file.close()

    def __len__(self):
        """Return the number of timesteps."""
        return len(self.seq_paths)

    def _add_seq_paths(self):
        """Add sample to seq_path.

        :returns:

        """
        idx = 0
        number_of_loaded_groups = 0
        for gname_seq, group_seq in self.h5_file.items():
            timesteps = []
            idx_time = 0
            number_of_loaded_timesteps = 0
            for gname_time, group_time in group_seq.items():
                # print(gname_time)
                timesteps.append(gname_time)
                number_of_loaded_timesteps += 1
                idx_time += 1
            assert len(timesteps) == number_of_loaded_timesteps

            # print(timesteps)
            self.seq_paths.append({
                'gname_seq': gname_seq,
                'timesteps': timesteps,
                'idx': idx,
                'num_timesteps': len(timesteps),
            })
            number_of_loaded_groups += 1
            idx += 1

        print("[utils_data] Loader restricted to {:}/{:} timesteps.".format(
            number_of_loaded_timesteps, idx_time))
        print("[utils_data] Loader restricted to {:}/{:} samples.".format(
            number_of_loaded_groups, idx))
        assert len(self.seq_paths) == number_of_loaded_groups

    def __getitem__(self, idx):
        """Return sample/group.

        :param idx: index of group 0<=idx<samples
        """
        return self.HDF5DatasetWrapper(idx, self.h5_file, self.seq_paths)

    def __iter__(self):
        """Implementd by next."""
        return self

    def __next__(self):
        """Implement iterator.

        :returns: the next element.
        """
        if self.start < self.end:
            ret = self[self.start]
            self.start += 1
            return ret
        else:
            raise StopIteration

    def tolist(self, timestep="all"):
        """Convert hdf5 data set to a list.

        :param timestep: timesteps to use.
        :returns a list
        """
        samples = []
        for s, sample in enumerate(self):
            if(timestep == "all"):
                feature_l = []
                for t, feature in enumerate(sample):
                    # convert to numpy array
                    feature = feature[:]
                    feature = feature.flatten()
                    feature_l.append(feature)
                samples.append(feature)
            else:
                # convert to numpy array
                f = sample[int(timestep)][:]
                samples.append([f.flatten()])
        return samples

    def to_numpy(self):
        """Convert hdf5 data set to numpy array."""
        pass
        # samples = len(self)
        # timesteps = len(self[0])
        # features = self[0][0].shape
        # print(f"samples {samples} timesteps {timesteps} features {features}")
        # array = np.zeros((samples, timesteps, features[0], features[1], features[2]))
        # for s, sample in enumerate(self):
        #     # for t, feature in enumerate(sample):
        #     array[s][0] = sample[0].value
        # return array
        # return self.seq_paths[self.sample_idx]["num_timesteps"]
