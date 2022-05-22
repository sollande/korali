#!/usr/bin/env python
import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import korali
import shutil
import time
from mpi4py import MPI
sys.path.append('./_models')
sys.path.append('./_scripts')
from cnn_autoencoder import configure_cnn_autencoder
from autoencoder import configure_autencoder
from utilities import min_max_scalar
from utilities import print_args
from utilities import print_header
from utilities import bcolors
from utilities import get_prediction
from utilities import get_minibatch
from utilities import move_dir
from utilities import copy_dir
from utilities import make_parser
from utilities import initialize_constants
from utilities import exp_dir_str
from utilities import DataLoader
import utilities as constants

initialize_constants()
CWD = os.getcwd()
REL_ROOT = os.path.relpath("/")
TIMESTEPS = 0
parser = make_parser()
isMaster = lambda: args.conduit != constants.DISTRIBUTED or (args.conduit == constants.DISTRIBUTED and MPIrank == MPIroot)

iPython = True
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        ipython = True

args = parser.parse_args()
# TODO: move this into argparser
args.latent_dim = int(args.latent_dim)
if iPython:
    pass


k = korali.Engine()
# = Initalize Korali Engine
k["Conduit"]["Type"] = args.conduit
# ===================== Model Selection == =================================
if args.conduit == constants.DISTRIBUTED:
    MPIcomm = MPI.COMM_WORLD
    MPIrank = MPIcomm.Get_rank()
    MPIsize = MPIcomm.Get_size()
    MPIroot = MPIsize - 1
    k.setMPIComm(MPI.COMM_WORLD)

    if args.verbosity != constants.SILENT:
        print_args(vars(args), color=bcolors.HEADER)
# ===================== Loading the data =================================
loader = DataLoader(args)
loader.fetch_data()
X_train = loader.X_train
X_test = loader.X_test
nb_train_samples = len(X_train)
nb_test_samples = len(X_test)
timesteps, input_channels, img_width, img_height = loader.shape
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
# = permute
# ===================== Preprocessing ====================================
# TODO make this generic and adapt to general time series with preprocessing
# ONLY FOR one time point at the moment
X_train = np.squeeze(X_train, axis=1)
X_test = np.squeeze(X_test, axis=1)
# TODO finished
scalar = MinMaxScaler()
scalar.fit_transform(X_train)
scalar.fit(X_test)
# TODO make this generic and adapt to general time series with preprocessing
X_train = np.expand_dims(X_train, axis=1).tolist()
X_test = np.expand_dims(X_test, axis=1).tolist()
# TODO finished
# ================ Getting the ground truth ==============================
y_train = get_prediction(X_train)
y_test = get_prediction(X_test)
assert np.shape(y_train[0]) == np.shape(y_test[0])
input_size = output_size = len(y_train[0])
# = Calculate Epochs and iterations
stepsPerEpoch = int(nb_train_samples / args.training_batch_size)
testingBatchSize = nb_test_samples
# == If this is test mode, run only one epoch
if False:
    args.epochs = 1
    stepsPerEpoch = 1

e = korali.Experiment()
# == Create result directories if they do not exist
if args.file_output:
    EXPERIMENT_DIR = exp_dir_str(args)
    RESULT_DIR = os.path.join(CWD, EXPERIMENT_DIR)
    RESULT_DIR_WITHOUT_SCRATCH = os.path.relpath(RESULT_DIR, constants.SCRATCH)
    RESULT_DIR_ON_HOME = os.path.join(constants.HOME, RESULT_DIR_WITHOUT_SCRATCH)
    # Note: korali appends ./ => requires relative path i.e. ../../../..
    e["File Output"]["Path"] = RESULT_DIR
    if isMaster():
        os.makedirs(RESULT_DIR, exist_ok=True)
        if args.overwrite:
            shutil.rmtree(RESULT_DIR, ignore_errors=True)
        os.makedirs(RESULT_DIR, exist_ok=True)
        os.makedirs(RESULT_DIR_ON_HOME, exist_ok=True)
    isStateFound = e.loadState(os.path.join(RESULT_DIR, "/latest"))
    if isMaster() and isStateFound and args.verbosity != constants.SILENT:
        print("[Script] Evaluating previous run...\n")

e["File Output"]["Frequency"] = 0
e["Problem"]["Type"] = "Supervised Learning"
e["Random Seed"] = 0xC0FFEE
e["Console Output"]["Verbosity"] = args.verbosity
e["Problem"]["Max Timesteps"] = TIMESTEPS+1
e["Solver"]["Batch Concurrency"] = args.batch_concurrency
e["Problem"]["Training Batch Size"] = args.training_batch_size
e["Problem"]["Testing Batch Size"] = testingBatchSize
# ====================================================================
e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Termination Criteria"]["Max Generations"] = args.max_generations-1
e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

#  Set the autencoder layers
# ===================== Model Selection ====================================
if args.model == constants.AUTOENCODER:
    configure_autencoder(e, img_width, img_height, input_channels, args.latent_dim)
else:
    configure_cnn_autencoder(e, args.latent_dim, img_width, img_height)
# ==========================================================================
if isMaster() and args.verbosity != constants.SILENT:
    print("[Script] Running MNIST solver.")
    print("[Script] Nb. Training Images: %s" % nb_train_samples)
    print("[Script] Nb. Testing Images: %s" % nb_test_samples)
    print("[Script] Algorithm: " + str(e["Solver"]["Neural Network"]["Optimizer"]))
    print("[Script] Database Size: " + str(len(X_train)))
    print("[Script] Batch Size: " + str(args.training_batch_size))
    print("[Script] Epochs: " + str(args.epochs))
    print("[Script] Initial Learning Rate: " + str(args.learning_rate))
    print("[Script] Decay: " + str(args.decay))
    # ### Running SGD loop
times = []
if isMaster() and args.file_output:
    ERROR_FILE = os.path.join(RESULT_DIR, args.result_file)
    with open(ERROR_FILE, 'w') as f:
        f.write("Epoch\tMeanSquaredError\tTime\n")
for epoch in range(args.epochs):
    if isMaster():
        time_start = time.time_ns()
    e["Solver"]["Mode"] = "Training"
    for step in range(stepsPerEpoch):
        e["File Output"]["Enabled"] = True if (epoch % args.frequency == 0 and step+1 == stepsPerEpoch) else False
        # Creating minibatch
        X_train_mini = get_minibatch(X_train, step, args.training_batch_size)
        y_train_mini = get_minibatch(y_train, step, args.training_batch_size)
        # Passing minibatch to Korali
        e["Problem"]["Input"]["Data"] = X_train_mini
        e["Problem"]["Solution"]["Data"] = y_train_mini
        # Reconfiguring solver
        e["Solver"]["Learning Rate"] = args.learning_rate
        e["Solver"]["Termination Criteria"]["Max Generations"] = (e["Solver"]["Termination Criteria"]["Max Generations"] + 1)
        # Running step
        if args.conduit == constants.DISTRIBUTED:
            k.setMPIComm(MPI.COMM_WORLD)
        k.run(e)
    # Printing Information
    if isMaster() and args.verbosity != constants.SILENT:
        print("[Script] --------------------------------------------------")
        print("[Script] Epoch: " + str(epoch+1) + "/" + str(args.epochs))
        print("[Script] Learning Rate: " + str(args.learning_rate))
        print("[Script] Current Training Loss: " + str(e["Solver"]["Current Loss"]))
    args.learning_rate = args.learning_rate * (1.0 / (1.0 + args.decay * (epoch + 1)))
    # Evaluating testing set
    e["Solver"]["Mode"] = "Testing"
    e["Problem"]["Input"]["Data"] = X_test
    e["Problem"]["Solution"]["Data"] = y_test
    if args.conduit == constants.DISTRIBUTED:
        k.setMPIComm(MPI.COMM_WORLD)
    k.run(e)
    # Getting MSE loss for testing set (only the korali master has the evaluated results)
    if isMaster():
        yhat_test = e["Solver"]["Evaluation"]
        assert len(yhat_test) == len(y_test),\
            "Inferred vector does not have the same sample size as the ground truth"
        MSE = 0.0
        for yhat, y in zip(yhat_test, y_test):
            for yhat_i, y_i in zip(yhat, y):
                diff = yhat_i - y_i
                MSE += diff * diff
        MSE /= (float(testingBatchSize) * 2.0)
        # if epoch % args.frequency == 0:
        #     # Writing testing error to output
        # Runtime of epochs
        times.append((time.time_ns()-time_start)/(10**9))
        if args.file_output:
            with open(ERROR_FILE, 'a') as f:
                f.write("{}\t{}\t{}\n".format(epoch+1, MSE, times[-1]))
        if args.verbosity != constants.SILENT:
            print("[Script] Current Testing Loss:  " + str(MSE))

if isMaster():
    if args.file_output:
        # Writing testing error to output
        if constants.SCRATCH:
            # move_dir(RESULT_DIR, RESULT_DIR_ON_HOME)
            # copy_dir(RESULT_DIR, RESULT_DIR_ON_HOME)
            pass
    print("[Script] Total Time {}s for {} Epochs".format(sum(times), args.epochs))
    print("[Script] Per Epoch Time: {}s ".format(sum(times)/len(times)))
    if args.plot:
        pass
