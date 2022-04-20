#!/usr/bin/env python
import argparse
import csv
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import get_newest_dir
from utilities import initialize_constants
import utilities as constants
import korali
sns.set()

def plot_reconstruction_error(X, Y, x_header = None, y_header = None):
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    # for x, y in d.items():
    #     ax.plot(x, y)
    ax.scatter(X, Y)
    if x_header:
        plt.xlabel(x_header)
    if y_header:
        plt.ylabel(y_header)

    plt.show()
    # plt.savefig("test.png")

def plot_img(img, img_pred):
    fig, axs = plt.subplots(1, 2)
    axs[0].imgshow(img)
    axs[1].imgshow(img_pred)
    plt.show()


def string_to_number(txt):
    return int("".join(char for char in txt if char.isdigit()))

def parse_file(file_path, delimiter = "\t", header=False):
    with open(file_path) as fh:
        values = list(csv.reader(fh, delimiter=delimiter))
    ncol = len(values[0])
    header = values.pop(0)
    d = {}
    for c in range(ncol):
        d[header[c]] = [float(r[c]) for r in values]
    return d


if __name__ == "__main__":
    initialize_constants()
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-f",
        "--file",
        help="filepath or directory",
        default=os.path.join(SCRIPT_DIR, "_korali_result", constants.AUTOENCODER),
        required=False,
    )
    parser.add_argument(
        "--header",
        help="indicate if file contains a header",
        action="store_true"
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        help="Input file delimiter",
        default="\t",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Models to plot",
        default=constants.AUTOENCODER,
        choices=["ALL", constants.AUTOENCODER, constants.CNN_AUTOENCODER],
        required=False,
    )
    parser.add_argument(
        "-t",
        "--type",
        help="plot type",
        default="latent-dim-vs_error",
        choices=["image", "epoch-vs-error", "latent-dim-vs_error"],
        required=False,
    )


    args = parser.parse_args()
    headers = []
    if args.type == "epoch-vs-error":
        values = parse_file(args.file, args.delimiter, args.header)
        if args.header:
            plot_reconstruction_error(values[0], values[1], headers[0], headers[1])
        else:
            plot_reconstruction_error(values[0], values[1])
    elif args.type == "image":
        k = korali.Engine()
        e = korali.Experiment()
        isStateFound = e.loadState(os.path.join(args.file, "/latest"))
        e["Solver"]["Mode"] = "Training"
        e["Random Seed"] = 0xC0FFEE
        if not isStateFound:
            sys.exit("No model found")
        inputData = e["Problem"]["Input"]["Data"]
        testingInferred = e["Solver"]["Evaluation"]
        idx = 0
        plot_img(inputData[idx], testingInferred[idx])
    elif args.type == "latent-dim-vs_error":
        if args.model == "ALL":
            pass
        else:
            latent_dim_dirs = os.listdir(args.file)
            latent_dim_dict = {}
            for lat_dim_dir in latent_dim_dirs:
                lat_dim = string_to_number(lat_dim_dir)
                path = os.path.join(args.file, lat_dim_dir)
                p = get_newest_dir(path, constants.DATE_FORMAT)
                d = parse_file(os.path.join(p, "testing_error.txt"))
                latent_dim_dict[lat_dim] = d[' MeanSqauredError'][-1]
            X, Y = zip(*list(latent_dim_dict.items()))
            plot_reconstruction_error(X, Y)
