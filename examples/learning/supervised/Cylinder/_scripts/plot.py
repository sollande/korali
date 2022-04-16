#!/usr/bin/env python
import argparse
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sns.set()

def plot_reconstruction_error(X, Y, x_header = None, y_header = None):
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.plot(X, Y)
    if x_header:
        plt.xlabel(x_header)
    if y_header:
        plt.ylabel(y_header)

    plt.show()

def parse_file(file_path, delimiter = "\t", header=False):
    with open(file_path) as fh:
        values = list(csv.reader(fh, delimiter=delimiter))
    if args.header:
        headers = values.pop(0)
    return [[int(l[0]) for l in values], [float(l[1]) for l in values]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-f",
        "--file",
        help="filepath to errors file",
        required=True,
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

    args = parser.parse_args()
    headers = []
    values = parse_file(args.file, args.delimiter, args.header)
    if args.header:
        plot_reconstruction_error(values[0], values[1], headers[0], headers[1])
    else:
        plot_reconstruction_error(values[0], values[1])
