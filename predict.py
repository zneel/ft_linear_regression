#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import mimetypes
import argparse
import matplotlib.pyplot as plt


def check_file_ext(filename):
    """
    Check file extension
    """
    if mimetypes.guess_type(filename)[0] != 'text/csv':
        raise argparse.ArgumentTypeError('wrong filetype or path')
    return filename


def estimate_price(mileage, theta0, theta1):
    """
    Estimate price function: 
    """
    return theta0 + theta1 * mileage


def normalize(x, min, max):
    """
    Data min-max normalization:
    """
    return (x - min) / (max - min)


def denormalize(x, min, max):
    """
    Data min-max denormalization:
    """
    return x * (max - min) + min


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=check_file_ext, help="CSV file path")
    args = parser.parse_args()
    train = [0, 0, 0, 0, 0, 0]
    if args.filename is not None:
        train = np.genfromtxt(args.filename, delimiter=',', skip_header=1)
    mileage = normalize(np.float(input("Enter a mileage for estimation: ")), train[2], train[3])
    print(denormalize(estimate_price(mileage, train[0], train[1]), train[4], train[5]))


if __name__ == "__main__":
    main()
