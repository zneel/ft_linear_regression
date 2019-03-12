#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import mimetypes
import argparse
import matplotlib.pyplot as plt
import csv


def check_file_ext(filename):
    """
    Check file extension
    """
    if mimetypes.guess_type(filename)[0] != 'text/csv':
        raise argparse.ArgumentTypeError('wrong filetype or path')
    return filename


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


def estimate_price(mileage, theta0, theta1):
    """
    Estimate price function: 
    """
    return theta0 + theta1 * mileage


def gradient_descent(X, Y, curr_t0, curr_t1, lr):
    """
    Gradient descent and its derivate
    """
    M = len(X)
    deriv_theta0 = 0
    deriv_theta1 = 0
    for i in range(M):
        deriv_theta0 += (1 / M) * ((curr_t0 + (curr_t1 * X[i])) - Y[i])
        deriv_theta1 += (1 / M) * (((curr_t0 + (curr_t1 * X[i])) - Y[i]) * X[i])
    tmp_theta0 = curr_t0 - lr * deriv_theta0
    tmp_theta1 = curr_t1 - lr * deriv_theta1
    return tmp_theta0, tmp_theta1


def cost_function(X, Y, theta0, theta1):
    """
    Evaluate the loss
    """
    M = len(X)
    err = 0.0
    for i in range(M):
        err += (Y[i] - estimate_price(X[i], theta0, theta1)) ** 2
    return err / M


def linear_regression(X, Y, lr, epochs):
    """
    The linear regression function to train the model
    """
    theta0 = 0
    theta1 = 0
    loss = []
    for _ in range(epochs):
        theta0, theta1 = gradient_descent(X, Y, theta0, theta1, lr)
        loss.append(cost_function(X, Y, theta0, theta1))
    return theta0, theta1, loss


def plot(x, y, t0, t1, loss):
    line_x = [0, 1]
    line_y = [(t1 * i) + t0 for i in line_x]
    # plt.figure(1)
    plt.subplot(211)
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(line_x, line_y)
    plt.scatter(x, y, None, 'orange')
    plt.title('Mileage vs. Price')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.grid(True)
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=check_file_ext, help="CSV file path")
    args = parser.parse_args()
    data = np.genfromtxt(args.filename, delimiter=',', skip_header=1)
    X = data[:, 0]
    Y = data[:, 1]
    xmin = np.min(X)
    xmax = np.max(X)
    ymin = np.min(Y)
    ymax = np.max(Y)
    normalized_X = [normalize(x, xmin, xmax) for x in X]
    normalized_Y = [normalize(y, ymin, ymax) for y in Y]
    learning_rate = 0.01
    epochs = 10000
    theta0, theta1, loss = linear_regression(normalized_X, normalized_Y, learning_rate, epochs)
    plot(normalized_X, normalized_Y, theta0, theta1, loss)
    print('Final loss = ', loss[-1])
    with open('out.csv', mode='w') as out:
        out_writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        out_writer.writerow(['theta0', 'theta1', 'xmin', 'xmax', 'ymin', 'ymax'])
        out_writer.writerow([theta0, theta1, xmin, xmax, ymin, ymax])


if __name__ == "__main__":
    main()
