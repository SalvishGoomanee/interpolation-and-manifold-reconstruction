import numpy as np
import argparse

from curve_fitting_algorithms import FittingSmoothCurveToData

# Define ground truth curve
x = np.linspace(-3, 4, 100)

def curve_1(x):
    poly_1 = (x ** 3) + ((x * 0.9 - 4) ** 2)
    return poly_1

def curve_2(x):
    poly_2 = (20 * np.sin(x * 3 + 4) + 20) + curve_1(x)
    return poly_2

# Implement curve fitting algorithm to noise data for approximating ground truth
fit_data = FittingSmoothCurveToData(x=x, curve_1=curve_1, curve_2=curve_2)

# TODO: Add argparse
def args_parse():
    parser = argparse.ArgumentParser(description="Curve fitting algorithms")
    parser.add_argument("--noise_data", action="store_true", help="display noise data")
    parser.add_argument("--sparse_data", action="store_true", help="display sparse data")
    parser.add_argument("--fit_lin_inter", action="store_true", help="display noise data with linear interpolation")
    parser.add_argument("--fit_lin_reg", action="store_true", help="display noise data with linear regression")
    parser.add_argument("--fit_poly_reg", action="store_true", help="display noise data with polynomial regression")
    parser.add_argument("--fit_moving_average", action="store_true", help="display noise data with moving average")
    parser.add_argument("--fit_lowess", action="store_true", help="display noise data with LOWESS")
    parser.add_argument("--fit_bspline", action="store_true", help="display noise and sparse data with a B-spline")
    args = parser.parse_args()

    if args.noise_data:
        fit_data.noise_data()

    if args.sparse_data:
        fit_data.sparse_data()

    if args.fit_lin_inter:
        fit_data.fit_linear_interpolation()

    if args.fit_lin_reg:
        fit_data.fit_linear_regression()

    if args.fit_poly_reg:
        fit_data.fit_polynomial_regression(20)

    if args.fit_moving_average:
        fit_data.fit_moving_average()

    if args.fit_lowess:
        fit_data.fit_lowess()

    if args.fit_bspline:
        fit_data.fit_bspline()


if __name__ == "__main__":
    args_parse()