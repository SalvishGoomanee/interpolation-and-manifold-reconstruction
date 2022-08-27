import numpy as np

from curve_fitting import FittingSmoothCurveToData

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
# fit_data.noise_data()
# fit_data.sparse_data()
# fit_data.fit_linear_interpolation()
fit_data.fit_linear_regression()
# fit_data.fit_polynomial_regression(degree=20)
# fit_data.fit_moving_average()
# fit_data.fit_lowess()
# fit_data.fit_bspline()
