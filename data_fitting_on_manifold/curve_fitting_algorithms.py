import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate

from numpy.polynomial import Polynomial as P
from statsmodels.nonparametric.smoothers_lowess import lowess

class PlotData:
    def __init__(self):
        self.x_axis_label = "Value X"
        self.y_axis_label = "Value Y"
        self.title = "Models"

    def plot_graph(self, x, y, title_curves, scatter=True, line=False, line_viz=None):
        global colors

        x_axis_label = self.x_axis_label
        y_axis_label = self.y_axis_label
        title = self.title
        figure_size = (10, 5)
        x_min, x_max = -3.3, 4.3
        y_max = 100
        title_curves = title + " - " + title_curves
        dot_opacity = line_opacity = 1
        size = 30 if len(x) > 15 else 100

        if line_viz is not None:
            if len(line_viz) == 1:
                colors = ["green"]
            if len(line_viz) == 2:
                colors = ["blue", "red"]
                dot_opacity = 0.3
                line_opacity = 3
            if len(line_viz) == 4:
                dot_opacity = 0.1
                line_opacity = 0.1
                colors = ["turquoise", "magenta", "yellow", "gold"]
            for i, (x, y) in enumerate(line_viz):
                plt.plot(x, y, color=colors[i], lw=3)

        if scatter:
            plt.scatter(x, y, color="green", marker="o", alpha=dot_opacity, s=size)
        if line:
            plt.plot(x, y, color="green", alpha=line_opacity)

        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)
        plt.title(title_curves)
        plt.figure(figsize=figure_size)
        plt.show()

class FittingSmoothCurveToData(PlotData):

    def __init__(self, x, curve_1, curve_2):
        super().__init__()
        self.x = x
        self.curve_1 = curve_1
        self.curve_2 = curve_2

    def piecewise_curve(self):
        curve = np.piecewise(self.x, [self.x < 0.2, self.x >= 0.2], [lambda x: self.curve_1(x), lambda x: self.curve_2(x)])
        return curve

    def noise_data(self):
        flux_range = 35
        random_flux = np.random.rand(len(self.piecewise_curve())) * flux_range - flux_range/2
        noise_data = self.piecewise_curve() + random_flux
        self.plot_graph(self.x, noise_data, title_curves="Noise data", scatter=True, line=False)
        return noise_data

    def sparse_data(self):
        sampling_increment = 9
        sparse_input = self.x[::sampling_increment]
        curve_sparse_data = self.piecewise_curve()
        sparse_data = curve_sparse_data[::sampling_increment]
        self.plot_graph(sparse_input, sparse_data, title_curves="Sparse data", scatter=True, line=False)
        return sparse_data

    def fit_linear_interpolation(self):
        flux_range = 35
        random_flux = np.random.rand(len(self.piecewise_curve())) * flux_range - flux_range/2
        noise_data = self.piecewise_curve() + random_flux
        self.plot_graph(self.x, noise_data, title_curves="Noise data with linear interpolation", scatter=True, line=True)

    def fit_linear_regression(self):
        noise_data = self.noise_data()
        m, b = P.fit(self.x, noise_data, 1)
        self.plot_graph(self.x, noise_data, title_curves="Noise data with linear regression", scatter=True, line=False, line_viz=[(self.x, m*self.x + b)])

    def fit_polynomial_regression(self, degree):
        noise_data = self.noise_data()
        z = np.polyfit(self.x, noise_data, degree)
        p = np.poly1d(z)
        xp = np.linspace(-3.3, 4.3, 100)
        self.plot_graph(self.x, noise_data, title_curves="Noise data with linear regression", scatter=True, line=False, line_viz=[(xp, p(xp))])

    def fit_moving_average(self):
        noise_data = self.noise_data()
        noise_data_series = pd.Series(self.noise_data())
        # print(noise_data_series.head(5))
        bin_sizes = [3, 5, 7, 10]
        moving_averages = [(self.x, noise_data_series.rolling(bin_size).mean()) for bin_size in bin_sizes]
        self.plot_graph(self.x, noise_data, title_curves="Noise data with moving averages", scatter=True, line=True, line_viz=moving_averages)

    # Apply locally weighted scatterplot smoothing
    # LOWESS combines piecewise bins aspect of moving averages with linear slope estimations of linear regression
    # Using Python's Statsmodels module
    def fit_lowess(self):
        noise_data = self.noise_data()
        lowess_tight = lowess(noise_data, self.x, frac=0.12)
        lowess_loose = lowess(noise_data, self.x, frac=0.2)
        lowess_list = [(lowess_tight[:, 0], lowess_tight[:, 1], lowess_loose[:, 0], lowess_loose[:, 1])]
        self.plot_graph(self.x, noise_data, title_curves="Noise data with LOWESS: tight (12 % bins) and looser (20 % bins)", scatter=True, line=False, line_viz=lowess_list)

    # Apply a type of curved spline - B-Splines
    # Piecewise polynomial interpolations between consecutive points called knots (control points for Bézier curves/splines)
    def fit_bspline(self):
        sampling_increment = 9
        sparse_input = self.x[::sampling_increment]
        sparse_data = self.sparse_data()
        BSpline = scipy.interpolate.make_interp_spline(sparse_input, sparse_data, 2)
        y_BSpline = BSpline(self.x)
        spline_tuple = [(self.x, y_BSpline)]
        self.plot_graph(self.x, sparse_data, title_curves="Sparse data with B-Spline smoothing", scatter=True, line=False, line_viz=spline_tuple)

























