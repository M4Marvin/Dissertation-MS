import pandas as pd
import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

a1_, a2_, a3_, a4_, a5_, a6_ = 307, -843, 24.8, -216, 198, 1110
b1_, b2_, b3_, b4_, b5_, b6_ = -1.5, 57.9, 104, 56.5, 225, 57.7
c1_, c2_, c3_, c4_, c5_, c6_ = 50.4, 25.8, 15.3, 18, 67.2, 24


class OutlierRemover:
    @staticmethod
    def remove_outliers(
        df: pd.DataFrame, col: str, threshold: float = 3
    ) -> pd.DataFrame:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        not_outliers = z_scores.abs() < threshold
        return df[not_outliers]


class DensityCalculator:
    @staticmethod
    def calculate_U(density: np.ndarray) -> np.ndarray:
        return -2.494 * np.log(density)


class DataBinner:
    def __init__(self, bins: int = 100):
        self.bins = bins

    def get_bins_U_norm(
        self, df: pd.DataFrame, value_col: str = "angle"
    ) -> Tuple[np.ndarray, np.ndarray]:
        angles = df[value_col].values
        hist, bins = np.histogram(angles, bins=self.bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        densities = hist / (len(angles) * (bins[1] - bins[0]))

        U = DensityCalculator.calculate_U(densities)

        bin_centers = bin_centers[~np.isinf(U)]
        U = U[~np.isinf(U)]
        bins = bin_centers[~np.isnan(U)]
        U = U[~np.isnan(U)]

        U -= U.min()
        bin_centers -= bin_centers.mean()

        return bin_centers, U


class GaussianFunction:
    @staticmethod
    def single_gaussian(x: np.ndarray, a1: float, b1: float, c1: float) -> np.ndarray:
        return a1 * np.exp(-(((x - b1) / c1) ** 2))

    @staticmethod
    def multi_gaussian(x: np.ndarray, *params) -> np.ndarray:
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            a, b, c = params[i : i + 3]
            y += GaussianFunction.single_gaussian(x, a, b, c)
        return y


class Fitter:
    def __init__(self, maxfev: int = 10000):
        self.maxfev = maxfev

    def fit_func(
        self, x: np.ndarray, y: np.ndarray, num_gaussians: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_params = num_gaussians * 3
        p0 = np.random.random(size=num_params)

        popt, pcov = curve_fit(
            GaussianFunction.multi_gaussian, x, y, p0=p0, maxfev=self.maxfev
        )
        return popt, pcov


class ModelSelector:
    @staticmethod
    def calculate_aic(n: int, mse: float, num_params: int) -> float:
        aic = n * np.log(mse) + 2 * num_params
        return aic

    def select_best_model(
        self, x: np.ndarray, y: np.ndarray, fits: dict, measure: str = "mse"
    ) -> Tuple[np.ndarray, Callable, float, int]:
        best_score = None
        best_fit = None
        best_func = None
        best_num_gaussians = None

        for num_gaussians, fit in fits.items():
            popt, pcov = fit
            fitted_func = GaussianFunction.multi_gaussian(x, *popt)
                    mse = mean_squared_error(y, fitted_func)

        if measure == "aic":
            score = self.calculate_aic(len(x), mse, num_gaussians * 3)
        else:
            score = mse

        if best_score is None or score < best_score:
            best_score = score
            best_fit = popt
            best_func = fitted_func
            best_num_gaussians = num_gaussians

    return best_fit, best_func, best_score, best_num_gaussians

def main():
    data_file = "path/to/your/csv/file.csv"
    angle_col = "angle"
    df = pd.read_csv(data_file)
    outlier_remover = OutlierRemover()
    df = outlier_remover.remove_outliers(df, angle_col)

    data_binner = DataBinner()
    x, y = data_binner.get_bins_U_norm(df, angle_col)

    fitter = Fitter()

    fits = {}
    for num_gaussians in range(1, 6):
        fits[num_gaussians] = fitter.fit_func(x, y, num_gaussians)

    model_selector = ModelSelector()
    best_fit, best_func, best_score, best_num_gaussians = model_selector.select_best_model(x, y, fits)

    print(f"Best model has {best_num_gaussians} Gaussians with a score of {best_score}")

    plt.plot(x, y, label="Data")
    plt.plot(x, best_func, label=f"Best fit ({best_num_gaussians} Gaussians)")
    plt.legend()
    plt.show()
