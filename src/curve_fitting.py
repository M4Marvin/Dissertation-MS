import pandas as pd
import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

a1_, a2_, a3_, a4_, a5_, a6_ = 307, -843, 24.8, -216, 198, 1110
b1_, b2_, b3_, b4_, b5_, b6_ = -1.5, 57.9, 104, 56.5, 225, 57.7
c1_, c2_, c3_, c4_, c5_, c6_ = 50.4, 25.8, 15.3, 18, 67.2, 24


def remove_outliers(df: pd.DataFrame, col: str, threshold: float = 3) -> pd.DataFrame:
    """
    Remove outliers from a dataframe using z-scores
    :param df: The dataframe
    :param col: The column name
    :return: The dataframe without outliers
    """

    # Get the z-scores
    z_scores = (df[col] - df[col].mean()) / df[col].std()

    # Get the rows that are not outliers
    not_outliers = z_scores.abs() < threshold

    # Return the dataframe without outliers
    return df[not_outliers]


def calculate_U(density):
    """
    Calculate the U values for a given density
    :param density: The density
    :return: The U value/s
    """
    return -2.494 * np.log(density)


def get_bins_U_norm(
    df: pd.DataFrame, bins: int = 100, value_col: str = "angle"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the bins and the U values for a given dataframe
    :param df: The dataframe
    :param bins: The number of bins
    :param value_col: The column name of the values
    :return: The bins and the U values
    """
    # Get the angles
    angles = df[value_col].values
    # Get the histogram
    hist, bins = np.histogram(angles, bins=bins)
    # Get the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Get the bin densities
    densities = hist / (len(angles) * (bins[1] - bins[0]))

    # Calculate the U values
    U = calculate_U(densities)

    # Remove the inf and nan values
    bin_centers = bin_centers[~np.isinf(U)]
    U = U[~np.isinf(U)]
    bins = bin_centers[~np.isnan(U)]
    U = U[~np.isnan(U)]

    # Scale the values
    U -= U.min()
    bin_centers -= bin_centers.mean()

    return bin_centers, U


def func1(x: np.ndarray, a1: float, b1: float, c1: float) -> np.ndarray:
    """
    The gaussian function
    :param x: The x values
    :param a1: The amplitude
    :param b1: The mean
    :param c1: The standard deviation
    :return: The y values
    """
    return a1 * np.exp(-(((x - b1) / c1) ** 2))


def func2(
    x: np.ndarray, a1: float, b1: float, c1: float, a2: float, b2: float, c2: float
) -> np.ndarray:
    return func1(x, a1, b1, c1) + func1(x, a2, b2, c2)


def func4(
    x: np.ndarray,
    a1: float,
    b1: float,
    c1: float,
    a2: float,
    b2: float,
    c2: float,
    a3: float,
    b3: float,
    c3: float,
    a4: float,
    b4: float,
    c4: float,
) -> np.ndarray:
    return func2(x, a1, b1, c1, a2, b2, c2) + func2(x, a3, b3, c3, a4, b4, c4)


def func6(
    x: np.ndarray,
    a1: float,
    b1: float,
    c1: float,
    a2: float,
    b2: float,
    c2: float,
    a3: float,
    b3: float,
    c3: float,
    a4: float,
    b4: float,
    c4: float,
    a5: float,
    b5: float,
    c5: float,
    a6: float,
    b6: float,
    c6: float,
) -> np.ndarray:
    return func4(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4) + func2(
        x, a5, b5, c5, a6, b6, c6
    )


def fit_func(
    x: np.ndarray, y: np.ndarray, maxfev: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the data for 2, 4 and 6 gaussians
    :param x: The x values
    :param y: The y values
    :param maxfev: The maximum number of function evaluations
    :return: The parameters for 2, 4 and 6 gaussians
    """

    # Fit the data for 2 gaussians
    popt2, pcov2 = curve_fit(
        func2, x, y, p0=[a1_, b1_, c1_, a2_, b2_, c2_], maxfev=maxfev
    )

    # Fit the data for 4 gaussians
    popt4, pcov4 = curve_fit(
        func4,
        x,
        y,
        p0=[a1_, b1_, c1_, a2_, b2_, c2_, a3_, b3_, c3_, a4_, b4_, c4_],
        maxfev=maxfev,
    )

    # Fit the data for 6 gaussians
    popt6, pcov6 = curve_fit(
        func6,
        x,
        y,
        p0=[
            a1_,
            b1_,
            c1_,
            a2_,
            b2_,
            c2_,
            a3_,
            b3_,
            c3_,
            a4_,
            b4_,
            c4_,
            a5_,
            b5_,
            c5_,
            a6_,
            b6_,
            c6_,
        ],
        maxfev=maxfev,
    )

    return popt2, popt4, popt6


def plot_all_fits(
    bin_centers: np.ndarray,
    U: np.ndarray,
    popt2: np.ndarray,
    popt4: np.ndarray,
    popt6: np.ndarray,
    save_path: str = None,
    return_fig: bool = False,
    show: bool = False,
) -> None or plt.Figure:
    """
    Plot the fitted curves
    :param bin_centers: The bin centers
    :param U: The U values
    :param popt2: The parameters for 2 gaussians
    :param popt4: The parameters for 4 gaussians
    :param popt6: The parameters for 6 gaussians
    :param save_path: The path to save the plot
    :param show: Whether to show the plot
    """

    # Plot the data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for ax in [ax1, ax2, ax3]:
        ax.plot(bin_centers, U, "o", label="Data")

    # Plot the fit for 2 gaussians
    ax1.plot(bin_centers, func2(bin_centers, *popt2), label="Fit")

    # Plot the fit for 4 gaussians
    ax2.plot(bin_centers, func4(bin_centers, *popt4), label="Fit")

    # Plot the fit for 6 gaussians
    ax3.plot(bin_centers, func6(bin_centers, *popt6), label="Fit")

    # Set the labels
    ax1.set_title("2 Gaussians")
    ax2.set_title("4 Gaussians")
    ax3.set_title("6 Gaussians")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("x")
        ax.set_ylabel("U")
        ax.legend()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()

    if return_fig:
        return fig


def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


def get_best_fit(
    bin_centers: np.ndarray,
    U: np.ndarray,
    popt2: np.ndarray,
    popt4: np.ndarray,
    popt6: np.ndarray,
    measure: str = "mse",
) -> Tuple[np.ndarray, int, float]:
    """
    Get the best fit curve using the AIC
    :param bin_centers: The bin centers
    :param U: The U values
    :param popt2: The parameters for 2 gaussians
    :param popt4: The parameters for 4 gaussians
    :param popt6: The parameters for 6 gaussians
    :param measure: The measure to use to get the best fit curve
    :return: The best fit curve
    """
    n = len(U)
    mse2 = mean_squared_error(U, func2(bin_centers, *popt2))
    mse4 = mean_squared_error(U, func4(bin_centers, *popt4))
    mse6 = mean_squared_error(U, func6(bin_centers, *popt6))

    if measure == "mse":
        # Get the best fit curve
        if mse2 < mse4 and mse2 < mse6:
            return popt2, func2, mse2, 2
        elif mse4 < mse2 and mse4 < mse6:
            return popt4, func4, mse4, 4
        else:
            return popt6, func6, mse6, 6

    elif measure == "aic":
        # Calculate the AIC for 2, 4 and 6 gaussians
        aic2 = calculate_aic(n, mse2, 6)
        aic4 = calculate_aic(n, mse4, 12)
        aic6 = calculate_aic(n, mse6, 18)

        # Get the best fit curve
        if aic2 < aic4 and aic2 < aic6:
            return popt2, func2, aic2, 2
        elif aic4 < aic2 and aic4 < aic6:
            return popt4, func4, aic4, 4
        else:
            return popt6, func6, aic6, 6

    elif measure == "rmse":
        # Calculate the RMSE for 2, 4 and 6 gaussians
        rmse2 = np.sqrt(mse2)
        rmse4 = np.sqrt(mse4)
        rmse6 = np.sqrt(mse6)

        # Get the best fit curve
        if rmse2 < rmse4 and rmse2 < rmse6:
            return popt2, func2, rmse2, 2
        elif rmse4 < rmse2 and rmse4 < rmse6:
            return popt4, func4, rmse4, 4
        else:
            return popt6, func6, rmse6, 6

    else:
        raise ValueError("Invalid measure")


def fit_df(
    df: pd.DataFrame,
    value_col: str = "angle",
    bins: int = 200,
    maxfev: int = 20000000,
    measure: str = "mse",
):
    """
    Fit the data in the dataframe
    :param df: The dataframe
    :param value_col: The column with the values
    :param bins: The number of bins
    :param maxfev: The maximum number of function evaluations
    :param measure: The measure to use to get the best fit curve
    :return: The best fit curve, the function, the score and the number of gaussians
    """
    df = remove_outliers(df, value_col)
    bin_centers, U = get_bins_U_norm(df, bins=bins, value_col=value_col)
    pop2, pop4, pop6 = fit_func(bin_centers, U, maxfev)
    best_fit, func, score, num_gaussians = get_best_fit(
        bin_centers, U, pop2, pop4, pop6, measure
    )

    return bin_centers, U, best_fit, func, score, num_gaussians
