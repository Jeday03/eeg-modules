import numpy as np

from scipy.signal import butter
from scipy.signal import filtfilt


def bandpass_filter(
    X,
    fs,
    low,
    high,
    order=4
):

    nyquist = fs / 2

    b, a = butter(
        order,
        [
            low / nyquist,
            high / nyquist
        ],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        X,
        axis=0
    )


def preprocess_mi(X, fs):

    X_filtered = bandpass_filter(
        X,
        fs,
        8,
        30
    )

    mean = np.mean(
        X_filtered,
        axis=0
    )

    std = np.std(
        X_filtered,
        axis=0
    ) + 1e-8

    return (X_filtered - mean) / std


def compute_power(X):

    return np.mean(
        X ** 2,
        axis=0
    )


def extract_features(
    X,
    fs
):

    mu = bandpass_filter(
        X,
        fs,
        8,
        12
    )

    beta = bandpass_filter(
        X,
        fs,
        13,
        30
    )

    mu_power = compute_power(mu)
    beta_power = compute_power(beta)

    return np.concatenate(
        (
            mu_power,
            beta_power
        )
    )