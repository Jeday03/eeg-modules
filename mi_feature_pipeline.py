import asyncio
import json
import websockets
import numpy as np
import time
import random

from collections import deque
from scipy.signal import butter, filtfilt


WS_URL = "ws://localhost:8080/ws"

WINDOW_SEC = 3.0
STEP_SEC = 0.25


buffer_ts = deque()
buffer_eeg = None

fs = None
n_channels = None
window_samples = None
step_samples = None

samples_since_last = 0
channel_names = None
trial_data = []

left_trials = 0
right_trials = 0

trial_number = 0

trial_labels = [
    "LEFT",
    "RIGHT"
]

def validate_chunk(msg):

    if msg.get("type") != "eeg_chunk":
        return False

    required = [
        "fs",
        "ch_names",
        "ts",
        "eeg"
    ]

    for key in required:

        if key not in msg:
            return False

    if msg["fs"] <= 0:
        return False

    eeg = msg["eeg"]
    ts = msg["ts"]
    ch_names = msg["ch_names"]

    if len(eeg) != len(ch_names):
        return False

    n = len(ts)

    for ch in eeg:

        if len(ch) != n:
            return False

    return True


def bandpass_filter(X, fs, low, high, order=4):

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


def extract_features(X, fs):

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


def baseline_decision(features):

    half = len(features) // 2

    mu = np.mean(
        features[:half]
    )

    beta = np.mean(
        features[half:]
    )

    score = beta - mu

    if score >= 0:

        return "RIGHT", score

    return "LEFT", score

def save_trial(
    label,
    ts_start,
    ts_end,
    X,
    X_pre,
    features
):

    trial = {

        "label": label,

        "ts_start": ts_start,

        "ts_end": ts_end,

        "X": X.copy(),

        "X_pre": X_pre.copy(),

        "features": features.copy()
    }

    trial_data.append(trial)


async def main():

    global fs
    global n_channels
    global window_samples
    global step_samples
    global buffer_eeg
    global channel_names
    global samples_since_last
    global trial_number
    global left_trials
    global right_trials

    while True:

        try:

            async with websockets.connect(WS_URL) as ws:

                print("Connected to acquisition\n")

                while True:

                    try:

                        msg = await ws.recv()
                        data = json.loads(msg)

                    except json.JSONDecodeError:

                        print("JSON inválido")
                        continue

                    except Exception as e:

                        print("Erro ao receber:", e)
                        break

                    if not validate_chunk(data):

                        print("Chunk inválido")
                        continue

                    fs_new = data["fs"]
                    ch_names = data["ch_names"]
                    ts = data["ts"]
                    eeg = data["eeg"]

                    if buffer_eeg is None:

                        fs = fs_new
                        n_channels = len(ch_names)
                        channel_names = ch_names

                        window_samples = int(
                            WINDOW_SEC * fs
                        )

                        step_samples = int(
                            STEP_SEC * fs
                        )

                        buffer_eeg = [
                            deque()
                            for _ in range(n_channels)
                        ]

                        print("Pipeline initialized")
                        print("Sampling rate:", fs)
                        print("Channels:", n_channels)
                        print("Window samples:", window_samples)
                        print("Step samples:", step_samples)
                        print()

                    if (
                        fs_new != fs
                        or
                        len(ch_names) != n_channels
                        or
                        ch_names != channel_names
                    ):

                        print("Configuração diferente.")
                        continue

                    chunk_samples = len(ts)

                    if chunk_samples == 0:
                        continue

                    for i in range(chunk_samples):

                        buffer_ts.append(
                            ts[i]
                        )

                        for ch in range(n_channels):

                            buffer_eeg[ch].append(
                                eeg[ch][i]
                            )

                    samples_since_last += chunk_samples

                    while len(buffer_ts) > window_samples:

                        buffer_ts.popleft()

                        for ch in range(n_channels):

                            buffer_eeg[ch].popleft()

                    if len(buffer_ts) < window_samples:
                        continue

                    if samples_since_last < step_samples:
                        continue

                    samples_since_last = 0
                        label = trial_labels[
                        trial_number % 2
                                ]

                        print()

                        print("Prepare...")

                        await asyncio.sleep(2)

                        print(label)

                        print()

                    X = np.zeros(
                        (
                            window_samples,
                            n_channels
                        )
                    )

                    for ch in range(n_channels):

                        X[:, ch] = list(
                            buffer_eeg[ch]
                        )

                    ts_window = list(buffer_ts)

                    duration = (
                        ts_window[-1]
                        - ts_window[0]
                    )

                    X_pre = preprocess_mi(
                        X,
                        fs
                    )

                    features = extract_features(
                        X_pre,
                        fs
                    )

                    decision, score = baseline_decision(
                        features
                    )

                    print("----------------------------------")
                    print("Window ready")
                    print("Window shape:", X.shape)
                    print("Duration:", round(duration, 3), "s")
                    print("Features:", len(features))
                    print("Feature sample:", features[:5])
                    print("Decision:", decision)
                    print("Score:", round(score, 4))
                    print()

        except Exception as e:

            print("Erro de conexão:", e)

        print("Reconectando em 2 segundos...\n")

        await asyncio.sleep(2)


if __name__ == "__main__":

    asyncio.run(main())
