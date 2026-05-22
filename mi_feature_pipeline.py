import asyncio
import websockets
import json
import numpy as np

from collections import deque
from scipy.signal import butter, filtfilt

WS_URL = "ws://localhost:8080/ws"

window_sec = 3.0

buffer_ts = deque()
buffer_eeg = None

fs = None
n_channels = None
window_samples = None

ch_names_ref = None


def validate_chunk(msg):

    if msg.get("type") != "eeg_chunk":
        return False

    required = ["fs", "ch_names", "ts", "eeg"]

    for k in required:

        if k not in msg:
            return False

    if msg["fs"] <= 0:
        return False

    ch_names = msg["ch_names"]
    eeg = msg["eeg"]
    ts = msg["ts"]

    if len(eeg) != len(ch_names):
        return False

    n_samples = len(ts)

    for ch in eeg:

        if len(ch) != n_samples:
            return False

    return True


def bandpass_filter(X, fs, low, high, order=4):

    nyquist = fs / 2

    low = low / nyquist
    high = high / nyquist

    b, a = butter(order, [low, high], btype="band")

    X_filtered = filtfilt(
        b,
        a,
        X,
        axis=0
    )

    return X_filtered


def preprocess_mi(X, fs):

    mean = np.mean(X, axis=0)

    std = np.std(X, axis=0) + 1e-8

    X_norm = (X - mean) / std

    return X_norm


def compute_power(X):

    return np.mean(X ** 2, axis=0)


def extract_baseline_features(X_pre, fs):

    X_mu = bandpass_filter(
        X_pre,
        fs,
        8,
        12
    )

    X_beta = bandpass_filter(
        X_pre,
        fs,
        13,
        30
    )

    mu_power = compute_power(X_mu)

    beta_power = compute_power(X_beta)

    features = np.concatenate([
        mu_power,
        beta_power
    ])

    return features


async def main():

    global fs
    global n_channels
    global window_samples
    global buffer_eeg
    global ch_names_ref

    while True:

        try:

            async with websockets.connect(WS_URL) as ws:

                print("Connected to acquisition")

                while True:

                    try:

                        msg = await ws.recv()

                        data = json.loads(msg)

                    except json.JSONDecodeError:

                        print("JSON inválido")
                        continue

                    except Exception as e:

                        print("Erro ao receber mensagem:", e)
                        break

                    if not validate_chunk(data):

                        print("Chunk inválido")
                        continue

                    fs_new = data["fs"]

                    ch_names = data["ch_names"]
                    ts = data["ts"]

                    print("ts len:", len(ts))

                    if len(ts) > 0:

                        print("first ts:", ts[0])

                        print("last ts:", ts[-1])

                    print("buffer size:", len(buffer_ts))

                    print()

                    eeg = data["eeg"]

                    n_channels_new = len(ch_names)

                    if buffer_eeg is None:

                        fs = fs_new

                        n_channels = n_channels_new

                        ch_names_ref = ch_names

                        window_samples = int(
                            window_sec * fs
                        )

                        buffer_eeg = [
                            deque()
                            for _ in range(n_channels)
                        ]

                        print("Configuração inicial definida")
                        print("fs:", fs)
                        print("channels:", n_channels)
                        print(
                            "window_samples:",
                            window_samples
                        )
                        print()

                    if (
                        fs_new != fs
                        or n_channels_new != n_channels
                        or ch_names != ch_names_ref
                    ):

                        print(
                            "Inconsistência detectada"
                        )

                        continue

                    chunk_samples = len(ts)
                    print("chunk_samples:", chunk_samples)

                    if chunk_samples == 0:
                        continue

                    for i in range(chunk_samples):

                        buffer_ts.append(ts[i])

                        for ch in range(n_channels):

                            buffer_eeg[ch].append(
                                eeg[ch][i]
                            )

                    while len(buffer_ts) > window_samples:

                        buffer_ts.popleft()

                        for ch in range(n_channels):

                            buffer_eeg[ch].popleft()

                    if len(buffer_ts) >= window_samples:

                        X = np.zeros(
                            (
                                window_samples,
                                n_channels
                            )
                        )

                        for ch in range(n_channels):

                            X[:, ch] = list(
                                buffer_eeg[ch]
                            )[-window_samples:]

                        ts_window = list(
                            buffer_ts
                        )[-window_samples:]

                        duration = (
                            ts_window[-1]
                            - ts_window[0]
                        )

                        X_pre = preprocess_mi(
                            X,
                            fs
                        )

                        features = (
                            extract_baseline_features(
                                X_pre,
                                fs
                            )
                        )

                        print("Window ready")

                        print(
                            "X shape:",
                            X.shape
                        )

                        print(
                            "X_pre shape:",
                            X_pre.shape
                        )

                        print(
                            "Duration:",
                            duration
                        )

                        print(
                            "Features shape:",
                            features.shape
                        )

                        print(
                            "Feature sample:",
                            features[:5]
                        )

                        print()

                        await asyncio.sleep(0.25)

        except Exception as e:

            print("Erro de conexão:", e)

        print("Tentando reconectar em 2 segundos...")

        await asyncio.sleep(2)


asyncio.run(main())