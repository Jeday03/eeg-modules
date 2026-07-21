import asyncio
import json
import websockets
import numpy as np
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
trial_buffer_ts = deque()
trial_buffer_eeg = None

left_trials = 0
right_trials = 0

trial_number = 0

trial_labels = [
    "LEFT",
    "RIGHT"
]

STATE_PREPARE = "PREPARE"
STATE_COLLECT = "COLLECT"
STATE_REST = "REST"


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


def save_dataset():
    if len(trial_data) == 0:
        return

    np.savez(
        "mi_trials.npz",
        labels=np.array(
            [t["label"] for t in trial_data]
        ),
        ts_start=np.array(
            [t["ts_start"] for t in trial_data]
        ),
        ts_end=np.array(
            [t["ts_end"] for t in trial_data]
        ),
        X=np.stack(
            [t["X"] for t in trial_data]
        ),
        X_pre=np.stack(
            [t["X_pre"] for t in trial_data]
        ),
        features=np.stack(
            [t["features"] for t in trial_data]
        )
    )

    print("Dataset salvo em mi_trials.npz")


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

    state = STATE_PREPARE
    current_label = None

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print("Connected to acquisition\n")

                while True:
                    try:
                        if state == STATE_REST:
                            print("Rest...\n")
                            await asyncio.sleep(2)
                            state = STATE_PREPARE
                            continue

                        msg = await ws.recv()
                        data = json.loads(msg)

                    except json.JSONDecodeError:
                        print("JSON inválido")
                        continue

                    except Exception as e:
                        save_dataset()
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
                        trial_buffer_ts = deque()

                        trial_buffer_eeg = [
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

                    if state == STATE_PREPARE:
                        print()
                        print("Prepare...")
                        await asyncio.sleep(2)

                        current_label = trial_labels[
                            trial_number % len(trial_labels)
                        ]

                        print(current_label)

                        trial_buffer_ts.clear()

                        for ch in trial_buffer_eeg:
                            ch.clear()

                        state = STATE_COLLECT

                    chunk_samples = len(ts)

                    if chunk_samples == 0:
                        continue

                    for i in range(chunk_samples):
                        trial_buffer_ts.append(
                            ts[i]
                        )

                        for ch in range(n_channels):
                            trial_buffer_eeg[ch].append(
                                eeg[ch][i]
                            )

                    samples_since_last += chunk_samples

                    while len(trial_buffer_ts) > window_samples:
                        trial_buffer_ts.popleft()

                        for ch in range(n_channels):
                            trial_buffer_eeg[ch].popleft()

                    if len(trial_buffer_ts) < window_samples:
                        continue

                    if samples_since_last < step_samples:
                        continue

                    samples_since_last = 0
<<<<<<< HEAD
=======
                    label = trial_labels[
                        trial_number % 2
                    ]

                    print()

                    print("Prepare...")

                    await asyncio.sleep(2)

                    print(label)

                    print()
>>>>>>> 8b26f1336424248019392a319ba3bc4a5c5a7f61

                    X = np.zeros(
                        (
                            window_samples,
                            n_channels
                        )
                    )

                    for ch in range(n_channels):
                        X[:, ch] = list(
                            trial_buffer_eeg[ch]
                        )

                    ts_window = list(trial_buffer_ts)

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

                    save_trial(
                        current_label,
                        ts_window[0],
                        ts_window[-1],
                        X,
                        X_pre,
                        features
                    )
                    save_dataset()
                    trial_number += 1

                    if current_label == "LEFT":
                        left_trials += 1
                    else:
                        right_trials += 1

                    print("----------------------------------")
                    print("Trial saved")
                    print("Label:", current_label)
                    print("X shape:", X.shape)
                    print("Features:", len(features))
                    print("Decision:", decision)
                    print("Score:", round(score, 4))
                    print("LEFT trials:", left_trials)
                    print("RIGHT trials:", right_trials)
                    print()

                    state = STATE_REST

        except Exception as e:
            print("Erro de conexão:", e)

        print("Reconectando em 2 segundos...\n")
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())