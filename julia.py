import asyncio
import websockets
import json
import numpy as np
from collections import deque

WS_URL = "ws://localhost:8080/ws"

window_sec = 3.0

buffer_ts = deque()
buffer_eeg = None

fs = None
n_channels = None
window_samples = None


def validate_chunk(msg):

    if msg["type"] != "eeg_chunk":
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


async def main():

    global fs, n_channels, window_samples, buffer_eeg

    async with websockets.connect(WS_URL) as ws:

        print("Connected to acquisition")

        while True:

            msg = await ws.recv()
            data = json.loads(msg)

            if not validate_chunk(data):
                print("Invalid chunk")
                continue

            fs = data["fs"]
            ch_names = data["ch_names"]
            ts = data["ts"]
            eeg = data["eeg"]

            n_channels = len(ch_names)

            if buffer_eeg is None:
                buffer_eeg = [deque() for _ in range(n_channels)]
                window_samples = int(window_sec * fs)

            
            for i in range(len(ts)):
                buffer_ts.append(ts[i])

                for ch in range(n_channels):
                    buffer_eeg[ch].append(eeg[ch][i])

            
            while len(buffer_ts) > window_samples:
                buffer_ts.popleft()

                for ch in range(n_channels):
                    buffer_eeg[ch].popleft()

            
            if len(buffer_ts) == window_samples:

                X = np.zeros((window_samples, n_channels))

                for ch in range(n_channels):
                    X[:, ch] = list(buffer_eeg[ch])

                ts_window = list(buffer_ts)

                duration = ts_window[-1] - ts_window[0]

                print("Window ready")
                print("X shape:", X.shape)
                print("Duration:", duration)
                print()
                

asyncio.run(main())