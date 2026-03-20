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


async def main():

    global fs, n_channels, window_samples, buffer_eeg, ch_names_ref

    while True:  

        try:
            async with websockets.connect(WS_URL) as ws:

                print("Connected to acquisition")

                while True:

                    try:
                        msg = await ws.recv()
                        data = json.loads(msg)

                    except json.JSONDecodeError:
                        print("Invalid JSON recebido")
                        continue

                    except websockets.exceptions.ConnectionClosed:
                        print("Conexão perdida. Reconectando...")
                        break

                    if not validate_chunk(data):
                        print("Invalid chunk")
                        continue

                
                    fs_new = data["fs"]
                    ch_names = data["ch_names"]
                    ts = data["ts"]
                    eeg = data["eeg"]

                    n_channels_new = len(ch_names)

                    
                    if buffer_eeg is None:
                        fs = fs_new
                        n_channels = n_channels_new
                        ch_names_ref = ch_names

                        buffer_eeg = [deque() for _ in range(n_channels)]
                        window_samples = int(window_sec * fs)

                        print("Configuração inicial definida")

                    
                    if (
                        fs_new != fs
                        or n_channels_new != n_channels
                        or ch_names != ch_names_ref
                    ):
                        print("Inconsistência detectada no stream. Chunk descartado.")
                        continue

                   
                    for i in range(len(ts)):

                        buffer_ts.append(ts[i])

                        for ch in range(n_channels):
                            buffer_eeg[ch].append(eeg[ch][i])

                    
                    while len(buffer_ts) > window_samples:

                        buffer_ts.popleft()

                        for ch in range(n_channels):
                            buffer_eeg[ch].popleft()

                
                    if len(buffer_ts) >= window_samples:

                        X = np.zeros((window_samples, n_channels))

                        for ch in range(n_channels):
                            X[:, ch] = list(buffer_eeg[ch])[-window_samples:]

                        ts_window = list(buffer_ts)[-window_samples:]

                        duration = ts_window[-1] - ts_window[0]

                        print("Window ready")
                        print("X shape:", X.shape)
                        print("Duration:", duration)
                        print()

        except Exception as e:
            print("Erro de conexão:", e)

        print("Tentando reconectar em 2 segundos...")
        await asyncio.sleep(2)


asyncio.run(main())