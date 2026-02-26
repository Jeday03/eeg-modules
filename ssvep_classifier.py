import asyncio
import json
import aiohttp
import numpy as np
from collections import deque
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, filtfilt


class SSVEPClassifier:
    def __init__(self, buffer_size=500, threshold=0.4, target_freqs=None):
        self.buffer_size = buffer_size
        self.threshold = threshold

        self.buffers = {}
        self.ts_buffer = deque(maxlen=buffer_size)
        self.last_ch_names = []

        self.fs = 250.0
        self.target_freqs = target_freqs or [8, 10, 12, 15]

        self.cca = CCA(n_components=1)
        self.recent_decisions = deque(maxlen=3)

    def update_buffers(self, eeg_data, ch_names, ts):
        if ts is not None:
            self.ts_buffer.extend(ts)

        self.last_ch_names = ch_names

        for i, name in enumerate(ch_names):
            if name not in self.buffers:
                self.buffers[name] = deque(maxlen=self.buffer_size)
            self.buffers[name].extend(eeg_data[i])


    def apply_bandpass_filter(self, data, lowcut=5.0, highcut=35.0, fs=250.0, order=4):

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")

        padlen = 3 * (max(len(a), len(b)) - 1)
        if len(data) <= padlen:
            return data

        return filtfilt(b, a, data)

    def generate_reference_signals(self, target_freq, fs, num_samples, num_harmonics=2):

        t = np.arange(num_samples) / fs
        refs = []
        for h in range(1, num_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * target_freq * t))
            refs.append(np.cos(2 * np.pi * h * target_freq * t))
        return np.array(refs).T


    def classify(self):
        if not self.last_ch_names:
            return None, None, -1.0

        if len(self.ts_buffer) < self.buffer_size:
            return None, None, -1.0

        all_channels = []
        for ch_name in self.last_ch_names:
            if ch_name not in self.buffers:
                return None, None, -1.0

            signal = np.array(self.buffers[ch_name], dtype=float)
            if len(signal) < self.buffer_size:
                return None, None, -1.0

            filtered = self.apply_bandpass_filter(signal, fs=self.fs)
            all_channels.append(filtered)

        X = np.array(all_channels).T

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-8] = 1.0
        X = (X - mean) / std

        max_corr = -1.0
        best_freq = None

        for freq in self.target_freqs:
            Y = self.generate_reference_signals(freq, self.fs, len(X), num_harmonics=2)

            self.cca.fit(X, Y)
            x_score, y_score = self.cca.transform(X, Y)

            corr = np.corrcoef(x_score[:, 0], y_score[:, 0])[0, 1]
            if np.isnan(corr):
                continue

            if corr > max_corr:
                max_corr = float(corr)
                best_freq = freq

        if best_freq is None:
            self.recent_decisions.append(None)
            return None, None, -1.0

        if max_corr >= self.threshold:
            self.recent_decisions.append(best_freq)
        else:
            self.recent_decisions.append(None)

        votos = list(self.recent_decisions).count(best_freq)
        if votos >= 2:
            self.recent_decisions.clear()
            return best_freq, best_freq, max_corr

        return None, best_freq, max_corr


async def run_client():
    url = "ws://localhost:8080/ws"
    classifier = SSVEPClassifier(buffer_size=500, threshold=0.4)

    samples_since_last_check = 0
    step_size = 50

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            print("Conexão aberta. Aguardando...")

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue

                if data.get("type") != "eeg_chunk":
                    continue

                classifier.fs = float(data["fs"])

                eeg = data["eeg"]
                ch_names = data["ch_names"]
                ts = data.get("ts")

                classifier.update_buffers(eeg, ch_names, ts)

                if ts is not None:
                    samples_since_last_check += len(ts)
                else:
                    samples_since_last_check += len(eeg[0])

                if len(classifier.ts_buffer) >= classifier.buffer_size and samples_since_last_check >= step_size:
                    duracao_real = classifier.ts_buffer[-1] - classifier.ts_buffer[0]

                    if duracao_real > 0 and duracao_real >= 1.9:
                        decision, best_freq, score = classifier.classify()

                        if decision is not None:
                            print(f"\nCOMANDO: {decision}Hz | Score: {score:.3f}")
                        else:
                            bf_txt = f"{best_freq}Hz" if best_freq is not None else "-"
                            print(f"Analisando... (Melhor: {bf_txt}, Score: {score:.3f})", end="\r")

                        samples_since_last_check = 0


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\nSaindo...")
