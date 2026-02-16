import asyncio
import json
import aiohttp
import numpy as np
from collections import deque
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, filtfilt

# Ideia geral:
# 1) Receber chunks de EEG (vários canais) via WebSocket (do publisher)
# 2) Guardar num buffer deslizante (janela)
# 3) Pré-processar (filtro + normalização)
# 4) Comparar o EEG com sinais de referência (sen/cos) usando CCA
# 5) Escolher a frequência com maior correlação e aplicar uma regra de decisão

class SSVEPClassifier:
    def __init__(self, buffer_size=500, threshold=0.4, target_freqs=None):
        # buffer_size: tamanho da janela (em número de amostras) usada para classificar
        # buffer_size=500 amostras => ~2s se fs=250 Hz
        # threshold: correlação mínima para aceitar que "tem SSVEP naquela frequência"
        # target_freqs: lista de frequências candidatas
        self.buffer_size = buffer_size
        self.threshold = threshold

        # buffers: dicionário {nome_do_canal: deque com as amostras recentes}
        # permite armazenar sinais por canal de forma incremental
        self.buffers = {}
        # ts_buffer: timestamps recentes (também em janela), seve para validar duração real
        self.ts_buffer = deque(maxlen=buffer_size)
        # last_ch_names: última lista de canais recebida, para manter ordem e consistência
        self.last_ch_names = []

        # fs: taxa de amostragem (Hz). Aqui começa com 250, mas é atualizada pelo JSON recebido.
        self.fs = 250.0
        # target_freqs: frequências que vamos "testar" no CCA
        self.target_freqs = target_freqs or [8, 10, 12, 15]

        # CCA: Método de analise do SSVEP.
        self.cca = CCA(n_components=1)
        # recent_decisions: histórico curto para "votação" (evitar comando por ruído)
        self.recent_decisions = deque(maxlen=3)

    def update_buffers(self, eeg_data, ch_names, ts):
        # Função chamada a cada chunk recebido do publisher.
        # Ela NÃO classifica: ela só atualiza os buffers (janela deslizante).

        # Se vier timestamp, atualiza o buffer de timestamps
        if ts is not None:
            self.ts_buffer.extend(ts)

        # Atualiza a lista de nomes/ordem dos canais com a última recebida
        self.last_ch_names = ch_names

        # Para cada canal, garante que existe um deque e adiciona as amostras do chunk
        # eeg_data vem como "lista por canal"
        for i, name in enumerate(ch_names):
            if name not in self.buffers:
                self.buffers[name] = deque(maxlen=self.buffer_size)
            self.buffers[name].extend(eeg_data[i])

    def apply_bandpass_filter(self, data, lowcut=5.0, highcut=35.0, fs=250.0, order=4):
        # Aplica um filtro passa-banda no sinal de 1 canal.
        # Para SSVEP, é interessante filtrar para reduzir ruído fora da banda.

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        # butter: cria coeficientes do filtro Butterworth
        b, a = butter(order, [low, high], btype="band")

        # filtfilt precisa de um mínimo de amostras (padlen) para funcionar bem.
        # Se o sinal for curto, retorna sem filtrar para não quebrar.
        padlen = 3 * (max(len(a), len(b)) - 1)
        if len(data) <= padlen:
            return data

        # filtfilt: filtra para frente e para trás (sem fase), bom para análise offline/janela
        return filtfilt(b, a, data)

    def generate_reference_signals(self, target_freq, fs, num_samples, num_harmonics=2):
        # Gera os sinais de referência usados no CCA:
        # seno e cosseno da frequência alvo e de alguns harmônicos.
        # Ex.: para 12 Hz e 2 harmônicos:
        # sin(12), cos(12), sin(24), cos(24)

        t = np.arange(num_samples) / fs
        refs = []
        for h in range(1, num_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * target_freq * t))
            refs.append(np.cos(2 * np.pi * h * target_freq * t))
        # Retorna matriz (num_samples x num_refs) para casar com X no CCA
        return np.array(refs).T

    def classify(self):
        # Executa a classificação usando a janela atual (buffer_size amostras).
        # Retorna 3 coisas:
        # - decision: frequência confirmada (após votação) ou None
        # - best_freq: melhor candidata na rodada (mesmo que não tenha passado no threshold)
        # - max_corr: score (correlação máxima encontrada)

        # Se ainda não recebemos nomes de canal, não dá pra montar X
        if not self.last_ch_names:
            return None, None, -1.0

        # Garante que já temos uma janela completa de timestamps (indicador de janela cheia)
        if len(self.ts_buffer) < self.buffer_size:
            return None, None, -1.0

        # Monta matriz X (amostras x canais) a partir dos buffers
        all_channels = []
        for ch_name in self.last_ch_names:
            # Se algum canal ainda não existe no dicionário, falha (inconsistência)
            if ch_name not in self.buffers:
                return None, None, -1.0

            # Pega o sinal do canal (últimas buffer_size amostras)
            signal = np.array(self.buffers[ch_name], dtype=float)
            if len(signal) < self.buffer_size:
                return None, None, -1.0

            # Filtra o canal individualmente (passa-banda)
            filtered = self.apply_bandpass_filter(signal, fs=self.fs)
            all_channels.append(filtered)

        # X: matriz final com shape (num_samples, num_channels)
        X = np.array(all_channels).T

        # Normalização por canal (z-score):
        # ajuda a deixar escalas comparáveis entre canais e reduzir efeito de amplitude absoluta
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-8] = 1.0
        X = (X - mean) / std

        # testar cada frequência alvo e pegar a que maximiza a correlação
        max_corr = -1.0
        best_freq = None

        for freq in self.target_freqs:
            # Y: referências seno/cosseno para essa frequência
            Y = self.generate_reference_signals(freq, self.fs, len(X), num_harmonics=2)

            # CCA encontra projeções de X e Y com máxima correlação
            self.cca.fit(X, Y)
            x_score, y_score = self.cca.transform(X, Y)

            # Correlação entre a 1ª componente canônica (mais forte)
            corr = np.corrcoef(x_score[:, 0], y_score[:, 0])[0, 1]
            if np.isnan(corr):
                continue

            # Guarda a melhor frequência até agora
            if corr > max_corr:
                max_corr = float(corr)
                best_freq = freq

        # Se não achou nenhuma frequência válida (tudo NaN, etc.), sem decisão
        if best_freq is None:
            self.recent_decisions.append(None)
            return None, None, -1.0

        # Regra de aceitação:
        # se a melhor correlação passa do limiar, conta como voto nessa frequência
        # senão, voto "None" (sem comando)
        if max_corr >= self.threshold:
            self.recent_decisions.append(best_freq)
        else:
            self.recent_decisions.append(None)

        # Votação simples: exige que a mesma frequência apareça 2 vezes no histórico curto
        votos = list(self.recent_decisions).count(best_freq)
        if votos >= 2:
            # Quando confirma o comando, limpa histórico para evitar repetição imediata
            self.recent_decisions.clear()
            return best_freq, best_freq, max_corr

        # Ainda não confirmou: retorna best_freq (candidata) e score para debug
        return None, best_freq, max_corr


async def run_client():
    # Cliente WebSocket que se conecta no publisher (módulo de aquisição)
    # e alimenta o classificador incrementalmente.

    url = "ws://localhost:8080/ws"
    classifier = SSVEPClassifier(buffer_size=500, threshold=0.4)

    # Contador de amostras recebidas desde a última checagem de classificação
    samples_since_last_check = 0
    # step_size: só tenta classificar de tempos em tempos (ex.: a cada 50 amostras)
    # evita rodar classify() a cada mensagem (pode ser pesado)
    step_size = 50

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            print("Conexão aberta. Aguardando...")

            # Loop principal: recebe mensagens do publisher indefinidamente
            async for msg in ws:
                # Só processa mensagens de texto (JSON)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                # Converte string JSON -> dict
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue

                # Só processa mensagens do tipo esperado
                if data.get("type") != "eeg_chunk":
                    continue

                # Atualiza fs (taxa de amostragem) com o valor vindo do publisher
                classifier.fs = float(data["fs"])

                # Extrai payload do chunk:
                # eeg: lista por canal
                # ch_names: nomes/ids dos canais
                # ts: timestamps (pode vir ou não, dependendo do publisher)
                eeg = data["eeg"]
                ch_names = data["ch_names"]
                ts = data.get("ts")

                # Atualiza buffers com o chunk recebido
                classifier.update_buffers(eeg, ch_names, ts)

                # Atualiza contador de amostras recebidas:
                # se tiver ts, usa len(ts); senão, usa tamanho do 1º canal
                if ts is not None:
                    samples_since_last_check += len(ts)
                else:
                    samples_since_last_check += len(eeg[0])

                # Só tenta classificar quando:
                # - já tem janela cheia (buffer_size)
                # - e já passaram step_size amostras desde a última tentativa
                if len(classifier.ts_buffer) >= classifier.buffer_size and samples_since_last_check >= step_size:
                    # duração real da janela (em segundos) usando timestamps
                    duracao_real = classifier.ts_buffer[-1] - classifier.ts_buffer[0]

                    # Garante que a janela representa ~2s (ou mais) antes de classificar
                    # Isso evita classificar com timestamps inconsistentes/curtos
                    if duracao_real > 0 and duracao_real >= 1.9:
                        decision, best_freq, score = classifier.classify()

                        # PRINT TEMPORÁRIO (debug): mostra comando quando confirma
                        if decision is not None:
                            print(f"\nCOMANDO: {decision}Hz | Score: {score:.3f}")
                        else:
                            # Mostra melhor candidata e score enquanto ainda não confirma
                            bf_txt = f"{best_freq}Hz" if best_freq is not None else "-"
                            print(f"Analisando... (Melhor: {bf_txt}, Score: {score:.3f})", end="\r")

                        # Reseta contador para a próxima rodada
                        samples_since_last_check = 0


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\nSaindo...")
