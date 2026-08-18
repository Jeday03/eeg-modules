import os
import mne
import numpy as np
from mi_processing import (
    preprocess_mi,
    extract_features
)


DATASET_PATH = "datasets"

gdf_files = []

for subject in range(1, 10):

    subject_id = f"B{subject:02d}"

    for session in range(1, 4):

        session_id = f"{session:02d}T"

        file_name = f"{subject_id}{session_id}.gdf"

        file_path = os.path.join(
            DATASET_PATH,
            file_name
        )

        if os.path.exists(file_path):
            gdf_files.append(file_path)
        else:
            print("Arquivo não encontrado:", file_path)

gdf_files = sorted(gdf_files)

trials = []
discarded_trials = []

window_sec = 3.0

print()
print("Arquivos encontrados:", len(gdf_files))
print()

fs = None
raw_eeg_ch_names = None

for file_path in gdf_files:

    file_name = os.path.basename(file_path)
    subject_id = file_name[:3]
    session_id = file_name[3:6]

    print("===================================")
    print("Arquivo:", file_name)
    print("Participante:", subject_id)
    print("Sessão:", session_id)
    print("===================================")

    raw = mne.io.read_raw_gdf(
        file_path,
        preload=False,
        verbose=False
    )
    raw.set_channel_types({
        "EOG:ch01": "eog",
        "EOG:ch02": "eog",
        "EOG:ch03": "eog"
    })

    print()
    print("Frequência:", raw.info["sfreq"])
    print()
    print("Quantidade de canais:")
    print(len(raw.ch_names))
    print()
    print("Nomes dos canais:")
    print(raw.ch_names)
    print()
    print("Tipos dos canais:")
    print(raw.get_channel_types())
    print()

    duration = raw.times[-1]
    print("Duração:")
    print(round(duration, 2), "segundos")
    print()

    print("Anotações encontradas:")
    annotation_types, annotation_counts = np.unique(
        raw.annotations.description,
        return_counts=True
    )
    for desc, count in zip(annotation_types, annotation_counts):
        print(desc, ":", count)
    print()

    events, event_dict = mne.events_from_annotations(raw)

    print("Dicionário de eventos:")
    print(event_dict)
    print()
    print("Quantidade total de eventos:")
    print(len(events))
    print()

    eeg_channels = [
        "EEG:C3",
        "EEG:Cz",
        "EEG:C4"
    ]

    raw_eeg = raw.copy().pick(eeg_channels)

    print("Canais utilizados no processamento:")
    print(raw_eeg.ch_names)
    print("Tipos:")
    print(raw_eeg.get_channel_types())
    print()

    fs = raw.info["sfreq"]
    raw_eeg_ch_names = raw_eeg.ch_names

    window_samples = int(window_sec * fs)

    print("Tamanho da janela:")
    print(window_samples, "amostras")
    print()

    artifact_count_file = 0
    pending_artifact = False

    for event in events:

        event_sample = event[0]
        event_code = event[2]
        event_onset = event_sample / fs

        if event_code == event_dict.get("1023"):

            # O 1023 marca o TRIAL como contaminado, mas não é o próprio
            # trial de imagética motora (não tem dado associado). O cue
            # 769/770 correspondente ainda vai aparecer no fluxo de eventos
            # e é ELE que precisa ser descartado.
            pending_artifact = True

            print(
                "Marcador de artefato encontrado:", file_name,
                "instante:", round(event_onset, 3),
                "-> próximo trial será descartado"
            )
            continue

        if event_code not in [event_dict.get("769"), event_dict.get("770")]:
            continue

        label = "LEFT" if event_code == event_dict.get("769") else "RIGHT"

        if pending_artifact:

            discarded_trials.append({
                "source_file": file_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "event_type": str(event_code),
                "event_onset": event_onset,
                "reason": "artifact"
            })

            artifact_count_file += 1

            print(
                "Trial descartado:", file_name,
                "evento:", event_code,
                "instante:", round(event_onset, 3),
                "motivo: artifact"
            )

            pending_artifact = False
            continue

        start = event_sample
        stop = start + window_samples

        if stop > raw_eeg.n_times:

            discarded_trials.append({
                "source_file": file_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "event_type": str(event_code),
                "event_onset": event_onset,
                "reason": "incomplete_window"
            })

            print(
                "Trial descartado:", file_name,
                "evento:", event_code,
                "instante:", round(event_onset, 3),
                "motivo: incomplete_window"
            )
            continue

        X = raw_eeg.get_data(start=start, stop=stop).T

        if X.shape != (window_samples, 3):

            discarded_trials.append({
                "source_file": file_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "event_type": str(event_code),
                "event_onset": event_onset,
                "reason": "invalid_shape"
            })

            print(
                "Trial descartado:", file_name,
                "evento:", event_code,
                "instante:", round(event_onset, 3),
                "motivo: invalid_shape"
            )
            continue

        if np.isnan(X).any():

            discarded_trials.append({
                "source_file": file_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "event_type": str(event_code),
                "event_onset": event_onset,
                "reason": "NaN"
            })

            print(
                "Trial descartado:", file_name,
                "evento:", event_code,
                "instante:", round(event_onset, 3),
                "motivo: NaN"
            )
            continue

        if np.isinf(X).any():

            discarded_trials.append({
                "source_file": file_name,
                "subject_id": subject_id,
                "session_id": session_id,
                "event_type": str(event_code),
                "event_onset": event_onset,
                "reason": "infinite"
            })

            print(
                "Trial descartado:", file_name,
                "evento:", event_code,
                "instante:", round(event_onset, 3),
                "motivo: infinite"
            )
            continue

        X_pre = preprocess_mi(X, fs)
        features = extract_features(X_pre, fs)

        trial = {
            "label": label,
            "subject_id": subject_id,
            "session_id": session_id,
            "source_file": file_name,
            "X": X.copy(),
            "X_pre": X_pre.copy(),
            "features": features.copy(),
            "event_onset": event_onset
        }

        trials.append(trial)

        print(
            "Trial válido:", file_name,
            "label:", label,
            "shape:", X.shape,
            "features:", len(features)
        )
        print("TOTAL DE TRIALS ACUMULADOS:", len(trials))
        print()

    print()
    print("Trials descartados por artefato neste arquivo:", artifact_count_file)
    print()


if len(trials) == 0:

    print("Nenhum trial válido foi encontrado.")

else:

    labels = np.array([trial["label"] for trial in trials])
    subject_ids = np.array([trial["subject_id"] for trial in trials])
    session_ids = np.array([trial["session_id"] for trial in trials])
    source_files = np.array([trial["source_file"] for trial in trials])
    X = np.stack([trial["X"] for trial in trials])
    X_pre = np.stack([trial["X_pre"] for trial in trials])
    features = np.stack([trial["features"] for trial in trials])
    event_onsets = np.array([trial["event_onset"] for trial in trials])

    discarded_source_files = np.array(
        [trial["source_file"] for trial in discarded_trials]
    )
    discarded_subject_ids = np.array(
        [trial["subject_id"] for trial in discarded_trials]
    )
    discarded_session_ids = np.array(
        [trial["session_id"] for trial in discarded_trials]
    )
    discarded_event_types = np.array(
        [trial["event_type"] for trial in discarded_trials]
    )
    discarded_event_onsets = np.array(
        [trial["event_onset"] for trial in discarded_trials]
    )
    discarded_reasons = np.array(
        [trial["reason"] for trial in discarded_trials]
    )

    np.savez_compressed(
        "mi_2b_trials.npz",
        labels=labels,
        subject_ids=subject_ids,
        session_ids=session_ids,
        source_files=source_files,
        X=X,
        X_pre=X_pre,
        features=features,
        event_onsets=event_onsets,
        fs=fs,
        ch_names=np.array(raw_eeg_ch_names),
        window_sec=window_sec,
        discarded_source_files=discarded_source_files,
        discarded_subject_ids=discarded_subject_ids,
        discarded_session_ids=discarded_session_ids,
        discarded_event_types=discarded_event_types,
        discarded_event_onsets=discarded_event_onsets,
        discarded_reasons=discarded_reasons
    )

    print()
    print("===================================")
    print("DATASET SALVO")
    print("===================================")
    print("Arquivo: mi_2b_trials.npz")
    print("Trials:", len(labels))
    print("X:", X.shape)
    print("X_pre:", X_pre.shape)
    print("Features:", features.shape)
    print("Descartados:", len(discarded_trials))
    print()