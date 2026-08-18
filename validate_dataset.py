import sys
import numpy as np


FILE_NAME = sys.argv[1] if len(sys.argv) > 1 else "mi_trials.npz"


data = np.load(
    FILE_NAME,
    allow_pickle=True
)


print("========== DATASET ==========")
print("Arquivo:", FILE_NAME)
print()

labels = data["labels"]

print("Total trials:", len(labels))


left = np.sum(labels == "LEFT")
right = np.sum(labels == "RIGHT")


print("LEFT trials :", left)
print("RIGHT trials:", right)
print()

X = data["X"]
X_pre = data["X_pre"]
features = data["features"]


print("Shape X:", X.shape)
print("Shape X_pre:", X_pre.shape)
print("Shape Features:", features.shape)
print()


print(
    "Features por trial:",
    features.shape[1]
)


print()

print(
    "NaN em X:",
    np.isnan(X).any()
)

print(
    "NaN em X_pre:",
    np.isnan(X_pre).any()
)

print(
    "NaN em Features:",
    np.isnan(features).any()
)

print()


print(
    "Inf em X:",
    np.isinf(X).any()
)

print(
    "Inf em X_pre:",
    np.isinf(X_pre).any()
)

print(
    "Inf em Features:",
    np.isinf(features).any()
)

print()


if "subject_ids" in data.files:

    subject_ids = data["subject_ids"]

    session_ids = data["session_ids"]

    source_files = data["source_files"]

    event_onsets = data["event_onsets"]

    fs = data["fs"]

    ch_names = data["ch_names"]

    window_sec = data["window_sec"]


    print("========== DATASET 2B ==========")
    print()

    print("Trials por participante:")

    unique_subjects = np.unique(
        subject_ids
    )

    for subject in unique_subjects:

        count = np.sum(
            subject_ids == subject
        )

        print(
            subject,
            ":",
            count
        )

    print()


    print("Trials por sessão:")

    unique_sessions = np.unique(
        session_ids
    )

    for session in unique_sessions:

        count = np.sum(
            session_ids == session
        )

        print(
            session,
            ":",
            count
        )

    print()

    print("Trials por arquivo:")

    unique_files = np.unique(
        source_files
    )

    for source_file in unique_files:

        count = np.sum(
            source_files == source_file
        )

        print(
            source_file,
            ":",
            count
        )

    print()

    print(
        "Frequência de amostragem:",
        fs,
        "Hz"
    )

    print()

    print(
        "Canais utilizados:"
    )

    print(
        list(ch_names)
    )

    print()


    print(
        "Ordem dos canais:"
    )

    for i, channel in enumerate(ch_names):

        print(
            i,
            "->",
            channel
        )

    print()

    print(
        "Duração da janela:",
        window_sec,
        "segundos"
    )

    print()


    print(
        "Primeiro event_onset:",
        event_onsets[0]
    )

    print()


    if "discarded_reasons" in data.files:

        discarded_reasons = data[
            "discarded_reasons"
        ]

        print(
            "Trials descartados:",
            len(discarded_reasons)
        )

        print()


        print(
            "Motivos dos descartes:"
        )


        unique_reasons = np.unique(
            discarded_reasons
        )


        for reason in unique_reasons:

            count = np.sum(
                discarded_reasons == reason
            )

            print(
                reason,
                ":",
                count
            )

    else:

        print(
            "Trials descartados: informação não encontrada"
        )

    print()


    print(
        "Participantes vazios:",
        np.any(
            subject_ids == ""
        )
    )

    print(
        "Sessões vazias:",
        np.any(
            session_ids == ""
        )
    )

    print(
        "Labels vazios:",
        np.any(
            labels == ""
        )
    )

    print()

    expected_x_shape = X.shape[1:]

    expected_x_pre_shape = X_pre.shape[1:]

    expected_feature_shape = features.shape[1:]


    different_x = False

    different_x_pre = False

    different_features = False


    for i in range(len(X)):

        if X[i].shape != expected_x_shape:

            different_x = True

            break


    for i in range(len(X_pre)):

        if X_pre[i].shape != expected_x_pre_shape:

            different_x_pre = True

            break


    for i in range(len(features)):

        if features[i].shape != expected_feature_shape:

            different_features = True

            break


    print(
        "Trials com shapes diferentes em X:",
        different_x
    )

    print(
        "Trials com shapes diferentes em X_pre:",
        different_x_pre
    )

    print(
        "Trials com shapes diferentes em Features:",
        different_features
    )

    print()

    print(
        "Verificação de quantidade por arquivo:"
    )


    counts = []


    for source_file in unique_files:

        count = np.sum(
            source_files == source_file
        )

        counts.append(count)


    counts = np.array(counts)


    print(
        "Mínimo:",
        counts.min()
    )

    print(
        "Máximo:",
        counts.max()
    )

    print(
        "Média:",
        round(counts.mean(), 2)
    )

    print()

    mean_count = counts.mean()
    threshold = 0.20

    outliers = []
    for source_file, count in zip(unique_files, counts):
        if mean_count > 0 and abs(count - mean_count) / mean_count > threshold:
            outliers.append((source_file, count))

    if outliers:
        print("Arquivos com quantidade de trials muito diferente da média:")
        for source_file, count in outliers:
            print(source_file, ":", count, "(média:", round(mean_count, 2), ")")
    else:
        print("Nenhum arquivo com quantidade de trials muito diferente da média.")

    print()


print("========== FIM ==========")
