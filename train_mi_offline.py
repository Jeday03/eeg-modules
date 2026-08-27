import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)


DATASET_FILE = "mi_2b_trials.npz"

TRAIN_SESSIONS = ["01T", "02T"]
TEST_SESSION = "03T"


print("===================================")
print(" Carregando dataset")
print("===================================\n")

data = np.load(DATASET_FILE, allow_pickle=True)

features = data["features"]
labels = data["labels"]
subject_ids = data["subject_ids"]
session_ids = data["session_ids"]

print("Dataset carregado.\n")

X = features
y_raw = labels

print("========== VALIDAÇÃO ==========\n")
print("Total de trials:", len(y_raw))
print("Número de features:", X.shape[1])
print()

left = np.sum(y_raw == "LEFT")
right = np.sum(y_raw == "RIGHT")

print("LEFT :", left)
print("RIGHT:", right)
print()

print("NaN:", np.isnan(X).any())
print("Inf:", np.isinf(X).any())
print()

if np.isnan(X).any():
    raise ValueError("Dataset possui NaN.")

if np.isinf(X).any():
    raise ValueError("Dataset possui valores infinitos.")

label_map = {
    "LEFT": 0,
    "RIGHT": 1
}
y = np.array([label_map[label] for label in y_raw])

print("Labels convertidos.\n")

participants = sorted(np.unique(subject_ids))

lda_accuracies = []
dummy_accuracies = []
lda_kappas = []

all_y_test = []
all_y_pred = []

print("===================================")
print(" Treinamento por participante")
print("===================================\n")

for subject in participants:

    print("===================================")
    print("Participante:", subject)
    print("Sessões de treinamento:", TRAIN_SESSIONS)
    print("Sessão de teste:", TEST_SESSION)
    print("===================================\n")

    train_mask = (
        (subject_ids == subject) &
        np.isin(session_ids, TRAIN_SESSIONS)
    )

    test_mask = (
        (subject_ids == subject) &
        (session_ids == TEST_SESSION)
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("Trials de treinamento:", len(y_train))
    print("Trials de teste:", len(y_test))
    print()

    train_left = np.sum(y_train == 0)
    train_right = np.sum(y_train == 1)

    test_left = np.sum(y_test == 0)
    test_right = np.sum(y_test == 1)

    print("Treinamento - LEFT:", train_left, "RIGHT:", train_right)
    print("Teste       - LEFT:", test_left, "RIGHT:", test_right)
    print()

    if len(y_train) == 0 or len(y_test) == 0:
        print("Participante ignorado: sem trials suficientes para treino/teste.\n")
        continue

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto"
                )
            )
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    print("Acurácia LDA:", f"{accuracy:.4f}")
    print("Cohen's kappa:", f"{kappa:.4f}")
    print()

    print("Matriz de Confusão:\n")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("Relatório de Classificação:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["LEFT", "RIGHT"]
        )
    )
    print()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    dummy_accuracy = accuracy_score(y_test, y_pred_dummy)

    print("Acurácia DummyClassifier:", f"{dummy_accuracy:.4f}")
    print()

    lda_accuracies.append(accuracy)
    dummy_accuracies.append(dummy_accuracy)
    lda_kappas.append(kappa)

    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)


print("===================================")
print(" Resumo Geral")
print("===================================\n")

print("Acurácia do LDA por participante:\n")
for subject, accuracy in zip(participants, lda_accuracies):
    print(subject, ":", f"{accuracy:.4f}")
print()

print("Acurácia do DummyClassifier por participante:\n")
for subject, accuracy in zip(participants, dummy_accuracies):
    print(subject, ":", f"{accuracy:.4f}")
print()

lda_accuracies = np.array(lda_accuracies)
dummy_accuracies = np.array(dummy_accuracies)

print("Acurácia média do LDA:", f"{lda_accuracies.mean():.4f}")
print("Desvio padrão do LDA:", f"{lda_accuracies.std():.4f}")
print()

print("Acurácia média do DummyClassifier:", f"{dummy_accuracies.mean():.4f}")
print()

print("Cohen's kappa do LDA por participante:\n")
for subject, kappa in zip(participants, lda_kappas):
    print(subject, ":", f"{kappa:.4f}")
print()

overall_kappa = cohen_kappa_score(all_y_test, all_y_pred)

print("Cohen's kappa geral do LDA:", f"{overall_kappa:.4f}")
print()

print("Matriz de Confusão geral:\n")
print(confusion_matrix(all_y_test, all_y_pred))
print()

print("Relatório de Classificação geral:\n")
print(
    classification_report(
        all_y_test,
        all_y_pred,
        target_names=["LEFT", "RIGHT"]
    )
)
print()

print("===================================")
print(" Avaliação concluída")
print("===================================")