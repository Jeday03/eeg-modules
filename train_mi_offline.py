import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

DATASET_FILE = "mi_trials.npz"

print("===================================")
print(" Carregando dataset")
print("===================================\n")

data = np.load(DATASET_FILE, allow_pickle=True)
features = data["features"]
labels = data["labels"]

print("Dataset carregado.\n")

X = features
y = labels

print("========== VALIDAÇÃO ==========\n")
print("Total de trials:", len(y))
print("Número de features:", X.shape[1])
print()

left = np.sum(y == "LEFT")
right = np.sum(y == "RIGHT")

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
y = np.array([label_map[label] for label in y])

print("Labels convertidos.\n")
print("===================================")
print(" Treinamento LDA")
print("===================================\n")


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


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


scores = cross_val_score(

    model,

    X,

    y,

    cv=cv,

    scoring="accuracy"

)


print("Acurácia por fold:\n")

for i, score in enumerate(scores):

    print(

        f"Fold {i+1}: {score:.4f}"

    )


print()

print(

    "Acurácia média:",

    f"{scores.mean():.4f}"

)

print(

    "Desvio padrão:",

    f"{scores.std():.4f}"

)

print()
print("===================================")
print(" Avaliação do Modelo")
print("===================================\n")


y_pred = cross_val_predict(

    model,

    X,

    y,

    cv=cv

)


print("Matriz de Confusão:\n")

print(

    confusion_matrix(

        y,

        y_pred

    )

)

print()

print("Relatório de Classificação:\n")

print(

    classification_report(

        y,

        y_pred,

        target_names=[

            "LEFT",

            "RIGHT"

        ]

    )

)

print()


print("===================================")
print(" Dummy Classifier")
print("===================================\n")


dummy = DummyClassifier(

    strategy="most_frequent"

)


dummy_scores = cross_val_score(

    dummy,

    X,

    y,

    cv=cv,

    scoring="accuracy"

)


print("Acurácia por fold:\n")

for i, score in enumerate(dummy_scores):

    print(

        f"Fold {i+1}: {score:.4f}"

    )


print()

print(

    "Acurácia média:",

    f"{dummy_scores.mean():.4f}"

)

print(

    "Desvio padrão:",

    f"{dummy_scores.std():.4f}"

)

print()
print("===================================")
print(" Treinamento Final")
print("===================================\n")


model.fit(
    X,
    y
)


joblib.dump(
    model,
    "mi_lda_model.joblib"
)


print("Modelo treinado com todo o dataset.")

print("Modelo salvo em:")

print("mi_lda_model.joblib")

print()

print("===================================")
print(" Treinamento concluído")
print("===================================")