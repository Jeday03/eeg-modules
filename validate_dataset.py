import numpy as np

FILE_NAME = "mi_trials.npz"

data = np.load(FILE_NAME, allow_pickle=True)

labels = data["labels"]
ts_start = data["ts_start"]
ts_end = data["ts_end"]
X = data["X"]
X_pre = data["X_pre"]
features = data["features"]

print("========== DATASET ==========\n")

print("Total trials:", len(labels))
print()

left = np.sum(labels == "LEFT")
right = np.sum(labels == "RIGHT")

print("LEFT trials :", left)
print("RIGHT trials:", right)
print()

print("Shape X:", X.shape)
print("Shape X_pre:", X_pre.shape)
print("Shape Features:", features.shape)
print()

print("Features por trial:", features.shape[1])
print()

print("NaN em X:", np.isnan(X).any())
print("NaN em X_pre:", np.isnan(X_pre).any())
print("NaN em Features:", np.isnan(features).any())
print()

print("Inf em X:", np.isinf(X).any())
print("Inf em X_pre:", np.isinf(X_pre).any())
print("Inf em Features:", np.isinf(features).any())
print()

print("Primeiros labels:")
print(labels[:10])

print()

print("Primeiro intervalo de timestamps:")
print(ts_start[0], "->", ts_end[0])