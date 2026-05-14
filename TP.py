import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────────
df_red   = pd.read_csv(r"wine+quality\winequality-red.csv",   sep=";")
df_white = pd.read_csv(r"wine+quality\winequality-white.csv", sep=";")

df_red["type"]   = 0
df_white["type"] = 1

df = pd.concat([df_red, df_white], ignore_index=True)

# Creamos el target binario: calidad >= 7 → buen vino (1), sino (0)
df["good_wine"] = (df["quality"] >= 7).astype(float)
df = df.drop(columns=["quality"])

# ── 2. ENTRADAS Y SALIDA ──────────────────────────────────────────────────────
all_inputs  = df.iloc[:, :-1].values   # shape: (6497, 12)
all_outputs = df.iloc[:, -1].values    # shape: (6497,)

# ── 3. NORMALIZACIÓN ──────────────────────────────────────────────────────────
scaler     = StandardScaler()
all_inputs = scaler.fit_transform(all_inputs)

# ── 4. DIVISIÓN TRAIN / TEST ──────────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    all_inputs, all_outputs, test_size=1/3, random_state=42
)

n = X_train.shape[0]   # cantidad de muestras de entrenamiento
print(f"Muestras de entrenamiento: {n}")

# ── 5. PESOS Y SESGOS ─────────────────────────────────────────────────────────
# 1 capa oculta con 8 neuronas, 12 entradas, 1 salida
w_hidden = np.random.rand(8, 12) * 0.01   # (8 x 12)
w_output = np.random.rand(1, 8)  * 0.01   # (1 x 8)
b_hidden = np.random.rand(8, 1)            # (8 x 1)
b_output = np.random.rand(1, 1)            # (1 x 1)

L = 0.05   # tasa de aprendizaje

# ── 6. FUNCIONES DE ACTIVACIÓN Y SUS DERIVADAS ───────────────────────────────
relu       = lambda x: np.maximum(x, 0)
logistic   = lambda x: 1 / (1 + np.exp(-x))
d_relu     = lambda x: (x > 0).astype(float)
d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

# ── 7. FORWARD PROPAGATION ────────────────────────────────────────────────────
# X: shape (12, 1) → una muestra en columna
# Z1: combinación lineal capa oculta  (8x12) @ (12x1) = (8x1)
# A1: ReLU aplicada a Z1              (8x1)
# Z2: combinación lineal capa salida  (1x8)  @ (8x1)  = (1x1)
# A2: logística aplicada a Z2         (1x1)  → probabilidad entre 0 y 1
def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

# ── 8. BACKWARD PROPAGATION ───────────────────────────────────────────────────
# Calcula el gradiente del error respecto a cada peso, usando regla de la cadena
def backward_prop(Z1, A1, Z2, A2, X, Y):
    dC_dA2 = 2 * A2 - 2 * Y            # derivada del error cuadrático

    dA2_dZ2 = d_logistic(Z2)           # derivada de la logística

    dC_dW2 = dC_dA2 * dA2_dZ2 * A1.T  # gradiente w_output  (1x8)
    dC_dB2 = dC_dA2 * dA2_dZ2          # gradiente b_output  (1x1)

    dC_dA1 = w_output.T * (dC_dA2 * dA2_dZ2)  # error hacia capa oculta (8x1)
    dA1_dZ1 = d_relu(Z1)               # derivada de ReLU    (8x1)

    dC_dW1 = dC_dA1 * dA1_dZ1 * X.T   # gradiente w_hidden  (8x12)
    dC_dB1 = dC_dA1 * dA1_dZ1          # gradiente b_hidden  (8x1)

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2

# ── 9. ENTRENAMIENTO ──────────────────────────────────────────────────────────
# En cada iteración tomamos 1 muestra aleatoria, propagamos, calculamos
# el error, retropropagamos y actualizamos los pesos (SGD)
for i in range(10_000):
    idx      = np.random.choice(n, 1, replace=False)
    X_sample = X_train[idx].transpose()   # (12, 1)
    Y_sample = Y_train[idx]               # 0 o 1

    Z1, A1, Z2, A2 = forward_prop(X_sample)
    dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    w_hidden -= L * dW1
    b_hidden -= L * dB1
    w_output -= L * dW2
    b_output -= L * dB2

    if i % 10_000 == 0:
        loss = float((A2 - Y_sample).flatten()[0] ** 2)
        print(f"Iteración {i:6d} — Loss: {loss:.4f}")

# ── 10. EVALUACIÓN ────────────────────────────────────────────────────────────
test_predictions = forward_prop(X_test.transpose())[3]   # A2: (1, n_test)
test_comparisons = np.equal(
    (test_predictions >= 0.5).flatten().astype(int), Y_test
)
accuracy = sum(test_comparisons.astype(int)) / X_test.shape[0]
print(f"\nACCURACY: {accuracy:.4f}")