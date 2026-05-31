import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
red = pd.read_csv("wine+quality\winequality-red.csv",   sep=";")
white = pd.read_csv("wine+quality\winequality-white.csv", sep=";")
df = pd.concat([red, white], ignore_index=True)

# Binarizar variable objetivo: 1 = bueno (calidad > promedio), 0 = malo (calidad <= promedio)
umbral = df['quality'].mean()
df['quality'] = (df['quality'] > umbral).astype(int)

# Separar entradas y salida
all_inputs  = df.iloc[:, :-1].values   # 11 columnas fisicoquímicas
all_outputs = df['quality'].values

# Split primero, normalizar después (evita data leakage)
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs, test_size=1/3, random_state=42)

# Calcular min/max SOLO sobre el conjunto de entrenamiento
input_min = X_train.min(axis=0)
input_max = X_train.max(axis=0)

# Aplicar la misma transformación a train y test
X_train = (X_train - input_min) / (input_max - input_min)
X_test  = (X_test  - input_min) / (input_max - input_min)

n = X_train.shape[0]  # cantidad de registros de entrenamiento
print(f"Registros de entrenamiento: {n}")
print(f"Registros de test: {X_test.shape[0]}")

# 2. ARQUITECTURA DE LA RED
# Capa de entrada: 11 neuronas (una por feature)
# Capa oculta:    8 neuronas + ReLU
# Capa de salida: 1 neurona  + Sigmoid

n_input  = 11   # features del dataset
n_hidden = 32    # neuronas capa oculta
n_output = 1    # clasificación binaria

# 3. INICIALIZACIÓN DE PESOS Y SESGOS
np.random.seed(3)
w_hidden = np.random.randn(n_hidden, n_input) * np.sqrt(2 / n_input)
w_output = np.random.randn(n_output, n_hidden) * np.sqrt(2 / n_hidden)

b_hidden = np.zeros((n_hidden, 1))
b_output = np.zeros((n_output, 1))


# 4. FUNCIONES DE ACTIVACIÓN Y SUS DERIVADAS
def relu(x):
    return np.maximum(x, 0)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def d_relu(x):
    return (x > 0).astype(float)

def d_logistic(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


# 5. FORWARD PROPAGATION
def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden   # combinación lineal capa oculta
    A1 = relu(Z1)                  # activación ReLU
    Z2 = w_output @ A1 + b_output  # combinación lineal capa de salida
    A2 = logistic(Z2)              # activación Sigmoid -> probabilidad [0,1]
    return Z1, A1, Z2, A2


# 6. FUNCIÓN DE COSTO (MSE)
def costo(A2, Y):
    return np.mean((A2 - Y) ** 2)


# 7. BACKPROPAGATION
def backward_prop(Z1, A1, Z2, A2, X, Y):
    dC_dA2 = 2 * A2 - 2 * Y           # (1,1)
    dA2_dZ2 = d_logistic(Z2)           # (1,1)
    dA1_dZ1 = d_relu(Z1)               # (n_hidden,1)

    # Gradientes capa de salida
    dC_dW2 = dC_dA2 * dA2_dZ2 @ A1.T  # (1,1)*(1,1) @ (1,n_hidden) = (1,n_hidden)
    dC_dB2 = dC_dA2 * dA2_dZ2         # (1,1)

    # Gradientes capa oculta (regla de la cadena: propagamos el error hacia atrás)
    dC_dA1 = w_output.T @ (dC_dA2 * dA2_dZ2)  # (n_hidden,1) @ (1,1) = (n_hidden,1)
    dC_dW1 = dC_dA1 * dA1_dZ1 @ X.T            # (n_hidden,1)*(n_hidden,1) @ (1,n_input) = (n_hidden,n_input)
    dC_dB1 = dC_dA1 * dA1_dZ1                   # (n_hidden,1)

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2


# 8. ENTRENAMIENTO - DESCENSO POR GRADIENTE ESTOCÁSTICO
L = 0.05         # tasa de aprendizaje
epochs = 100_000   # iteraciones

train_losses   = []
test_losses    = []
train_accuracy = []
test_accuracy  = []

for i in range(epochs):
    # Seleccionar un sample aleatorio (SGD)
    idx = np.random.choice(n, 1, replace=False)
    #idx = np.random.choice(n, 32, replace=False)
    X_sample = X_train[idx].T
    Y_sample = Y_train[idx]

    # Forward
    Z1, A1, Z2, A2 = forward_prop(X_sample)

    # Backward
    dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    # Actualizar pesos
    w_hidden -= L * dW1
    b_hidden -= L * dB1
    w_output -= L * dW2
    b_output -= L * dB2

    # Registrar métricas cada 1000 epochs
    if i % 1000 == 0:
        # Loss
        _, _, _, A2_train = forward_prop(X_train.T)
        _, _, _, A2_test  = forward_prop(X_test.T)
        train_losses.append(costo(A2_train, Y_train))
        test_losses.append(costo(A2_test, Y_test))

        # Accuracy
        acc_train = np.mean((A2_train.flatten() >= 0.5).astype(int) == Y_train)
        acc_test  = np.mean((A2_test.flatten()  >= 0.5).astype(int) == Y_test)
        train_accuracy.append(acc_train)
        test_accuracy.append(acc_test)


# 9. RESULTADOS FINALES
_, _, _, A2_final = forward_prop(X_test.T)
acc_final = np.mean((A2_final.flatten() >= 0.5).astype(int) == Y_test)
print(f"\nAccuracy final en test: {acc_final:.4f} ({acc_final*100:.2f}%)")

# 10. CURVAS DE ENTRENAMIENTO
epochs_range = range(0, epochs, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Curva de pérdida
ax1.plot(epochs_range, train_losses, label='Train Loss', color='steelblue')
ax1.plot(epochs_range, test_losses,  label='Test Loss',  color='tomato')
ax1.set_title('Función de pérdida (MSE)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Curva de accuracy
ax2.plot(epochs_range, train_accuracy, label='Train Accuracy', color='steelblue')
ax2.plot(epochs_range, test_accuracy,  label='Test Accuracy',  color='tomato')
ax2.set_title('Precisión (Accuracy)')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curvas_entrenamiento.png', dpi=150)
plt.show()
print("Gráfico guardado como curvas_entrenamiento.png")


# Vino tinto clásico de buena calidad
vino1 = np.array([[8.0, 0.25, 0.45, 2.5, 0.05, 15.0, 80.0, 0.994, 3.3, 0.7, 11.5]])

# Vino barato con muchos sulfatos y alta acidez
vino2 = np.array([[10.0, 0.65, 0.05, 5.0, 0.12, 8.0, 180.0, 0.999, 3.1, 0.4, 9.0]])

# Vino en el límite, parámetros mediocres
vino3 = np.array([[7.5, 0.35, 0.25, 3.0, 0.07, 20.0, 110.0, 0.996, 3.25, 0.55, 10.8]])

for i, vino in enumerate([vino1, vino2, vino3], 1):
    vino_norm = (vino - input_min) / (input_max - input_min)
    _, _, _, pred = forward_prop(vino_norm.T)
    prob = pred[0][0]
    resultado = "BUEN VINO" if prob >= 0.5 else "MAL VINO"
    print(f"Vino {i}: {prob*100:.2f}% => {resultado}")