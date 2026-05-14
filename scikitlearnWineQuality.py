import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ── 1. CARGA DEL DATASET ──────────────────────────────────────────────────────
# El dataset viene en dos archivos separados: vinos tintos y blancos.
# Los combinamos en uno solo para tener más datos (~6500 filas en total).
# El separador es ";" en lugar de "," — importante especificarlo.
df_red   = pd.read_csv("wine+quality\winequality-red.csv",   sep=";")
df_white = pd.read_csv("wine+quality\winequality-white.csv", sep=";")

# Agregamos una columna para distinguir el tipo de vino: 0 = tinto, 1 = blanco
# Esto agrega una feature extra que puede ser útil para la red.
df_red["type"]   = 0
df_white["type"] = 1

# Unimos los dos dataframes en uno solo y reiniciamos el índice
df = pd.concat([df_red, df_white], ignore_index=True)

# ── 2. CREAR LA VARIABLE OBJETIVO BINARIA ─────────────────────────────────────
# La columna "quality" tiene valores del 0 al 10 (escala de calidad del vino).
# Como necesitamos clasificación BINARIA, la convertimos:
#   calidad >= 7 → vino bueno → 1
#   calidad <  7 → vino malo  → 0
df["good_wine"] = (df["quality"] >= 7).astype(int)

# Eliminamos la columna original "quality" porque ya no la necesitamos
# (si la dejamos, la red haría trampa usándola como entrada)
df = df.drop(columns=["quality"])

# ── 3. SEPARAR ENTRADAS Y SALIDA ──────────────────────────────────────────────
# X: todas las columnas menos la última (good_wine)
# Y: solo la columna good_wine, que es lo que queremos predecir
X = df.iloc[:, :-1].values  # 12 features: las 11 físicoquímicas + type
Y = df.iloc[:, -1].values   # 1 = buen vino, 0 = mal vino

# ── 4. DIVISIÓN TRAIN / TEST ──────────────────────────────────────────────────
# Separamos 1/3 de los datos para evaluar el modelo al final.
# random_state=42 fija la semilla aleatoria para que el resultado sea
# reproducible: si volvés a correr el código, obtenés la misma división.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=42
)

# ── 5. NORMALIZACIÓN ──────────────────────────────────────────────────────────
# Las features tienen escalas muy distintas:
#   - alcohol: 8 a 15 %
#   - total sulfur dioxide: 0 a 440 mg/L
#   - density: ~0.99 a ~1.04 g/cm³
# Si no normalizamos, las features con valores grandes dominan el aprendizaje.
# Usamos StandardScaler (Z-score): lleva todo a media 0 y desvío estándar 1.
scaler = StandardScaler()

# fit_transform: aprende la media y desvío DE LOS DATOS DE ENTRENAMIENTO
# y los transforma. Nunca se ajusta con los datos de test.
X_train = scaler.fit_transform(X_train)

# transform: aplica la misma transformación aprendida en train al test.
# NO vuelve a calcular media/desvío — eso sería hacer trampa.
X_test  = scaler.transform(X_test)

# ── 6. DEFINICIÓN DE LA RED NEURONAL ─────────────────────────────────────────
# MLPClassifier = Multi-Layer Perceptron Classifier (red neuronal)
# Parámetros:
#   solver='sgd'              → descenso por gradiente estocástico
#   hidden_layer_sizes=(8, 4) → dos capas ocultas: 8 neuronas y luego 4
#                               Con 12 entradas esto es suficiente y simple
#   activation='relu'         → función ReLU en las capas ocultas
#   max_iter=1000             → máximo de épocas de entrenamiento
#   learning_rate_init=0.01   → tamaño del paso al actualizar los pesos
#   random_state=42           → semilla para reproducibilidad
nn = MLPClassifier(
    solver='sgd',
    hidden_layer_sizes=(8, 4),
    activation='relu',
    max_iter=1000,
    learning_rate_init=0.01,
    random_state=42
)

# ── 7. ENTRENAMIENTO ──────────────────────────────────────────────────────────
# fit() hace todo el trabajo internamente:
#   1. Inicializa los pesos aleatoriamente
#   2. Propaga los datos hacia adelante (forward propagation)
#   3. Calcula el error entre la predicción y el valor real
#   4. Propaga el error hacia atrás (backpropagation)
#   5. Actualiza los pesos con el learning rate
#   6. Repite max_iter veces
nn.fit(X_train, Y_train)

# ── 8. PESOS Y SESGOS ─────────────────────────────────────────────────────────
# coefs_: lista con las matrices de pesos de cada capa
# intercepts_: lista con los vectores de sesgos (bias) de cada capa
print("Forma de los pesos por capa:")
for i, w in enumerate(nn.coefs_):
    print(f"  Capa {i+1}: {w.shape}")

print("\nForma de los sesgos por capa:")
for i, b in enumerate(nn.intercepts_):
    print(f"  Capa {i+1}: {b.shape}")

# ── 9. EVALUACIÓN ─────────────────────────────────────────────────────────────
# score() calcula el accuracy: porcentaje de predicciones correctas
# Se compara la etiqueta predicha (0 o 1) con la etiqueta real
print("\nPuntaje en entrenamiento: %.4f" % nn.score(X_train, Y_train))
print("Puntaje en prueba:        %.4f" % nn.score(X_test,  Y_test))