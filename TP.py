import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ── 1. CARGA DEL DATASET ──────────────────────────────────────────────────────
# Leemos el CSV desde disco. El dataset tiene 145.460 filas y 23 columnas.
df = pd.read_csv(r"D:\UNSAM\Matematica III\TP dataset\TPRedNeuronal-Mate3\weatherAUS.csv")

# ── 2. LIMPIEZA BÁSICA ────────────────────────────────────────────────────────
# Eliminamos columnas de texto que no podemos usar directamente como números:
# Date (fecha), Location (ciudad), y las tres columnas de dirección del viento.
df = df.drop(columns=["Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm"])

# Convertimos RainToday y RainTomorrow de "Yes"/"No" a 1 y 0.
# Esto es necesario porque la red solo entiende números.
df["RainToday"] = (df["RainToday"] == "Yes").astype(float)
df["RainTomorrow"] = (df["RainTomorrow"] == "Yes").astype(float)

# Eliminamos todas las filas que tengan al menos un valor nulo.
# El dataset tiene columnas con hasta 48% de nulos (Sunshine, Evaporation),
# por eso esta limpieza reduce bastante las filas, pero es la opción más simple.
df = df.dropna()

# ── 3. SEPARAR ENTRADAS Y SALIDA ──────────────────────────────────────────────
# X: todas las columnas menos la última (RainTomorrow)
# Y: solo la columna RainTomorrow, que es lo que queremos predecir
X = df.iloc[:, :-1].values   # shape: (filas, 16)
Y = df.iloc[:, -1].values    # shape: (filas,)

# ── 4. NORMALIZACIÓN ──────────────────────────────────────────────────────────
# A diferencia del ejemplo original (que dividía por 255 porque eran píxeles),
# acá las columnas tienen escalas muy distintas:
#   - Temperatura: -8 a 48 °C
#   - Presión: ~990 a ~1040 hPa
#   - Humedad: 0 a 100 %
# Usamos StandardScaler (Z-score): resta la media y divide por el desvío estándar,
# dejando todas las variables con media 0 y desvío 1.
# MUY IMPORTANTE: el scaler se ajusta SOLO con los datos de entrenamiento,
# y luego se aplica igual a test. Si lo ajustamos con test, "hacemos trampa".
scaler = StandardScaler()

# ── 5. DIVISIÓN TRAIN / TEST ──────────────────────────────────────────────────
# Separamos 1/3 de los datos para evaluar el modelo al final.
# random_state=42 fija la semilla para que el resultado sea reproducible.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=42)

# Ajustamos el scaler con train y transformamos ambos conjuntos
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 6. DEFINICIÓN DE LA RED NEURONAL ─────────────────────────────────────────
# MLPClassifier es la red neuronal de scikit-learn.
# Parámetros:
#   solver='sgd'            → usa descenso por gradiente estocástico (igual al ejemplo)
#   hidden_layer_sizes=(16,8) → dos capas ocultas: primera con 16 neuronas, segunda con 8
#   activation='relu'       → función de activación ReLU en las capas ocultas
#   max_iter=500            → máximo de épocas de entrenamiento
#   learning_rate_init=0.01 → tamaño del paso en cada actualización de pesos
#   random_state=42         → semilla para reproducibilidad
nn = MLPClassifier(
    solver='sgd',
    hidden_layer_sizes=(16, 8),
    activation='relu',
    max_iter=500,
    learning_rate_init=0.01,
    random_state=42
)

# ── 7. ENTRENAMIENTO ──────────────────────────────────────────────────────────
# fit() hace todo: inicializa pesos, propaga hacia adelante, calcula el error,
# retropropaga los gradientes y actualiza los pesos. Repite esto max_iter veces.
nn.fit(X_train, Y_train)

# ── 8. PESOS Y SESGOS ────────────────────────────────────────────────────────
# coefs_: lista de matrices de pesos por capa
# intercepts_: lista de vectores de sesgos (bias) por capa
print("Forma de los pesos por capa:")
for i, w in enumerate(nn.coefs_):
    print(f"  Capa {i+1}: {w.shape}")

# ── 9. EVALUACIÓN ─────────────────────────────────────────────────────────────
# score() calcula el accuracy: proporción de predicciones correctas
print("\nPuntaje en entrenamiento: %.4f" % nn.score(X_train, Y_train))
print("Puntaje en prueba:        %.4f" % nn.score(X_test, Y_test))