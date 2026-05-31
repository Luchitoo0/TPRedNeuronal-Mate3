import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# ── 1. CARGA DEL DATASET ──────────────────────────────────────────────────────
red   = pd.read_csv(r"wine+quality\winequality-red.csv",   sep=";")
white = pd.read_csv(r"wine+quality\winequality-white.csv", sep=";")
df    = pd.concat([red, white], ignore_index=True)

# ── 2. VARIABLE OBJETIVO BINARIA ──────────────────────────────────────────────
umbral = df['quality'].mean()
df['quality'] = (df['quality'] > umbral).astype(int)

all_inputs  = df.iloc[:, :-1].values
all_outputs = df['quality'].values

# ── 3. DIVISIÓN TRAIN / TEST ──────────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    all_inputs, all_outputs, test_size=1/3, random_state=42
)

# ── 4. NORMALIZACIÓN MIN-MAX ──────────────────────────────────────────────────
input_min = X_train.min(axis=0)
input_max = X_train.max(axis=0)
X_train = (X_train - input_min) / (input_max - input_min)
X_test  = (X_test  - input_min) / (input_max - input_min)

print(f"Registros de entrenamiento: {X_train.shape[0]}")
print(f"Registros de test:          {X_test.shape[0]}")

# ── 5. LAS 4 CONFIGURACIONES ──────────────────────────────────────────────────
# Config 1: 8 neuronas, lr=0.05
nn_c1 = MLPClassifier(hidden_layer_sizes=(8,), activation='relu',
                       solver='sgd', learning_rate_init=0.05,
                       max_iter=100_000, random_state=42)
nn_c1.fit(X_train, Y_train)
acc_c1_train = nn_c1.score(X_train, Y_train)
acc_c1_test  = nn_c1.score(X_test,  Y_test)
print("\n=== CONFIG 1: 8 neuronas, lr=0.05 ===")
print(f"Accuracy train: {acc_c1_train*100:.2f}%")
print(f"Accuracy test:  {acc_c1_test*100:.2f}%")

# Config 2: 8 neuronas, lr=0.01
nn_c2 = MLPClassifier(hidden_layer_sizes=(8,), activation='relu',
                       solver='sgd', learning_rate_init=0.01,
                       max_iter=100_000, random_state=42)
nn_c2.fit(X_train, Y_train)
acc_c2_train = nn_c2.score(X_train, Y_train)
acc_c2_test  = nn_c2.score(X_test,  Y_test)
print("\n=== CONFIG 2: 8 neuronas, lr=0.01 ===")
print(f"Accuracy train: {acc_c2_train*100:.2f}%")
print(f"Accuracy test:  {acc_c2_test*100:.2f}%")

# Config 3: 32 neuronas, lr=0.05  ← MEJOR
nn_c3 = MLPClassifier(hidden_layer_sizes=(32,), activation='relu',
                       solver='sgd', learning_rate_init=0.05,
                       max_iter=100_000, random_state=42)
nn_c3.fit(X_train, Y_train)
acc_c3_train = nn_c3.score(X_train, Y_train)
acc_c3_test  = nn_c3.score(X_test,  Y_test)
print("\n=== CONFIG 3: 32 neuronas, lr=0.05 ★ MEJOR ===")
print(f"Accuracy train: {acc_c3_train*100:.2f}%")
print(f"Accuracy test:  {acc_c3_test*100:.2f}%")

# Config 4: 32 neuronas, lr=0.01
nn_c4 = MLPClassifier(hidden_layer_sizes=(32,), activation='relu',
                       solver='sgd', learning_rate_init=0.01,
                       max_iter=100_000, random_state=42)
nn_c4.fit(X_train, Y_train)
acc_c4_train = nn_c4.score(X_train, Y_train)
acc_c4_test  = nn_c4.score(X_test,  Y_test)
print("\n=== CONFIG 4: 32 neuronas, lr=0.01 ===")
print(f"Accuracy train: {acc_c4_train*100:.2f}%")
print(f"Accuracy test:  {acc_c4_test*100:.2f}%")

# ── 6. GRÁFICO COMPARATIVO ────────────────────────────────────────────────────
modelos = [
    'Config 1\n(8n, lr=0.05)',
    'Config 2\n(8n, lr=0.01)',
    'Config 3 ★\n(32n, lr=0.05)',
    'Config 4\n(32n, lr=0.01)',
]
acc_train_vals = [acc_c1_train, acc_c2_train, acc_c3_train, acc_c4_train]
acc_test_vals  = [acc_c1_test,  acc_c2_test,  acc_c3_test,  acc_c4_test]

x     = np.arange(len(modelos))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, acc_train_vals, width, label='Train Accuracy',
               color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, acc_test_vals,  width, label='Test Accuracy',
               color='tomato', alpha=0.85)

# Resaltar la mejor (Config 3, índice 2)
bars1[2].set_edgecolor('green'); bars1[2].set_linewidth(2.5)
bars2[2].set_edgecolor('green'); bars2[2].set_linewidth(2.5)
ax.text(x[2], max(acc_train_vals[2], acc_test_vals[2]) + 0.018,
        '★ Mejor', ha='center', fontsize=10, color='green', fontweight='bold')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Comparación de 4 configuraciones — 1 capa oculta, SGD, 100k iteraciones',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modelos, fontsize=10)
ax.set_ylim(0.60, 0.85)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparacion_arquitecturas.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico guardado como comparacion_arquitecturas.png")