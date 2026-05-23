import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── CARGA Y PREPARACIÓN ───────────────────────────────────────────────────────
df_red   = pd.read_csv(r"wine+quality\winequality-red.csv",   sep=";")
df_white = pd.read_csv(r"wine+quality\winequality-white.csv", sep=";")
df_red['type']   = 0
df_white['type'] = 1
df = pd.concat([df_red, df_white], ignore_index=True)
umbral = df['quality'].mean()          # calcula el promedio automáticamente
df['good_wine'] = (df['quality'] >= umbral).astype(float)  # clasifica según ese umbral
df = df.drop(columns=['quality'])

# ── CONFIGURACIÓN DE LA GRILLA ────────────────────────────────────────────────
# 12 features → grilla de 3 filas x 4 columnas
# figsize=(18,13) controla el tamaño total de la imagen en pulgadas
features = ['alcohol', 'density', 'volatile acidity', 'chlorides',
            'residual sugar', 'total sulfur dioxide', 'fixed acidity',
            'citric acid', 'free sulfur dioxide', 'sulphates', 'pH', 'type']

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
fig.suptitle('Distribución de cada feature según calidad del vino',
             fontsize=16, fontweight='bold', y=1.01)

color_malo  = '#E57373'   # rojo suave
color_bueno = '#64B5F6'   # azul suave

# ── LOOP: un boxplot por feature ──────────────────────────────────────────────
for ax, feat in zip(axes.flatten(), features):
    # Separamos los datos según la clase
    malo  = df[df['good_wine'] == 0][feat].dropna().values
    bueno = df[df['good_wine'] == 1][feat].dropna().values

    # boxplot recibe una lista con los dos grupos
    # patch_artist=True permite colorear las cajas
    bp = ax.boxplot(
        [malo, bueno],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color='black', linewidth=2),   # línea de mediana en negro
        whiskerprops=dict(linewidth=1.2),               # bigotes
        capprops=dict(linewidth=1.2),                   # extremos de bigotes
        flierprops=dict(marker='o', markersize=2,       # outliers: puntos chicos
                        alpha=0.3, linestyle='none')
    )

    # Coloreamos cada caja por separado
    bp['boxes'][0].set_facecolor(color_malo)
    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_facecolor(color_bueno)
    bp['boxes'][1].set_alpha(0.8)

    # Agregamos un diamante (marker='D') para marcar la media
    # Es útil porque la mediana y la media no siempre coinciden
    ax.plot(1, np.mean(malo),  marker='D', color='darkred',  markersize=6, zorder=5)
    ax.plot(2, np.mean(bueno), marker='D', color='darkblue', markersize=6, zorder=5)

    # Estética de cada subgráfico
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Mal vino', 'Buen vino'], fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)     # ocultamos bordes innecesarios
    ax.spines['right'].set_visible(False)

# ── LEYENDA GLOBAL ────────────────────────────────────────────────────────────
# mpatches.Patch crea un cuadradito de color para la leyenda
patch_malo  = mpatches.Patch(color=color_malo,  alpha=0.8, label='Mal vino (quality < 7)')
patch_bueno = mpatches.Patch(color=color_bueno, alpha=0.8, label='Buen vino (quality ≥ 7)')
from matplotlib.lines import Line2D
linea_media = Line2D([0], [0], marker='D', color='gray',
                     markersize=7, linestyle='none', label='Media')

fig.legend(handles=[patch_malo, patch_bueno, linea_media],
           loc='lower center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('boxplots_features.png', dpi=150, bbox_inches='tight')
plt.show()