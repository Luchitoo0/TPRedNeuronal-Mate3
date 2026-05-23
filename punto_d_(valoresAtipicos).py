import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── CARGA Y PREPARACIÓN ───────────────────────────────────────────────────────
df_red   = pd.read_csv("wine+quality\winequality-red.csv",   sep=";")
df_white = pd.read_csv("wine+quality\winequality-white.csv", sep=";")
df_red["type"]   = 0
df_white["type"] = 1
df = pd.concat([df_red, df_white], ignore_index=True)

umbral = df["quality"].mean()
df["good_wine"] = (df["quality"] >= umbral).astype(float)
df = df.drop(columns=["quality", "type"])

features = ["alcohol", "density", "volatile acidity", "chlorides",
            "residual sugar", "total sulfur dioxide", "fixed acidity",
            "citric acid", "free sulfur dioxide", "sulphates", "pH"]

# ── GRILLA DE SUBGRÁFICOS ─────────────────────────────────────────────────────
# 11 features → grilla 3x4 (el último subplot queda vacío)
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
fig.suptitle("Media vs Mediana por feature\n(diferencia grande indica outliers influyentes)",
             fontsize=15, fontweight="bold", y=1.01)

# ── UN BOXPLOT POR FEATURE ────────────────────────────────────────────────────
for ax, feat in zip(axes.flatten(), features):
    valores = df[feat].dropna().values
    media   = np.mean(valores)
    mediana = np.median(valores)

    # Diferencia relativa entre media y mediana (en %)
    # Si supera el 5% consideramos que los outliers son influyentes
    dif_pct     = abs(media - mediana) / mediana * 100
    es_problema = dif_pct > 5

    # Rojo si hay problema, azul si no
    color_caja = "#EF9A9A" if es_problema else "#90CAF9"

    # Dibujamos el boxplot
    # medianprops con linewidth=0 oculta la línea de mediana default
    # porque la vamos a dibujar nosotros con axhline para que sea más visible
    bp = ax.boxplot(
        valores,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=2.5,
                        alpha=0.3, linestyle="none",
                        markerfacecolor="gray")
    )
    bp["boxes"][0].set_facecolor(color_caja)
    bp["boxes"][0].set_alpha(0.8)

    # Línea horizontal de MEDIANA (azul sólida)
    ax.axhline(mediana, color="blue", linewidth=2, linestyle="-", zorder=5)

    # Línea horizontal de MEDIA (roja punteada)
    ax.axhline(media,   color="red",  linewidth=2, linestyle="--", zorder=5)

    # Anotación con el porcentaje de diferencia
    ax.annotate(
        f"Dif: {dif_pct:.1f}%",
        xy=(1.32, (media + mediana) / 2),
        fontsize=9,
        color="darkred" if es_problema else "gray",
        fontweight="bold" if es_problema else "normal",
        va="center"
    )

    # Título con advertencia si hay problema
    titulo      = f"{feat}\n⚠ OUTLIERS INFLUYENTES" if es_problema else feat
    color_titulo = "darkred" if es_problema else "black"
    ax.set_title(titulo, fontsize=9.5, fontweight="bold",
                 pad=6, color=color_titulo)

    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Ocultamos el subplot vacío (posición 12 de una grilla 3x4)
axes.flatten()[-1].set_visible(False)

# ── LEYENDA GLOBAL ────────────────────────────────────────────────────────────
linea_mediana = Line2D([0], [0], color="blue", linewidth=2,
                        label="Mediana")
linea_media   = Line2D([0], [0], color="red",  linewidth=2,
                        linestyle="--", label="Media")
patch_prob    = mpatches.Patch(color="#EF9A9A", alpha=0.8,
                                label="Outliers influyentes (dif > 5%)")
patch_ok      = mpatches.Patch(color="#90CAF9", alpha=0.8,
                                label="Sin problema (dif ≤ 5%)")

fig.legend(handles=[linea_mediana, linea_media, patch_prob, patch_ok],
           loc="lower right", fontsize=10, frameon=True,
           bbox_to_anchor=(0.98, 0.02))

plt.tight_layout()
plt.savefig("media_vs_mediana.png", dpi=150, bbox_inches="tight")
plt.show()