from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Configuración y carga de datos
ruta_csv = Path("resultados/resultados_por_prueba.csv")
salida_dir = Path("resultados/contrastes")
salida_dir.mkdir(parents=True, exist_ok=True)

v_nom = 5.0 / 3.6  # 1.388888... m/s
mu_ref = 0.35
m_kg = 65.0
w_ref = m_kg * 9.81  # 637.65 N

if ruta_csv.exists():
    df = pd.read_csv(ruta_csv)
    df.columns = [c.strip() for c in df.columns]
else:
    # Respaldo con los valores registrados de P1 a P5 si aún no está el CSV en la ruta
    df = pd.DataFrame({
        "prueba": ["P1", "P2", "P3", "P4", "P5"],
        "v_exp": [1.4365037127721623, 1.4584468654164044, 1.453833678384557, 1.4451585481430722, 1.349302585593998],
        "pct_error": [3.4282673195956868, 5.008174309981124, 4.676024843688101, 4.050839466301202, 2.8502138372321433],
        "Fx_max": [122.68797811492843, 103.61887625330145, 134.89623909499448, 133.98818662537843, 121.57813620749914],
        "mu_req": [1.5154340648406037, 0.1478147403360097, 4.911944704512949, 3.0976188120333614, 0.368814870764455],
        "mu_p95": [1.3208995823888616, 0.14582359783177864, 3.0887723667048976, 2.541368849560255, 0.30794785256688806],
        "F0": [118.03150548217951, 98.98881188774878, 130.6220333117701, 129.24332379524628, 112.9821804012904],
        "f": [1.8199335123573444, 1.8000338297400977, 1.8499221163974329, 1.8398237186566142, 1.7797832455235738],
        "rmse_x_B": [0.0030332433640855435, 0.002663650630461077, 0.0031325828773436626, 0.0031691830380779083, 0.0028575561691704327],
        "rmse_v_B": [0.01060, 0.0090812, 0.011781492, 0.0116066, 0.0102542],
    })

# Columnas calculadas para contraste
df["v_nom"] = v_nom
df["diff_v"] = df["v_exp"] - v_nom
df["a_cm_max_est"] = df["Fx_max"] / m_kg
df["cumple_5pct"] = np.where(df["pct_error"] <= 5.0, "Sí", "Límite/No")

# 2. Tabla dinámica / consolidada con Promedio y Desviación Estándar (ddof=1)
cols_resumen = ["v_exp", "diff_v", "pct_error", "Fx_max", "a_cm_max_est", "mu_req", "mu_p95", "f", "rmse_x_B", "rmse_v_B"]
promedios = df[cols_resumen].mean()
desv_std = df[cols_resumen].std(ddof=1)

tabla_dinamica = pd.DataFrame({
    "Promedio": promedios,
    "Desv. Est. (s)": desv_std
})

print("\n=== TABLA DE CONTRASTE POR PRUEBA ===")
print(df[["prueba", "v_nom", "v_exp", "diff_v", "pct_error", "Fx_max", "mu_req", "mu_p95", "f", "rmse_x_B", "rmse_v_B"]].round(4).to_string(index=False))

print("\n=== RESUMEN ESTADÍSTICO (P1–P5) ===")
print(tabla_dinamica.round(4).to_string())

df.to_csv(salida_dir / "tabla_contraste_pruebas.csv", index=False)
tabla_dinamica.to_csv(salida_dir / "tabla_resumen_dinamico.csv")

# 3. Gráfico 1: Velocidad experimental vs. velocidad nominal
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
barras = ax.bar(df["prueba"], df["v_exp"], width=0.52, color="#2B6CB0", label="Velocidad experimental ($v_{exp}$)")
ax.axhline(v_nom, color="#C53030", linestyle="--", linewidth=1.6, label=f"Velocidad nominal ($v_{{nom}} = {v_nom:.4f}\\text{{ m/s}}$)")
ax.axhline(promedios["v_exp"], color="#2F855A", linestyle=":", linewidth=1.6, label=f"Promedio $v_{{exp}}$ ({promedios['v_exp']:.4f} m/s)")

for b, err in zip(barras, df["pct_error"]):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 0.004, f"{h:.3f}\n({err:.2f}%)",
            ha="center", va="bottom", fontsize=8.5)

ax.set_ylim(1.30, 1.50)
ax.set_ylabel("Velocidad (m/s)")
ax.set_xlabel("Prueba")
ax.set_title("Velocidad experimental en apoyo medio frente a velocidad nominal (eje recortado)")
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig(salida_dir / "contraste_velocidades.png")
plt.close()

# 4. Gráfico 2: Fuerza horizontal máxima y coeficiente requerido por prueba
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=300)

ax1.bar(df["prueba"], df["Fx_max"], width=0.52, color="#3182CE")
ax1.axhline(promedios["Fx_max"], color="#2D3748", linestyle="--", linewidth=1.3, label=f"Media: {promedios['Fx_max']:.1f} N")
for i, val in enumerate(df["Fx_max"]):
    ax1.text(i, val + 2.5, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)
ax1.set_ylim(0, 155)
ax1.set_ylabel("Fuerza horizontal máxima $F_{x,\\text{máx}}$ (N)")
ax1.set_xlabel("Prueba")
ax1.set_title("Fuerza horizontal máxima por ensayo")
ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.legend(fontsize=8)

x_idx = np.arange(len(df))
ancho = 0.36
ax2.bar(x_idx - ancho / 2, df["mu_req"], width=ancho, color="#DD6B20", label="$\\mu_{\\text{req}}$ (máx)")
ax2.bar(x_idx + ancho / 2, df["mu_p95"], width=ancho, color="#ECC94B", label="$\\mu_{p95}$ (percentil 95)")
ax2.axhline(mu_ref, color="#C53030", linestyle="--", linewidth=1.5, label=f"Referencia $\\mu_s = {mu_ref:.2f}$")
ax2.set_xticks(x_idx)
ax2.set_xticklabels(df["prueba"])
ax2.set_ylabel("Coeficiente adimensional $\\mu$")
ax2.set_xlabel("Prueba")
ax2.set_title("Coeficiente de fricción requerido vs. $\\mu_s = 0.35$")
ax2.grid(axis="y", linestyle="--", alpha=0.35)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(salida_dir / "contraste_dinamica.png")
plt.close()

print(f"\nArchivos generados en: {salida_dir.resolve()}")