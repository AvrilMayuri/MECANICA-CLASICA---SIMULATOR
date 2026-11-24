import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# =====================================================================
# CONFIGURACIÓN ADAPTADA A TUS EXPORTACIONES DE TRACKER
# =====================================================================
ARCHIVOS = [
    "PRUEBA 1 - ANÁLISIS.csv",
    "PRUEBA 2 - ANÁLISIS.csv",
    "PRUEBA 3 - ANÁLISIS.csv",
    "PRUEBA 4 - ANÁLISIS.csv",
    "PRUEBA 5 - ANÁLISIS.csv",
]

M = 65.0                 # masa del usuario (kg)
G = 9.81                 # gravedad (m/s^2)
FPS = 60.0               # fotogramas por segundo del video
V_NOM = 5.0 / 3.6        # velocidad nominal de la banda (m/s)
MU_S = 0.35              # coeficiente de fricción estática de referencia (cota)
UMBRAL_ERROR = 5.0       # meta de H1 (%)
ESCALA_M_PX = None       # escala de Tracker en m/px (opcional)
SUAVIZADO = {"metodo": "savgol", "ventana": 9, "orden": 3}
APOYO = {
    "fraccion_altura": 0.15,
    "min_muestras": 6,
    "fraccion_central": 0.5,
}
CARPETA_SALIDA = "resultados"


# =====================================================================
# LECTURA Y PREPARACIÓN
# =====================================================================
def cargar_tracker_txt(ruta):
    """Lee archivos exportados de Tracker (#multi: t, x_cad, y_cad, x_tob, y_tob)."""
    with open(ruta, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    filas = []
    for line in lines:
        if line.startswith("#") or line.startswith(",") or line.lower().startswith("t,"):
            continue
        parts = [p.strip().replace(",", ".") for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            t_val = float(parts[0])
        except ValueError:
            continue

        def a_float(idx):
            return float(parts[idx]) if parts[idx] != "" else np.nan

        filas.append({
            "t": t_val,
            "x_cadera": a_float(1),
            "y_cadera": a_float(2),
            "x_tobillo": a_float(3),
            "y_tobillo": a_float(4),
        })

    df = pd.DataFrame(filas).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    n_nan = int(df.isna().sum().sum())
    if n_nan:
        print(f"  [aviso] {ruta}: {n_nan} valores faltantes; se interpolan huecos.")
        df = df.interpolate(method="linear", limit_direction="both").dropna().reset_index(drop=True)

    t = df["t"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    if abs(dt - 1.0 / FPS) > 0.02 / FPS:
        print(f"  [aviso] {ruta}: dt medido = {dt:.5f} s, esperado 1/{FPS:.0f} = {1/FPS:.5f} s.")

    datos = {
        "x_cadera": df["x_cadera"].to_numpy(float),
        "y_cadera": df["y_cadera"].to_numpy(float),
        "x_tobillo": df["x_tobillo"].to_numpy(float),
        "y_tobillo": df["y_tobillo"].to_numpy(float),
    }
    return t, datos, dt


def suavizar(y):
    """Suaviza una serie de posición según SUAVIZADO."""
    v = SUAVIZADO["ventana"]
    if len(y) < 3:
        return y.copy()
    if SUAVIZADO["metodo"] == "savgol":
        v = min(v if v % 2 else v + 1, len(y) - (1 - len(y) % 2))
        if v > SUAVIZADO["orden"]:
            return savgol_filter(y, v, SUAVIZADO["orden"], mode="interp")
    return pd.Series(y).rolling(v, center=True, min_periods=1).mean().to_numpy()


def derivadas(x, dt):
    """Diferencias finitas centrales: v_i = (x_{i+1}-x_{i-1})/(2dt), a_i = (x_{i+1}-2x_i+x_{i-1})/dt^2."""
    v = (x[2:] - x[:-2]) / (2 * dt)
    a = (x[2:] - 2 * x[1:-1] + x[:-2]) / dt ** 2
    return v, a


# =====================================================================
# FASE DE APOYO
# =====================================================================
def segmentos_apoyo(t, x_raw, y_suav):
    """Detecta apoyos completos y ajusta x(t) del tobillo en su parte central."""
    n = len(t)
    umbral = y_suav.min() + APOYO["fraccion_altura"] * (y_suav.max() - y_suav.min())
    mask = y_suav <= umbral
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    inicios, fines = np.where(d == 1)[0], np.where(d == -1)[0]
    segs, mask_central = [], np.zeros(n, bool)
    for i0, i1 in zip(inicios, fines):
        if i0 == 0 or i1 == n:
            continue
        L = i1 - i0
        if L < APOYO["min_muestras"]:
            continue
        k = int(round(L * (1 - APOYO["fraccion_central"]) / 2))
        j0, j1 = i0 + k, i1 - k
        if j1 - j0 < 4:
            continue
        pend = np.polyfit(t[j0:j1], x_raw[j0:j1], 1)[0]
        segs.append({"i0": i0, "i1": i1, "j0": j0, "j1": j1, "pendiente": pend})
        mask_central[j0:j1] = True
    return umbral, mask, mask_central, segs


# =====================================================================
# ANÁLISIS DE UNA PRUEBA (cinemática y dinámica)
# =====================================================================
def analizar_prueba(ruta):
    t, d, dt = cargar_tracker_txt(ruta)

    y_tob_s = suavizar(d["y_tobillo"])
    umbral, mask, mask_c, segs = segmentos_apoyo(t, d["x_tobillo"], y_tob_s)
    pend = np.array([s["pendiente"] for s in segs])
    if len(pend) == 0:
        print(f"  [aviso] {ruta}: no se detectó ningún apoyo completo; ajusta APOYO.")
    v_exp = float(np.mean(np.abs(pend))) if len(pend) else np.nan
    pct = abs(v_exp - V_NOM) / V_NOM * 100 if len(pend) else np.nan

    x_cm, y_cm = suavizar(d["x_cadera"]), suavizar(d["y_cadera"])
    tc = t[1:-1]
    vx, ax = derivadas(x_cm, dt)
    _, ay = derivadas(y_cm, dt)
    Fx = M * ax
    N = M * (G + ay)
    mu = np.where(N > 0, np.abs(Fx) / np.where(N > 0, N, np.nan), np.nan)

    mu_valid = mu[~np.isnan(mu)]
    res = {
        "archivo": ruta, "t": t, "dt": dt, "d": d,
        "v_exp": v_exp, "pct_error": pct, "n_apoyos": len(segs), "segs": segs,
        "umbral_apoyo": umbral, "mask_apoyo": mask, "mask_central": mask_c, "y_tob_s": y_tob_s,
        "tc": tc, "x_cm": x_cm, "vx": vx, "ax": ax, "ay": ay, "Fx": Fx, "N": N, "mu": mu,
        "Fx_max": float(np.max(np.abs(Fx))),
        "mu_req": float(np.max(mu_valid)) if len(mu_valid) else np.nan,
        "mu_p95": float(np.percentile(mu_valid, 95)) if len(mu_valid) else np.nan,
        "sigma": None,
    }
    if ESCALA_M_PX is not None:
        sx = ESCALA_M_PX
        res["sigma"] = {
            "sx": sx,
            "sv": sx / (np.sqrt(2) * dt),
            "sa": np.sqrt(6) * sx / dt ** 2,
            "sF": M * np.sqrt(6) * sx / dt ** 2,
        }
    return res


# =====================================================================
# SIMULACIÓN DE EULER
# =====================================================================
def euler(t, a, x0, v0):
    """v_{i+1} = v_i + a_i dt ;  x_{i+1} = x_i + v_i dt."""
    n = len(t)
    x, v = np.empty(n), np.empty(n)
    x[0], v[0] = x0, v0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        v[i + 1] = v[i] + a[i] * h
        x[i + 1] = x[i] + v[i] * h
    return x, v


def modelo_fuerza(t, F0, f, phi):
    return F0 * np.sin(2 * np.pi * f * t + phi)


def ajustar_fuerza(t, Fx, dt):
    """Ajusta F0, f, phi de F_x(t) = F0 sin(2 pi f t + phi) a la F_x medida."""
    tt = t - t[0]
    y = Fx - np.mean(Fx)
    freqs = np.fft.rfftfreq(len(y), dt)
    esp = np.abs(np.fft.rfft(y))
    banda = (freqs >= 0.3) & (freqs <= 4.0)
    f0 = freqs[banda][np.argmax(esp[banda])] if banda.any() else 1.0
    F00 = max(np.sqrt(2) * np.std(y), 1e-3)
    mejor, sse_min = None, np.inf
    for phi0 in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        try:
            p, _ = curve_fit(
                modelo_fuerza, tt, Fx,
                p0=[F00, f0, phi0],
                bounds=([0, 0.3, -2 * np.pi], [np.inf, 4.0, 2 * np.pi]),
                maxfev=20000,
            )
        except RuntimeError:
            continue
        sse = np.sum((modelo_fuerza(tt, *p) - Fx) ** 2)
        if sse < sse_min:
            mejor, sse_min = p, sse
    if mejor is None:
        raise RuntimeError("No convergió el ajuste de F_x(t); revisa los datos.")
    return mejor


def metricas(sim, ref):
    err = sim - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rango = float(np.max(ref) - np.min(ref))
    return {"rmse": rmse, "nrmse": rmse / rango if rango > 0 else np.nan, "max_abs": float(np.max(np.abs(err)))}


def simular(res, ci="media_cero"):
    tc, dt = res["tc"], res["dt"]
    x_ref, v_ref = res["x_cm"][1:-1], res["vx"]
    x0 = x_ref[0]

    xA, vA = euler(tc, res["ax"], x0, v_ref[0])

    F0, f, phi = ajustar_fuerza(tc, res["Fx"], dt)
    tt = tc - tc[0]
    aB = modelo_fuerza(tt, F0, f, phi) / M
    if ci == "media_cero":
        _, v_sin = euler(tc, aB, x0, 0.0)
        v0 = np.mean(v_ref) - np.mean(v_sin)
    else:
        v0 = v_ref[0]
    xB, vB = euler(tc, aB, x0, v0)

    n = len(tc)
    t_f = tc[0] + np.arange(2 * (n - 1) + 1) * dt / 2
    a_f = modelo_fuerza(t_f - tc[0], F0, f, phi) / M
    xF, vF = euler(t_f, a_f, x0, v0)
    dif = {"x_max": float(np.max(np.abs(xF[::2] - xB))), "v_max": float(np.max(np.abs(vF[::2] - vB)))}

    return {
        "F0": F0, "f": f, "phi": phi, "v0": v0, "ci": ci,
        "xA": xA, "vA": vA, "xB": xB, "vB": vB, "x_ref": x_ref, "v_ref": v_ref,
        "A_x": metricas(xA, x_ref), "A_v": metricas(vA, v_ref),
        "B_x": metricas(xB, x_ref), "B_v": metricas(vB, v_ref), "dif_dt": dif,
    }


# =====================================================================
# GRÁFICAS
# =====================================================================
def graficar_prueba(nombre, res, sim, carpeta):
    t, d = res["t"], res["d"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t, d["x_cadera"], ".", ms=2.5, label="x cadera")
    ax[0, 0].plot(res["tc"], res["x_cm"][1:-1], "-", lw=1.5, label="x_CM suavizado")
    ax[0, 0].set(title="Cadera: x(t)", xlabel="t (s)", ylabel="x (m)")
    ax[0, 0].legend()

    ax[0, 1].plot(t, d["x_tobillo"], "-", lw=1, color="gray", label="x tobillo")
    for s in res["segs"]:
        sl = slice(s["j0"], s["j1"])
        ax[0, 1].plot(t[sl], d["x_tobillo"][sl], "o", ms=3, color="tab:blue")
        ax[0, 1].plot(t[sl], np.polyval(np.polyfit(t[sl], d["x_tobillo"][sl], 1), t[sl]), "r-", lw=2)
    ax[0, 1].set(title=f"Tobillo: apoyo medio (v_exp = {res['v_exp']:.3f} m/s)", xlabel="t (s)", ylabel="x (m)")
    ax2 = ax[0, 1].twinx()
    ax2.plot(t, res["y_tob_s"], color="tab:green", lw=0.8, alpha=0.7)
    ax2.axhline(res["umbral_apoyo"], color="tab:green", ls="--", lw=0.8)
    ax2.set_ylabel("y tobillo (m)", color="tab:green")

    ax[1, 0].plot(res["tc"], res["Fx"], label="F_x medida (m·a_x)")
    ax[1, 0].plot(res["tc"], modelo_fuerza(res["tc"] - res["tc"][0], sim["F0"], sim["f"], sim["phi"]),
                  "--", label=f"modelo seno (f = {sim['f']:.2f} Hz)")
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].set(title="Fuerza horizontal de contacto", xlabel="t (s)", ylabel="F_x (N)")
    ax[1, 0].legend()

    ax[1, 1].plot(res["tc"], res["mu"], label="|F_x|/N")
    ax[1, 1].axhline(MU_S, color="r", ls="--", label=f"μ_s = {MU_S}")
    ax[1, 1].axhline(res["mu_req"], color="k", ls=":", label=f"μ_req = {res['mu_req']:.3f}")
    ax[1, 1].set(title="Coeficiente de fricción requerido", xlabel="t (s)", ylabel="μ")
    ax[1, 1].legend()
    fig.suptitle(nombre)
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, f"{nombre}_analisis.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    tc = res["tc"]
    ax[0].plot(tc, sim["x_ref"], "k-", label="Tracker")
    ax[0].plot(tc, sim["xA"], "--", label="Euler A (a medida)")
    ax[0].plot(tc, sim["xB"], "-.", label="Euler B (modelo seno)")
    ax[0].set(ylabel="x_CM (m)", title=f"{nombre}: Tracker vs Euler")
    ax[0].legend()
    ax[1].plot(tc, sim["v_ref"], "k-", label="Tracker")
    ax[1].plot(tc, sim["vA"], "--", label="Euler A")
    ax[1].plot(tc, sim["vB"], "-.", label="Euler B")
    ax[1].set(xlabel="t (s)", ylabel="v_CM (m/s)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, f"{nombre}_euler.png"), dpi=150)
    plt.close(fig)


# =====================================================================
# PROGRAMA PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    filas = []
    for i, ruta in enumerate(ARCHIVOS, start=1):
        nombre = f"P{i}"
        print(f"\n== {nombre}: {ruta}")
        res = analizar_prueba(ruta)
        sim = simular(res, ci="media_cero")
        graficar_prueba(nombre, res, sim, CARPETA_SALIDA)
        print(f"  v_exp = {res['v_exp']:.4f} m/s | %Error = {res['pct_error']:.2f}% | "
              f"mu_req = {res['mu_req']:.3f} | p95 = {res['mu_p95']:.3f}")
        filas.append({
            "prueba": nombre,
            "v_exp": res["v_exp"],
            "pct_error": res["pct_error"],
            "Fx_max": res["Fx_max"],
            "mu_req": res["mu_req"],
            "mu_p95": res["mu_p95"],
            "F0": sim["F0"],
            "f": sim["f"],
            "rmse_x_B": sim["B_x"]["rmse"],
            "rmse_v_B": sim["B_v"]["rmse"],
        })

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(CARPETA_SALIDA, "resultados_por_prueba.csv"), index=False)
    cols = ["v_exp", "pct_error", "Fx_max", "mu_req", "mu_p95", "rmse_x_B", "rmse_v_B"]
    resumen = pd.DataFrame({
        "promedio": df[cols].mean(),
        "desv_est_s": df[cols].std(ddof=1),
    })
    resumen.to_csv(os.path.join(CARPETA_SALIDA, "resumen_estadistico.csv"))
    print("\nRESUMEN:")
    print(resumen.round(4))