# app.py
# EFILabs – Índice energético (100 = óptimo) y MJ extra por 100 km
#
# Ejecutar:
#   streamlit run app.py
#
# Requisitos:
#   pip install streamlit pandas

import sys
import re
import unicodedata
import streamlit as st
import pandas as pd

# ---------------------------
# Bloque anti-"python app.py"
# ---------------------------
def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if __name__ == "__main__" and not _running_in_streamlit():
    print("\nEste archivo es una app de Streamlit.")
    print("Ejecuta así:\n")
    print("  streamlit run app.py\n")
    sys.exit(0)


# ---------------------------
# Utilidades
# ---------------------------
def normalize_text(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


def find_column(df, candidates):
    cols = {normalize_text(c): c for c in df.columns}
    for c in candidates:
        key = normalize_text(c)
        if key in cols:
            return cols[key]
    return None


def read_csv_safe(file):
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=";")


# ---------------------------
# Normalización CSV (Nazar + genérico)
# ---------------------------
def normalize_dataframe(df, filename):
    out = df.copy()
    out["__archivo"] = filename

    col_patente = find_column(out, ["vehiculo", "vehículo", "patente", "ppu", "unidad"])
    col_presion = find_column(out, ["valor presion", "valor presión", "presion", "presión", "psi", "presion_psi"])
    col_optima  = find_column(out, ["presion correcta", "presión correcta", "presion optima", "presion óptima", "presion_optima"])

    if not col_patente or not col_presion or not col_optima:
        return pd.DataFrame()

    out["patente"] = out[col_patente].astype(str).str.upper().str.strip()
    out["presion_psi"] = pd.to_numeric(out[col_presion], errors="coerce")
    out["presion_optima_psi"] = pd.to_numeric(out[col_optima], errors="coerce")

    out = out.dropna(subset=["presion_psi", "presion_optima_psi"])
    if out.empty:
        return pd.DataFrame()

    # Campos opcionales (si no existen → "NA")
    def opt(names):
        c = find_column(out, names)
        return out[c].astype(str).str.strip() if c else "NA"

    out["operacion"] = opt(["operacion", "operación", "flota", "faena", "cliente"])
    out["sede"] = opt(["sede", "terminal", "base", "planta"])
    out["ruta"] = opt(["ruta", "tramo", "origen destino", "origen-destino", "od"])

    out["marca_camion"] = opt(["marca camion", "marca camión", "marca vehiculo", "marca vehículo", "marca_camion"])
    out["modelo_camion"] = opt(["modelo camion", "modelo camión", "modelo vehiculo", "modelo vehículo", "modelo_camion"])

    out["marca_neumatico"] = opt(["marca neumatico", "marca neumático", "marca_neumatico", "marca"])
    out["modelo_neumatico"] = opt(["modelo neumatico", "modelo neumático", "modelo_neumatico", "modelo"])

    out["ciclo"] = opt(["ciclo", "etapa", "vida", "recauchado", "n recauchaje", "num recauchaje", "n_recauchaje"])

    return out


# ---------------------------
# Métricas EFILabs (normalizado a 100 km)
# ---------------------------
def compute_metrics(df, k, mj_base_100km):
    df = df.copy()
    df["delta_psi"] = df["presion_psi"] - df["presion_optima_psi"]
    df["desvio_presion_pct"] = (df["delta_psi"].abs() / df["presion_optima_psi"]) * 100.0

    # % energía extra
    df["energia_extra_pct"] = df["desvio_presion_pct"] * float(k)

    # Índice 100 = óptimo
    df["indice_energia"] = 100.0 + df["energia_extra_pct"]

    # MJ extra por 100 km
    df["mj_extra_100km"] = float(mj_base_100km) * (df["energia_extra_pct"] / 100.0)
    return df


def resumen(df, col):
    return (
        df.groupby(col, dropna=False)
        .agg(
            registros=("mj_extra_100km", "count"),
            indice_prom=("indice_energia", "mean"),
            energia_extra_prom_pct=("energia_extra_pct", "mean"),
            mj_extra_prom=("mj_extra_100km", "mean"),
            mj_extra_total=("mj_extra_100km", "sum"),
        )
        .reset_index()
        .sort_values("mj_extra_prom", ascending=False)
    )


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("EFILabs – MJ extra por 100 km (Índice 100 = óptimo)")

st.markdown(
    """
**Cómo leer esto (simple):**
- **Índice 100** = presión óptima  
- **Índice > 100** = % de energía extra por desviación de presión  
- **MJ extra / 100 km** = impacto energético comparable (normalizado)
"""
)

st.sidebar.header("Parámetros")
k = st.sidebar.number_input("k = % energía extra por 1% desvío de presión", 0.0, 5.0, 0.5, 0.1)
mj_base_100km = st.sidebar.number_input("Energía base cada 100 km (MJ/100km)", 100.0, 50000.0, 1000.0, 100.0)
top_n = st.sidebar.number_input("Top N", 5, 200, 15, 1)

files = st.file_uploader("Sube uno o más archivos CSV", type=["csv"], accept_multiple_files=True)

# Si no hay archivos, no hacemos nada (no concatena)
if not files:
    st.info("Sube un CSV para comenzar.")
    st.stop()

dfs = []
errores = []

for f in files:
    raw = read_csv_safe(f)
    norm = normalize_dataframe(raw, f.name)
    if norm.empty:
        errores.append(f.name)
    else:
        dfs.append(norm)

if errores:
    st.warning("No se pudieron procesar estos archivos (faltan columnas mínimas): " + ", ".join(errores))

# Si nada se pudo procesar, no concatena
if not dfs:
    st.error("No hay datos para mostrar (ningún archivo cumplió con columnas mínimas).")
    st.stop()

df = pd.concat(dfs, ignore_index=True)
df = compute_metrics(df, k, mj_base_100km)

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Índice promedio", f"{df['indice_energia'].mean():.2f}")
c2.metric("MJ extra promedio / 100 km", f"{df['mj_extra_100km'].mean():.2f}")
c3.metric("MJ extra total / 100 km", f"{df['mj_extra_100km'].sum():.0f}")

# Detalle
st.subheader("Detalle (ordenado por peor MJ extra/100km)")
st.dataframe(df.sort_values("mj_extra_100km", ascending=False), use_container_width=True)

# Comparaciones (solo si hay más de 1 valor)
st.subheader("Comparaciones (peores promedios arriba)")

categorias = [
    ("Patente", "patente"),
    ("Operación", "operacion"),
    ("Sede", "sede"),
    ("Ruta", "ruta"),
    ("Marca camión", "marca_camion"),
    ("Modelo camión", "modelo_camion"),
    ("Marca neumático", "marca_neumatico"),
    ("Modelo neumático", "modelo_neumatico"),
    ("Ciclo", "ciclo"),
]

for label, col in categorias:
    if col in df.columns and df[col].nunique(dropna=False) > 1:
        st.markdown(f"### {label}")
        r = resumen(df, col).head(int(top_n))
        st.dataframe(r, use_container_width=True)
        st.bar_chart(r.set_index(col)["mj_extra_prom"])
