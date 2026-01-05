"""
Aplicación Streamlit simplificada para analizar datos de presiones de neumáticos a partir
de archivos CSV. Esta versión evita dependencias externas como openpyxl y utiliza
únicamente `pandas` para leer archivos CSV. La aplicación detecta el formato de
origen (Nazar o TCCU) según los nombres de las columnas y unifica los datos para
calcular desvíos respecto a la presión óptima. También genera resúmenes por
patente y por operación y ofrece la descarga de los resultados en un archivo ZIP
con múltiples CSV.

Requisitos: pip install streamlit pandas
"""

import streamlit as st
import pandas as pd
from io import BytesIO
import zipfile
# Eliminado import de Altair; usaremos funciones integradas de Streamlit para gráficos.

# Factor de conversión de desvío de presión a pérdida de energía.
# Ajusta este valor según las métricas proporcionadas por EFILabs.
ENERGY_FACTOR = 1.0

# Mapeo de alias de columnas para normalización flexible
COLUMN_ALIASES = {
    "patente": ["patente", "PPU", "vehiculo", "Vehículo", "camion", "camión"],
    "ruta": ["ruta", "Ruta", "tramo", "Trayecto", "origen_destino", "origen-destino", "od"],
    "sede": ["sede", "Sede", "terminal", "Terminal", "site", "planta", "Flota", "flota"],
    "operacion": ["operacion", "operación", "Operacion", "operation", "Flota", "flota"],
    "eje_tipo": ["eje_tipo", "Tipo de Eje", "tipo_eje", "eje"],
    "posicion": ["posicion", "Posición", "pos", "position"],
    "fecha": ["fecha", "Fecha Inspección", "Fecha Inspeccion", "date"],
    "presion_psi": ["presion_psi", "Valor Presión", "presion", "PRESION CONTROLADA NEU"],
    "presion_optima_psi": ["presion_optima_psi", "Presión Correcta", "PRESION OPTIMA NEU"],
}


# -----------------------------
# Funciones auxiliares
# -----------------------------

def leer_csv_subido(uploaded_file: BytesIO) -> pd.DataFrame:
    """Lee un archivo CSV subido y devuelve un DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Error al leer el archivo CSV: {exc}")
        return pd.DataFrame()
    df["__archivo"] = uploaded_file.name
    return df


def normalizar_unificar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas utilizando alias predefinidos (ver `COLUMN_ALIASES`).
    Si una columna esperada no existe, se crea con valores NA. Además se aplican
    transformaciones básicas: patente en mayúsculas, eje_tipo en minúsculas,
    conversión de fechas y campos numéricos.
    """
    out = df.copy()
    # Crear un mapa de las columnas originales en minúsculas a sus nombres originales
    lower_cols = {c.lower(): c for c in out.columns}
    rename_map: dict[str, str] = {}
    # Recorrer los alias para renombrar columnas encontradas
    for std_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in lower_cols:
                orig_col = lower_cols[alias_lower]
                rename_map[orig_col] = std_col
                break
    # Renombrar columnas al estándar
    out = out.rename(columns=rename_map)
    # Asegurar que todas las columnas estándar existan, creando NaN si faltan
    for std_col in COLUMN_ALIASES:
        if std_col not in out.columns:
            out[std_col] = pd.NA
    # Asegurar columna __archivo
    if "__archivo" not in out.columns:
        out["__archivo"] = pd.NA
    # Limpieza específica
    # Patente a mayúsculas y sin espacios
    if "patente" in out.columns:
        out["patente"] = out["patente"].astype(str).str.strip().str.upper()
    # Eje tipo a minúsculas y reemplazo de sinónimos comunes
    if "eje_tipo" in out.columns:
        out["eje_tipo"] = out["eje_tipo"].astype(str).str.strip().str.lower()
        out["eje_tipo"] = out["eje_tipo"].replace(
            {"dirección": "direccional", "tracción": "traccion", "libre": "arrastre"}
        )
    # Convertir fecha
    if "fecha" in out.columns:
        out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
    # Convertir campos numéricos
    for numeric_col in ["presion_psi", "presion_optima_psi"]:
        if numeric_col in out.columns:
            out[numeric_col] = pd.to_numeric(out[numeric_col], errors="coerce")
    # Si la columna 'operacion' está vacía pero existe 'sede', usar sede como operación
    if "operacion" in out.columns:
        if out["operacion"].isna().all() and "sede" in out.columns:
            out["operacion"] = out["sede"]
    return out


def agregar_metricas(df: pd.DataFrame, tol_psi: float = 3.0) -> pd.DataFrame:
    """Calcula la diferencia respecto al óptimo y clasifica en OK/SUBINFLADO/SOBREINFLADO."""
    out = df.copy()
    # Calcular desvío absoluto respecto al óptimo
    out["delta_psi"] = out["presion_psi"] - out["presion_optima_psi"]
    # Evita división por cero o NaN al calcular porcentaje
    out["delta_pct"] = (
        out["delta_psi"] / out["presion_optima_psi"]
    ) * 100.0
    # Calcular pérdida de energía (valor absoluto del desvío multiplicado por factor)
    out["energia_perdida"] = out["delta_psi"].abs() * ENERGY_FACTOR
    # Clasificar estado
    out["estado"] = "OK"
    out.loc[out["delta_psi"] <= -tol_psi, "estado"] = "SUBINFLADO"
    out.loc[out["delta_psi"] >= tol_psi, "estado"] = "SOBREINFLADO"
    return out


def resumen(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Genera un resumen agregando por las columnas indicadas.

    Además del recuento y promedios de desvío, calcula la energía total perdida
    sumando la columna `energia_perdida`. El resultado se ordena por mayor
    desvío absoluto promedio y número de registros.
    """
    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_registros=("presion_psi", "count"),
            delta_prom=("delta_psi", "mean"),
            delta_abs_prom=("delta_psi", lambda s: s.abs().mean()),
            energia_total=("energia_perdida", "sum"),
            pct_sub=("estado", lambda s: (s == "SUBINFLADO").mean() * 100),
            pct_sobre=("estado", lambda s: (s == "SOBREINFLADO").mean() * 100),
        )
        .reset_index()
    )
    return g.sort_values(["delta_abs_prom", "n_registros"], ascending=[False, False])


def to_zip_download(df_detalle: pd.DataFrame) -> bytes:
    """
    Genera un archivo ZIP en memoria con los datos detallados y sus resúmenes
    en formato CSV. Devuelve los bytes del archivo ZIP.
    """
    por_patente = resumen(df_detalle, ["patente"])
    por_operacion = resumen(df_detalle, ["operacion"])
    por_patente_operacion = resumen(df_detalle, ["patente", "operacion"])

    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("detalle.csv", df_detalle.to_csv(index=False))
        zf.writestr("por_patente.csv", por_patente.to_csv(index=False))
        zf.writestr("por_operacion.csv", por_operacion.to_csv(index=False))
        zf.writestr(
            "patente_x_operacion.csv", por_patente_operacion.to_csv(index=False)
        )
    mem_zip.seek(0)
    return mem_zip.getvalue()


# -----------------------------
# Interfaz Streamlit
# -----------------------------

st.set_page_config(page_title="Análisis de presiones", layout="wide")
st.title("Análisis de presiones vs óptimo (formato CSV)")

st.markdown(
    """
    Sube uno o varios archivos CSV con mediciones de presión de neumáticos.
    La app detecta el formato (Nazar o TCCU), unifica las columnas y calcula el
    desvío respecto al óptimo. Si tus datos están en Excel, conviértelos a CSV
    antes de cargarlos.
    """
)

uploaded_files = st.file_uploader(
    "Selecciona uno o varios archivos CSV",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        df_file = leer_csv_subido(f)
        if not df_file.empty:
            df_norm = normalizar_unificar(df_file)
            if not df_norm.empty:
                dfs.append(df_norm)

    if dfs:
        df_unificado = pd.concat(dfs, ignore_index=True)
        df_metrics = agregar_metricas(df_unificado)

        st.subheader("Detalle de mediciones")
        st.dataframe(
            df_metrics[
                [
                    "patente",
                    "operacion",
                    "eje_tipo",
                    "posicion",
                    "fecha",
                    "presion_psi",
                    "presion_optima_psi",
                    "delta_psi",
                    "delta_pct",
                    "estado",
                ]
            ],
            use_container_width=True,
        )

        st.subheader("Resumen por patente")
        st.dataframe(resumen(df_metrics, ["patente"]), use_container_width=True)

        st.subheader("Resumen por operación")
        st.dataframe(resumen(df_metrics, ["operacion"]), use_container_width=True)

        st.subheader("Resumen patente x operación")
        st.dataframe(
            resumen(df_metrics, ["patente", "operacion"]),
            use_container_width=True,
        )


        # -----------------------------
        # Resumen de energía perdida promedio por agrupación
        # -----------------------------
        # Patentes (camiones) con mayor energía perdida promedio
        st.subheader("Patentes con mayor energía perdida promedio")
        por_patente = resumen(df_metrics, ["patente"])
        if not por_patente.empty:
            por_patente["energia_prom"] = por_patente["energia_total"] / por_patente["n_registros"]
            por_patente = por_patente.sort_values("energia_prom", ascending=False)
            st.dataframe(
                por_patente[["patente", "energia_prom", "energia_total", "n_registros"]].head(10),
                use_container_width=True,
            )
            # Mostrar gráfico de barras usando funciones integradas
            st.bar_chart(data=por_patente.set_index("patente")["energia_prom"].head(10))

        # Operaciones con mayor energía perdida promedio
        st.subheader("Operaciones con mayor energía perdida promedio")
        por_operacion_chart = resumen(df_metrics, ["operacion"])
        por_operacion_chart = por_operacion_chart.dropna(subset=["operacion"])
        if not por_operacion_chart.empty:
            por_operacion_chart["energia_prom"] = por_operacion_chart["energia_total"] / por_operacion_chart["n_registros"]
            por_operacion_chart = por_operacion_chart.sort_values("energia_prom", ascending=False)
            st.dataframe(
                por_operacion_chart[["operacion", "energia_prom", "energia_total", "n_registros"]].head(10),
                use_container_width=True,
            )
            st.bar_chart(data=por_operacion_chart.set_index("operacion")["energia_prom"].head(10))

        # Rutas con mayor energía perdida promedio (si la columna existe)
        if "ruta" in df_metrics.columns:
            st.subheader("Rutas con mayor energía perdida promedio")
            por_ruta = resumen(df_metrics, ["ruta"])
            por_ruta = por_ruta.dropna(subset=["ruta"])
            if not por_ruta.empty:
                por_ruta["energia_prom"] = por_ruta["energia_total"] / por_ruta["n_registros"]
                por_ruta = por_ruta.sort_values("energia_prom", ascending=False)
                st.dataframe(
                    por_ruta[["ruta", "energia_prom", "energia_total", "n_registros"]].head(10),
                    use_container_width=True,
                )
                st.bar_chart(data=por_ruta.set_index("ruta")["energia_prom"].head(10))

        # Sedes con mayor energía perdida promedio (si la columna existe)
        if "sede" in df_metrics.columns:
            st.subheader("Sedes con mayor energía perdida promedio")
            por_sede = resumen(df_metrics, ["sede"])
            por_sede = por_sede.dropna(subset=["sede"])
            if not por_sede.empty:
                por_sede["energia_prom"] = por_sede["energia_total"] / por_sede["n_registros"]
                por_sede = por_sede.sort_values("energia_prom", ascending=False)
                st.dataframe(
                    por_sede[["sede", "energia_prom", "energia_total", "n_registros"]].head(10),
                    use_container_width=True,
                )
                st.bar_chart(data=por_sede.set_index("sede")["energia_prom"].head(10))

        # Descarga del reporte ZIP con CSVs
        zip_bytes = to_zip_download(df_metrics)
        st.download_button(
            "Descargar reportes en ZIP",
            data=zip_bytes,
            file_name="reporte_presiones_vs_optimo.zip",
            mime="application/zip",
        )
    else:
        st.error("No se encontraron datos reconocibles en los archivos subidos.")
else:
    st.info("Sube al menos un archivo CSV para empezar.")