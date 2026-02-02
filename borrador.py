import hashlib
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

from config import DATA_PROCESADA_PATH
from src.filtros import filtrar_geo
from src.estadisticas import resumen_por_sitio, MIN_OP_MIN, MAX_OP_MIN


# -----------------------------
# Configuración Streamlit
# -----------------------------
st.set_page_config(page_title="Promedio de alarmas por sitio", layout="wide")
st.title("📍 Promedio de alarmas por sitio")

# Hacer el cuadro de diálogo más ancho
st.markdown(
    """
    <style>
    /* Contenedor principal del diálogo */
    [data-testid="stDialog"] > div {
        width: 80vw !important;
        max-width: 80vw !important;
    }

    /* Bloque interno para que el contenido use todo el ancho */
    [data-testid="stDialog"] [data-testid="stVerticalBlock"] {
        max-width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=True)
def cargar_procesada() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESADA_PATH)


df = cargar_procesada()

# -----------------------------
# Sidebar filtros geo
# -----------------------------
st.sidebar.header("Filtros geográficos")

nivel = st.sidebar.radio(
    "Nivel",
    ["Departamento", "Departamento + Provincia", "Departamento + Provincia + Distrito"],
)

deps = sorted(df["Departamento"].dropna().unique().tolist())
dep = st.sidebar.selectbox("Departamento", deps)

df_f = filtrar_geo(df, departamento=dep)

prov = None
dist = None

if nivel != "Departamento":
    provs = sorted(df_f["Provincia"].dropna().unique().tolist())
    prov = st.sidebar.selectbox("Provincia", provs)
    df_f = filtrar_geo(df, departamento=dep, provincia=prov)

if nivel == "Departamento + Provincia + Distrito":
    if "Distrito" in df_f.columns:
        dists = sorted(df_f["Distrito"].dropna().unique().tolist())
    else:
        dists = []

    if not dists:
        dist = st.sidebar.selectbox("Distrito", ["(sin datos)"])
    else:
        dist = st.sidebar.selectbox("Distrito", dists)
        df_f = filtrar_geo(df, departamento=dep, provincia=prov, distrito=dist)


# -----------------------------
# Ventana aparte (detalle)
# -----------------------------
@st.dialog("Detalle del sitio: duraciones, motivo e imputación")
def mostrar_detalle_sitio(df_sitio: pd.DataFrame, nombre: str):
    st.markdown(f"**Sitio:** {nombre}")

    df_det = df_sitio.copy()
    df_det["duracion_minutos"] = pd.to_numeric(df_det["duracion_minutos"], errors="coerce")

    # Clasificación por motivo según reglas de negocio
    es_peq = df_det["duracion_minutos"] < MIN_OP_MIN
    es_ext = df_det["duracion_minutos"] > MAX_OP_MIN
    es_op = (~es_peq) & (~es_ext) & df_det["duracion_minutos"].notna()

    df_det["motivo"] = np.select(
        [
            es_peq,
            es_op,
            es_ext,
        ],
        [
            f"Pequeño (< {MIN_OP_MIN} min)",
            "Operacional",
            f"Extremo (> {MAX_OP_MIN} min)",
        ],
        default="Sin dato",
    )

    # Imputación: solo a los EXTREMOS
    # Usamos la mediana de los cortes operacionales del sitio como valor imputado.
    vals_op = df_det.loc[es_op, "duracion_minutos"].dropna()
    if len(vals_op) > 0:
        valor_imputacion = float(vals_op.median())
    else:
        # Si el sitio no tiene cortes operacionales, usamos MAX_OP_MIN como imputación
        valor_imputacion = float(MAX_OP_MIN)

    df_det["imputado"] = es_ext
    df_det["duracion_imputada_min"] = np.where(
        df_det["imputado"],
        valor_imputacion,
        df_det["duracion_minutos"],
    )

    # Conversión a horas y días (original e imputado)
    df_det["duracion_horas"] = df_det["duracion_minutos"] / 60.0
    df_det["duracion_dias"] = df_det["duracion_minutos"] / (60.0 * 24.0)

    df_det["duracion_imputada_horas"] = df_det["duracion_imputada_min"] / 60.0
    df_det["duracion_imputada_dias"] = df_det["duracion_imputada_min"] / (60.0 * 24.0)

    # KPIs
    cortes_totales = int(df_det["duracion_minutos"].notna().sum())
    cortes_imputados = int(df_det["imputado"].sum())
    prom_bruto_h = float(df_det["duracion_minutos"].mean() / 60.0) if cortes_totales > 0 else 0.0
    prom_imputado_h = float(df_det["duracion_imputada_min"].mean() / 60.0) if cortes_totales > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cortes totales", f"{cortes_totales:,}")
    col2.metric("Cortes imputados (extremos)", f"{cortes_imputados:,}")
    col3.metric("Promedio bruto (h)", f"{prom_bruto_h:,.2f}")
    col4.metric("Promedio imputado (h)", f"{prom_imputado_h:,.2f}")

    tab_orig, tab_imp = st.tabs(["Detalle original", "Detalle imputado"])

    # --- TABLA ORIGINAL (sin Last Occurred / Cleared On) ---
    with tab_orig:
        cols_orig = [
            "duracion_minutos",
            "duracion_horas",
            "duracion_dias",
            "motivo",
        ]
        cols_orig = [c for c in cols_orig if c in df_det.columns]
        st.dataframe(
            df_det[cols_orig].sort_values("duracion_minutos", ascending=False),
            use_container_width=True,
            height=420,
        )
        st.download_button(
            "⬇️ Descargar detalle original (CSV)",
            data=df_det[cols_orig]
            .sort_values("duracion_minutos", ascending=False)
            .to_csv(index=False)
            .encode("utf-8"),
            file_name=f"{nombre}_detalle_original.csv",
            mime="text/csv",
        )

    # --- TABLA IMPUTADA (sin Last Occurred / Cleared On) ---
    with tab_imp:
        cols_imp = [
            "duracion_minutos",
            "duracion_horas",
            "duracion_imputada_min",
            "duracion_imputada_horas",
            "duracion_imputada_dias",
            "motivo",
            "imputado",
        ]
        cols_imp = [c for c in cols_imp if c in df_det.columns]
        st.dataframe(
            df_det[cols_imp].sort_values("duracion_minutos", ascending=False),
            use_container_width=True,
            height=420,
        )
        st.download_button(
            "⬇️ Descargar detalle imputado (CSV)",
            data=df_det[cols_imp]
            .sort_values("duracion_minutos", ascending=False)
            .to_csv(index=False)
            .encode("utf-8"),
            file_name=f"{nombre}_detalle_imputado.csv",
            mime="text/csv",
        )


# -----------------------------
# Gráfica de BARRAS por rangos de horas
# plomo = Departamento, rojo = filtro completo
# filtros: mes + día del mes
# -----------------------------
st.subheader("Distribución de duración del corte (horas) por rangos")

serie = st.radio(
    "Serie a graficar",
    ["Antes (duracion_minutos)", "Después (duracion_cap)"],
    horizontal=True,
)

# Columna base según selección (con fallback)
if serie.startswith("Antes"):
    col_base = "duracion_minutos"
else:
    col_base = "duracion_cap" if "duracion_cap" in df.columns else "duracion_minutos"

MESES_ES = {
    1: "Ene",
    2: "Feb",
    3: "Mar",
    4: "Abr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dic",
}
orden_meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Plomo = Departamento (solo dep)
df_dep = df[df["Departamento"] == dep].copy()

# Rojo = filtro completo (dep+prov(+dist))
if nivel == "Departamento":
    df_rojo = df_dep.copy()
elif nivel == "Departamento + Provincia":
    df_rojo = df_dep[df_dep["Provincia"] == prov].copy()
else:
    df_rojo = df_dep[(df_dep["Provincia"] == prov) & (df_dep["Distrito"] == dist)].copy()

for _d in (df_dep, df_rojo):
    _d["Last Occurred"] = pd.to_datetime(_d["Last Occurred"], errors="coerce")


def prep(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.dropna(subset=["Last Occurred", col_base]).copy()
    out["mes_num"] = out["Last Occurred"].dt.month
    out = out[out["mes_num"].between(1, 12)].copy()
    out["mes"] = out["mes_num"].map(MESES_ES)
    out["dia_mes"] = out["Last Occurred"].dt.day
    out["duracion_horas"] = pd.to_numeric(out[col_base], errors="coerce") / 60.0
    out = out.dropna(subset=["duracion_horas"]).copy()
    return out


dep_p = prep(df_dep)
rojo_p = prep(df_rojo)

if len(dep_p) == 0:
    st.info("No hay datos suficientes para graficar con el Departamento seleccionado.")
else:
    # ---- Filtros: mes y día ----
    meses_disponibles = [m for m in orden_meses if m in set(dep_p["mes"].unique())]
    meses_sel = st.multiselect("Filtrar por mes", options=orden_meses, default=meses_disponibles)

    dia_min, dia_max = st.slider("Filtrar por día del mes", 1, 31, (1, 31), 1)

    dep_pf = dep_p.copy()
    rojo_pf = rojo_p.copy()

    if meses_sel:
        dep_pf = dep_pf[dep_pf["mes"].isin(meses_sel)]
        rojo_pf = rojo_pf[rojo_pf["mes"].isin(meses_sel)]
    else:
        dep_pf = dep_pf.iloc[0:0]
        rojo_pf = rojo_pf.iloc[0:0]

    dep_pf = dep_pf[(dep_pf["dia_mes"] >= dia_min) & (dep_pf["dia_mes"] <= dia_max)]
    rojo_pf = rojo_pf[(rojo_pf["dia_mes"] >= dia_min) & (rojo_pf["dia_mes"] <= dia_max)]

    # ---- KPIs: cantidad de cortes ----
    c1, c2 = st.columns(2)
    c1.metric("Cortes del Departamento (plomo)", f"{len(dep_pf):,}")
    c2.metric("Cortes del filtro completo (rojo)", f"{len(rojo_pf):,}")

    if len(dep_pf) == 0 and len(rojo_pf) == 0:
        st.info("Con esos filtros de mes/día no hay puntos para mostrar.")
    else:
        # Definir rangos de horas
        hour_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 48, np.inf]
        hour_labels = [
            "0–1h",
            "1–2h",
            "2–3h",
            "3–4h",
            "4–5h",
            "5–6h",
            "6–7h",
            "7–8h",
            "8–9h",
            "9–10h",
            "10–11h",
            "11–12h",
            "12–24h",
            "24–48h",
            ">48h",
        ]

        def agg_rangos(df_in: pd.DataFrame, label: str) -> pd.DataFrame:
            if df_in.empty:
                return pd.DataFrame(columns=["rango_horas", "cortes", "grupo"])
            tmp = df_in.copy()
            tmp["rango_horas"] = pd.cut(
                tmp["duracion_horas"],
                bins=hour_bins,
                labels=hour_labels,
                right=False,
                include_lowest=True,
            )
            tmp = tmp.dropna(subset=["rango_horas"])
            if tmp.empty:
                return pd.DataFrame(columns=["rango_horas", "cortes", "grupo"])
            agg = (
                tmp.groupby("rango_horas")
                .size()
                .rename("cortes")
                .reset_index()
            )
            agg["grupo"] = label
            return agg

        dep_agg = agg_rangos(dep_pf, f"Departamento: {dep}")
        rojo_agg = agg_rangos(rojo_pf, "Filtro completo")

        plot_df = pd.concat([dep_agg, rojo_agg], ignore_index=True)

        if plot_df.empty:
            st.info("No hay datos para los rangos de horas seleccionados.")
        else:
            color_map = {f"Departamento: {dep}": "gray", "Filtro completo": "red"}

            fig_bar = px.bar(
                plot_df,
                x="rango_horas",
                y="cortes",
                color="grupo",
                color_discrete_map=color_map,
                barmode="group",
                category_orders={"rango_horas": hour_labels},
                text="cortes",
                labels={
                    "rango_horas": "Rango de duración (horas)",
                    "cortes": "Cantidad de cortes",
                    "grupo": "",
                },
                title=f"Cantidad de cortes por rango de horas — {serie}",
            )

            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                xaxis_title="Rango de duración (horas)",
                yaxis_title="Cantidad de cortes",
            )

            st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------
# Tabla principal: promedios + lupa por fila
# (usa resumen_por_sitio actualizado)
# -----------------------------
tabla_sitios = resumen_por_sitio(df_f)

st.subheader(f"Lista de sitios ({len(tabla_sitios)})")

cols_show = [
    "Site Name",
    "cortes_antes",
    "cortes_despues",
    "promedio_antes",
    "promedio_despues",
]
cols_show = [c for c in cols_show if c in tabla_sitios.columns]

df_tabla = tabla_sitios[cols_show].copy()
df_tabla.insert(0, "🔍", False)

filtro_id = f"{nivel}|{dep}|{prov}|{dist}"
editor_key = "editor_sitios_" + hashlib.md5(filtro_id.encode("utf-8")).hexdigest()[:10]

editado = st.data_editor(
    df_tabla,
    use_container_width=True,
    height=560,
    hide_index=True,
    column_config={
        "🔍": st.column_config.CheckboxColumn(
            "🔍", help="Marca para abrir el detalle del sitio", default=False
        ),
        "promedio_antes": st.column_config.NumberColumn("promedio_antes", format="%.2f"),
        "promedio_despues": st.column_config.NumberColumn("promedio_despues", format="%.2f"),
    },
    disabled=[c for c in df_tabla.columns if c != "🔍"],
    key=editor_key,
)

st.download_button(
    "⬇️ Descargar tabla (CSV)",
    data=tabla_sitios[cols_show].to_csv(index=False).encode("utf-8"),
    file_name="sitios_promedios_antes_despues.csv",
    mime="text/csv",
)

seleccion = editado.loc[editado["🔍"] == True]
if not seleccion.empty:
    sitio_sel = str(seleccion.iloc[0]["Site Name"])
    last_key = f"ultimo_sitio_abierto_{editor_key}"
    if st.session_state.get(last_key) != sitio_sel:
        st.session_state[last_key] = sitio_sel
        df_sitio = df_f[df_f["Site Name"] == sitio_sel].copy()
        mostrar_detalle_sitio(df_sitio, sitio_sel)
