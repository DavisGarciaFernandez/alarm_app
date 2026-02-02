import numpy as np
import pandas as pd

# -----------------------------
# Parámetros de negocio
# -----------------------------
# Duración mínima para considerar un corte "operacional" (en minutos)
MIN_OP_MIN = 20         # ej: < 5 min se consideran muy pequeños

# Duración máxima "operacional" (en minutos)
# Ejemplo: 2 días = 48 * 60 = 2880 minutos
MAX_OP_MIN = 48 * 60

# Mínimo de cortes para considerar que un promedio por sitio es "confiable"
N_MIN_SITIO = 5


def resumen_por_sitio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas robustas por sitio para la duración de cortes.

    Retorna por cada 'Site Name':

      - cortes_antes: número total de cortes (todas las duraciones)
      - promedio_antes: promedio bruto (todas las duraciones, en minutos)

      - cortes_pequenos: nº cortes con duración < MIN_OP_MIN
      - cortes_extremos: nº cortes con duración > MAX_OP_MIN
      - cortes_operacionales: nº cortes en rango [MIN_OP_MIN, MAX_OP_MIN]

      - promedio_operacional: promedio SOLO en rango operacional (minutos)
      - mediana_operacional: mediana SOLO en rango operacional (minutos)

      - cortes_despues: alias de cortes_operacionales (para compatibilidad con app.py)
      - promedio_despues: alias de promedio_operacional (para compatibilidad con app.py)

      - pct_extremos: porcentaje de cortes extremos sobre el total
      - pct_pequenos: porcentaje de cortes pequeños sobre el total

      - flag_poca_data: True si cortes_antes < N_MIN_SITIO (poca data en el sitio)
    """

    # Copia para no tocar el DF original
    df = df.copy()

    # Aseguramos tipo numérico en duracion_minutos
    df["duracion_minutos"] = pd.to_numeric(df["duracion_minutos"], errors="coerce")

    # Definimos máscaras por registro
    mask_peq = df["duracion_minutos"] < MIN_OP_MIN
    mask_ext = df["duracion_minutos"] > MAX_OP_MIN
    mask_op = (~mask_peq) & (~mask_ext) & df["duracion_minutos"].notna()

    # Guardamos (por si luego quieres usar estas columnas en otros análisis)
    df["es_pequeno"] = mask_peq
    df["es_extremo"] = mask_ext
    df["es_operacional"] = mask_op

    # Agrupación por sitio
    g = df.groupby("Site Name", dropna=False)

    # --- Estadísticos básicos (antes / bruto) ---
    cortes_antes = g["duracion_minutos"].size().rename("cortes_antes")
    promedio_antes = g["duracion_minutos"].mean().rename("promedio_antes")

    # --- Contadores por categoría ---
    cortes_pequenos = g["es_pequeno"].sum().rename("cortes_pequenos")
    cortes_extremos = g["es_extremo"].sum().rename("cortes_extremos")
    cortes_operacionales = g["es_operacional"].sum().rename("cortes_operacionales")

    # --- Estadísticos sobre rango operacional ---
    def _promedio_operacional(grp: pd.DataFrame) -> float:
        vals = grp.loc[grp["es_operacional"], "duracion_minutos"].dropna()
        if len(vals) == 0:
            return np.nan
        return float(vals.mean())

    def _mediana_operacional(grp: pd.DataFrame) -> float:
        vals = grp.loc[grp["es_operacional"], "duracion_minutos"].dropna()
        if len(vals) == 0:
            return np.nan
        return float(vals.median())

    promedio_operacional = g.apply(_promedio_operacional).rename("promedio_operacional")
    mediana_operacional = g.apply(_mediana_operacional).rename("mediana_operacional")

    # --- Construir tabla final ---
    out = pd.concat(
        [
            cortes_antes,
            promedio_antes,
            cortes_pequenos,
            cortes_extremos,
            cortes_operacionales,
            promedio_operacional,
            mediana_operacional,
        ],
        axis=1,
    ).reset_index()

    # Aliases para compatibilidad con app.py (antes/después)
    out["cortes_despues"] = out["cortes_operacionales"]
    out["promedio_despues"] = out["promedio_operacional"]

    # Flags y porcentajes
    out["flag_poca_data"] = out["cortes_antes"] < N_MIN_SITIO

    out["pct_extremos"] = np.where(
        out["cortes_antes"] > 0,
        out["cortes_extremos"] / out["cortes_antes"],
        0.0,
    )

    out["pct_pequenos"] = np.where(
        out["cortes_antes"] > 0,
        out["cortes_pequenos"] / out["cortes_antes"],
        0.0,
    )

    # Orden sugerido:
    #   1) primero sitios con suficiente data (flag_poca_data = False)
    #   2) dentro de ellos, mayor promedio_operacional primero
    #   3) a igualdad, más cortes_operacionales primero
    out = out.sort_values(
        ["flag_poca_data", "promedio_operacional", "cortes_operacionales"],
        ascending=[True, False, False],
    )

    return out
