import pandas as pd

def filtrar_geo(
    df: pd.DataFrame,
    departamento: str,
    provincia: str | None = None,
    distrito: str | None = None
) -> pd.DataFrame:
    out = df[df["Departamento"] == departamento].copy()
    if provincia is not None:
        out = out[out["Provincia"] == provincia].copy()
    if distrito is not None:
        out = out[out["Distrito"] == distrito].copy()
    return out
