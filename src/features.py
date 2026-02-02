import numpy as np
import pandas as pd

def agregar_duracion_minutos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Last Occurred"] = pd.to_datetime(out["Last Occurred"], errors="coerce")
    out["Cleared On"] = pd.to_datetime(out["Cleared On"], errors="coerce")

    out["esta_abierta"] = out["Cleared On"].isna()

    out["duracion_minutos"] = (out["Cleared On"] - out["Last Occurred"]).dt.total_seconds() / 60.0
    out.loc[out["duracion_minutos"] < 0, "duracion_minutos"] = np.nan

    return out