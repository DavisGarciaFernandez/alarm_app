import numpy as np
import pandas as pd

def winsorizar_por_grupo(
    df: pd.DataFrame,
    col_grupo: str = "Site Name",
    col_val: str = "duracion_minutos",
    min_n: int = 20,
    p_cap: float = 0.99,
) -> pd.DataFrame:
    """
    No elimina registros.
    Crea:
      - duracion_cap: valor capeado (winsorizado) por grupo
      - fue_capeado: bool si se recortó
      - cap_valor: el percentil usado como tope por grupo
    """
    out = df.copy()
    out[col_val] = pd.to_numeric(out[col_val], errors="coerce")

    out["duracion_cap"] = out[col_val].astype(float)
    out["fue_capeado"] = False
    out["cap_valor"] = np.nan

    tamaños = out[col_grupo].value_counts(dropna=False)
    elegibles = set(tamaños[tamaños >= min_n].index)

    def _proc(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp[col_grupo].iloc[0]
        x = grp[col_val].dropna()

        if (g not in elegibles) or (len(x) == 0):
            return grp

        cap = float(x.quantile(p_cap))
        grp["cap_valor"] = cap

        # capear
        v = grp[col_val].astype(float)
        grp["duracion_cap"] = np.minimum(v, cap)
        grp["fue_capeado"] = v > cap
        return grp

    out = out.groupby(col_grupo, dropna=False, group_keys=False, sort=False).apply(_proc)
    return out
