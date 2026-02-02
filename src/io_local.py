import pandas as pd

COLUMNAS_REQUERIDAS = [
    "Severity", "Site Name", "Departamento", "Provincia", "Distrito",
    "Last Occurred", "Cleared On"
]

COLUMNAS_TEXTO = [
    "Severity","Name","Site Name","Codigo",
    "Departamento","Provincia","Distrito",
    "Device Type","Location Info","Device","Manage Domain"
]

COLUMNAS_NUM = ["Altitud","Latitud","Longitud","Prioridad"]

def cargar_datos_local(ruta) -> pd.DataFrame:
    ruta = str(ruta)
    if ruta.lower().endswith(".csv"):
        df = pd.read_csv(ruta)
    else:
        df = pd.read_excel(ruta)

    df.columns = [c.strip() for c in df.columns]
    _validar_columnas(df)
    df = _normalizar_tipos(df)
    return df

def _validar_columnas(df: pd.DataFrame) -> None:
    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas requeridas: {faltan}")

def _normalizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in COLUMNAS_TEXTO:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    for c in COLUMNAS_NUM:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out
