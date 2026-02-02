from pathlib import Path
from config import DATA_RAW_PATH, DATA_PROCESADA_PATH
from src.io_local import cargar_datos_local
from src.features import agregar_duracion_minutos
from src.limpieza_outliers import winsorizar_por_grupo

def main():
    df = cargar_datos_local(Path(DATA_RAW_PATH))
    df = agregar_duracion_minutos(df)
    df = df.dropna(subset=["duracion_minutos"]).copy()

    # CAP por sitio (no elimina)
    df = winsorizar_por_grupo(
        df,
        col_grupo="Site Name",
        col_val="duracion_minutos",
        min_n=20,
        p_cap=0.99
    )

    Path(DATA_PROCESADA_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESADA_PATH, index=False)
    print("OK ->", DATA_PROCESADA_PATH)

if __name__ == "__main__":
    main()