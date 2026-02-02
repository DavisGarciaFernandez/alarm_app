from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # carpeta donde está config.py
DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "HistoricalAlarms202501-202512.xlsx"
DATA_PROCESADA_PATH = BASE_DIR / "data" / "procesada" / "alarmas_con_limpieza.parquet"
