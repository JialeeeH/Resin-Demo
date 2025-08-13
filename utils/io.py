import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'

def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)

def save_df(df: pd.DataFrame, name: str):
    (DATA_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / name, index=False)
