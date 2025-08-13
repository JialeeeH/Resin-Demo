import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pathlib import Path
from utils.io import read_csv, save_df
from etl.validation import validate_qc_result

def main():
    stage = read_csv('stage_features.csv')
    qc = read_csv('qc_result.csv')
    if not validate_qc_result(qc)["success"]:
        raise ValueError("qc_result.csv failed validation")
    # aggregate stage features to batch level (mean/max of per-step stats)
    agg = stage.groupby('batch_id').agg('mean').add_suffix('_mean').reset_index()
    agg_max = stage.groupby('batch_id').agg('max').add_suffix('_max').reset_index()
    F = pd.merge(agg, agg_max, on='batch_id', how='left')
    data = pd.merge(F, qc, on='batch_id', how='inner')
    out = Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'features.csv'
    data.to_csv(out, index=False)
    print(f"Saved features -> {out}")

if __name__ == '__main__':
    main()
