import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pathlib import Path
from utils.io import read_csv, save_df
from utils.preprocessing import savgol_slope
from utils.constants import STEPS, TAGS
from etl.validation import validate_batch, validate_ts_signal

def build_segments_from_events(op_event: pd.DataFrame):
    # assume events cover all steps with timestamps; build (start,end) per step per batch
    op_event = op_event.sort_values(['batch_id','ts'])
    segs = []
    for bid, g in op_event.groupby('batch_id'):
        g = g.reset_index(drop=True)
        ts_list = list(pd.to_datetime(g['ts']))
        # Ensure 8 steps by expanding if needed
        for i in range(len(ts_list)-1):
            segs.append({'batch_id': bid, 'step': int(g.loc[i,'step']), 
                         'start_ts': ts_list[i], 'end_ts': ts_list[i+1]})
    return pd.DataFrame(segs)

def build_stage_features(ts: pd.DataFrame, segs: pd.DataFrame):
    feats = []
    ts['ts'] = pd.to_datetime(ts['ts'])
    for (bid, step), seg in segs.groupby(['batch_id','step']):
        start = seg['start_ts'].min(); end = seg['end_ts'].max()
        X = ts[(ts['batch_id']==bid) & (ts['ts']>=start) & (ts['ts']<end)]
        if X.empty: 
            continue
        # pivot to wide
        P = X.pivot_table(index='ts', columns='tag', values='value').sort_index()
        # slopes
        P['dTdt'] = savgol_slope(P['T'].interpolate().fillna(method='bfill').fillna(method='ffill').values, window=19)
        # stats
        row = {'batch_id': bid, 'step': step, 'duration_min': (end-start).total_seconds()/60.0}
        for col in ['T','pH','Vac','Flow','RPM','DehydV','dTdt']:
            if col in P:
                s = P[col].dropna()
                if s.empty: 
                    continue
                row[f'{col}_mean'] = float(s.mean())
                row[f'{col}_std']  = float(s.std())
                row[f'{col}_max']  = float(s.max())
                row[f'{col}_min']  = float(s.min())
        feats.append(row)
    return pd.DataFrame(feats)

def main():
    batch = read_csv('batch.csv')
    ts = read_csv('ts_signal.csv')
    # fail fast if inputs do not meet expectations
    if not validate_batch(batch)["success"]:
        raise ValueError("batch.csv failed validation")
    if not validate_ts_signal(ts)["success"]:
        raise ValueError("ts_signal.csv failed validation")
    op = read_csv('op_event.csv')
    op['ts'] = pd.to_datetime(op['ts'])
    segs = build_segments_from_events(op)
    segs.to_csv(Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'segments.csv', index=False)
    feat = build_stage_features(ts, segs)
    save_df(feat, 'stage_features.csv')

if __name__ == '__main__':
    main()
