import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pathlib import Path
from utils.io import read_csv, save_df
from utils.preprocessing import savgol_slope
from utils.constants import STEPS, TAGS
from etl.validation import validate_batch, validate_ts_signal
import ruptures as rpt
from typing import List

WINDOW = pd.Timedelta(minutes=3)


def detect_changepoints(ts_wide: pd.DataFrame) -> pd.DataFrame:
    """Detect changepoints for key signals using PELT algorithm.

    Parameters
    ----------
    ts_wide : pd.DataFrame
        Wide time series indexed by timestamps with columns among
        ['T', 'pH', 'Vac', 'Flow'].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['signal', 'ts'] giving detected changepoint
        timestamps for each signal.
    """

    cp_rows: List[dict] = []
    for col in ['T', 'pH', 'Vac', 'Flow']:
        if col not in ts_wide:
            continue
        s = ts_wide[col].dropna()
        if len(s) < 10:
            continue
        # run PELT to find up to 7 breakpoints (8 steps)
        algo = rpt.Pelt(model="rbf").fit(s.values)
        try:
            bkps = algo.predict(n_bkps=7)
        except Exception:
            # fallback with penalty if n_bkps not feasible
            bkps = algo.predict(pen=3)
        for b in bkps[:-1]:  # last breakpoint is len(series)
            ts = s.index[min(b, len(s)-1)]
            cp_rows.append({'signal': col, 'ts': pd.Timestamp(ts)})
    return pd.DataFrame(cp_rows)


def adjust_segments_with_changepoints(ts: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    """Adjust segment boundaries if changepoints agree across signals.

    Parameters
    ----------
    ts : pd.DataFrame
        Long-format timeseries with columns ['batch_id','ts','tag','value'].
    segs : pd.DataFrame
        Initial segments from events with ['batch_id','step','start_ts','end_ts'].

    Returns
    -------
    pd.DataFrame
        Adjusted segments DataFrame.
    """

    ts = ts.copy()
    ts['ts'] = pd.to_datetime(ts['ts'])
    out = segs.copy()
    for bid, g in out.groupby('batch_id'):
        batch_ts = ts[ts['batch_id'] == bid]
        P = batch_ts.pivot_table(index='ts', columns='tag', values='value').sort_index()
        cps = detect_changepoints(P)
        if cps.empty:
            continue
        seg_idx = g.sort_values('start_ts').index.tolist()
        g_sorted = out.loc[seg_idx]
        g_sorted = g_sorted.sort_values('start_ts').reset_index()
        for i in range(len(g_sorted)):
            # adjust start boundary (not for first step)
            if i > 0:
                boundary = g_sorted.loc[i, 'start_ts']
                near = cps[(cps['ts'] >= boundary - WINDOW) & (cps['ts'] <= boundary + WINDOW)]
                if near['signal'].nunique() >= 2:
                    new_ts = near['ts'].median()
                    g_sorted.loc[i, 'start_ts'] = new_ts
                    g_sorted.loc[i-1, 'end_ts'] = new_ts
            # adjust end boundary (not for last step)
            if i < len(g_sorted) - 1:
                boundary = g_sorted.loc[i, 'end_ts']
                near = cps[(cps['ts'] >= boundary - WINDOW) & (cps['ts'] <= boundary + WINDOW)]
                if near['signal'].nunique() >= 2:
                    new_ts = near['ts'].median()
                    g_sorted.loc[i, 'end_ts'] = new_ts
                    g_sorted.loc[i+1, 'start_ts'] = new_ts
        out.update(g_sorted.set_index('index'))
    return out

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
        # state flags
        if 'T' in P and P['T'].dropna().size:
            tmax = P['T'].max()
            row['heat_ok'] = bool((tmax >= 95) and (tmax <= 98))
        else:
            row['heat_ok'] = False
        if 'pH' in P and P['pH'].dropna().size:
            ph_max = P['pH'].max(); ph_min = P['pH'].min()
            row['pH_window'] = bool((ph_min >= 8.4) and (ph_max <= 9.6))
        else:
            row['pH_window'] = False
        if 'DehydV' in P and P['DehydV'].dropna().size:
            row['vacuum_leq_16L'] = bool(P['DehydV'].max() <= 16000)
        else:
            row['vacuum_leq_16L'] = False
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
    segs = adjust_segments_with_changepoints(ts, segs)
    segs.to_csv(Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'segments.csv', index=False)
    feat = build_stage_features(ts, segs)
    save_df(feat, 'stage_features.csv')

if __name__ == '__main__':
    main()
