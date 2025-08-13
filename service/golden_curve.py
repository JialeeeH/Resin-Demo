import pandas as pd, numpy as np, json
from pathlib import Path
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw_path

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'

def softdtw_proto(curves):
    """Compute Soft-DTW barycenter and align curves.

    Parameters
    ----------
    curves : list[np.ndarray]
        Variable-length univariate series.

    Returns
    -------
    proto : np.ndarray
        Barycenter curve.
    aligned : np.ndarray
        Curves warped to the prototype's timeline.
    """
    L = max(len(c) for c in curves)
    grid = np.linspace(0, 1, L)
    rs = np.array([
        np.interp(grid, np.linspace(0, 1, len(c)), c) for c in curves
    ])
    proto = softdtw_barycenter(rs[:, :, None]).ravel()
    aligned = []
    for r in rs:
        path, _ = dtw_path(r[:, None], proto[:, None])
        arr = np.zeros(len(proto))
        cnt = np.zeros(len(proto))
        for i, j in path:
            arr[j] += r[i]
            cnt[j] += 1
        aligned.append(arr / np.maximum(cnt, 1))
    return proto, np.stack(aligned, axis=0)

def build(step=2, tag='T'):
    qc = pd.read_csv(DATA / 'qc_result.csv')
    ev = pd.read_csv(DATA / 'op_event.csv')
    ts = pd.read_csv(DATA / 'ts_signal.csv')
    ev['ts'] = pd.to_datetime(ev['ts'])
    # duration filter: top quartile among passing batches
    durations = (ev.groupby('batch_id')['ts'].max() -
                 ev.groupby('batch_id')['ts'].min()).dt.total_seconds()
    q75 = durations.quantile(0.75)
    eligible = set(qc[qc['pass_flag'] == 1]['batch_id']).intersection(
        durations[durations >= q75].index
    )
    if not eligible:
        print('No passing batches with sufficient duration')
        return
    ts = ts[ts['batch_id'].isin(eligible)]
    ev = ev[ev['batch_id'].isin(eligible)]
    # derive segments (simple): consecutive events define [start,end)
    segs = []
    for bid, g in ev.groupby('batch_id'):
        g = g.sort_values('ts').reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, 'step'] == step:
                segs.append((bid, g.loc[i, 'ts'], g.loc[i + 1, 'ts']))
    curves = []
    for bid, s, e in segs:
        X = ts[(ts['batch_id'] == bid) & (ts['tag'] == tag)].copy()
        X['ts'] = pd.to_datetime(X['ts'])
        cur = X[(X['ts'] >= s) & (X['ts'] < e)].sort_values('ts')['value'].values
        if len(cur) > 10:
            curves.append(cur)
    if not curves:
        print('No curves for step', step)
        return
    proto, aligned = softdtw_proto(curves)
    env_lo = np.percentile(aligned, 5, axis=0)
    env_hi = np.percentile(aligned, 95, axis=0)
    outd = Path(__file__).resolve().parents[0] / 'artifacts'
    outd.mkdir(exist_ok=True)
    np.save(outd / f'proto_step{step}_{tag}.npy', proto)
    np.save(outd / f'envlo_step{step}_{tag}.npy', env_lo)
    np.save(outd / f'envhi_step{step}_{tag}.npy', env_hi)
    meta = {'step': step, 'tag': tag, 'n_samples': len(curves)}
    with open(outd / f'meta_step{step}_{tag}.json', 'w') as f:
        json.dump(meta, f)
    print('Saved golden curve for step', step, 'tag', tag)

if __name__ == '__main__':
    build(step=2, tag='T')
