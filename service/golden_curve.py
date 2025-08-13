import json
import pandas as pd, numpy as np
from pathlib import Path
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw_path
from tslearn.utils import to_time_series_dataset

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'

def dba_proto(curves):
    """Compute prototype using Soft-DTW barycenter averaging."""
    dataset = to_time_series_dataset([c.reshape(-1, 1) for c in curves])
    proto = softdtw_barycenter(dataset)
    return proto.ravel()

def build(step=2, tag='T'):
    qc = pd.read_csv(DATA / 'qc_result.csv')
    good = qc[qc['pass_flag'] == True]['batch_id'].tolist()
    ts = pd.read_csv(DATA / 'ts_signal.csv')
    ev = pd.read_csv(DATA / 'op_event.csv')
    ts = ts[ts['batch_id'].isin(good)]
    ev = ev[ev['batch_id'].isin(good)]
    ts['ts'] = pd.to_datetime(ts['ts'])
    ev['ts'] = pd.to_datetime(ev['ts'])
    # derive segments (simple): consecutive events define [start,end)
    segs = []
    for bid, g in ev.groupby('batch_id'):
        g = g.sort_values('ts').reset_index(drop=True)
        for i in range(len(g)-1):
            if g.loc[i,'step']==step:
                segs.append((bid, g.loc[i,'ts'], g.loc[i+1,'ts']))
    curves = []
    for bid, s, e in segs:
        X = ts[(ts['batch_id'] == bid) & (ts['tag'] == tag)]
        cur = X[(X['ts'] >= s) & (X['ts'] < e)].sort_values('ts')['value'].values
        if len(cur) > 10:
            curves.append(cur)
    if not curves:
        print('No curves for step', step)
        return
    proto = dba_proto(curves)
    aligned = []
    for c in curves:
        path, _ = dtw_path(c.reshape(-1, 1), proto.reshape(-1, 1))
        L = len(proto)
        tmp = [[] for _ in range(L)]
        for i, j in path:
            tmp[j].append(c[i])
        aligned.append(np.array([np.mean(v) for v in tmp]))
    env_lo = np.percentile(np.stack(aligned, axis=0), 5, axis=0)
    env_hi = np.percentile(np.stack(aligned, axis=0), 95, axis=0)
    outd = Path(__file__).resolve().parents[0] / 'artifacts'
    outd.mkdir(exist_ok=True)
    np.save(outd / f"proto_step{step}_{tag}.npy", proto)
    np.save(outd / f"envlo_step{step}_{tag}.npy", env_lo)
    np.save(outd / f"envhi_step{step}_{tag}.npy", env_hi)
    meta = {"step": step, "tag": tag, "sample_count": len(curves)}
    with open(outd / f"meta_step{step}_{tag}.json", "w") as f:
        json.dump(meta, f)
    print('Saved golden curve for step', step, 'tag', tag)

if __name__ == '__main__':
    build(step=2, tag='T')
