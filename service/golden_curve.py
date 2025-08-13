import pandas as pd, numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'

def dba_proto(curves):
    # very small DBA-like: return mean after aligning to same length
    L = max(len(c) for c in curves)
    rs = [np.interp(np.linspace(0,1,L), np.linspace(0,1,len(c)), c) for c in curves]
    return np.mean(np.stack(rs,axis=0), axis=0)

def build(step=2, tag='T'):
    qc = pd.read_csv(DATA/'qc_result.csv')
    good = qc[qc['pass_flag']==True]['batch_id'].tolist()
    ts = pd.read_csv(DATA/'ts_signal.csv')
    ev = pd.read_csv(DATA/'op_event.csv')
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
        X = ts[(ts['batch_id']==bid)&(ts['tag']==tag)]
        X['ts'] = pd.to_datetime(X['ts'])
        cur = X[(X['ts']>=s)&(X['ts']<e)].sort_values('ts')['value'].values
        if len(cur)>10:
            curves.append(cur)
    if not curves:
        print('No curves for step', step)
        return
    proto = dba_proto(curves)
    env_lo = np.percentile(np.stack([np.interp(np.linspace(0,1,len(proto)), np.linspace(0,1,len(c)), c) for c in curves], axis=0), 5, axis=0)
    env_hi = np.percentile(np.stack([np.interp(np.linspace(0,1,len(proto)), np.linspace(0,1,len(c)), c) for c in curves], axis=0), 95, axis=0)
    outd = Path(__file__).resolve().parents[0] / 'artifacts'
    outd.mkdir(exist_ok=True)
    np.save(outd/f'proto_step{step}_{tag}.npy', proto)
    np.save(outd/f'envlo_step{step}_{tag}.npy', env_lo)
    np.save(outd/f'envhi_step{step}_{tag}.npy', env_hi)
    print('Saved golden curve for step', step, 'tag', tag)

if __name__ == '__main__':
    build(step=2, tag='T')
