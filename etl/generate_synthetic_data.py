"""Generate synthetic batches with 8-step recipe and multi-signal timeseries.
Outputs CSVs into data/synthetic/.
"""
import numpy as np, pandas as pd, json, random
from datetime import datetime, timedelta
from pathlib import Path

rng = np.random.default_rng(2025)
N_BATCH = 3
BASE_START = datetime(2025,5,1,8,0,0)  # arbitrary
KETTLES = ['KRPF-02']
PROCESS_CARD = 'B2024S3003385R'

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'
DATA_DIR.mkdir(parents=True, exist_ok=True)

materials = ['A','B','C','D','E','G','H']
lots = [f'L{y}{i:03d}' for y in (24,25) for i in range(1,200)]

def draw_recipe():
    # base around the card, with noise
    return {
        'A': rng.normal(500, 20),
        'B': rng.normal(50, 5),
        'C': rng.normal(46, 4),
        'D': rng.normal(50, 5),
        'E': rng.normal(22, 3),
        'G': rng.normal(4, 0.5),
        'H': rng.normal(65, 6)
    }

def step_durations():
    # minutes per step (1..8)
    base = np.array([10, 60, 20, 60, 25, 90, 40, 30], dtype=float)
    noise = rng.normal(0, [2,5,3,5,3,7,5,3])
    return np.maximum(5, base + noise)

def make_timegrid(durs, dt=30):
    # dt seconds per sample
    times = [0]
    for m in durs:
        times.append(times[-1] + int(m*60/dt))
    # list of segment index -> (start_idx, end_idx)
    segs = []
    for i in range(8):
        segs.append((times[i], times[i+1]))
    total = times[-1]
    return segs, total

def synth_curve(durs, segs):
    n_total = segs[-1][1]
    # signals arrays
    T = np.zeros(n_total)
    pH = np.zeros(n_total)
    Vac = np.zeros(n_total)             # kPa (absolute), lower means higher vacuum
    Flow = np.zeros(n_total)
    RPM = np.zeros(n_total)
    DehydV = np.zeros(n_total)          # cumulative ml

    # baseline
    T0 = rng.normal(25, 2)
    pH0 = rng.normal(9.4, 0.2)
    vac_base = 101.3                     # kPa ~ atmosphere
    rpm_base = rng.normal(150, 10)

    # Helper to fill a segment linearly
    def fill_linear(arr, i0, i1, v0, v1, jitter=0.01):
        n = i1 - i0
        if n <= 0: return
        arr[i0:i1] = np.linspace(v0, v1, n) + rng.normal(0, jitter, n)

    # Step mapping (simplified):
    # 1 mix + adjust pH to 9.0-10.0
    i0,i1 = segs[0]
    fill_linear(T, i0, i1, T0, T0+5)
    fill_linear(pH, i0, i1, pH0, rng.normal(9.5, 0.1))
    fill_linear(RPM, i0, i1, rpm_base, rpm_base+20)

    # 2 heat to 95-98C, hold
    i0,i1 = segs[1]
    Tset = rng.normal(96.5, 0.8)
    fill_linear(T, i0, i1, T[i0-1] if i0>0 else T0, Tset)
    pH[i0:i1] = pH[i0-1] if i0>0 else pH0
    RPM[i0:i1] = rpm_base+30

    # 3 cool to 80C + acid to pH 9.0-10.0
    i0,i1 = segs[2]
    fill_linear(T, i0, i1, Tset, rng.normal(80,1.5))
    fill_linear(pH, i0, i1, pH[i0-1], rng.normal(9.4,0.15))
    RPM[i0:i1] = rpm_base+10

    # 4 heat to 95-98C, hold
    i0,i1 = segs[3]
    Tset2 = rng.normal(97, 0.6)
    fill_linear(T, i0, i1, T[i0-1], Tset2)
    pH[i0:i1] = rng.normal(9.6,0.1)
    RPM[i0:i1] = rpm_base+35

    # 5 cool to 70C + add A/D and adjust pH 8.4-9.0
    i0,i1 = segs[4]
    fill_linear(T, i0, i1, Tset2, rng.normal(70,1.2))
    fill_linear(pH, i0, i1, pH[i0-1], rng.normal(8.7,0.15))
    RPM[i0:i1] = rpm_base+15

    # 6 heat to 95C, hold 90 min
    i0,i1 = segs[5]
    Tset3 = rng.normal(95.0, 0.5)
    fill_linear(T, i0, i1, 70, Tset3)
    pH[i0:i1] = rng.normal(9.2,0.15)
    RPM[i0:i1] = rpm_base+30

    # 7 cool to 70-80C + vacuum dehydration (≤16000 ml), continue 30 min
    i0,i1 = segs[6]
    Ttarget = rng.normal(76, 2.0)
    fill_linear(T, i0, i1, Tset3, Ttarget)
    pH[i0:i1] = rng.normal(9.0,0.2)
    RPM[i0:i1] = rpm_base+20
    # vacuum on: Vac lower (e.g., 60-70 kPa), DehydV cumulative
    n = i1 - i0
    vac_level = rng.normal(70,5)
    Vac[i0:i1] = vac_level + rng.normal(0,1.0,n)
    total_dehyd = max(8000, min(16000, rng.normal(12000, 2000)))
    # generate bell-shaped rate
    x = np.linspace(-2,2,n)
    rate = np.exp(-x**2)                 # bell curve
    rate = rate / rate.sum() * total_dehyd
    DehydV[i0:i1] = np.cumsum(rate)

    # 8 cool to 70C + mixing 30 min
    i0,i1 = segs[7]
    fill_linear(T, i0, i1, Ttarget, 70+rng.normal(0,0.5))
    pH[i0:i1] = rng.normal(8.9,0.1)
    RPM[i0:i1] = rpm_base+10

    # defaults for other segments
    # Flow ~ correlated with heating/cooling activity
    Flow = (np.abs(np.gradient(T)) * 20.0) + rng.normal(1.0, 0.5, len(T))
    Flow = np.maximum(0, Flow)

    # vacuum outside step 7 ~ atmospheric ~101.3
    Vac[Vac==0] = 101.3 + rng.normal(0,0.3,(Vac==0).sum())

    # ensure monotonic cumulative DehydV
    for i in range(1,len(DehydV)):
        if DehydV[i] < DehydV[i-1]:
            DehydV[i] = DehydV[i-1]

    return {
        'T':T, 'pH':pH, 'Vac':Vac, 'Flow':Flow, 'RPM':RPM, 'DehydV':DehydV
    }

# Quality function (hidden ground-truth): depends on hold quality, dehydration, pH windows, temperature exposure.
def qc_from_features(durs, signals):
    T = signals['T']; pH = signals['pH']; DehydV = signals['DehydV']
    # simple features
    vis = 55 - 0.0006*(DehydV[-1]-11000) + 0.02*(np.clip(np.max(T)-97, -2, 2))           - 0.5*(np.mean((pH-9.2)**2))
    vis += np.random.normal(0, 1.5)
    free_hcho = 0.35 - 0.00002*(np.trapz(T/100)) + 0.01*(np.mean(np.maximum(pH-9.6,0)))
    free_hcho = max(0.02, free_hcho + np.random.normal(0, 0.03))
    moisture = 20.5 - 0.0002*DehydV[-1] + np.random.normal(0,0.3)
    dextrin = 48 + np.random.normal(0,1.0)

    # pass rules
    pass_flag = (40 <= vis <= 60) and (free_hcho <= 0.30) and (19.0 <= moisture <= 20.5)
    return vis, free_hcho, moisture, dextrin, pass_flag

def main():
    batch_rows = []
    qc_rows = []
    ts_rows = []
    ev_rows = []
    rm_rows = []

    dt = 60  # seconds sampling (lite tiny)

    for k in range(N_BATCH):
        b_id = f"BATCH_{k+1:04d}"
        start = BASE_START + timedelta(hours=int(k*3))
        kettle = random.choice(KETTLES)
        shift = random.choice(['A','B','C'])
        team = random.choice(['T1','T2','T3'])

        durs = step_durations()
        segs, total = make_timegrid(durs, dt=dt)
        sigs = synth_curve(durs, segs)

        # events (one per step boundary, simplified)
        cursor = start
        for step, (i0,i1) in enumerate(segs, start=1):
            ts = cursor + timedelta(seconds=i0*dt)
            if step in (1,5,8):
                # add materials
                rec = draw_recipe()
                for m,v in rec.items():
                    ev_rows.append({
                        'batch_id': b_id, 'step': step, 'action': 'add',
                        'param': json.dumps({'material':m, 'mass_kg': round(max(v,0.1),2)}),
                        'ts': ts.isoformat()
                    })
                    rm_rows.append({'material': m, 'lot': random.choice(lots), 'specs': json.dumps({'purity': round(np.random.uniform(0.95,0.99),3)})})
            if step in (2,4,6):
                setT = float(np.max(sigs['T'][i0:i1]))
                ev_rows.append({'batch_id': b_id, 'step': step, 'action': 'heat_hold',
                                'param': json.dumps({'target_T': round(setT,2)}), 'ts': ts.isoformat()})
            if step == 7:
                ev_rows.append({'batch_id': b_id, 'step': step, 'action': 'vacuum',
                                'param': json.dumps({'max_total_ml': int(sigs['DehydV'][i1-1])}), 'ts': ts.isoformat()})
        end = start + timedelta(seconds=segs[-1][1]*dt)

        # ts signals
        idx = np.arange(segs[-1][1])
        ts_base = start
        for tag, arr in sigs.items():
            for i,val in enumerate(arr):
                ts_rows.append({'ts': (ts_base + timedelta(seconds=int(i*dt))).isoformat(),
                                'batch_id': b_id, 'tag': tag, 'value': float(val)})

        # quality
        vis, fhcho, moist, dextrin, pflag = qc_from_features(durs, sigs)

        batch_rows.append({'batch_id': b_id, 'kettle_id': kettle, 'process_card_id': PROCESS_CARD,
                           'start_ts': start.isoformat(), 'end_ts': end.isoformat(),
                           'shift': shift, 'team': team})
        qc_rows.append({'batch_id': b_id, 'viscosity': round(float(vis),3),
                        'free_hcho': round(float(fhcho),4),
                        'moisture': round(float(moist),3),
                        'dextrin': round(float(dextrin),3),
                        'sec_cut_2h': round(float(np.random.normal(30,3)),3),
                        'sec_cut_24h': round(float(np.random.normal(50,5)),3),
                        'hardness': round(float(np.random.normal(80,10)),3),
                        'penetration': round(float(np.random.normal(60,8)),3),
                        'pass_flag': bool(pflag)})

    # Save CSVs
    pd.DataFrame(batch_rows).to_csv(DATA_DIR/'batch.csv', index=False)
    pd.DataFrame(qc_rows).to_csv(DATA_DIR/'qc_result.csv', index=False)
    pd.DataFrame(ts_rows).to_csv(DATA_DIR/'ts_signal.csv', index=False)
    pd.DataFrame(ev_rows).to_csv(DATA_DIR/'op_event.csv', index=False)
    pd.DataFrame(rm_rows).drop_duplicates().to_csv(DATA_DIR/'raw_material.csv', index=False)

if __name__ == '__main__':
    main()
