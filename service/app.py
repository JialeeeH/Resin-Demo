from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib, os, json, random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tslearn.metrics import soft_dtw

app = FastAPI(title='Industrial AI Process Optimization')

ART = Path(__file__).resolve().parents[0] / 'artifacts'
ART.mkdir(exist_ok=True)

# lazy load
_cls = None
_reg = None
_proto = None
_env_lo = None
_env_hi = None
_state: Dict[str, Dict] = {}
_audit_log: List[Dict] = []

try:
    _cls = joblib.load(ART / 'cls_pass.pkl')
    _reg = joblib.load(ART / 'reg_viscosity.pkl')
except Exception as e:
    print('Model load warning:', e)

try:
    _proto = np.load(ART / 'proto_step2_T.npy')
    _env_lo = np.load(ART / 'envlo_step2_T.npy')
    _env_hi = np.load(ART / 'envhi_step2_T.npy')
except Exception as e:
    print('Golden curve load warning:', e)

class Tick(BaseModel):
    ts: str
    batch_id: str
    signals: Dict[str, float]  # e.g., {'T':96.1, 'pH':9.4, 'Vac':70, 'Flow':10, 'RPM':180, 'DehydV':5000}

@app.get('/health')
def health():
    return {'ok': True}

@app.post('/batch/start')
def start_batch(batch_id: str, kettle_id: str, process_card_id: str):
    # In real impl, create state in store; here return echo
    return {'status':'started','batch_id':batch_id}

@app.post('/batch/{batch_id}/tick')
def tick(batch_id: str, payload: Tick):
    """Update batch state with new signals and run predictions."""
    st = _state.setdefault(batch_id, {'history': []})
    st['history'].append(payload.signals)

    df = pd.DataFrame(st['history'])
    feats = df.mean().to_frame().T
    deviation = None
    if _proto is not None:
        curve = df.get('T', pd.Series(dtype=float)).dropna().values.astype(float)
        if len(curve) > 1:
            deviation = float(soft_dtw(curve, _proto))
    st['deviation'] = deviation

    prob = None
    if _cls is not None:
        try:
            prob = float(_cls.predict_proba(feats)[:, 1][0])
        except Exception:
            prob = None
    st['pass_prob'] = prob

    visc = None
    if _reg is not None:
        try:
            visc = float(_reg.predict(feats)[0])
        except Exception:
            visc = None
    st['viscosity'] = visc
    st['features'] = feats.iloc[0].to_dict()

    return {'batch_id': batch_id, 'deviation': deviation,
            'pass_prob': prob, 'viscosity': visc}


@app.get('/dashboard/kpis')
def dashboard_kpis():
    """Return high level KPI cards with status color."""
    data = [
        {'name': 'Yield', 'value': 0.95, 'status': 'green'},
        {'name': 'Throughput', 'value': 0.80, 'status': 'yellow'},
        {'name': 'Scrap Rate', 'value': 0.15, 'status': 'red'},
    ]
    return {'kpis': data}


@app.get('/dashboard/trends/{metric}')
def dashboard_trend(metric: str):
    """Return simple trend data for a metric."""
    now = datetime.utcnow()
    pts = [
        {'ts': (now - timedelta(minutes=i)).isoformat(), 'value': random.random()}
        for i in range(10)
    ][::-1]
    return {'metric': metric, 'trend': pts}


@app.get('/dashboard/stage-kpis')
def stage_kpis():
    """Return KPIs for each production stage."""
    stages = {
        'mixing': {
            'temperature': {'value': 98, 'status': 'green'},
            'pH': {'value': 9.1, 'status': 'yellow'},
        },
        'reaction': {
            'pressure': {'value': 80, 'status': 'red'},
        },
    }
    return {'stages': stages}


@app.get('/dashboard/setpoints')
def recommended_setpoints():
    """Return recommended setpoint ranges for optimization/SOP."""
    data = {
        'temperature': {'low': 95, 'high': 100},
        'pH': {'low': 8.5, 'high': 9.5},
        'pressure': {'low': 60, 'high': 75},
    }
    return {'setpoints': data}

@app.get('/advice/{batch_id}')
def advice(batch_id: str):
    st = _state.get(batch_id)
    if not st or st.get('pass_prob') is None:
        return {'batch_id': batch_id, 'advice': ['Insufficient data']}

    adv = []
    prob = st.get('pass_prob')
    dev = st.get('deviation')
    if prob is not None and prob < 0.7:
        adv.append('Low pass probability; review process parameters')
    if dev is not None and _env_hi is not None and _env_lo is not None:
        if dev > float(np.mean(_env_hi - _env_lo)):
            adv.append('Temperature profile deviates from prototype')
    if not adv:
        adv.append('Process on track')
    _audit_log.append({'batch_id': batch_id, 'advice': adv})
    return {'batch_id': batch_id, 'pass_prob': prob,
            'viscosity': st.get('viscosity'), 'advice': adv}
