from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tslearn.metrics import soft_dtw

app = FastAPI(title='Industrial AI Process Optimization')

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

ART = Path(__file__).resolve().parents[0] / 'artifacts'
ART.mkdir(exist_ok=True)

_cls = None
_reg = None
_proto = None
_env_lo = None
_env_hi = None

# per-batch in-memory state
_state: Dict[str, Dict] = {}


@app.on_event('startup')
def _load_artifacts() -> None:
    """Load ML models and golden-curve artifacts when service starts."""
    global _cls, _reg, _proto, _env_lo, _env_hi
    try:
        _cls = joblib.load(ART / 'cls_pass.pkl')
        _reg = joblib.load(ART / 'reg_viscosity.pkl')
    except Exception as e:  # pragma: no cover - informational
        print('Model load warning:', e)

    try:
        _proto = np.load(ART / 'proto_step2_T.npy')
        _env_lo = np.load(ART / 'envlo_step2_T.npy')
        _env_hi = np.load(ART / 'envhi_step2_T.npy')
    except Exception as e:  # pragma: no cover - informational
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
    window = min(len(df), 5)
    feats = df.rolling(window=window).mean().iloc[-1].to_frame().T

    deviation = None
    envelope_violations = None
    curve = df.get('T', pd.Series(dtype=float)).dropna().values.astype(float)
    if _proto is not None and len(curve) > 1:
        deviation = float(soft_dtw(curve, _proto))
    if _env_lo is not None and _env_hi is not None and len(curve) > 0:
        L = min(len(curve), len(_env_lo))
        env_lo = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(_env_lo)), _env_lo)
        env_hi = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(_env_hi)), _env_hi)
        segment = curve[:L]
        viol = (segment < env_lo) | (segment > env_hi)
        envelope_violations = int(viol.sum())

    prob = None
    if _cls is not None:
        try:
            prob = float(_cls.predict_proba(feats)[:, 1][0])
        except Exception:
            pass

    visc = None
    if _reg is not None:
        try:
            visc = float(_reg.predict(feats)[0])
        except Exception:
            pass

    advice = []
    if prob is not None and prob < 0.7:
        advice.append('Low pass probability; review process parameters')
    if deviation is not None and _env_hi is not None and _env_lo is not None:
        thresh = float(np.mean(_env_hi - _env_lo))
        if deviation > thresh:
            advice.append('Temperature profile deviates from prototype')
    if envelope_violations is not None and envelope_violations > 0:
        advice.append('Temperature outside control envelope')
    if not advice:
        advice.append('Process on track')

    st.update({
        'features': feats.iloc[0].to_dict(),
        'deviation': deviation,
        'envelope_violations': envelope_violations,
        'pass_prob': prob,
        'viscosity': visc,
        'advice': advice,
    })

    return {
        'batch_id': batch_id,
        'pass_prob': prob,
        'viscosity': visc,
        'deviation': deviation,
        'envelope_violations': envelope_violations,
        'advice': advice,
    }

@app.get('/state/{batch_id}')
def get_state(batch_id: str):
    st = _state.get(batch_id)
    if not st:
        return {'batch_id': batch_id, 'error': 'unknown batch'}
    data = {k: v for k, v in st.items() if k != 'history'}
    return {'batch_id': batch_id, **data}


@app.get('/advice/{batch_id}')
def advice(batch_id: str):
    st = _state.get(batch_id)
    if not st:
        return {'batch_id': batch_id, 'advice': ['Insufficient data']}
    return {
        'batch_id': batch_id,
        'pass_prob': st.get('pass_prob'),
        'viscosity': st.get('viscosity'),
        'deviation': st.get('deviation'),
        'envelope_violations': st.get('envelope_violations'),
        'advice': st.get('advice', ['Process on track']),
    }
