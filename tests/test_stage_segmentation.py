import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from etl.stage_segmentation import (
    build_segments_from_events,
    adjust_segments_with_changepoints,
    build_stage_features,
)
from etl.generate_synthetic_data import step_durations, make_timegrid, synth_curve


def make_synthetic_batch():
    rng = np.random.default_rng(0)
    durs = step_durations()
    seg_idx, total = make_timegrid(durs, dt=60)
    signals = synth_curve(durs, seg_idx)
    start = datetime(2025, 1, 1, 8, 0, 0)
    times = [start + timedelta(seconds=i * 60) for i in range(total)]

    # build long-format timeseries
    rows = []
    for i, ts in enumerate(times):
        for tag, arr in signals.items():
            rows.append({"batch_id": "B1", "ts": ts, "tag": tag, "value": arr[i]})
    ts_long = pd.DataFrame(rows)

    # event times with jitter up to ±2 min
    events = []
    for step, (i0, i1) in enumerate(seg_idx, start=1):
        jitter = int(rng.integers(-120, 120))
        ts_event = start + timedelta(seconds=i0 * 60 + jitter)
        events.append({"batch_id": "B1", "step": step, "ts": ts_event})
    # final end event
    events.append({"batch_id": "B1", "step": 8, "ts": start + timedelta(seconds=seg_idx[-1][1] * 60)})
    op_event = pd.DataFrame(events)

    true_bounds = [start + timedelta(seconds=i0 * 60) for i0, _ in seg_idx]
    true_bounds.append(start + timedelta(seconds=seg_idx[-1][1] * 60))
    return ts_long, op_event, true_bounds


def test_adjust_segments_with_changepoints():
    ts_long, op_event, true_bounds = make_synthetic_batch()
    segs = build_segments_from_events(op_event)
    segs_adj = adjust_segments_with_changepoints(ts_long, segs)
    segs_adj = segs_adj.sort_values("start_ts").reset_index(drop=True)
    for i in range(len(segs_adj)):
        assert abs((segs_adj.loc[i, "start_ts"] - true_bounds[i]).total_seconds()) <= 180
        assert abs((segs_adj.loc[i, "end_ts"] - true_bounds[i + 1]).total_seconds()) <= 180


def test_build_stage_features_flags():
    ts_long, op_event, _ = make_synthetic_batch()
    segs = build_segments_from_events(op_event)
    segs_adj = adjust_segments_with_changepoints(ts_long, segs)
    feat = build_stage_features(ts_long, segs_adj)
    # pH window and heat ok should hold for majority of steps
    assert feat["pH_window"].all()
    assert feat["heat_ok"].any()
    # vacuum check specifically for step 7
    step7 = feat[feat["step"] == 7].iloc[0]
    assert step7["vacuum_leq_16L"]

