from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from etl.stage_segmentation import build_segments_from_events, detect_changepoints, build_stage_features


def make_ts_with_change():
    base = datetime(2025, 1, 1, 0, 0, 0)
    times = [base + timedelta(minutes=i) for i in range(10)]
    rows = []
    for t in times:
        val = 1.0 if t < times[5] else 10.0
        rows.extend([
            {"batch_id": "B1", "ts": t, "tag": "T", "value": val},
            {"batch_id": "B1", "ts": t, "tag": "pH", "value": 7.0},
            {"batch_id": "B1", "ts": t, "tag": "Vac", "value": 100.0},
            {"batch_id": "B1", "ts": t, "tag": "Flow", "value": 1.0},
        ])
    ts = pd.DataFrame(rows)

    events = pd.DataFrame([
        {"batch_id": "B1", "step": 1, "ts": times[0]},
        {"batch_id": "B1", "step": 2, "ts": times[-1]},
    ])
    return ts, events


def test_detect_changepoints_splits_segment():
    ts, events = make_ts_with_change()
    segs = build_segments_from_events(events)
    refined = detect_changepoints(ts, segs, thresholds={"T": 5, "pH": 1, "Vac": 1, "Flow": 1}, penalty=1.0)

    # Expect the segment to split at the change time (times[5])
    assert len(refined) == 2
    refined = refined.sort_values("segment_id")
    boundary = ts[ts["tag"] == "T"].sort_values("ts").iloc[5]["ts"]
    assert refined.iloc[0]["end_ts"] == boundary
    assert refined.iloc[1]["start_ts"] == boundary

    feats = build_stage_features(ts, refined)
    assert len(feats) == 2
    # Second segment should have higher mean temperature
    assert feats.sort_values("segment_id").iloc[0]["T_mean"] < feats.sort_values("segment_id").iloc[1]["T_mean"]
