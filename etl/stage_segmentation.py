import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.io import read_csv, save_df
from utils.preprocessing import savgol_slope
from utils.constants import STEPS, TAGS
from etl.validation import validate_batch, validate_ts_signal


def _pelt_multivariate(X: np.ndarray, penalty: float = 1.0):
    """Basic PELT implementation for small multivariate series.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Multivariate signal.
    penalty : float
        Penalty for adding a changepoint.

    Returns
    -------
    list[int]
        Sorted list of changepoint indices (0-based, excluding n).
    """
    X = np.asarray(X)
    n, d = X.shape
    csum = np.vstack([np.zeros(d), np.cumsum(X, axis=0)])
    csq = np.vstack([np.zeros(d), np.cumsum(X ** 2, axis=0)])

    def seg_cost(s, t):
        seg_sum = csum[t] - csum[s]
        seg_sq = csq[t] - csq[s]
        length = t - s
        mean = seg_sum / length
        # sum of squared errors for all dimensions
        return float(np.sum(seg_sq - 2 * mean * seg_sum + length * mean * mean))

    F = [0.0] + [float("inf")] * n
    last = [0] * (n + 1)
    for t in range(1, n + 1):
        for s in range(t):
            val = F[s] + seg_cost(s, t) + penalty
            if val < F[t]:
                F[t] = val
                last[t] = s

    cps = []
    t = n
    while t > 0:
        s = last[t]
        if s == 0:
            break
        cps.append(s)
        t = s
    return sorted(cps)


def detect_changepoints(
    ts: pd.DataFrame,
    segs: pd.DataFrame,
    thresholds: dict | None = None,
    penalty: float = 10.0,
):
    """Refine segments by detecting changepoints within each batch.

    A lightweight PELT-style algorithm is applied on the multivariate
    signal consisting of ``T``, ``pH``, ``Vac`` and ``Flow``. Detected
    changepoints are used to split existing segments when the mean shift
    across any signal exceeds the provided threshold.

    Parameters
    ----------
    ts : DataFrame
        Timeseries data with columns ``batch_id``, ``ts``, ``tag`` and
        ``value``.
    segs : DataFrame
        Initial segments produced from operation events.
    thresholds : dict, optional
        Mapping of tag -> minimal mean shift to consider a changepoint.
        Defaults to ``{'T':2.0, 'pH':0.1, 'Vac':5.0, 'Flow':1.0}``.
    penalty : float, optional
        Penalty value used in the PELT cost function.

    Returns
    -------
    DataFrame
        Refined segments with an added ``segment_id`` column.
    """

    thresholds = thresholds or {"T": 2.0, "pH": 0.1, "Vac": 5.0, "Flow": 1.0}
    ts = ts.copy()
    ts["ts"] = pd.to_datetime(ts["ts"])
    segs = segs.copy()
    segs["start_ts"] = pd.to_datetime(segs["start_ts"])
    segs["end_ts"] = pd.to_datetime(segs["end_ts"])

    tags = list(thresholds.keys())
    refined = []
    for bid, gseg in segs.groupby("batch_id"):
        # wide dataframe of signals for this batch
        P = (
            ts[ts["batch_id"] == bid]
            .pivot_table(index="ts", columns="tag", values="value")
            .sort_index()
        )
        if P.empty:
            continue
        P = P.reindex(columns=tags).interpolate().fillna(method="bfill").fillna(method="ffill")

        cps = _pelt_multivariate(P.values, penalty=penalty)
        cps = sorted(set(cps))
        # filter by mean shift
        valid_cp_times = []
        bounds = [0] + cps + [len(P)]
        for i, cp in enumerate(cps, start=1):
            left, right = bounds[i - 1], bounds[i + 1]
            mean_left = P.iloc[left:cp].mean()
            mean_right = P.iloc[cp:right].mean()
            diff = (mean_right - mean_left).abs()
            if any(diff[tag] >= thresholds[tag] for tag in tags):
                valid_cp_times.append(P.index[cp])

        # split original segments at detected changepoints
        for _, row in gseg.iterrows():
            start = row["start_ts"]
            end = row["end_ts"]
            internal = [t for t in valid_cp_times if start < t < end]
            boundaries = [start] + internal + [end]
            for idx in range(len(boundaries) - 1):
                refined.append(
                    {
                        "batch_id": bid,
                        "step": row["step"],
                        "segment_id": idx,
                        "start_ts": boundaries[idx],
                        "end_ts": boundaries[idx + 1],
                    }
                )

    return pd.DataFrame(refined)

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
    """Compute simple statistics for each (batch, step, segment)."""

    feats = []
    ts = ts.copy()
    ts["ts"] = pd.to_datetime(ts["ts"])

    if "segment_id" not in segs.columns:
        segs = segs.copy()
        segs["segment_id"] = 0

    for (bid, step, seg_id), seg in segs.groupby(["batch_id", "step", "segment_id"]):
        start = seg["start_ts"].min()
        end = seg["end_ts"].max()
        X = ts[(ts["batch_id"] == bid) & (ts["ts"] >= start) & (ts["ts"] < end)]
        if X.empty:
            continue
        P = X.pivot_table(index="ts", columns="tag", values="value").sort_index()
        P["dTdt"] = savgol_slope(
            P["T"].interpolate().fillna(method="bfill").fillna(method="ffill").values,
            window=19,
        )
        row = {
            "batch_id": bid,
            "step": step,
            "segment_id": seg_id,
            "duration_min": (end - start).total_seconds() / 60.0,
        }
        for col in ["T", "pH", "Vac", "Flow", "RPM", "DehydV", "dTdt"]:
            if col in P:
                s = P[col].dropna()
                if s.empty:
                    continue
                row[f"{col}_mean"] = float(s.mean())
                row[f"{col}_std"] = float(s.std())
                row[f"{col}_max"] = float(s.max())
                row[f"{col}_min"] = float(s.min())
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
    segs = detect_changepoints(ts, segs)
    segs.to_csv(Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'segments.csv', index=False)
    feat = build_stage_features(ts, segs)
    save_df(feat, 'stage_features.csv')

if __name__ == '__main__':
    main()
