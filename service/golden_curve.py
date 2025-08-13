import json
from pathlib import Path

import numpy as np
import pandas as pd
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw_path

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def softdtw_proto(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute Soft-DTW barycenter and align curves to the prototype.

    Parameters
    ----------
    curves : list[np.ndarray]
        Variable-length univariate series.

    Returns
    -------
    proto : np.ndarray
        Barycenter curve (length = max length among inputs).
    aligned : np.ndarray
        Array of shape (n_curves, len(proto)) with curves warped
        to the prototype's timeline.
    """
    # Resample all curves to a common grid for stable barycenter estimation
    L = max(len(c) for c in curves)
    grid = np.linspace(0, 1, L)
    rs = np.array([np.interp(grid, np.linspace(0, 1, len(c)), c) for c in curves])

    # Soft-DTW barycenter expects (n_series, length, dim)
    proto = softdtw_barycenter(rs[:, :, None]).ravel()

    # Align each resampled curve to the prototype using DTW path averaging
    aligned = []
    for r in rs:
        path, _ = dtw_path(r[:, None], proto[:, None])
        acc = np.zeros(len(proto))
        cnt = np.zeros(len(proto))
        for i, j in path:
            acc[j] += r[i]
            cnt[j] += 1
        aligned.append(acc / np.maximum(cnt, 1))
    return proto, np.stack(aligned, axis=0)


def build(step: int = 2, tag: str = "T"):
    qc = pd.read_csv(DATA / "qc_result.csv")
    ts = pd.read_csv(DATA / "ts_signal.csv")
    ev = pd.read_csv(DATA / "op_event.csv")

    # Keep only passing batches (robust to 1/True)
    pass_mask = (qc["pass_flag"] == 1) | (qc["pass_flag"] == True)
    good = set(qc.loc[pass_mask, "batch_id"])

    ts = ts[ts["batch_id"].isin(good)]
    ev = ev[ev["batch_id"].isin(good)]

    ts["ts"] = pd.to_datetime(ts["ts"])
    ev["ts"] = pd.to_datetime(ev["ts"])

    # Duration filter: keep top quartile among passing batches
    durations = (ev.groupby("batch_id")["ts"].max() - ev.groupby("batch_id")["ts"].min()).dt.total_seconds()
    q75 = durations.quantile(0.75)
    eligible = good.intersection(durations[durations >= q75].index)

    if not eligible:
        print("No passing batches with sufficient duration")
        return

    ts = ts[ts["batch_id"].isin(eligible)]
    ev = ev[ev["batch_id"].isin(eligible)]

    # Derive segments: consecutive events define [start, end) for a given step
    segs = []
    for bid, g in ev.groupby("batch_id"):
        g = g.sort_values("ts").reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, "step"] == step:
                segs.append((bid, g.loc[i, "ts"], g.loc[i + 1, "ts"]))

    # Slice timeseries by segment and tag
    curves = []
    for bid, s, e in segs:
        X = ts[(ts["batch_id"] == bid) & (ts["tag"] == tag)]
        cur = X[(X["ts"] >= s) & (X["ts"] < e)].sort_values("ts")["value"].values
        if len(cur) > 10:
            curves.append(cur)

    if not curves:
        print("No curves for step", step)
        return

    # Prototype + aligned ensemble
    proto, aligned = softdtw_proto(curves)

    # 5–95% envelope
    env_lo = np.percentile(aligned, 5, axis=0)
    env_hi = np.percentile(aligned, 95, axis=0)

    # Persist artifacts
    outd = Path(__file__).resolve().parents[0] / "artifacts"
    outd.mkdir(exist_ok=True)
    np.save(outd / f"proto_step{step}_{tag}.npy", proto)
    np.save(outd / f"envlo_step{step}_{tag}.npy", env_lo)
    np.save(outd / f"envhi_step{step}_{tag}.npy", env_hi)

    meta = {"step": step, "tag": tag, "n_samples": len(curves)}
    with open(outd / f"meta_step{step}_{tag}.json", "w") as f:
        json.dump(meta, f)

    print("Saved golden curve for step", step, "tag", tag)


if __name__ == "__main__":
    build(step=2, tag="T")
