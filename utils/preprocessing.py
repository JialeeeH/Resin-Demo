import numpy as np

def savgol_slope(series, window=21, poly=3, step=1):
    # Simple poly fit derivative (naive replacement for Savitzky-Golay to avoid scipy dep)
    x = np.arange(len(series))
    y = np.asarray(series, dtype=float)
    half = window // 2
    out = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        lo = max(0, i-half)
        hi = min(len(y), i+half+1)
        xx = x[lo:hi] - x[lo]
        yy = y[lo:hi]
        if len(xx) > poly:
            coef = np.polyfit(xx, yy, poly)
            # derivative of poly at position (center)
            dcoef = np.polyder(coef)
            out[i] = np.polyval(dcoef, (i-lo))
        else:
            out[i] = np.nan
    # approximate per-step derivative
    return out / max(step, 1)
