import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Tuple

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import norm


def estimate_cate(X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> Tuple[np.ndarray, LinearRegression]:
    """Estimate conditional average treatment effect using a simple DR-Learner."""
    X = np.asarray(X)
    t = np.asarray(treatment)
    y = np.asarray(outcome)

    # Propensity score
    prop = LogisticRegression(max_iter=200).fit(X, t)
    p = prop.predict_proba(X)[:, 1]

    # Outcome models for treated and control
    mu1_model = RandomForestRegressor(n_estimators=200, random_state=0)
    mu0_model = RandomForestRegressor(n_estimators=200, random_state=0)
    mu1_model.fit(X[t == 1], y[t == 1])
    mu0_model.fit(X[t == 0], y[t == 0])

    mu1 = mu1_model.predict(X)
    mu0 = mu0_model.predict(X)

    pseudo = mu1 - mu0 + (t - p) * (y - (t * mu1 + (1 - t) * mu0)) / np.clip(p * (1 - p), 1e-3, None)
    final = LinearRegression().fit(X, pseudo)
    cate = final.predict(X)
    return cate, final


def safe_bayesian_opt(
    objective: Callable[[np.ndarray], float],
    bounds: Dict[str, Tuple[float, float]],
    safe_bounds: Dict[str, Tuple[float, float]],
    n_iter: int = 20,
    random_state: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Perform a naive safe Bayesian optimisation.

    Parameters
    ----------
    objective : Callable
        Function to maximise; receives an array of shape (d,).
    bounds : dict
        Search space for each variable.
    safe_bounds : dict
        Region considered safe; exploration is limited here.
    n_iter : int
        Number of iterations.

    Returns
    -------
    tuple
        Best setpoint and corresponding objective value within the safe region.
    """

    rng = np.random.default_rng(random_state)
    keys = list(bounds.keys())
    d = len(keys)

    def sample(n: int) -> np.ndarray:
        lows = np.array([bounds[k][0] for k in keys])
        highs = np.array([bounds[k][1] for k in keys])
        return rng.uniform(lows, highs, size=(n, d))

    def in_safe(x: np.ndarray) -> np.ndarray:
        mask = np.ones(len(x), dtype=bool)
        for i, k in enumerate(keys):
            sl, sh = safe_bounds.get(k, bounds[k])
            mask &= (x[:, i] >= sl) & (x[:, i] <= sh)
        return mask

    # Start from centre of safe region
    x0 = np.array([(safe_bounds.get(k, v)[0] + safe_bounds.get(k, v)[1]) / 2 for k, v in bounds.items()])
    X = [x0]
    y = [objective(x0)]

    kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    for _ in range(n_iter):
        gp.fit(np.array(X), np.array(y))
        cand = sample(200)
        cand = cand[in_safe(cand)]
        if len(cand) == 0:
            break
        mu, sigma = gp.predict(cand, return_std=True)
        best_y = np.max(y)
        imp = mu - best_y
        z = imp / np.clip(sigma, 1e-9, None)
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        x_next = cand[np.argmax(ei)]
        y_next = objective(x_next)
        X.append(x_next)
        y.append(y_next)

    best = np.argmax(y)
    return np.array(X[best]), y[best]
