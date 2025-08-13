import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Tuple


def compare_batches(problem: pd.DataFrame, reference: pd.DataFrame, output_pdf: str | Path) -> pd.Series:
    """Compare problem vs. reference batches and export a PDF summary.

    Parameters
    ----------
    problem : pd.DataFrame
        Data for the problematic batches.
    reference : pd.DataFrame
        Data for normal/reference batches.
    output_pdf : str or Path
        Location to save the generated PDF report.

    Returns
    -------
    pd.Series
        Top‑5 factors ranked by absolute mean difference.
    """

    if problem.empty or reference.empty:
        raise ValueError("`problem` and `reference` must be non-empty dataframes")

    # Align columns and compute mean differences
    common_cols = problem.columns.intersection(reference.columns)
    diff = problem[common_cols].mean() - reference[common_cols].mean()
    top5 = diff.abs().sort_values(ascending=False).head(5)

    pdf_path = Path(output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # Bar chart of top factors
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(top5.index.astype(str), top5.values, color="tab:blue")
        ax.set_ylabel("Mean difference")
        ax.set_title("Top-5 factor differences")
        plt.xticks(rotation=45, ha="right")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Textual summary page
        summary = [f"Problem batches: {len(problem)}", f"Reference batches: {len(reference)}", "", "Top factors:"]
        summary.extend([f"{k}: {v:.3f}" for k, v in top5.items()])
        fig_txt = plt.figure(figsize=(8.27, 11.69))  # A4 size
        fig_txt.text(0.01, 0.99, "\n".join(summary), va="top", family="monospace")
        pdf.savefig(fig_txt, bbox_inches="tight")
        plt.close(fig_txt)

    return top5
