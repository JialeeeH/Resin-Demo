from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_weekly_report(metrics: Dict[str, float], output_pdf: str | Path) -> None:
    """Create a simple management-facing PDF with weekly KPIs."""
    pdf_path = Path(output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        lines = ["Weekly Production Report", ""]
        lines.extend([f"{k}: {v}" for k, v in metrics.items()])
        fig.text(0.5, 0.95, "\n".join(lines), ha="center", va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
