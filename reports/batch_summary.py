from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_batch_summary(top_factors: Iterable[tuple], output_pdf: str | Path) -> None:
    """Create a technical PDF outlining top drivers for a batch issue."""
    pdf_path = Path(output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        lines = ["Batch Issue Summary", "", "Top contributing factors:"]
        lines.extend([f"{k}: {v:.3f}" for k, v in top_factors])
        fig.text(0.01, 0.99, "\n".join(lines), va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
