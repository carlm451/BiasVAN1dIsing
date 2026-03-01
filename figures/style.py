"""Shared plotting style for all figures."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (7, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(fig, name):
    """Save figure as both PDF and PNG."""
    ensure_output_dir()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    print(f"Saved {name}.pdf and {name}.png to {OUTPUT_DIR}")
