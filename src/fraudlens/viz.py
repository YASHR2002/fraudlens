"""Shared matplotlib style: validated colour-blind-safe palette and recessive axes."""

from __future__ import annotations

import matplotlib as mpl

# Categorical slots 1-2 of the validated palette (light mode): legit / fraud, or
# "this model" / "comparison". Text never uses series colours.
BLUE, ORANGE = "#2a78d6", "#eb6834"
QUIET = "#b4b2ac"  # de-emphasised marks
INK, INK2 = "#0b0b0b", "#52514e"  # primary / secondary text
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
SERIES = (BLUE, ORANGE, "#1baf7a")  # first three slots validate all-pairs

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "figure.dpi": 110,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.edgecolor": QUIET,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
}


def apply_style() -> None:
    """Apply the FraudLens chart style globally (headless-safe backend for scripts)."""
    mpl.rcParams.update(STYLE)
