"""Shared test setup."""

import matplotlib

# Headless backend: SHAP draws with pyplot, and the default Tk backend fails intermittently
# when no display is available (seen as TclError on Windows and in CI).
matplotlib.use("Agg")
