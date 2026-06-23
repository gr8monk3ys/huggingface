"""Make the per-Space helper modules importable from the test suite.

Spaces deploy independently, so their modules live in subfolders rather than an
installed package. We add the relevant folders to ``sys.path`` for testing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for sub in ("code-explainer-space", "trading-dashboard-space"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
