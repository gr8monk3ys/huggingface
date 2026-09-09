"""Test helpers for importing the per-Space modules.

Spaces deploy independently, so their modules live in subfolders rather than an
installed package. Vendored modules (``hf_client``, ``papers``) have unique
names, so one folder holding each goes on ``sys.path``. The per-Space
``core.py`` files share a name, so load those by path under a unique module
name via :func:`load_local_module`.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Vendored modules are byte-identical across the Spaces that carry them (a
# guard in test_vendored.py enforces that), so one copy of each on the path is
# enough. Each entry names a folder holding one such module: code-explainer for
# hf_client.py, paper-recommender for papers.py.
_VENDOR_DIRS = ("code-explainer-space", "paper-recommender-space")
for _name in _VENDOR_DIRS:
    _dir = str(ROOT / _name)
    if _dir not in sys.path:
        sys.path.append(_dir)


def load_local_module(module_name: str, relative_path: str):
    """Import a module from a file path under a unique *module_name*."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
