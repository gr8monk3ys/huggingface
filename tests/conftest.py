"""Test helpers for importing the per-Space modules.

Spaces deploy independently, so their modules live in subfolders rather than an
installed package. ``hf_client`` is unique, so we put its folder on ``sys.path``.
The per-Space ``core.py`` files share a name, so load those by path under a
unique module name via :func:`load_local_module`.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# hf_client.py is identical across Spaces; one copy on the path is enough.
_helper_dir = str(ROOT / "code-explainer-space")
if _helper_dir not in sys.path:
    sys.path.insert(0, _helper_dir)


def load_local_module(module_name: str, relative_path: str):
    """Import a module from a file path under a unique *module_name*."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
