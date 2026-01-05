"""Example runners.

This package exposes each example script's `run(...)` function directly at the
package level, aliased to the *module filename*.

For example, if there is a file `global_bo_rosenbrock2d.py` that defines
`run(smoke: bool = False)`, you can call:

    import examples
    examples.global_bo_rosenbrock2d(smoke=True)

Any `*.py` file in this directory that defines a top-level callable named
`run` will be auto-imported and re-exported.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType
from typing import List

__all__: List[str] = []

_pkg_dir = Path(__file__).resolve().parent

def _has_module_level_run(src: str) -> bool:
    """Return True iff `src` defines a *module-level* `def run(...):`.

    We intentionally require `run` to be a top-level function (not a class method)
    so we don't import unrelated helper modules.
    """

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return True
    return False


def _discover_example_modules() -> List[str]:
    """Return module names (filestems) for files that define `def run(...):`."""
    mods: List[str] = []
    for p in sorted(_pkg_dir.glob("*.py")):
        name = p.stem
        if name in {"__init__", "common"} or name.startswith("_"):
            continue
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if _has_module_level_run(src):
            mods.append(name)
    return mods


def _export_runs() -> None:
    for mod_name in _discover_example_modules():
        # Force a *relative* import from THIS package (i.e., this ./examples dir).
        module: ModuleType = importlib.import_module(f".{mod_name}", package=__name__)

        # Extra safety: ensure we're loading from the same filesystem directory.
        mod_file = Path(getattr(module, "__file__", "")).resolve()
        if mod_file.parent != _pkg_dir:
            continue

        run = getattr(module, "run", None)
        if callable(run):
            globals()[mod_name] = run  # re-export under filename
            __all__.append(mod_name)


_export_runs()
