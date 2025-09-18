# SPDX-License-Identifier: Apache-2.0
"""Utility helpers for orchestrator runtime."""
from __future__ import annotations

import importlib
from typing import Any


def resolve_callable(qualname: str) -> Any:
    """Resolve dotted path to a callable.

    Supports shorthands:
    - `module.function`
    - `package.module:function`
    If no package prefix is provided, defaults to `orchestrator.plugins` namespace.
    """
    if ":" in qualname:
        module_name, func_name = qualname.split(":", 1)
    else:
        parts = qualname.split(".")
        module_name, func_name = ".".join(parts[:-1]), parts[-1]
        if not module_name.startswith("orchestrator."):
            module_name = f"orchestrator.plugins.{module_name}" if module_name else "orchestrator.plugins"
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(f"callable '{qualname}' not found in module '{module_name}'") from exc
