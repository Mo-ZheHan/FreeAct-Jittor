"""Register FreeAct's task suites and configure the LIBERO runtime.

Importing this module registers the suites. Call ``prepare_suite_env`` in each
worker process before creating an environment to apply its scene configuration.
"""

import importlib.util
import os
import sys
import types

import yaml


def _ensure_libero_config() -> None:
    """Write LIBERO's default ``config.yaml`` so that importing it never prompts interactively."""
    config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    config_file = os.path.join(config_dir, "config.yaml")
    if os.path.exists(config_file):
        return
    root = os.path.join(os.path.dirname(importlib.util.find_spec("libero").origin), "libero")
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(
            {
                "benchmark_root": root,
                "bddl_files": os.path.join(root, "bddl_files"),
                "init_states": os.path.join(root, "init_files"),
                "datasets": os.path.join(root, "../datasets"),
                "assets": os.path.join(root, "assets"),
            },
            f,
        )


def _import_libero_benchmark() -> None:
    """Import ``libero.libero.benchmark`` without PyTorch.

    The module imports ``torch`` at the top only to ``torch.load`` the initial states; FreeAct
    reads them with :func:`freeact.utils.checkpoint_io.load_torch_file` instead.  If PyTorch is
    not installed, a stand-in exposing just ``load`` is provided for the duration of the import.
    """
    if "libero.libero.benchmark" in sys.modules or importlib.util.find_spec("torch") is not None:
        import libero.libero.benchmark
        return
    from freeact.utils.checkpoint_io import load_torch_file

    stub = types.ModuleType("torch")
    stub.load = lambda path, *args, **kwargs: load_torch_file(path)
    sys.modules["torch"] = stub
    try:
        import libero.libero.benchmark  # noqa: F401
    finally:
        del sys.modules["torch"]


_ensure_libero_config()
_import_libero_benchmark()

from .suites import SUITE_DEFS, prepare_suite_env, register_suites  # noqa: E402

register_suites()

__all__ = ["SUITE_DEFS", "prepare_suite_env", "register_suites"]
