"""Registration of the target-environment suites (with LIBERO's benchmark registry).

All suites reuse ``libero_spatial`` scenes and ship frozen initial states
(``assets/init_files/<suite>``, 50 per task) so that every policy is scored on identical
configurations.

``libero_target`` is the target environment of the paper; its tasks 0-4 are the paper's tasks A-E.
The additional suites test a +5-degree camera azimuth change on Task B
(``libero_camera``) and an object replacement on Task C (``libero_bottle``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
INIT_FILES_DIR = os.path.join(_PACKAGE_DIR, "assets", "init_files")
_registered = False


@dataclass(frozen=True)
class SuiteDef:
    name: str
    source: str
    task_letters: str = ""
    perturbation: dict = field(default_factory=dict)  # kwargs for PerturbationConfig
    scenes: tuple[str, ...] = ()  # libero_spatial scenes, in task order
    max_steps: int = 500  # episode horizon


SUITE_DEFS: dict[str, SuiteDef] = {
    "libero_target": SuiteDef(
        name="libero_target",
        source="libero_spatial",
        task_letters="ABCDE",
        perturbation=dict(
            lighting="bright",
            texture="checkerboard",
            robot="white_arm",
            object_swap={"akita_black_bowl": "milk"},
            language_swap={"black bowl": "milk carton"},
        ),
        scenes=(  # tasks A-E
            "between_the_plate_and_the_ramekin",
            "from_table_center",
            "on_the_cookie_box",
            "next_to_the_cookie_box",
            "on_the_stove",
        ),
    ),
    "libero_camera": SuiteDef(
        name="libero_camera",
        source="libero_spatial",
        task_letters="B",
        perturbation=dict(
            lighting="bright",
            texture="checkerboard",
            robot="white_arm",
            camera_azimuth=5.0,
            object_swap={"akita_black_bowl": "milk"},
            language_swap={"black bowl": "milk carton"},
        ),
        scenes=("from_table_center",),
    ),
    "libero_bottle": SuiteDef(
        name="libero_bottle",
        source="libero_spatial",
        task_letters="C",
        perturbation=dict(
            lighting="bright",
            texture="checkerboard",
            robot="white_arm",
            object_swap={"akita_black_bowl": "ketchup"},
            language_swap={"black bowl": "ketchup bottle"},
        ),
        scenes=("on_the_cookie_box",),
    ),
}


def _apply_swaps(text: str, swaps: dict[str, str]) -> str:
    for old, new in swaps.items():
        text = text.replace(old, new)
    return text


def _scene(task_name: str) -> str:
    """``pick_up_the_black_bowl_<scene>_and_place_it_on_the_plate`` -> ``<scene>``."""
    return task_name.removeprefix("pick_up_the_black_bowl_").removesuffix("_and_place_it_on_the_plate")


def _make_benchmark_cls(suite_name: str, base):
    class _Suite(base):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = suite_name
            self._make_benchmark()

        def _make_benchmark(self):
            from libero.libero.benchmark import task_maps

            # task order is the order of ``SuiteDef.scenes``
            self.tasks = list(task_maps[self.name].values())
            self.n_tasks = len(self.tasks)

    _Suite.__name__ = _Suite.__qualname__ = suite_name.upper()
    return _Suite


def _link_init_states(sdef: SuiteDef, tasks: dict) -> None:
    """Expose ``assets/init_files/<suite>/*`` inside LIBERO's init-state directory.

    Safe under concurrent processes: each link is created under a unique name and atomically
    renamed into place, so readers never see a missing file.
    """
    from libero.libero import get_libero_path

    dst_dir = os.path.join(get_libero_path("init_states"), sdef.source)
    for task in tasks.values():
        src = os.path.join(INIT_FILES_DIR, sdef.name, task.init_states_file)
        dst = os.path.join(dst_dir, task.init_states_file)
        if not os.path.exists(src):
            continue
        try:
            if os.readlink(dst) == src:
                continue
        except OSError:  # missing, or a regular file
            pass
        tmp = f"{dst}.{os.getpid()}.tmp"
        os.symlink(src, tmp)
        os.replace(tmp, dst)


def register_suites() -> None:
    """Register every suite in :data:`SUITE_DEFS` with LIBERO (idempotent)."""
    global _registered
    if _registered:
        return

    from libero.libero.benchmark import Benchmark, Task, register_benchmark, task_maps

    from .libero_env import TASK_SUITE_MAX_STEPS

    for sdef in SUITE_DEFS.values():
        by_scene = {_scene(t.name): t for t in task_maps[sdef.source].values()}
        task_maps[sdef.name] = {
            (n := f"{sdef.name}_task_{letter}"): Task(
                name=n,
                language=_apply_swaps(t.language, sdef.perturbation["language_swap"]),
                problem="Libero",
                problem_folder=sdef.source,
                bddl_file=t.bddl_file,
                init_states_file=f"{n}.pruned_init",
            )
            for letter, sc in zip(sdef.task_letters, sdef.scenes, strict=True)
            for t in (by_scene[sc],)
        }
        register_benchmark(_make_benchmark_cls(sdef.name, Benchmark))
        _link_init_states(sdef, task_maps[sdef.name])
        TASK_SUITE_MAX_STEPS[sdef.name] = sdef.max_steps
        logger.info("Registered '%s' (%d tasks)", sdef.name, len(task_maps[sdef.name]))

    _registered = True


def prepare_suite_env(task_group: str) -> None:
    """Apply the suite's perturbations (no-op for standard LIBERO suites)."""
    sdef = SUITE_DEFS.get(task_group)
    if sdef is None:
        return
    from .config import PerturbationConfig
    from .patches import apply_perturbations

    apply_perturbations(PerturbationConfig(**sdef.perturbation))
