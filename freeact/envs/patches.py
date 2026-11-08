"""Runtime patches that turn ``libero_spatial`` into the target environments.

All patches are runtime monkey-patches of LIBERO and of :class:`freeact.envs.libero_env.LiberoEnv`.  ``apply_perturbations(config)``
must run once per process before any environment is created.
"""

import logging
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .config import PerturbationConfig
from .libero_env import LiberoEnv

logger = logging.getLogger(__name__)

_ASSETS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_active_config: PerturbationConfig | None = None
_patches_applied: set[str] = set()


def get_active_config() -> PerturbationConfig | None:
    return _active_config


# "specular_scale" multiplies each light's original specular so scene-specific lights keep
# their relative intensity.
_LIGHTING_PRESETS = {
    "bright": {"diffuse": "4.0 4.0 4.0", "specular_scale": 16 / 3, "castshadow": "false"},
}

# Table texture element name in the scene XML -> replacement file (relative to _ASSETS_ROOT).
_TEXTURE_PRESETS = {
    "checkerboard": {
        "tex-table": "textures/checkerboard.png",
        "tex-living_room_table": "textures/checkerboard.png",
        "tex-desk": "textures/checkerboard.png",
    },
}

_scene_xml_cache: dict[tuple, str] = {}
_swapped_bddl_cache: dict[tuple, str] = {}


def _libero_assets_dir() -> str:
    """LIBERO's asset root (hf-libero downloads it on first use; upstream ships it in-tree)."""
    import libero.libero as libero_pkg

    if hasattr(libero_pkg, "get_assets_path"):
        return libero_pkg.get_assets_path()
    return os.path.join(os.path.dirname(libero_pkg.__file__), "assets")


def _swap_bddl_objects(bddl_path: str, swap_map: dict[str, str]) -> str:
    """Textually replace object type names in a BDDL file; returns a cached temp file."""
    key = (bddl_path, tuple(sorted(swap_map.items())))
    if key in _swapped_bddl_cache:
        return _swapped_bddl_cache[key]

    with open(bddl_path, "r", encoding="utf-8") as f:
        text = f.read()
    for old_type, new_type in swap_map.items():
        text = text.replace(old_type, new_type)

    fd, tmp_path = tempfile.mkstemp(suffix=".bddl", prefix="freeact_swap_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    _swapped_bddl_cache[key] = tmp_path
    return tmp_path


def _generate_scene_xml(base_xml_path: str, cfg: PerturbationConfig) -> str:
    """Write a copy of the scene XML with the lighting / texture overrides applied."""
    key = (base_xml_path, cfg.lighting, cfg.texture)
    if key in _scene_xml_cache:
        return _scene_xml_cache[key]

    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    xml_dir = os.path.dirname(os.path.abspath(base_xml_path))

    if cfg.lighting:
        preset = _LIGHTING_PRESETS[cfg.lighting]
        spec_scale = preset["specular_scale"]
        for light in root.iter("light"):
            light.set("diffuse", preset["diffuse"])
            light.set("castshadow", preset["castshadow"])
            if orig := light.get("specular"):
                light.set("specular", " ".join(f"{float(v) * spec_scale:.4g}" for v in orig.split()))

    if cfg.texture:
        tex_map = _TEXTURE_PRESETS[cfg.texture]
        for tex in root.iter("texture"):
            if replacement := tex_map.get(tex.get("name", "")):
                tex.set("file", os.path.join(_ASSETS_ROOT, replacement))

    # Resolve relative file references so the XML can live in a temp dir
    for tag in ("texture", "mesh"):
        for elem in root.iter(tag):
            file_attr = elem.get("file")
            if file_attr and not os.path.isabs(file_attr):
                elem.set("file", os.path.abspath(os.path.join(xml_dir, file_attr)))
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="freeact_scene_")
    os.close(fd)
    tree.write(tmp_path, xml_declaration=False)
    _scene_xml_cache[key] = tmp_path
    return tmp_path


def _patch_scene():
    """Intercept the scene XML chosen by the BDDL problem class and apply lighting / texture."""
    if "scene" in _patches_applied:
        return
    from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

    _orig_bddl_init = BDDLBaseDomain.__init__
    assets_dir = _libero_assets_dir()

    def _patched_bddl_init(self, bddl_file_name, *args, **kwargs):
        cfg = get_active_config()
        if cfg and cfg.has_scene_override:
            scene_xml = kwargs.get("scene_xml", "scenes/libero_base_style.xml")
            base_xml_path = os.path.join(os.path.abspath(kwargs.get("custom_asset_dir") or assets_dir), scene_xml)
            if not os.path.isfile(base_xml_path):
                raise FileNotFoundError(f"Scene XML not found: {base_xml_path}")
            kwargs["scene_xml"] = _generate_scene_xml(base_xml_path, cfg)
        _orig_bddl_init(self, bddl_file_name, *args, **kwargs)

    BDDLBaseDomain.__init__ = _patched_bddl_init
    _patches_applied.add("scene")


def _patch_env_create():
    """Swap objects in the BDDL and rewrite the instruction when the env is built."""
    if "env_create" in _patches_applied:
        return
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    _orig_make = LiberoEnv._make_envs_task

    def _patched_make(self, task_suite, task_id=0):
        cfg = get_active_config()
        if cfg is None or cfg.is_empty:
            return _orig_make(self, task_suite, task_id)

        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = cfg.apply_language_swap(task.language)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        if cfg.object_swap:
            bddl = _swap_bddl_objects(bddl, cfg.object_swap)

        env_args = dict(
            bddl_file_name=bddl,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
        )
        if cfg.camera_azimuth:
            env_args["camera_azimuth"] = cfg.camera_azimuth
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    LiberoEnv._make_envs_task = _patched_make
    _patches_applied.add("env_create")


def _camera_pose_offset(pos, quat_wxyz, d_azimuth):
    """Orbit a camera around its look-at point on the table by ``d_azimuth`` degrees."""
    import robosuite.utils.transform_utils as T

    mat = T.quat2mat(np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]))
    forward = -mat[:, 2]

    # Look-at point: intersection of the optical axis with the table plane (fall back to floor)
    target = None
    if abs(forward[2]) > 1e-3:
        t_table = -(pos[2] - 0.85) / forward[2]
        if t_table > 0.1:
            target = pos + t_table * forward
        elif (t_floor := -pos[2] / forward[2]) > 0.1:
            target = pos + t_floor * forward
    if target is None:
        target = pos + forward

    rel = pos - target
    r = np.linalg.norm(rel)
    azimuth = np.arctan2(rel[1], rel[0]) + np.deg2rad(d_azimuth)
    elevation = np.clip(np.arcsin(np.clip(rel[2] / r, -1.0, 1.0)), -1.5, 1.5)
    new_rel = r * np.array(
        [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
    )
    new_pos = target + new_rel

    z_axis = new_rel / np.linalg.norm(new_rel)
    if abs(z_axis[2]) > 0.99:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
        x_axis /= np.linalg.norm(x_axis)
    q = T.mat2quat(np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis)))
    return new_pos, np.array([q[3], q[0], q[1], q[2]])


def _patch_view():
    """Accept a ``camera_azimuth`` kwarg on BDDL environments and rotate the agent-view camera accordingly."""
    if "view" in _patches_applied:
        return
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING, BDDLBaseDomain
    from robosuite.utils.mjcf_utils import string_to_array

    def _wrap_setup_camera(original):
        def _setup_camera(self, mujoco_arena):
            original(self, mujoco_arena)
            if (d_azimuth := getattr(self, "camera_azimuth", None)) is None:
                return
            cam = mujoco_arena.worldbody.find(".//camera[@name='agentview']")
            if cam is None:
                return
            pos = string_to_array(cam.get("pos")) if cam.get("pos") else np.zeros(3)
            quat = string_to_array(cam.get("quat")) if cam.get("quat") else np.array([1.0, 0.0, 0.0, 0.0])
            new_pos, new_quat = _camera_pose_offset(pos, quat, d_azimuth)
            mujoco_arena.set_camera("agentview", pos=new_pos, quat=new_quat)

        return _setup_camera

    _orig_init = BDDLBaseDomain.__init__

    def _patched_init(self, *args, **kwargs):
        self.camera_azimuth = kwargs.pop("camera_azimuth", None)
        _orig_init(self, *args, **kwargs)

    BDDLBaseDomain.__init__ = _patched_init
    base_setup = BDDLBaseDomain._setup_camera
    BDDLBaseDomain._setup_camera = _wrap_setup_camera(base_setup)
    for problem_class in TASK_MAPPING.values():
        if problem_class._setup_camera is not BDDLBaseDomain._setup_camera and problem_class._setup_camera is not base_setup:
            problem_class._setup_camera = _wrap_setup_camera(problem_class._setup_camera)
    _patches_applied.add("view")


def _arm_geom_idx(name: str) -> int | None:
    """Robot arm geom index for names like ``robot0_g{N}_vis``; 1000 for the gripper hand."""
    if name == "gripper0_hand_visual":
        return 1000
    if name.startswith("robot0_g") and name.endswith("_vis"):
        try:
            return int(name[8:-4])
        except ValueError:
            return None
    return None


_HIDDEN_ARM_GEOM_MAX = 13  # robot0_g0-g13 = link0-2
_ROBOT_WHITE = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)


def _paint_robot_white(control_env):
    """Hide the rear arm links and the mount, paint the remaining visible arm white."""
    sim = control_env.env.sim
    model = sim.model
    hide_group = 2

    for i in range(model.ngeom):
        name = model.geom_id2name(i) or ""
        idx = _arm_geom_idx(name)
        if idx is not None and idx <= _HIDDEN_ARM_GEOM_MAX:
            model.geom_group[i] = hide_group
        elif idx is not None:
            model.geom_rgba[i] = _ROBOT_WHITE
        elif "mount0" in name and "_vis" in name:
            model.geom_group[i] = hide_group
        elif name.startswith("gripper0") and "_vis" in name and "finger" not in name:
            model.geom_rgba[i] = _ROBOT_WHITE
    sim._render_context_offscreen.vopt.geomgroup[hide_group] = 0
    sim.forward()


def _patch_reset():
    """Paint the robot white after every reset and re-render the observation."""
    if "reset" in _patches_applied:
        return

    _orig_reset = LiberoEnv.reset

    def _patched_reset(self, seed=None, **kwargs):
        obs, info = _orig_reset(self, seed=seed, **kwargs)
        cfg = get_active_config()
        if cfg and cfg.robot == "white_arm":
            _paint_robot_white(self._env)
            obs = self._format_raw_obs(self._env.env._get_observations(force_update=True))
        return obs, info

    LiberoEnv.reset = _patched_reset
    _patches_applied.add("reset")


def apply_perturbations(config: PerturbationConfig) -> None:
    """Activate ``config`` and install the monkey-patches it needs."""
    global _active_config
    _active_config = config
    if config.is_empty:
        return
    logger.info("Target environment perturbations: %s", config.summary())
    if config.has_scene_override:
        _patch_scene()
    if config.object_swap or config.language_swap or config.camera_azimuth:
        _patch_env_create()
    if config.camera_azimuth:
        _patch_view()
    if config.robot:
        _patch_reset()
