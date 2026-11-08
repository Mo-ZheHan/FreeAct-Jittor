"""GPU worker serving policy inference for many environments."""

import logging

import ray

from .step_task import StepResult, StepTask

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InferenceWorker:
    """Ray Actor for GPU-based policy inference."""

    def __init__(self, vla_path: str, seed: int | None = None):
        import jittor as jt

        jt.flags.use_cuda = 1
        from freeact.vla.inference import load_vla_model

        self.policy = load_vla_model(vla_path=vla_path, seed=seed)
        logger.info("InferenceWorker: window_size=%d", self.policy.action_decoder.window_size)

    def infer_single(self, task: StepTask, max_retries: int = 3) -> StepResult:
        """Sample an action chunk; resample when the sampled tokens do not decode into a chunk of the
        expected shape, and return ``action=None`` after ``max_retries`` failures."""
        for attempt in range(max_retries):
            try:
                action = self.policy.select_action(task.observation)
                return StepResult(env_id=task.env_id, action=action[0])
            except ValueError as e:
                if "could not be broadcast" not in str(e):
                    raise
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: {e}")
                else:
                    logger.error(f"Inference failed for env {task.env_id}")
                    return StepResult(env_id=task.env_id, action=None)

    def __ray_shutdown__(self) -> None:
        """Ray lifecycle hook for graceful actor termination."""
        if hasattr(self, "policy"):
            del self.policy
