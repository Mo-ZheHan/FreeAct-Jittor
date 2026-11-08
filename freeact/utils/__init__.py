"""Small helpers shared across the package.

Jittor is imported lazily: the I/O helpers of this package (``checkpoint_io``) are also used by the
GPU-less environment workers of the evaluation, where loading Jittor's CUDA runtime fails.
"""


def freeze(module):
    """Mark every parameter of ``module`` (a ``jittor.nn.Module``) as frozen (no gradient).

    Unlike PyTorch's ``requires_grad_(False)``, Jittor's ``Module.requires_grad_(False)`` runs the
    whole forward of the module under ``jt.no_grad()``; freezing the parameters one by one keeps
    the module's forward unchanged.
    """
    for p in module.parameters():
        p.stop_grad()
    return module
