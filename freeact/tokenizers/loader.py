"""Copying a checkpoint's tensors into a Jittor module, with optional key renaming.

The released baseline LAMs store their weights under names that differ from this port's module
tree (a ``lam.`` wrapper prefix, or LAPA's numeric layer indices); :func:`copy_state` applies the
mapping and reports every key that neither side accounts for.
"""

from typing import Callable, Dict, Iterable, Optional

import jittor as jt
import numpy as np


def copy_state(module: jt.nn.Module, state: Dict[str, np.ndarray], *, prefix: str = "",
               rename: Optional[Callable[[str], str]] = None,
               ignore: Iterable[str] = ()) -> Dict[str, list]:
    """Assign ``state`` into ``module``, matched on parameter names.

    ``prefix`` is stripped from the checkpoint's keys first, then ``rename`` maps a stripped key to a
    parameter name (identity when omitted).  ``ignore`` lists stripped checkpoint keys that have no
    counterpart here by design.  Returns ``{"missing": [...], "unexpected": [...], "ignored": [...]}``.
    """
    params = module.state_dict()
    ignore = set(ignore)
    renamed = {rename(k[len(prefix):]) if rename else k[len(prefix):]: k[len(prefix):]
               for k in state if k.startswith(prefix)}  # parameter name -> checkpoint key (without prefix)

    for name, var in params.items():
        if name not in renamed:
            continue
        value = state[prefix + renamed[name]]
        if tuple(value.shape) != tuple(var.shape):
            raise ValueError(f"{prefix + renamed[name]}: checkpoint {value.shape} vs model {var.shape}")
        var.assign(jt.array(value.astype(np.float32) if value.dtype != np.float32 else value))
    return {
        "missing": sorted(set(params) - set(renamed)),
        "unexpected": sorted(n for n, k in renamed.items() if n not in params and k not in ignore),
        "ignored": sorted(k for n, k in renamed.items() if n not in params and k in ignore),
    }
