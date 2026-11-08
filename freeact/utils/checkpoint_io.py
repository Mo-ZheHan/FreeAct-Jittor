"""Torch-free reading of the checkpoint formats used by FreeAct.

The released checkpoints (``Mo-ZheHan/FreeAct``) are shared with the PyTorch implementation:

* ``*.safetensors`` (LoRA adapters, Qwen3-VL base weights) are parsed directly;
* ``*.pt`` / ``*.ckpt`` / ``*.pruned_init`` are PyTorch zip archives (a ``data.pkl`` pickle that
  references raw tensor storages).  They are unpickled here with stand-ins for the handful of
  ``torch`` globals they reference, so that PyTorch is not needed.

Everything is returned as NumPy arrays; bfloat16 tensors are widened to float32.
"""

from __future__ import annotations

import collections
import io
import json
import pickle
import struct
import zipfile
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------ bfloat16


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """``uint16`` bfloat16 bit patterns -> float32."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


# ------------------------------------------------------------------ safetensors

_ST_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def safetensors_header(path: str) -> Tuple[Dict[str, Any], int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def load_safetensors(
    path: str,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load (a subset of) a safetensors file as NumPy arrays (memory-mapped, then copied)."""
    header, offset = safetensors_header(path)
    header.pop("__metadata__", None)
    wanted = set(keys) if keys is not None else None
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if wanted is not None and name not in wanted:
            continue
        start, end = info["data_offsets"]
        raw = mm[offset + start : offset + end]
        arr = np.frombuffer(raw, dtype=_ST_DTYPES[info["dtype"]]).reshape(info["shape"]).copy()
        if info["dtype"] == "BF16":
            arr = bf16_to_f32(arr)
        out[name] = arr
    return out


# ------------------------------------------------------------------ PyTorch zip archives

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": np.uint16,  # widened below
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name
        self.dtype = np.dtype(_STORAGE_DTYPES[name])


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    arr, is_bf16 = storage
    itemsize = arr.dtype.itemsize
    if len(size) == 0:
        out = arr[storage_offset : storage_offset + 1].reshape(())
    else:
        out = np.lib.stride_tricks.as_strided(
            arr[storage_offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
        )
    out = np.array(out)  # own the memory (contiguous)
    if is_bf16:
        out = bf16_to_f32(out)
    return out


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


class _TorchUnpickler(pickle.Unpickler):
    def __init__(self, f, zf: zipfile.ZipFile, prefix: str):
        super().__init__(f)
        self._zf, self._prefix = zf, prefix
        self._cache: Dict[str, Tuple[np.ndarray, bool]] = {}

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module.startswith("numpy") or module in ("builtins", "_codecs"):
            return super().find_class(module, name)
        # Unknown objects (e.g. Lightning hyper-parameter containers) become plain dicts
        return lambda *a, **k: dict(*a, **k) if a and isinstance(a[0], dict) else {}

    def persistent_load(self, pid):
        typename, storage_type, key, location, numel = pid
        assert typename == "storage", pid
        if key not in self._cache:
            raw = self._zf.read(f"{self._prefix}data/{key}")
            arr = np.frombuffer(raw, dtype=storage_type.dtype)
            self._cache[key] = (arr, storage_type.name == "BFloat16Storage")
        return self._cache[key]


def load_torch_file(path: str) -> Any:
    """Load a PyTorch ``torch.save`` zip archive into nested Python / NumPy objects."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("data.pkl")]
        return _TorchUnpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()


class _SkeletonUnpickler(_TorchUnpickler):
    """Unpickles the structure of an archive; every tensor becomes its shape tuple and no storage is read."""

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, *a, **k: tuple(size)
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return None


def load_torch_skeleton(path: str) -> Any:
    """Like :func:`load_torch_file` but with each tensor replaced by its shape tuple: cheap key / shape listing."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        return _SkeletonUnpickler(io.BytesIO(zf.read(pkl)), zf, pkl[: -len("data.pkl")]).load()
