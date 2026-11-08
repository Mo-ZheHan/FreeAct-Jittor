"""Background data loading with worker processes (no PyTorch DataLoader).

Workers are forked once and receive lists of sample indices; each returns the collated batch.
Batches are produced in order with ``prefetch`` batches in flight per worker.  The resampling
stream of the dataset is reseeded per batch from ``(seed, batch index)``, so a batch does not
depend on the number of workers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import traceback
from typing import Callable, Iterable, Iterator, List

import numpy as np


def batch_seed(seed: int, k: int) -> int:
    # [seed, 0, k]: the resampling stream of batch k
    return int(np.random.SeedSequence([seed, 0, k]).generate_state(1)[0])


def _load(dataset, collate, seed, idxs):
    if hasattr(dataset, "reseed"):
        dataset.reseed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return collate([dataset[i] for i in idxs])


def _worker_loop(dataset, collate, in_q, out_q):
    try:
        _serve(dataset, collate, in_q, out_q)
    finally:
        os._exit(0)  # skip the parent's atexit handlers (e.g. Jittor / CUDA teardown)


def _serve(dataset, collate, in_q, out_q):
    while True:
        job = in_q.get()
        if job is None:
            return
        k, idxs, seed = job
        try:
            out_q.put((k, _load(dataset, collate, seed, idxs), None))
        except Exception:  # surface the worker's traceback in the main process
            out_q.put((k, None, traceback.format_exc()))


class DataLoader:
    def __init__(self, dataset, batch_size: int, sampler: Iterable[int], collate_fn: Callable,
                 num_workers: int = 8, prefetch: int = 2, seed: int = 0):
        self.dataset, self.batch_size, self.sampler = dataset, batch_size, sampler
        self.collate_fn, self.num_workers, self.prefetch = collate_fn, num_workers, prefetch
        self.seed = seed
        self._workers: List[mp.Process] = []
        if num_workers > 0:
            ctx = mp.get_context("fork")
            self._in = [ctx.Queue() for _ in range(num_workers)]
            self._out = ctx.Queue(maxsize=num_workers * prefetch * 2)
            for w in range(num_workers):
                p = ctx.Process(target=_worker_loop, args=(dataset, collate_fn, self._in[w], self._out), daemon=True)
                p.start()
                self._workers.append(p)

    def _batches(self) -> List[List[int]]:
        idx = list(iter(self.sampler))
        return [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]

    def __len__(self) -> int:
        return -(-len(self.sampler) // self.batch_size)

    def __iter__(self) -> Iterator:
        batches = self._batches()
        n = len(batches)
        if self.num_workers == 0:
            for k in range(n):
                yield _load(self.dataset, self.collate_fn, batch_seed(self.seed, k), batches[k])
            return
        job = lambda k: (k, batches[k], batch_seed(self.seed, k))
        sent, nxt, buf = 0, 0, {}
        while sent < min(n, self.num_workers * self.prefetch):
            self._in[sent % self.num_workers].put(job(sent))
            sent += 1
        while nxt < n:
            while nxt not in buf:
                k, batch, err = self._out.get()
                if err is not None:
                    raise RuntimeError(f"Data loader worker failed:\n{err}")
                buf[k] = batch
            if sent < n:
                self._in[sent % self.num_workers].put(job(sent))
                sent += 1
            yield buf.pop(nxt)
            nxt += 1

    def close(self):
        for q in getattr(self, "_in", []):
            q.put(None)
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._workers = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
