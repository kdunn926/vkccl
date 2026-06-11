# SPDX-License-Identifier: Apache-2.0
"""Multi-process smoke test for the Python bindings.

Runs standalone (no pytest needed):  python3 python/tests/test_bindings.py
Uses numpy when available, falling back to array/bytearray buffers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vccl  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None

NRANKS = 3
COUNT = 4099  # deliberately not divisible by NRANKS


def child(rank: int, uid_hex: str, q: mp.Queue) -> None:
    try:
        uid = vccl.UniqueId.from_hex(uid_hex)
        with vccl.Communicator(NRANKS, uid, rank) as comm:
            if np is not None:
                _numpy_tests(comm, rank)
            _bytearray_tests(comm, rank)
        q.put((rank, None))
    except Exception as exc:  # noqa: BLE001
        q.put((rank, f"{type(exc).__name__}: {exc}"))


def _numpy_tests(comm: vccl.Communicator, rank: int) -> None:
    # all_reduce, dtype inferred from the array
    send = np.arange(COUNT, dtype=np.float32) + rank
    recv = np.zeros_like(send)
    comm.all_reduce(send, recv)
    want = np.arange(COUNT, dtype=np.float32) * NRANKS + sum(range(NRANKS))
    assert np.array_equal(recv, want), "all_reduce mismatch"

    # in-place all_reduce
    buf = np.arange(COUNT, dtype=np.float32) + rank
    comm.all_reduce(buf)
    assert np.array_equal(buf, want), "in-place all_reduce mismatch"

    # all_gather
    send = np.full(17, float(rank), dtype=np.float32)
    recv = np.zeros(17 * NRANKS, dtype=np.float32)
    comm.all_gather(send, recv)
    for r in range(NRANKS):
        assert (recv[r * 17:(r + 1) * 17] == r).all(), "all_gather mismatch"

    # reduce_scatter
    send = np.tile(np.arange(NRANKS * 5, dtype=np.float32), 1) + rank
    recv = np.zeros(5, dtype=np.float32)
    comm.reduce_scatter(send, recv)
    base = np.arange(rank * 5, (rank + 1) * 5, dtype=np.float32)
    want = base * NRANKS + sum(range(NRANKS))
    assert np.array_equal(recv, want), "reduce_scatter mismatch"

    # broadcast + registration
    buf = (np.arange(33, dtype=np.float32)
           if rank == 1 else np.zeros(33, dtype=np.float32))
    handle = comm.register(buf)
    comm.broadcast(buf, root=1)
    assert np.array_equal(buf, np.arange(33, dtype=np.float32)), "bcast"
    comm.deregister(handle)

    # avg
    send = np.full(7, float(rank + 1), dtype=np.float32)
    comm.all_reduce(send, op=vccl.RedOp.AVG)
    assert np.allclose(send, (NRANKS + 1) / 2), "avg mismatch"


def _bytearray_tests(comm: vccl.Communicator, rank: int) -> None:
    import struct

    n = 16
    send = bytearray(struct.pack(f"{n}i", *([rank] * n)))
    recv = bytearray(n * NRANKS * 4)
    comm.all_gather(send, recv, sendcount=n, dtype=vccl.DataType.INT32)
    vals = struct.unpack(f"{n * NRANKS}i", bytes(recv))
    for r in range(NRANKS):
        assert all(v == r for v in vals[r * n:(r + 1) * n]), "bytearray AG"


def main() -> int:
    os.environ.setdefault("VCCL_SOCKET_ADDR", "127.0.0.1")
    if not os.environ.get("VCCL_LIBRARY"):
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("libvccl.dylib", "libvccl.so"):
            cand = os.path.join(here, "..", "..", "build", name)
            if os.path.exists(cand):
                os.environ["VCCL_LIBRARY"] = os.path.abspath(cand)
                break

    uid = vccl.get_unique_id()
    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=child, args=(r, uid.hex(), q))
             for r in range(NRANKS)]
    for p in procs:
        p.start()
    failures = 0
    for _ in procs:
        rank, err = q.get(timeout=120)
        if err is not None:
            print(f"rank {rank}: {err}", file=sys.stderr)
            failures += 1
    for p in procs:
        p.join(timeout=30)
    print("python bindings:", "FAIL" if failures else
          ("OK" if np is not None else "OK (numpy not installed; subset)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
