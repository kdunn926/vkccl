# SPDX-License-Identifier: Apache-2.0
"""VcclCommunicator test against torch.distributed (gloo) references.

Standalone:  python3 python/tests/test_torch_comm.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NRANKS = 3


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def child(rank: int, port: int, q: mp.Queue) -> None:
    try:
        import torch
        import torch.distributed as dist

        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=NRANKS,
        )
        from vccl.torch_comm import VcclCommunicator

        comm = VcclCommunicator(dist.group.WORLD)

        # all_reduce vs gloo
        torch.manual_seed(42 + rank)
        x = torch.randn(33, 65, dtype=torch.float32)
        ref = x.clone()
        dist.all_reduce(ref)
        out = comm.all_reduce(x.clone())
        assert torch.allclose(out, ref, atol=1e-5), "all_reduce vs gloo"

        # M8/H1(b): repeat calls on the same shape reuse one registered
        # buffer instead of allocating/registering per call.
        ptr_before = comm._buf("ar", x.shape, x.dtype).data_ptr()
        comm.all_reduce(x.clone())
        comm.all_reduce(x.clone())
        assert len(comm._buffers) == 1, "expected a single cached ar buffer"
        assert comm._buf("ar", x.shape, x.dtype).data_ptr() == ptr_before, \
            "expected the cached buffer's address to be stable across calls"

        # all_reduce bf16
        xb = torch.randn(257, dtype=torch.bfloat16)
        refb = xb.clone().float()
        dist.all_reduce(refb)
        outb = comm.all_reduce(xb.clone())
        assert torch.allclose(outb.float(), refb, rtol=0.05, atol=0.1), \
            "bf16 all_reduce"

        # all_gather along both dims vs gloo
        y = torch.randn(6, 5, dtype=torch.float32) + rank
        for dim in (0, 1, -1):
            gathered = comm.all_gather(y, dim=dim)
            refs = [torch.empty_like(y) for _ in range(NRANKS)]
            dist.all_gather(refs, y)
            want = torch.cat(refs, dim=dim if dim >= 0 else y.dim() - 1)
            assert torch.equal(gathered, want), f"all_gather dim={dim}"

        # reduce_scatter vs an independent gloo path (sum then slice with the
        # same movedim layout the implementation promises)
        z = torch.randn(NRANKS * 4, 7, dtype=torch.float32) * (rank + 1)
        out = comm.reduce_scatter(z, dim=0)
        zsum = z.clone()
        dist.all_reduce(zsum)
        chunk = zsum.shape[0] // NRANKS
        want = zsum[rank * chunk:(rank + 1) * chunk]
        assert torch.allclose(out, want, atol=1e-5), "reduce_scatter dim=0"

        # broadcast
        b = (torch.arange(11, dtype=torch.float32)
             if rank == 2 else torch.zeros(11))
        comm.broadcast(b, src=2)
        assert torch.equal(b, torch.arange(11, dtype=torch.float32)), "bcast"

        # send/recv ring: pass rank stamp to the right
        stamp = torch.full((4,), float(rank), dtype=torch.float32)
        if rank % 2 == 0:
            comm.send(stamp)  # to next
            got = comm.recv(stamp.size(), stamp.dtype)  # from prev
        else:
            got = comm.recv(stamp.size(), stamp.dtype)
            comm.send(stamp)
        prev = (rank - 1) % NRANKS
        assert torch.equal(got, torch.full((4,), float(prev))), "send/recv"

        comm.destroy()
        dist.destroy_process_group()
        q.put((rank, None))
    except Exception as exc:  # noqa: BLE001
        import traceback

        q.put((rank, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def main() -> int:
    os.environ.setdefault("VCCL_SOCKET_ADDR", "127.0.0.1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0"
                          if sys.platform == "darwin" else "lo")
    if not os.environ.get("VCCL_LIBRARY"):
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("libvccl.dylib", "libvccl.so"):
            cand = os.path.join(here, "..", "..", "build", name)
            if os.path.exists(cand):
                os.environ["VCCL_LIBRARY"] = os.path.abspath(cand)
                break

    port = _free_port()
    ctx = mp.get_context("spawn")  # torch + fork is unreliable
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=child, args=(r, port, q))
             for r in range(NRANKS)]
    for p in procs:
        p.start()
    failures = 0
    for _ in procs:
        rank, err = q.get(timeout=300)
        if err is not None:
            print(f"rank {rank}: {err}", file=sys.stderr)
            failures += 1
    for p in procs:
        p.join(timeout=60)
    print("torch communicator:", "FAIL" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
