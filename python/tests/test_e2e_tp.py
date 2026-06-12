# SPDX-License-Identifier: Apache-2.0
"""End-to-end tensor-parallel inference test through VcclCommunicator.

Simulates exactly what vLLM's TP layers do on the vllm-vulkan platform
(tensors in host memory, collectives via the device communicator):

  - Megatron MLP block: column-parallel W1 + gelu, row-parallel W2,
    partial results summed with all_reduce.
  - Column-parallel LM head: per-rank logits shard, assembled with
    all_gather along the last dim.
  - Pipeline-parallel boundary: hidden states passed rank->rank with
    send/recv.

Every sharded result is compared against the unsharded single-process
reference. Also verifies the vllm-vulkan integration patch: the patched
VulkanPlatform must resolve its device communicator to VcclCommunicator.

Standalone:  python3 python/tests/test_e2e_tp.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NRANKS = 2
HIDDEN = 64
FFN = 128  # per full model; each rank holds FFN // NRANKS
VOCAB = 96
BATCH = 5


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def child(rank: int, port: int, q: mp.Queue) -> None:
    try:
        import torch
        import torch.distributed as dist

        torch.manual_seed(1234)  # identical full weights on every rank
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=NRANKS,
        )
        from vccl.torch_comm import VcclCommunicator

        comm = VcclCommunicator(dist.group.WORLD)

        dtype = torch.float32
        x = torch.randn(BATCH, HIDDEN, dtype=dtype)
        w1 = torch.randn(HIDDEN, FFN, dtype=dtype) * 0.1
        w2 = torch.randn(FFN, HIDDEN, dtype=dtype) * 0.1
        lm_head = torch.randn(HIDDEN, VOCAB, dtype=dtype) * 0.1

        # ── reference (unsharded) ────────────────────────────────────────
        ref_h = torch.nn.functional.gelu(x @ w1) @ w2 + x
        ref_logits = ref_h @ lm_head

        # ── tensor-parallel forward (what vLLM layers do) ────────────────
        # Column-parallel W1: rank r holds columns [r*FFN/n, (r+1)*FFN/n).
        shard = FFN // NRANKS
        w1_shard = w1[:, rank * shard:(rank + 1) * shard]
        # Row-parallel W2: rank r holds rows [r*FFN/n, (r+1)*FFN/n).
        w2_shard = w2[rank * shard:(rank + 1) * shard, :]

        h_partial = torch.nn.functional.gelu(x @ w1_shard) @ w2_shard
        h = comm.all_reduce(h_partial) + x  # row-parallel sum + residual
        assert torch.allclose(h, ref_h, atol=1e-5), "TP MLP mismatch"

        # Column-parallel LM head + all_gather of logit shards.
        vshard = VOCAB // NRANKS
        head_shard = lm_head[:, rank * vshard:(rank + 1) * vshard]
        logits = comm.all_gather(h @ head_shard, dim=-1)
        assert torch.allclose(logits, ref_logits, atol=1e-5), \
            "TP logits mismatch"

        # ── pipeline-parallel boundary: hidden states rank -> rank ──────
        if rank == 0:
            comm.send(h, dst=1)
        else:
            h_recv = comm.recv(h.size(), h.dtype, src=0)
            assert torch.allclose(h_recv, ref_h, atol=1e-5), "PP mismatch"

        # ── integration patch resolution ─────────────────────────────────
        # The patched vllm-vulkan platform must point at VcclCommunicator.
        vv = os.environ.get("VLLM_VULKAN_DIR")
        if vv and os.path.isdir(vv):
            sys.path.insert(0, vv)
            from vllm_vulkan.platform import VulkanPlatform

            cls_path = VulkanPlatform.get_device_communicator_cls()
            assert cls_path == "vccl.torch_comm.VcclCommunicator", cls_path
            module, name = cls_path.rsplit(".", 1)
            import importlib

            resolved = getattr(importlib.import_module(module), name)
            assert resolved is VcclCommunicator
            assert VulkanPlatform.dist_backend == "gloo"

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
    ctx = mp.get_context("spawn")
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
    checked_patch = bool(os.environ.get("VLLM_VULKAN_DIR"))
    print("e2e TP inference:",
          "FAIL" if failures else
          f"OK (platform patch {'verified' if checked_patch else 'not checked; set VLLM_VULKAN_DIR'})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
