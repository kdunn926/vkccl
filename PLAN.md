# vccl — Vulkan Collective Communications Library

A NCCL/RCCL-style collective communications library for **multi-node, single-device-per-node**
clusters running open-source Vulkan stacks (Mesa/RADV), with **RoCE v2 / RDMA** as the
high-performance transport. The end goal is to give
[vllm-vulkan](https://github.com/ericcurtin/vllm-vulkan) working tensor-parallel /
distributed inference across nodes without CUDA, ROCm, NCCL, or RCCL.

## Scope decisions (amended)

- **Multi-node first, single device per node.** The original plan's Phase 2
  (multi-GPU single node via `VK_KHR_device_group` / peer copies) is **dropped** — device
  groups are poorly supported in practice, and the target cluster is one GPU per node.
- **Primitives that matter, not the full NCCL API.** vLLM tensor parallelism needs
  **all-reduce** (= ring **reduce-scatter** + ring **all-gather**), all-gather (for
  vocab/logits gathers), broadcast, and eventually send/recv (pipeline parallelism).
  We implement exactly those. No `ncclCommSplit`, no graph/group fusion, no tree
  algorithms initially.
- **NCCL-*like*, not NCCL-ABI-compatible.** We export a clean `vccl.h` C API modeled on
  `nccl.h` (`vcclCommInitRank`, `vcclAllGather`, …). A binary `libnccl.so` shim is out of
  scope: vllm-vulkan keeps PyTorch tensors on CPU and forces
  `distributed_executor_backend="uni"`, so nothing consumes the NCCL ABI there anyway. The
  integration point is a vLLM `DeviceCommunicator` calling vccl, not `pynccl`.
- **C++17 core, C ABI.** Loadable from Python (ctypes/cffi), Rust (bindgen for the
  vllm-vulkan core), and potentially a torch.distributed backend later.
- **Host-staged RDMA first, GPU-direct (dmabuf) second.** Crucially, the primary target
  (ASRock BC-250, an AMD APU) has **unified memory**: Vulkan device-local heaps are
  host-visible, so "host-staged" is effectively zero-copy there — we `ibv_reg_mr` the
  mapped pages directly. dmabuf MRs (`VK_EXT_external_memory_dma_buf` +
  `ibv_reg_dmabuf_mr`, kernel ≥ 5.12) are a later phase that matters for discrete GPUs
  with non-host-visible VRAM.
- **TCP transport kept forever**, not just as a stopgap: it enables development/CI on
  machines without RDMA NICs (including macOS, where vllm-vulkan itself runs via
  KosmicKrisp) and is the automatic fallback when ibverbs is unavailable.

## Target environment

| | |
|---|---|
| Nodes | ASRock BC-250 class (AMD Cyan Skillfish APU, GFX1013, unified memory) |
| Driver | Mesa RADV (Vulkan 1.3+), ACO compiler |
| Network | RoCE v2 capable NICs (ConnectX class), libibverbs/rdma-core |
| Dev/CI | macOS (TCP transport only) + Linux containers; Soft-RoCE (rxe) for ibverbs testing without hardware |

## Architecture

```
 vcclAllReduce(send, recv, count, dtype, op, comm)
        │
 ┌──────▼───────────────────────────────────────────┐
 │ collectives: ring reduce-scatter / ring all-gather│   src/core/
 │ chunking & pipelining, CPU reduction kernels      │
 ├───────────────────────────────────────────────────┤
 │ transport interface: setup(peer), send, recv,     │   src/transport/
 │ regMr/deregMr                                     │
 │   ├── tcp    : nonblocking sockets, duplex poll   │
 │   └── verbs  : ibverbs RC QPs, RoCE v2 GIDs,      │
 │                pinned MRs, CQ polling             │
 ├───────────────────────────────────────────────────┤
 │ bootstrap: vcclUniqueId = root ip:port, TCP       │   src/core/bootstrap.cc
 │ rendezvous, endpoint/QP info exchange             │
 └───────────────────────────────────────────────────┘
```

- **Bootstrap**: `vcclGetUniqueId` binds a listening socket in the caller and embeds its
  address in the 128-byte opaque id (same usage contract as NCCL: the id is distributed
  out-of-band; rank 0 must be the process that created it). `vcclCommInitRank` performs an
  all-to-root gather of per-rank endpoint info, root broadcasts the table, then each rank
  establishes ring links (connect→next, accept←prev).
- **Collectives** (rank `r`, `n` ranks, data in `n` chunks):
  - *all-gather*: step `s` sends chunk `(r−s) mod n`, receives `(r−s−1) mod n`; `n−1` steps.
  - *reduce-scatter*: step `s` sends partial chunk `(r−s−1) mod n`, receives and
    accumulates chunk `(r−s−2) mod n`; rank `r` ends owning reduced chunk `r`.
  - *all-reduce* = reduce-scatter + all-gather (bandwidth-optimal, 2(n−1)/n × size on the wire).
  - Send and receive in each step proceed concurrently (different sockets/QPs; duplex
    progress loop) so per-step time is max(send, recv), not the sum.
- **Reduction kernels**: CPU first (f32/f64/i32 native; f16/bf16 via float conversion;
  sum/prod/min/max/avg). A Vulkan compute-shader reduction is only needed once buffers
  live in non-host-visible VRAM — on the APU target, CPU reduction over unified memory is
  acceptable for v1 and avoids a GPU dependency in the core library.
- **API surface (v1)**: `vcclGetUniqueId`, `vcclCommInitRank`, `vcclCommDestroy`,
  `vcclCommCount`, `vcclCommUserRank`, `vcclAllGather`, `vcclReduceScatter`,
  `vcclAllReduce`, `vcclBroadcast`, `vcclSend`, `vcclRecv`, `vcclGetErrorString`.
  Calls are **host-blocking** in v1 (no stream argument). Async submission integrated with
  Vulkan timeline semaphores is deferred until the Vulkan-buffer path exists.

## Milestones

1. **M0 — Functional core over TCP** ✅ *(runs anywhere, incl. macOS)*
   Scaffold, `vccl.h`, bootstrap, TCP transport, ring all-gather. Multi-process loopback
   tests pass.
2. **M1 — Full v1 primitive set + perf harness** ✅
   Reduce-scatter, all-reduce, broadcast, send/recv; all dtypes/ops; `vccl-bench`
   (nccl-tests-style size sweep, algbw/busbw). Correctness vs CPU reference for
   non-divisible counts, n = 1..5.
3. **M2 — ibverbs RC transport (RoCE v2)** ✅ *(rxe-validated; hardware pending)*
   RC QP per ring neighbor, QP exchange over bootstrap, registered host MRs, completion
   polling. Validated on Soft-RoCE (rxe) in a QEMU arm64 VM; RoCE v2 GID auto-selection
   prefers IPv4-mapped GIDs (link-local GIDs fail neighbor resolution). Runtime transport
   selection (`VCCL_TRANSPORT=verbs|tcp`, auto-detect with fallback). Still to do:
   validate on BC-250 + real NICs, chunked pipelining.
4. **M3 — Vulkan memory path** ✅ *(API level; GPU validation pending)*
   `vcclCommRegister`/`vcclCommDeregister` (NCCL-style buffer pinning) and
   `vcclCommRegisterDmaBuf` (import a Vulkan allocation exported via
   `VK_EXT_external_memory_dma_buf`). The verbs transport registers the dmabuf itself
   (`ibv_reg_dmabuf_mr`) when supported; otherwise it transparently bounces through a
   registered staging buffer — necessary because dmabuf CPU mappings cannot be pinned
   with `ibv_reg_mr` (get_user_pages fails on those VMAs; confirmed with udmabuf+rxe).
   `vccl-vk-demo` exercises the full Vulkan export → register → all-gather flow.
   Still to do: run on RADV/BC-250, GPU reduction shader (subgroup add).
5. **M4 — GPU-direct + vllm-vulkan integration**
   Validate dmabuf MRs against dGPU VRAM (no CPU mapping) — needs NIC-side collectives
   for reduce ops or GPU reduction. Python bindings; a `DeviceCommunicator`/worker patch
   for vllm-vulkan enabling `distributed_executor_backend != "uni"` and multi-node TP as
   the end-to-end demo.
6. **M5 — Hardening/perf** *(ongoing)*
   Pipelined multi-chunk steps, inline sends for small messages, latency-oriented
   algorithms (recursive doubling) if profiles demand, error propagation/abort.

## Key risks & mitigations

- **No RDMA hardware on the dev machine (macOS).** All logic is testable over TCP locally;
  the verbs transport is developed against Soft-RoCE in a Linux container/VM, so hardware
  only validates performance, not correctness.
- **BC-250 RADV maturity (GFX1013).** Mesa support exists but is niche; mitigated because
  M0–M2 don't touch Vulkan at all — the comm library is useful even with vllm-vulkan's
  current CPU-resident tensors.
- **dmabuf MR support** varies by NIC/kernel; mitigated by capability probing with
  automatic fallback to registered host memory (free on APUs).
- **Performance vs NCCL** is not the bar — the bar is beating vLLM's gloo/TCP path on the
  same hardware, which ring-RDMA with zero-copy registration clears comfortably.

## Build & layout

CMake ≥ 3.20, C++17, no required deps beyond a POSIX socket stack; `libibverbs` optional
(`-DVCCL_ENABLE_VERBS=ON`, default ON when headers found); Vulkan optional from M3.

```
include/vccl.h          public C API
src/core/               comm, bootstrap, ring algorithms, CPU reduction
src/transport/          transport.h, tcp.cc, verbs.cc
tests/                  fork-based multi-process correctness tests
bench/                  vccl-bench harness
```
