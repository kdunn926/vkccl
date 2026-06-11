# vccl — Vulkan Collective Communications Library

NCCL/RCCL-style collectives for **multi-node, single-GPU-per-node** clusters on
open-source Vulkan stacks (Mesa/RADV) — no CUDA, ROCm, NCCL, or RCCL required.
Built to give [vllm-vulkan](https://github.com/ericcurtin/vllm-vulkan)
multi-node tensor-parallel inference over **RoCE v2 / RDMA**, with a TCP
fallback that runs anywhere (including macOS).

See [PLAN.md](PLAN.md) for the architecture and roadmap.

## Status

- ✅ M0/M1: bootstrap, TCP transport, ring all-gather / reduce-scatter /
  all-reduce / broadcast / send-recv, all NCCL datatypes, sum/prod/min/max/avg,
  multi-process correctness tests, `vccl-bench` harness.
- ✅ M2: ibverbs RC (RoCE v2) transport — validated against Soft-RoCE (rxe)
  on Linux; RoCE v2 GID auto-selection (prefers IPv4-mapped), RNR-safe
  two-sided data path, MR region cache.
- ✅ M3 (API): memory registration (`vcclCommRegister`) and dmabuf import
  (`vcclCommRegisterDmaBuf`) — uses `ibv_reg_dmabuf_mr` when the NIC supports
  it, transparently bouncing through a registered staging buffer otherwise
  (validated with udmabuf over rxe). `vccl-vk-demo` runs an all-gather inside
  Vulkan device memory exported with `VK_EXT_external_memory_dma_buf`.
- ✅ M4 (software side): Python bindings (`python/vccl`, pure ctypes) and a
  vLLM `DeviceCommunicator` (`vccl.torch_comm.VcclCommunicator`, validated
  against torch.distributed/gloo references). True multi-node validated:
  two QEMU VMs, one rank each, all-reduce/all-gather over Soft-RoCE across
  an inter-VM network with `--check` passing. `VCCL_TIMEOUT` watchdog on all
  blocking paths.
- ⬜ Pending: BC-250 + RoCE NIC validation, GPU reduction shaders, the
  vllm-vulkan platform patch (see `python/vccl/torch_comm.py` docstring).

## Build

```sh
cmake -B build && cmake --build build -j
ctest --test-dir build          # multi-process correctness tests
```

`libibverbs-dev` (Linux) enables the RDMA transport automatically; otherwise
only TCP is built.

## Quick start

```c
#include <vccl.h>

vcclUniqueId id;
if (rank == 0) vcclGetUniqueId(&id);   // distribute id out-of-band
vcclComm_t comm;
vcclCommInitRank(&comm, nranks, id, rank);
vcclAllReduce(sendbuf, recvbuf, count, vcclFloat32, vcclSum, comm);
vcclCommDestroy(comm);
```

## Benchmarks

Local (forks ranks on one machine):

```sh
./build/vccl-bench --launch 4 --op allreduce -b 4K -e 64M -f 2
```

Multi-node (one process per node; the unique id is exchanged through a shared
file — NFS, or scp it before starting the other ranks):

```sh
node0$ VCCL_ID_FILE=/shared/vccl.id VCCL_RANK=0 VCCL_NRANKS=2 ./vccl-bench --op allgather
node1$ VCCL_ID_FILE=/shared/vccl.id VCCL_RANK=1 VCCL_NRANKS=2 ./vccl-bench --op allgather
```

## Environment variables

| Variable | Meaning |
|---|---|
| `VCCL_DEBUG` | `NONE`/`ERROR`/`WARN`/`INFO`/`TRACE` (default `WARN`) |
| `VCCL_TIMEOUT` | Seconds before a stuck transfer/wait fails (default 600, 0 = no timeout) |
| `VCCL_TRANSPORT` | `verbs` or `tcp` (default: verbs when a RDMA device exists, else tcp) |
| `VCCL_SOCKET_ADDR` | IP to advertise for bootstrap/TCP data (default: first non-loopback interface) |
| `VCCL_SOCKET_IFNAME` | Interface to advertise instead of an explicit IP |
| `VCCL_IB_HCA` | RDMA device name (default: first with an active port) |
| `VCCL_GID_INDEX` | GID table index (default: first RoCE v2 entry) |

## Python

```python
import vccl
uid = vccl.get_unique_id()                 # rank 0; distribute out-of-band
comm = vccl.Communicator(nranks, uid, rank)
comm.all_reduce(numpy_or_torch_cpu_array)  # in place; dtype inferred
```

`python/vccl/torch_comm.py` provides `VcclCommunicator`, a vLLM
`DeviceCommunicatorBase` implementation (bootstraps the unique id over the
gloo process group, runs collectives through vccl). Point
`VCCL_LIBRARY` at `libvccl.so` if it is not on the loader path.

## Testing the RDMA path without hardware (Soft-RoCE)

On any Linux machine or VM:

```sh
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0
./build/test_collectives        # picks up the rxe device automatically
```
