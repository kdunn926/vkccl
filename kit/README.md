# vccl e2e kit — bare-metal 2-node verification

Fully static (musl) x86_64 binaries for BC-250-class node bring-up:
**zero runtime dependencies** — drop into any initramfs, squashfs, or
netboot image and run. RDMA providers are linked in statically
(mlx4, mlx5 ConnectX, bnxt_re Broadcom, irdma Intel E810, rxe Soft-RoCE);
no `/etc/libibverbs.d`, no `.so` files, no glibc needed.

| file | purpose |
|---|---|
| `vccl-info` | RDMA device/port/GID survey; marks the GID vccl auto-selects |
| `vccl-bench` | nccl-tests-style sweep with `--check` correctness verification |
| `test_collectives` | full multi-process correctness suite (single node) |
| `test_dmabuf` | dmabuf registration test (udmabuf-based; single node) |
| `vccl-e2e.sh` | orchestrates bring-up checks + verified sweeps across nodes |

## Quick start (2 nodes)

Copy `dist/` to both nodes. With node A reachable at `10.0.0.1`:

```sh
nodeA# ./vccl-e2e.sh 0 2 10.0.0.1
nodeB# ./vccl-e2e.sh 1 2 10.0.0.1
```

That's it — rank coordination uses `VCCL_COMM_ID` (rank 0 listens on
`10.0.0.1:5555+`); no shared filesystem or id files. Each op prints a
correctness check result and a bandwidth table; the script exits non-zero
on any failure.

Manual single runs:

```sh
nodeA# VCCL_COMM_ID=10.0.0.1:5555 VCCL_RANK=0 VCCL_NRANKS=2 ./vccl-bench --op allreduce --check
nodeB# VCCL_COMM_ID=10.0.0.1:5555 VCCL_RANK=1 VCCL_NRANKS=2 ./vccl-bench --op allreduce --check
```

Single-node sanity (no second node needed): `./test_collectives`,
`./vccl-bench --launch 2 --op allreduce --check`.

## Kernel requirements (initramfs/squashfs checklist)

Userspace is fully self-contained; the kernel side needs:

- `CONFIG_INFINIBAND` + `CONFIG_INFINIBAND_USER_ACCESS` (`ib_uverbs.ko`) —
  creates `/dev/infiniband/uverbs*` (devtmpfs handles the nodes).
- The NIC's RDMA driver: `mlx5_ib`+`mlx5_core` (ConnectX), `bnxt_re`
  (Broadcom), `irdma` (Intel E810).
- For Soft-RoCE on non-RDMA NICs: `rdma_rxe.ko`, created with iproute2's
  `rdma` tool: `rdma link add rxe0 type rxe netdev <if>` (the `rdma`
  binary is the one thing not bundled here; include `iproute2-rdma` in the
  image if you need rxe).
- `/sys` mounted (GID type detection reads
  `/sys/class/infiniband/*/ports/*/gid_attrs/types/*`).
- `lo` up (`ip link set lo up`) — bare initramfs environments often leave
  loopback down, which breaks local multi-process runs with connect
  timeouts. `vccl-e2e.sh` does this automatically.
- The test/bench fall back to TCP automatically when no RDMA device is
  usable, so the kit still verifies networking before RDMA is up.

## Useful environment knobs

| var | meaning |
|---|---|
| `VCCL_COMM_ID` | `<rank0-ip>:<port>` rendezvous; identical on all ranks |
| `VCCL_TRANSPORT` | `verbs` or `tcp` (default: verbs when a device exists) |
| `VCCL_SOCKET_ADDR` / `VCCL_SOCKET_IFNAME` | which IP/interface to advertise for bootstrap+TCP |
| `VCCL_IB_HCA` / `VCCL_GID_INDEX` | pin RDMA device / GID table entry |
| `VCCL_DEBUG` | `INFO` or `TRACE` for bring-up debugging |
| `VCCL_TIMEOUT` | seconds before a stuck transfer fails (default 600) |

## Rebuilding

`kit/build-kit.sh` reproduces the binaries on any Alpine x86_64 host
(container or VM): builds rdma-core v57 with `ENABLE_STATIC`, then links
vccl with `-static` and `-DRDMA_STATIC_PROVIDERS=...`. See the script for
the package list. The same recipe works for aarch64 by running it on an
Alpine aarch64 host.
