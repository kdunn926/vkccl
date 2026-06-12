# BC-250 bring-up kit — build & verification notes

What's in `kit/dist/` (also packaged as `kit/vccl-e2e-kit-x86_64.tar.gz`,
~2.6 MB, sha256 `d9cacd2165716ff5eeaae96c740a191d8ca90e4cafb43edc109890c61a9efcbf`),
how it was produced, and how it was verified — captured 2026-06-12.

## Artifacts

Fully static x86_64 binaries — **zero runtime dependencies**. No glibc, no
`.so` files, no `/etc/libibverbs.d`; drop into any initramfs/squashfs and
run. RDMA providers are linked in statically (rdma-core v57 with `mlx4`,
`mlx5` ConnectX, `bnxt_re` Broadcom, `irdma` Intel E810, `rxe` Soft-RoCE):

| binary | size | purpose |
|---|---|---|
| `vccl-info` | 752 KB | device/port/GID survey; marks the GID vccl auto-selects |
| `vccl-bench` | 1.9 MB | bandwidth sweep with `--check` correctness verification |
| `test_collectives` | 1.9 MB | full single-node correctness suite |
| `test_dmabuf` | 1.9 MB | dmabuf registration check |
| `vccl-e2e.sh` | — | orchestrates everything across nodes |

## Usage on the BC-250s

```sh
nodeA# ./vccl-e2e.sh 0 2 <nodeA-ip>
nodeB# ./vccl-e2e.sh 1 2 <nodeA-ip>
```

No shared filesystem or id files: `VCCL_COMM_ID=<ip:port>` (the
`NCCL_COMM_ID` analog) lets every rank derive the same unique id locally;
rank 0 binds the well-known port at init. The script brings up `lo`, loads
RDMA modules best-effort, surveys devices, then runs correctness-checked
sweeps of all-reduce / all-gather / reduce-scatter / broadcast, falling
back to TCP automatically when no RDMA port is active.

Knobs for optimization work: `VCCL_E2E_MAX` / `VCCL_E2E_ITERS` (sweep
size/iterations), `VCCL_IB_HCA`, `VCCL_GID_INDEX`, `VCCL_TRANSPORT=tcp`,
`VCCL_DEBUG=TRACE`, `VCCL_TIMEOUT`.

## How it was built

Natively on Alpine 3.22 x86_64 (QEMU/TCG VM on the dev Mac):

1. `apk add build-base cmake ninja libnl3-dev libnl3-static linux-headers python3`
2. rdma-core v57.0 with `-DENABLE_STATIC=1 -DNO_PYVERBS=1 -DNO_MAN_PAGES=1`
3. vccl linked with `g++ -static` and
   `-DRDMA_STATIC_PROVIDERS=mlx4,mlx5,rxe,bnxt_re,irdma`, provider archives
   in a `--start-group/--end-group` block plus `libnl-3.a`/`libnl-route-3.a`.

`kit/build-kit.sh` reproduces this on any Alpine x86_64 host (container or
VM); the same recipe works on aarch64.

## How it was verified

- Full correctness suite (`test_collectives`: every collective × dtypes ×
  sizes × 1–5 ranks, group semantics, comm split, PreMulSum) passed over
  **Soft-RoCE on x86_64** with the exact shipped binaries (kernel 6.12).
- **True 2-node run passed**: two Alpine x86_64 VMs joined by a private
  network segment, `rxe0` on the inter-VM NIC, literally
  `./vccl-e2e.sh 0 2 10.99.1.1` and `./vccl-e2e.sh 1 2 10.99.1.1` —
  both ranks PASSED with all correctness checks green.

## Gotchas discovered (relevant to the BC-250 initramfs)

- **`lo` is down in a bare initramfs** — local multi-process runs and the
  TCP fallback then fail with connect timeouts that look like network
  bugs. `vccl-e2e.sh` now runs `ip link set lo up` itself.
- Kernel-side checklist: `ib_uverbs` + the NIC's RDMA driver (`mlx5_ib`,
  `bnxt_re`, `irdma`, …), `/sys` mounted (GID-type detection reads sysfs),
  devtmpfs for `/dev/infiniband/uverbs*`.
- Soft-RoCE needs iproute2's `rdma` tool
  (`rdma link add rxe0 type rxe netdev <if>`) — the one thing not bundled;
  only needed until real RoCE NICs are in place.
- `udmabuf` test requires `CONFIG_UDMABUF` (`/dev/udmabuf`); it skips
  cleanly (exit 77) when absent.

## What to look for on real hardware

- `vccl-info` should show the NIC port ACTIVE with an IPv4-mapped RoCE v2
  GID marked `<- vccl default`. If neighbor resolution fails, pin
  `VCCL_GID_INDEX`.
- With real RoCE NICs (not rxe), `ibv_reg_dmabuf_mr` should take the
  zero-copy path for dmabuf imports rather than the staging bounce — watch
  for the "dmabuf MR registered" INFO line under `VCCL_DEBUG=INFO`.
- Compare `VCCL_TRANSPORT=verbs` vs `tcp` sweeps; verbs should win clearly
  on real NICs (on emulated rxe it does not — that's expected).
