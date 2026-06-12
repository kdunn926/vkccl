#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# vccl 2-node e2e: bring-up checks + correctness-verified bandwidth sweep.
# POSIX sh (BusyBox-compatible). Run one instance per node:
#
#   node A (rank 0):  ./vccl-e2e.sh 0 2 <rankA-ip>
#   node B (rank 1):  ./vccl-e2e.sh 1 2 <rankA-ip>
#
# The third argument is always rank 0's reachable IP (the bootstrap root).
# Optional: VCCL_PORT (default 5555), VCCL_TRANSPORT=tcp to force TCP,
# VCCL_IB_HCA / VCCL_GID_INDEX to pin the device/GID.

set -u
RANK=${1:?usage: vccl-e2e.sh <rank> <nranks> <rank0-ip>}
NRANKS=${2:?usage: vccl-e2e.sh <rank> <nranks> <rank0-ip>}
ROOT_IP=${3:?usage: vccl-e2e.sh <rank> <nranks> <rank0-ip>}
PORT=${VCCL_PORT:-5555}
MAXBYTES=${VCCL_E2E_MAX:-64M}
ITERS=${VCCL_E2E_ITERS:-20}
HERE=$(dirname "$0")

echo "== vccl e2e: rank $RANK/$NRANKS, root $ROOT_IP:$PORT =="

# Minimal-environment fixups: loopback is often down in a bare initramfs
# (local multi-process runs and the TCP fallback need it).
ip link set lo up 2>/dev/null || ifconfig lo up 2>/dev/null

# Best-effort module load; harmless if built-in or already loaded.
modprobe ib_uverbs 2>/dev/null
modprobe mlx5_ib 2>/dev/null
modprobe bnxt_re 2>/dev/null
modprobe irdma 2>/dev/null

echo "== RDMA devices =="
if ! "$HERE/vccl-info"; then
  echo "!! no active RDMA port. For Soft-RoCE bring-up:"
  echo "     modprobe rdma_rxe && rdma link add rxe0 type rxe netdev <if>"
  echo "   (needs iproute2's rdma tool). Continuing over TCP."
  export VCCL_TRANSPORT=tcp
fi

export VCCL_COMM_ID="$ROOT_IP:$PORT"
export VCCL_RANK="$RANK" VCCL_NRANKS="$NRANKS"
# Advertise the address peers can reach us on. Rank 0 advertises the root
# ip; others let vccl pick their first non-loopback interface unless
# VCCL_SOCKET_ADDR/IFNAME is set by the caller.
if [ "$RANK" = "0" ] && [ -z "${VCCL_SOCKET_ADDR:-}" ]; then
  export VCCL_SOCKET_ADDR="$ROOT_IP"
fi

fail=0
for op in allreduce allgather reducescatter; do
  echo "== $op (correctness + sweep) =="
  # Ports must differ per run: rank 0 binds VCCL_COMM_ID's port each time.
  VCCL_COMM_ID="$ROOT_IP:$((PORT + 1))" "$HERE/vccl-bench" \
      --op "$op" --check -b 4K -e "$MAXBYTES" -f 4 --iters "$ITERS" || fail=1
  PORT=$((PORT + 1))
done

echo "== broadcast (sweep) =="
VCCL_COMM_ID="$ROOT_IP:$((PORT + 1))" "$HERE/vccl-bench" \
    --op broadcast -b 64K -e "$MAXBYTES" -f 8 --iters "$ITERS" || fail=1

if [ "$fail" = "0" ]; then
  echo "== vccl e2e PASSED (rank $RANK) =="
else
  echo "== vccl e2e FAILED (rank $RANK) =="
fi
exit $fail
