#!/usr/bin/env bash
# Deploy libvccl to a cluster node, backing up the existing lib first so a bad
# push is one command to roll back. Pairs with the VCCL_LIBRARY escape hatch:
# if you would rather not touch the system lib, point the consumer at a custom
# path with VCCL_LIBRARY=/path/to/libvccl.so instead of running this.
#
# Usage: tools/deploy.sh <node> [local-so] [remote-path]
#   node         ssh target, e.g. 192.168.1.53 (user defaults to root)
#   local-so     local library to push   (default: build/libvccl.so)
#   remote-path  destination on the node  (default: /lib64/libvccl.so)
#
# Env overrides: VCCL_SSH_USER (root), VCCL_SSH (ssh), VCCL_SCP (scp).
set -euo pipefail

NODE="${1:?usage: tools/deploy.sh <node> [local-so] [remote-path]}"
LOCAL_SO="${2:-build/libvccl.so}"
REMOTE="${3:-/lib64/libvccl.so}"
USER="${VCCL_SSH_USER:-root}"
SSH="${VCCL_SSH:-ssh}"
SCP="${VCCL_SCP:-scp}"
TARGET="${USER}@${NODE}"

if [ ! -f "$LOCAL_SO" ]; then
  echo "deploy: no such library: $LOCAL_SO" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
echo "deploy: ${LOCAL_SO} -> ${TARGET}:${REMOTE} (backup suffix .bak.${STAMP})"

# Back up the current lib if present.
$SSH "$TARGET" "if [ -f '${REMOTE}' ]; then cp -a '${REMOTE}' '${REMOTE}.bak.${STAMP}' && echo 'backed up ${REMOTE} -> ${REMOTE}.bak.${STAMP}'; else echo 'no existing ${REMOTE}, skipping backup'; fi"

# Upload beside the target, then swap into place atomically.
$SCP "$LOCAL_SO" "${TARGET}:${REMOTE}.new"
$SSH "$TARGET" "mv -f '${REMOTE}.new' '${REMOTE}' && (ldconfig 2>/dev/null || true) && echo 'installed ${REMOTE}'"

echo "deploy: done."
echo "rollback: ${SSH} ${TARGET} mv -f '${REMOTE}.bak.${STAMP}' '${REMOTE}'"
