#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Builds the static x86_64 vccl e2e kit. Run on Alpine Linux x86_64
# (native, container, or VM) from the vccl source root:
#
#   apk add build-base cmake ninja libnl3-dev libnl3-static linux-headers python3
#   sh kit/build-kit.sh
#
# Produces kit/dist/ with fully static (musl) binaries: vccl-bench,
# test_collectives, test_dmabuf, vccl-info — zero runtime dependencies,
# suitable for any initramfs/squashfs.

set -eu
RDMA_CORE_VERSION=57.0
# Provider drivers statically linked into the binaries. rxe = Soft-RoCE;
# mlx4/mlx5 = Mellanox ConnectX; bnxt_re = Broadcom; irdma = Intel E810.
PROVIDERS="mlx4,mlx5,rxe,bnxt_re,irdma"

cd "$(dirname "$0")/.."
mkdir -p kit/dist build-kit
cd build-kit

if [ ! -d "rdma-core-$RDMA_CORE_VERSION" ]; then
  wget -q "https://github.com/linux-rdma/rdma-core/releases/download/v$RDMA_CORE_VERSION/rdma-core-$RDMA_CORE_VERSION.tar.gz"
  tar xzf "rdma-core-$RDMA_CORE_VERSION.tar.gz"
fi
RC="$PWD/rdma-core-$RDMA_CORE_VERSION/build"
if [ ! -f "$RC/lib/libibverbs.a" ]; then
  cmake -GNinja -S "rdma-core-$RDMA_CORE_VERSION" -B "$RC" \
    -DENABLE_STATIC=1 -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 \
    -DCMAKE_BUILD_TYPE=Release -DIOCTL_MODE=both
  ninja -C "$RC"
fi

cd ..
SRCS="src/core/types.cc src/core/reduce.cc src/core/bootstrap.cc \
      src/core/comm.cc src/core/collectives.cc src/core/group.cc \
      src/transport/tcp.cc src/transport/verbs.cc \
      src/util/socket.cc src/util/log.cc"
PROVIDER_VER=$(ls "$RC/lib" | sed -n 's/librxe-\(rdmav[0-9]*\)\.a/\1/p')
LIBS="-Wl,--start-group -libverbs -lmlx5 -lmlx4 \
      -l:librxe-$PROVIDER_VER.a -l:libbnxt_re-$PROVIDER_VER.a \
      -l:libirdma-$PROVIDER_VER.a -Wl,--end-group \
      /usr/lib/libnl-route-3.a /usr/lib/libnl-3.a -lpthread"
CXXFLAGS="-static -O2 -std=c++17 -DVCCL_HAVE_VERBS=1 \
          -DRDMA_STATIC_PROVIDERS=$PROVIDERS \
          -Iinclude -Isrc -I$RC/include"

for tgt in tests/test_collectives.cc:test_collectives \
           tests/test_dmabuf.cc:test_dmabuf \
           bench/vccl_bench.cc:vccl-bench; do
  src=${tgt%%:*}; out=${tgt##*:}
  # shellcheck disable=SC2086
  g++ $CXXFLAGS $SRCS "$src" -L"$RC/lib" $LIBS -o "kit/dist/$out"
done
# shellcheck disable=SC2086
g++ -static -O2 -std=c++17 -DRDMA_STATIC_PROVIDERS=$PROVIDERS \
  -I"$RC/include" tools/vccl_info.cc -L"$RC/lib" $LIBS -o kit/dist/vccl-info

strip kit/dist/*
cp kit/vccl-e2e.sh kit/dist/
chmod +x kit/dist/vccl-e2e.sh
ls -la kit/dist/
echo "kit ready: kit/dist/"
