/* SPDX-License-Identifier: Apache-2.0
 *
 * Bootstrap reject-and-keep-listening test (TCP loopback). A bogus client with
 * the wrong nranks connects to the root and is rejected; the root must keep
 * listening so a legitimate 2-rank job still forms. Uses fork + a raw socket
 * that replays the 32-byte bootstrap Hello. No RDMA.
 */
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/bootstrap.h"   // UniqueIdInternal, kVcclMagic
#include "util/socket.h"      // SocketAddr, connectTo, sendAll, recvAll, closeQuiet
#include "vccl.h"

namespace {

// Mirror of bootstrap.cc's anonymous-namespace Hello (32 bytes, verified layout).
struct HelloWire {
  uint64_t magic;
  uint64_t salt;
  uint32_t version;
  int32_t nranks;
  int32_t rank;
  int32_t pad;      // low byte = endianness marker (1 on little-endian)
};
static_assert(sizeof(HelloWire) == 32, "HelloWire must match Hello");

int32_t leMarker() {
  const uint16_t probe = 1;
  return (*reinterpret_cast<const uint8_t*>(&probe) == 1) ? 1 : 0;
}

// A well-formed Hello EXCEPT nranks is wrong -> the root must reject it.
// Returns 0 iff the root replied reject(0).
int bogusClient(const vcclUniqueId& id) {
  const auto* internal =
      reinterpret_cast<const vccl::UniqueIdInternal*>(id.internal);
  int fd = -1;
  if (vccl::connectTo(internal->getAddr(), 3000, &fd) != vcclSuccess) return 2;
  HelloWire h{};
  h.magic = internal->magic;
  h.salt = internal->salt;
  h.version = static_cast<uint32_t>(VCCL_VERSION_CODE);
  h.nranks = 99;   // WRONG: real job is nranks=2
  h.rank = 1;
  h.pad = leMarker();
  if (vccl::sendAll(fd, &h, sizeof(h)) != vcclSuccess) { vccl::closeQuiet(fd); return 3; }
  uint8_t reply = 0xff;
  vcclResult_t r = vccl::recvAll(fd, &reply, sizeof(reply));
  vccl::closeQuiet(fd);
  return (r == vcclSuccess && reply == 0) ? 0 : 1;  // expect explicit reject
}

// Minimal real rank: init, sanity-check, destroy. Exit 0 on success.
int realRank(int rank, const vcclUniqueId& id) {
  vcclComm_t comm = nullptr;
  if (vcclCommInitRank(&comm, 2, id, rank) != vcclSuccess || comm == nullptr)
    return 1;
  int got = -1, cnt = -1;
  vcclCommUserRank(comm, &got);
  vcclCommCount(comm, &cnt);
  int rc = (got == rank && cnt == 2) ? 0 : 1;
  vcclCommDestroy(comm);
  return rc;
}

}  // namespace

int main() {
  setenv("VCCL_SOCKET_ADDR", "127.0.0.1", 0);
  setenv("VCCL_TRANSPORT", "tcp", 1);
  setenv("VCCL_TIMEOUT", "5", 1);

  vcclUniqueId id;
  if (vcclGetUniqueId(&id) != vcclSuccess) { printf("bootstrap-reject: FAIL (uid)\n"); return 1; }

  // rank 0 (root): forked so it inherits the registry-held listen fd, then
  // blocks accepting one valid peer.
  pid_t root = fork();
  if (root == 0) _exit(realRank(0, id));

  // Bogus client runs to completion (connect -> wrong Hello -> reject reply)
  // BEFORE rank 1 exists, forcing the reject-and-keep-listening path.
  int bogus = bogusClient(id);

  pid_t peer = fork();
  if (peer == 0) _exit(realRank(1, id));

  int st0 = 0, st1 = 0;
  waitpid(root, &st0, 0);
  waitpid(peer, &st1, 0);
  bool rootOk = WIFEXITED(st0) && WEXITSTATUS(st0) == 0;
  bool peerOk = WIFEXITED(st1) && WEXITSTATUS(st1) == 0;
  bool ok = (bogus == 0) && rootOk && peerOk;
  if (!ok)
    fprintf(stderr, "bogus=%d root=%d peer=%d\n", bogus, WEXITSTATUS(st0), WEXITSTATUS(st1));
  printf("bootstrap-reject: %s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
