/* SPDX-License-Identifier: Apache-2.0 */
#include "core/bootstrap.h"

#include <arpa/inet.h>

#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>

#include "util/deadline.h"
#include "util/log.h"

namespace vccl {
namespace {

// Listen sockets created by vcclGetUniqueId, waiting to be claimed by the
// rank-0 vcclCommInitRank in the same process.
std::mutex g_registryMutex;
std::map<std::string, int> g_listenRegistry;

std::string idKey(const vcclUniqueId& id) {
  return std::string(id.internal, VCCL_UNIQUE_ID_BYTES);
}

struct Hello {
  uint64_t magic;
  uint64_t salt;
  uint32_t version;  // VCCL_VERSION_CODE
  int32_t  nranks;
  int32_t  rank;
  int32_t  pad;      // M11: low byte carries the sender's endianness marker
                     // (kWireLittleEndian); rest kept zero on the wire.
};
static_assert(sizeof(Hello) == 32, "Hello must have no implicit padding");

// M11: a 1-byte endianness marker folded into Hello.pad (no struct size
// change, still static_assert(sizeof(Hello)==32)). The BC-250 cluster is
// homogeneous little-endian, so this is a no-op on the wire there; it closes
// the mixed-endian Mac-dev / Linux-cluster hole the review names by having
// root reject a mismatched rank cleanly via the existing accept/reject reply.
constexpr int32_t kWireLittleEndian = 1;

int32_t hostEndiannessMarker() {
  const uint16_t probe = 1;
  return (*reinterpret_cast<const uint8_t*>(&probe) == 1) ? kWireLittleEndian
                                                            : 0;
}

}  // namespace

void UniqueIdInternal::setAddr(const SocketAddr& sa) {
  family = sa.storage.ss_family;
  memset(addr, 0, sizeof(addr));
  if (family == AF_INET) {
    const auto* in = reinterpret_cast<const sockaddr_in*>(&sa.storage);
    port = in->sin_port;
    memcpy(addr, &in->sin_addr, 4);
  } else {
    const auto* in6 = reinterpret_cast<const sockaddr_in6*>(&sa.storage);
    port = in6->sin6_port;
    memcpy(addr, &in6->sin6_addr, 16);
  }
}

SocketAddr UniqueIdInternal::getAddr() const {
  SocketAddr sa;
  if (family == AF_INET) {
    auto* in = reinterpret_cast<sockaddr_in*>(&sa.storage);
    in->sin_family = AF_INET;
    in->sin_port = port;
    memcpy(&in->sin_addr, addr, 4);
    sa.len = sizeof(sockaddr_in);
  } else {
    auto* in6 = reinterpret_cast<sockaddr_in6*>(&sa.storage);
    in6->sin6_family = AF_INET6;
    in6->sin6_port = port;
    memcpy(&in6->sin6_addr, addr, 16);
    sa.len = sizeof(sockaddr_in6);
  }
  return sa;
}

vcclResult_t uniqueIdCreate(vcclUniqueId* out) {
  memset(out->internal, 0, VCCL_UNIQUE_ID_BYTES);
  auto* id = reinterpret_cast<UniqueIdInternal*>(out->internal);

  // VCCL_COMM_ID=<ip>:<port> (NCCL_COMM_ID analog): every rank derives the
  // same id locally — no out-of-band distribution. Rank 0 binds the given
  // port at vcclCommInitRank time instead of here.
  if (const char* env = std::getenv("VCCL_COMM_ID")) {
    // Contract: one live communicator per VCCL_COMM_ID value. All ranks derive
    // the same id locally, so the id (and its embedded port) is shared; two
    // OVERLAPPING comms on the same VCCL_COMM_ID would both try to bind that
    // port (EADDRINUSE) — give each concurrent comm its own <ip>:<port>. For
    // SEQUENTIAL reuse, every rank must recreate its comm in the same order.
    const char* colon = strrchr(env, ':');
    SocketAddr addr;
    memset(&addr.storage, 0, sizeof(addr.storage));
    auto* in = reinterpret_cast<sockaddr_in*>(&addr.storage);
    char host[64] = {0};
    if (colon == nullptr || colon == env ||
        static_cast<size_t>(colon - env) >= sizeof(host)) {
      VCCL_ERR("VCCL_COMM_ID must be <ip>:<port>, got '%s'", env);
      return vcclInvalidArgument;
    }
    memcpy(host, env, colon - env);
    int port = atoi(colon + 1);
    if (inet_pton(AF_INET, host, &in->sin_addr) != 1 || port <= 0 ||
        port > 65535) {
      VCCL_ERR("VCCL_COMM_ID must be <ipv4>:<port>, got '%s'", env);
      return vcclInvalidArgument;
    }
    in->sin_family = AF_INET;
    in->sin_port = htons(static_cast<uint16_t>(port));
    addr.len = sizeof(sockaddr_in);
    id->magic = kVcclMagic;
    id->salt = kCommIdEnvSalt;
    id->setAddr(addr);
    VCCL_INFO("unique id derived from VCCL_COMM_ID=%s", env);
    return vcclSuccess;
  }

  SocketAddr addr;
  VCCLCHECK(getAdvertisedAddr(&addr));

  int listenFd = -1;
  uint16_t port = 0;
  VCCLCHECK(createListenSocket(&listenFd, 0, &port));
  addr.setPort(port);
  if (addr.len == 0) addr.len = sizeof(sockaddr_in);

  id->magic = kVcclMagic;
  id->setAddr(addr);
  {
    std::random_device rd;
    id->salt = (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }

  std::lock_guard<std::mutex> lock(g_registryMutex);
  // Defensive: if a prior uid with this exact key was created and never
  // claimed by a rank-0 vcclCommInitRank, close its leaked listen fd before
  // overwriting the slot. (A random 64-bit salt makes real collisions
  // astronomically unlikely; this bounds the abandoned-uid leak regardless.)
  auto existing = g_listenRegistry.find(idKey(*out));
  if (existing != g_listenRegistry.end()) closeQuiet(existing->second);
  g_listenRegistry[idKey(*out)] = listenFd;
  VCCL_INFO("unique id created, root at %s", addr.str().c_str());
  return vcclSuccess;
}

vcclResult_t Bootstrap::init(const vcclUniqueId& id, int rank, int nranks,
                             std::unique_ptr<Bootstrap>* out) {
  const auto* internal =
      reinterpret_cast<const UniqueIdInternal*>(id.internal);
  if (internal->magic != kVcclMagic) {
    VCCL_ERR("invalid unique id (bad magic)");
    return vcclInvalidArgument;
  }

  std::unique_ptr<Bootstrap> bs(new Bootstrap(rank, nranks));

  if (rank == 0) {
    int listenFd = -1;
    {
      std::lock_guard<std::mutex> lock(g_registryMutex);
      auto it = g_listenRegistry.find(idKey(id));
      if (it != g_listenRegistry.end()) {
        listenFd = it->second;
        g_listenRegistry.erase(it);
      }
    }
    if (listenFd < 0 && internal->salt == kCommIdEnvSalt) {
      // Id came from VCCL_COMM_ID: bind the well-known port now.
      SocketAddr addr = internal->getAddr();
      VCCLCHECK(createListenSocket(&listenFd, addr.port(), nullptr));
    }
    if (listenFd < 0) {
      VCCL_ERR(
          "rank 0 must call vcclCommInitRank in the process that called "
          "vcclGetUniqueId");
      return vcclInvalidUsage;
    }
    bs->peerFds_.assign(nranks, -1);
    int connected = 0;
    while (connected < nranks - 1) {
      int fd = -1;
      vcclResult_t res = acceptOne(listenFd, &fd);
      if (res != vcclSuccess) {
        // accept() itself failed: the listen socket is unusable, not a single
        // client's fault. Stay fatal.
        closeQuiet(listenFd);
        return res;
      }
      Hello hello{};
      res = recvAll(fd, &hello, sizeof(hello));
      const bool ok =
          res == vcclSuccess && hello.magic == internal->magic &&
          hello.salt == internal->salt &&
          hello.version == static_cast<uint32_t>(VCCL_VERSION_CODE) &&
          hello.nranks == nranks && hello.rank > 0 && hello.rank < nranks &&
          bs->peerFds_[hello.rank] == -1 &&
          hello.pad == hostEndiannessMarker();
      // Reply accept(1)/reject(0) so a wrong-job client fails fast instead of
      // hanging in sendAll/recvAll. Best-effort on the reject path.
      uint8_t reply = ok ? 1 : 0;
      sendAll(fd, &reply, sizeof(reply));
      if (!ok) {
        // Reject THIS client only and keep listening: a stray/wrong-nranks
        // client (or one that dropped mid-Hello) must not abort a legitimate
        // forming job. A genuinely missing rank instead surfaces via its own
        // connect/collective timeout, not a hard abort here.
        VCCL_ERR(
            "bootstrap: rejected a client (magic/salt/version/nranks/rank/"
            "endianness or recv error); still listening for %d more peer(s)",
            nranks - 1 - connected);
        closeQuiet(fd);
        continue;
      }
      bs->peerFds_[hello.rank] = fd;
      connected++;
    }
    closeQuiet(listenFd);
  } else {
    VCCLCHECK(connectTo(internal->getAddr(), connectTimeoutMs(), &bs->rootFd_));
    Hello hello{};
    hello.magic = internal->magic;
    hello.salt = internal->salt;
    hello.version = static_cast<uint32_t>(VCCL_VERSION_CODE);
    hello.nranks = nranks;
    hello.rank = rank;
    hello.pad = hostEndiannessMarker();
    VCCLCHECK(sendAll(bs->rootFd_, &hello, sizeof(hello)));
    uint8_t reply = 0;
    VCCLCHECK(recvAll(bs->rootFd_, &reply, sizeof(reply)));
    if (reply != 1) {
      VCCL_ERR("bootstrap: root rejected rank %d (salt/version/nranks mismatch)",
                rank);
      return vcclInvalidUsage;
    }
  }

  *out = std::move(bs);
  return vcclSuccess;
}

Bootstrap::~Bootstrap() {
  closeQuiet(rootFd_);
  for (int fd : peerFds_) closeQuiet(fd);
}

vcclResult_t Bootstrap::allGather(void* buf, size_t sliceBytes) {
  char* base = static_cast<char*>(buf);
  if (rank_ == 0) {
    for (int r = 1; r < nranks_; r++) {
      VCCLCHECK(recvAll(peerFds_[r], base + r * sliceBytes, sliceBytes));
    }
    for (int r = 1; r < nranks_; r++) {
      VCCLCHECK(sendAll(peerFds_[r], base, sliceBytes * nranks_));
    }
  } else {
    VCCLCHECK(sendAll(rootFd_, base + rank_ * sliceBytes, sliceBytes));
    VCCLCHECK(recvAll(rootFd_, base, sliceBytes * nranks_));
  }
  return vcclSuccess;
}

vcclResult_t Bootstrap::barrier() {
  std::vector<char> tmp(nranks_, 0);
  return allGather(tmp.data(), 1);
}

}  // namespace vccl
