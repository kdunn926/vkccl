/* SPDX-License-Identifier: Apache-2.0 */
#include "core/bootstrap.h"

#include <arpa/inet.h>

#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>

#include "util/log.h"

namespace vccl {
namespace {

constexpr int kConnectTimeoutMs = 120000;

// Listen sockets created by vcclGetUniqueId, waiting to be claimed by the
// rank-0 vcclCommInitRank in the same process.
std::mutex g_registryMutex;
std::map<std::string, int> g_listenRegistry;

std::string idKey(const vcclUniqueId& id) {
  return std::string(id.internal, VCCL_UNIQUE_ID_BYTES);
}

struct Hello {
  uint64_t magic;
  int32_t rank;
};

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
    for (int i = 0; i < nranks - 1; i++) {
      int fd = -1;
      vcclResult_t res = acceptOne(listenFd, &fd);
      if (res != vcclSuccess) {
        closeQuiet(listenFd);
        return res;
      }
      Hello hello{};
      res = recvAll(fd, &hello, sizeof(hello));
      if (res != vcclSuccess || hello.magic != internal->magic ||
          hello.rank <= 0 || hello.rank >= nranks ||
          bs->peerFds_[hello.rank] != -1) {
        VCCL_ERR("bootstrap: bad hello from peer");
        closeQuiet(fd);
        closeQuiet(listenFd);
        return res != vcclSuccess ? res : vcclInternalError;
      }
      bs->peerFds_[hello.rank] = fd;
    }
    closeQuiet(listenFd);
  } else {
    VCCLCHECK(
        connectTo(internal->getAddr(), kConnectTimeoutMs, &bs->rootFd_));
    Hello hello{internal->magic, rank};
    VCCLCHECK(sendAll(bs->rootFd_, &hello, sizeof(hello)));
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
