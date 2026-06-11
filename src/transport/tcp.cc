/* SPDX-License-Identifier: Apache-2.0 */
#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/bootstrap.h"
#include "transport/transport.h"
#include "util/log.h"
#include "util/socket.h"

namespace vccl {
namespace {

constexpr int kConnectTimeoutMs = 120000;

class TcpTransport final : public Transport {
 public:
  static vcclResult_t create(Bootstrap* bs, std::unique_ptr<Transport>* out) {
    std::unique_ptr<TcpTransport> t(new TcpTransport());
    const int rank = bs->rank();
    const int nranks = bs->nranks();
    t->fds_.assign(nranks, -1);

    // Publish a per-rank data listening address, then build the full mesh:
    // connect to lower ranks, accept from higher ranks.
    int listenFd = -1;
    uint16_t port = 0;
    VCCLCHECK(createListenSocket(&listenFd, 0, &port));

    std::vector<SocketAddr> addrs(nranks);
    VCCLCHECK(getAdvertisedAddr(&addrs[rank]));
    addrs[rank].setPort(port);
    vcclResult_t res = bs->allGather(addrs.data(), sizeof(SocketAddr));
    if (res != vcclSuccess) {
      closeQuiet(listenFd);
      return res;
    }

    for (int peer = 0; peer < rank; peer++) {
      int fd = -1;
      res = connectTo(addrs[peer], kConnectTimeoutMs, &fd);
      if (res != vcclSuccess) break;
      int32_t myRank = rank;
      res = sendAll(fd, &myRank, sizeof(myRank));
      if (res != vcclSuccess) {
        closeQuiet(fd);
        break;
      }
      t->fds_[peer] = fd;
    }
    if (res == vcclSuccess) {
      for (int i = rank + 1; i < nranks; i++) {
        int fd = -1;
        res = acceptOne(listenFd, &fd);
        if (res != vcclSuccess) break;
        int32_t peerRank = -1;
        res = recvAll(fd, &peerRank, sizeof(peerRank));
        if (res != vcclSuccess || peerRank <= rank || peerRank >= nranks ||
            t->fds_[peerRank] != -1) {
          VCCL_ERR("tcp transport: bad peer handshake (rank %d)", peerRank);
          closeQuiet(fd);
          if (res == vcclSuccess) res = vcclInternalError;
          break;
        }
        t->fds_[peerRank] = fd;
      }
    }
    closeQuiet(listenFd);
    if (res != vcclSuccess) return res;

    // Non-blocking mode lets sendAll/recvAll/duplexTransfer multiplex.
    for (int fd : t->fds_) {
      if (fd >= 0) fcntl(fd, F_SETFL, O_NONBLOCK);
    }
    VCCL_INFO("tcp transport connected (%d ranks)", nranks);
    *out = std::move(t);
    return vcclSuccess;
  }

  ~TcpTransport() override {
    for (int fd : fds_) closeQuiet(fd);
  }

  vcclResult_t send(int peer, const void* buf, size_t bytes) override {
    return sendAll(fds_[peer], buf, bytes);
  }

  vcclResult_t recv(int peer, void* buf, size_t bytes) override {
    return recvAll(fds_[peer], buf, bytes);
  }

  vcclResult_t sendRecv(int sendPeer, const void* sbuf, size_t sbytes,
                        int recvPeer, void* rbuf, size_t rbytes) override {
    return duplexTransfer(fds_[sendPeer], sbuf, sbytes, fds_[recvPeer], rbuf,
                          rbytes);
  }

  const char* name() const override { return "tcp"; }

 private:
  TcpTransport() = default;
  std::vector<int> fds_;
};

}  // namespace

vcclResult_t tcpTransportCreate(Bootstrap* bootstrap,
                                std::unique_ptr<Transport>* out) {
  return TcpTransport::create(bootstrap, out);
}

vcclResult_t transportCreate(Bootstrap* bootstrap,
                             std::unique_ptr<Transport>* out) {
  const char* env = std::getenv("VCCL_TRANSPORT");
#ifdef VCCL_HAVE_VERBS
  if (env == nullptr || strcmp(env, "verbs") == 0) {
    if (verbsAvailable()) return verbsTransportCreate(bootstrap, out);
    if (env != nullptr) {
      VCCL_ERR("VCCL_TRANSPORT=verbs but no usable RDMA device");
      return vcclSystemError;
    }
    VCCL_INFO("no RDMA device found, falling back to tcp transport");
  }
#else
  if (env != nullptr && strcmp(env, "verbs") == 0) {
    VCCL_ERR("vccl built without ibverbs support");
    return vcclInvalidUsage;
  }
#endif
  if (env != nullptr && strcmp(env, "tcp") != 0
#ifdef VCCL_HAVE_VERBS
      && strcmp(env, "verbs") != 0
#endif
  ) {
    VCCL_ERR("unknown VCCL_TRANSPORT='%s'", env);
    return vcclInvalidArgument;
  }
  return tcpTransportCreate(bootstrap, out);
}

}  // namespace vccl
