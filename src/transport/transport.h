/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_TRANSPORT_TRANSPORT_H_
#define VCCL_TRANSPORT_TRANSPORT_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "vccl.h"

namespace vccl {

class Bootstrap;

// One point-to-point transfer inside a batch.
struct P2pOp {
  bool isSend;
  int peer;
  void* buf;  // const-cast for sends; never written
  size_t bytes;
};

// Point-to-point byte transport between the ranks of one communicator.
// Connections to all peers are established during create(). All calls are
// blocking; send/recv pairs must match in size. sendRecv makes progress on
// both directions concurrently, which ring algorithms rely on to get
// max(send, recv) step time instead of the sum.
class Transport {
 public:
  virtual ~Transport() = default;

  virtual vcclResult_t send(int peer, const void* buf, size_t bytes) = 0;
  virtual vcclResult_t recv(int peer, void* buf, size_t bytes) = 0;
  virtual vcclResult_t sendRecv(int sendPeer, const void* sbuf, size_t sbytes,
                                int recvPeer, void* rbuf, size_t rbytes) = 0;

  // Progress a set of transfers concurrently (vcclGroupStart/End semantics):
  // a send+recv pair with one peer must not deadlock against the mirrored
  // pair on the other side. Multiple same-direction ops to one peer complete
  // in issue order.
  virtual vcclResult_t batch(const std::vector<P2pOp>& ops) = 0;

  // One receive to pre-post (M5). Buffer must stay valid until the matching
  // sendRecv/recv consumes it.
  struct RecvReq { int peer; void* buf; size_t bytes; };

  // Pre-post every recv of a multi-step collective up front. On RDMA this
  // lifts the per-step recv-post off the critical path and makes the path
  // structurally RNR-free; the subsequent sendRecv for the same (peer,buf,
  // bytes) skips its own post and just waits. Default (and TCP): no-op hint;
  // sendRecv is unchanged, so there is no send/send deadlock. reqs must list
  // the recvs in the order the collective will issue its sendRecv recvs.
  virtual vcclResult_t postRecvs(const std::vector<RecvReq>& reqs) {
    (void)reqs;
    return vcclSuccess;
  }

  // Make any blocked operation fail promptly (vcclCommAbort). Thread-safe.
  virtual void abort() { aborted_.store(true, std::memory_order_relaxed); }
  bool isAborted() const { return aborted_.load(std::memory_order_relaxed); }

  virtual const char* name() const = 0;

  // Pin a buffer with the transport ahead of use. Collectives touching any
  // range inside a registered region reuse its registration. The default
  // implementations are no-ops for transports without registration (TCP).
  virtual vcclResult_t regMr(void* buf, size_t bytes, void** handle) {
    (void)buf;
    (void)bytes;
    *handle = nullptr;
    return vcclSuccess;
  }
  // Register a dmabuf whose CPU mapping is [ptr, ptr+bytes). RDMA transports
  // register the dmabuf itself when the device supports it (zero-copy from
  // the NIC's view), falling back to registering the mapping.
  virtual vcclResult_t regDmaBuf(int fd, uint64_t offset, void* ptr,
                                 size_t bytes, void** handle) {
    (void)fd;
    (void)offset;
    return regMr(ptr, bytes, handle);
  }
  virtual vcclResult_t deregMr(void* handle) {
    (void)handle;
    return vcclSuccess;
  }

  // True when [buf, buf+bytes) already lies inside a persistent
  // registration (user-pinned or dmabuf), so a caller should not pin it
  // again — notably dmabuf mappings, which cannot be pinned via regMr.
  virtual bool covers(const void* buf, size_t bytes) const {
    (void)buf;
    (void)bytes;
    return false;
  }

 protected:
  std::atomic<bool> aborted_{false};
};

// TCP sockets, full mesh. Always available.
vcclResult_t tcpTransportCreate(Bootstrap* bootstrap,
                                std::unique_ptr<Transport>* out);

#ifdef VCCL_HAVE_VERBS
// ibverbs RC queue pairs (RoCE v2 / InfiniBand).
vcclResult_t verbsTransportCreate(Bootstrap* bootstrap,
                                  std::unique_ptr<Transport>* out);
bool verbsAvailable();
#endif

// Select per VCCL_TRANSPORT=tcp|verbs, defaulting to verbs when usable.
vcclResult_t transportCreate(Bootstrap* bootstrap,
                             std::unique_ptr<Transport>* out);

}  // namespace vccl

#endif  // VCCL_TRANSPORT_TRANSPORT_H_
