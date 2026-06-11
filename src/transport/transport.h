/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_TRANSPORT_TRANSPORT_H_
#define VCCL_TRANSPORT_TRANSPORT_H_

#include <cstddef>
#include <memory>

#include "vccl.h"

namespace vccl {

class Bootstrap;

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
