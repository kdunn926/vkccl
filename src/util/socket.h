/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_UTIL_SOCKET_H_
#define VCCL_UTIL_SOCKET_H_

#include <netinet/in.h>
#include <sys/socket.h>

#include <cstddef>
#include <string>

#include "vccl.h"

namespace vccl {

// A self-contained socket address (IPv4 or IPv6).
struct SocketAddr {
  sockaddr_storage storage{};
  socklen_t len = 0;

  const sockaddr* sa() const {
    return reinterpret_cast<const sockaddr*>(&storage);
  }
  sockaddr* sa() { return reinterpret_cast<sockaddr*>(&storage); }
  std::string str() const;
  uint16_t port() const;
  void setPort(uint16_t port);
};

// Pick the local IP to advertise to peers: VCCL_SOCKET_ADDR if set, else the
// address of VCCL_SOCKET_IFNAME, else the first non-loopback interface, else
// loopback. Port is left as 0.
vcclResult_t getAdvertisedAddr(SocketAddr* addr);

// Bind a TCP listening socket on the wildcard address with an ephemeral (or
// given) port. Returns the fd and fills boundPort.
vcclResult_t createListenSocket(int* fd, uint16_t port, uint16_t* boundPort);

// Connect to addr, retrying with backoff for up to timeoutMs.
vcclResult_t connectTo(const SocketAddr& addr, int timeoutMs, int* fd);

// Accept one connection (blocking).
vcclResult_t acceptOne(int listenFd, int* fd);

// Blocking exact-size helpers (work on blocking or non-blocking fds).
vcclResult_t sendAll(int fd, const void* buf, size_t bytes);
vcclResult_t recvAll(int fd, void* buf, size_t bytes);

// Concurrently send sbytes to sendFd and receive rbytes from recvFd using a
// poll() progress loop, so neither direction blocks the other.
vcclResult_t duplexTransfer(int sendFd, const void* sbuf, size_t sbytes,
                            int recvFd, void* rbuf, size_t rbytes);

void closeQuiet(int fd);

}  // namespace vccl

#endif  // VCCL_UTIL_SOCKET_H_
