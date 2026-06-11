/* SPDX-License-Identifier: Apache-2.0 */
#include "util/socket.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "util/deadline.h"
#include "util/log.h"

namespace vccl {

std::string SocketAddr::str() const {
  char host[INET6_ADDRSTRLEN] = "?";
  uint16_t p = 0;
  if (storage.ss_family == AF_INET) {
    const auto* in = reinterpret_cast<const sockaddr_in*>(&storage);
    inet_ntop(AF_INET, &in->sin_addr, host, sizeof(host));
    p = ntohs(in->sin_port);
  } else if (storage.ss_family == AF_INET6) {
    const auto* in6 = reinterpret_cast<const sockaddr_in6*>(&storage);
    inet_ntop(AF_INET6, &in6->sin6_addr, host, sizeof(host));
    p = ntohs(in6->sin6_port);
  }
  char buf[INET6_ADDRSTRLEN + 16];
  snprintf(buf, sizeof(buf), "%s:%u", host, p);
  return buf;
}

uint16_t SocketAddr::port() const {
  if (storage.ss_family == AF_INET)
    return ntohs(reinterpret_cast<const sockaddr_in*>(&storage)->sin_port);
  if (storage.ss_family == AF_INET6)
    return ntohs(reinterpret_cast<const sockaddr_in6*>(&storage)->sin6_port);
  return 0;
}

void SocketAddr::setPort(uint16_t port) {
  if (storage.ss_family == AF_INET)
    reinterpret_cast<sockaddr_in*>(&storage)->sin_port = htons(port);
  else if (storage.ss_family == AF_INET6)
    reinterpret_cast<sockaddr_in6*>(&storage)->sin6_port = htons(port);
}

vcclResult_t getAdvertisedAddr(SocketAddr* addr) {
  memset(&addr->storage, 0, sizeof(addr->storage));

  if (const char* env = std::getenv("VCCL_SOCKET_ADDR")) {
    auto* in = reinterpret_cast<sockaddr_in*>(&addr->storage);
    if (inet_pton(AF_INET, env, &in->sin_addr) == 1) {
      in->sin_family = AF_INET;
      addr->len = sizeof(sockaddr_in);
      return vcclSuccess;
    }
    auto* in6 = reinterpret_cast<sockaddr_in6*>(&addr->storage);
    if (inet_pton(AF_INET6, env, &in6->sin6_addr) == 1) {
      in6->sin6_family = AF_INET6;
      addr->len = sizeof(sockaddr_in6);
      return vcclSuccess;
    }
    VCCL_ERR("VCCL_SOCKET_ADDR='%s' is not a valid IP address", env);
    return vcclInvalidArgument;
  }

  const char* ifname = std::getenv("VCCL_SOCKET_IFNAME");

  ifaddrs* ifas = nullptr;
  if (getifaddrs(&ifas) != 0) return vcclSystemError;

  // Two passes: exact interface-name match first (if requested), then the
  // first non-loopback IPv4, then loopback as a last resort.
  const sockaddr_in* loopback = nullptr;
  const sockaddr_in* candidate = nullptr;
  for (ifaddrs* ifa = ifas; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET)
      continue;
    const auto* in = reinterpret_cast<const sockaddr_in*>(ifa->ifa_addr);
    if (ifname != nullptr) {
      if (strcmp(ifa->ifa_name, ifname) == 0) {
        candidate = in;
        break;
      }
      continue;
    }
    if (ntohl(in->sin_addr.s_addr) >> 24 == 127) {
      if (loopback == nullptr) loopback = in;
    } else if (candidate == nullptr) {
      candidate = in;
    }
  }
  if (candidate == nullptr) candidate = loopback;
  if (candidate == nullptr) {
    freeifaddrs(ifas);
    VCCL_ERR("no usable IPv4 interface found%s%s",
             ifname ? " matching VCCL_SOCKET_IFNAME=" : "",
             ifname ? ifname : "");
    return vcclSystemError;
  }
  memcpy(&addr->storage, candidate, sizeof(sockaddr_in));
  addr->len = sizeof(sockaddr_in);
  freeifaddrs(ifas);
  return vcclSuccess;
}

static void setSockOpts(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

vcclResult_t createListenSocket(int* fd, uint16_t port, uint16_t* boundPort) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) return vcclSystemError;
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(s, 128) != 0) {
    VCCL_ERR("bind/listen failed: %s", strerror(errno));
    closeQuiet(s);
    return vcclSystemError;
  }
  socklen_t alen = sizeof(addr);
  if (getsockname(s, reinterpret_cast<sockaddr*>(&addr), &alen) != 0) {
    closeQuiet(s);
    return vcclSystemError;
  }
  if (boundPort != nullptr) *boundPort = ntohs(addr.sin_port);
  *fd = s;
  return vcclSuccess;
}

vcclResult_t connectTo(const SocketAddr& addr, int timeoutMs, int* fd) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  int delayMs = 20;
  for (;;) {
    int s = socket(addr.storage.ss_family, SOCK_STREAM, 0);
    if (s < 0) return vcclSystemError;
    if (connect(s, addr.sa(), addr.len) == 0) {
      setSockOpts(s);
      *fd = s;
      return vcclSuccess;
    }
    int err = errno;
    closeQuiet(s);
    if (std::chrono::steady_clock::now() >= deadline) {
      VCCL_ERR("connect to %s failed: %s", addr.str().c_str(), strerror(err));
      return vcclSystemError;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    if (delayMs < 500) delayMs *= 2;
  }
}

vcclResult_t acceptOne(int listenFd, int* fd) {
  Deadline deadline;
  for (;;) {
    pollfd pfd{listenFd, POLLIN, 0};
    int rc = poll(&pfd, 1, deadline.pollMs());
    if (rc < 0 && errno != EINTR) {
      VCCL_ERR("poll(listen) failed: %s", strerror(errno));
      return vcclSystemError;
    }
    if (deadline.expired()) {
      VCCL_ERR("timed out waiting for peer connection (VCCL_TIMEOUT)");
      return vcclSystemError;
    }
    if (rc <= 0) continue;
    int s = accept(listenFd, nullptr, nullptr);
    if (s >= 0) {
      setSockOpts(s);
      *fd = s;
      return vcclSuccess;
    }
    if (errno == EINTR) continue;
    VCCL_ERR("accept failed: %s", strerror(errno));
    return vcclSystemError;
  }
}

vcclResult_t sendAll(int fd, const void* buf, size_t bytes) {
  Deadline deadline;
  const char* p = static_cast<const char*>(buf);
  size_t sent = 0;
  while (sent < bytes) {
    ssize_t n = send(fd, p + sent, bytes - sent, 0);
    if (n > 0) {
      sent += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && (errno == EINTR)) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      pollfd pfd{fd, POLLOUT, 0};
      poll(&pfd, 1, deadline.pollMs());
      if (deadline.expired()) {
        VCCL_ERR("send timed out (VCCL_TIMEOUT)");
        return vcclSystemError;
      }
      continue;
    }
    VCCL_ERR("send failed: %s", n == 0 ? "peer closed" : strerror(errno));
    return vcclSystemError;
  }
  return vcclSuccess;
}

vcclResult_t recvAll(int fd, void* buf, size_t bytes) {
  Deadline deadline;
  char* p = static_cast<char*>(buf);
  size_t got = 0;
  while (got < bytes) {
    ssize_t n = recv(fd, p + got, bytes - got, 0);
    if (n > 0) {
      got += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      pollfd pfd{fd, POLLIN, 0};
      poll(&pfd, 1, deadline.pollMs());
      if (deadline.expired()) {
        VCCL_ERR("recv timed out (VCCL_TIMEOUT)");
        return vcclSystemError;
      }
      continue;
    }
    VCCL_ERR("recv failed: %s", n == 0 ? "peer closed" : strerror(errno));
    return vcclSystemError;
  }
  return vcclSuccess;
}

vcclResult_t duplexTransfer(int sendFd, const void* sbuf, size_t sbytes,
                            int recvFd, void* rbuf, size_t rbytes) {
  Deadline deadline;
  const char* sp = static_cast<const char*>(sbuf);
  char* rp = static_cast<char*>(rbuf);
  size_t sent = 0, got = 0;
  while (sent < sbytes || got < rbytes) {
    pollfd pfds[2];
    int n = 0;
    int sendIdx = -1, recvIdx = -1;
    if (sent < sbytes) {
      sendIdx = n;
      pfds[n++] = {sendFd, POLLOUT, 0};
    }
    if (got < rbytes) {
      recvIdx = n;
      pfds[n++] = {recvFd, POLLIN, 0};
    }
    if (poll(pfds, n, deadline.pollMs()) < 0) {
      if (errno == EINTR) continue;
      return vcclSystemError;
    }
    if (deadline.expired()) {
      VCCL_ERR("transfer timed out (VCCL_TIMEOUT)");
      return vcclSystemError;
    }
    if (sendIdx >= 0 && (pfds[sendIdx].revents & (POLLOUT | POLLERR | POLLHUP))) {
      ssize_t k = send(sendFd, sp + sent, sbytes - sent, 0);
      if (k > 0) {
        sent += static_cast<size_t>(k);
      } else if (k < 0 && errno != EINTR && errno != EAGAIN &&
                 errno != EWOULDBLOCK) {
        VCCL_ERR("duplex send failed: %s", strerror(errno));
        return vcclSystemError;
      }
    }
    if (recvIdx >= 0 && (pfds[recvIdx].revents & (POLLIN | POLLERR | POLLHUP))) {
      ssize_t k = recv(recvFd, rp + got, rbytes - got, 0);
      if (k > 0) {
        got += static_cast<size_t>(k);
      } else if (k == 0) {
        VCCL_ERR("duplex recv: peer closed");
        return vcclSystemError;
      } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
        VCCL_ERR("duplex recv failed: %s", strerror(errno));
        return vcclSystemError;
      }
    }
  }
  return vcclSuccess;
}

void closeQuiet(int fd) {
  if (fd >= 0) close(fd);
}

}  // namespace vccl
