/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_CORE_BOOTSTRAP_H_
#define VCCL_CORE_BOOTSTRAP_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "util/socket.h"
#include "vccl.h"

namespace vccl {

// Wire layout of vcclUniqueId.internal. The id embeds the address of a TCP
// listening socket bound by vcclGetUniqueId in the rank-0 process; the socket
// itself is kept in a process-local registry until vcclCommInitRank claims it.
// The address is stored compactly (sockaddr_storage is too large and not
// portable across nodes).
struct UniqueIdInternal {
  uint64_t magic;
  uint64_t salt;
  uint16_t family;  // AF_INET or AF_INET6
  uint16_t port;    // network byte order
  uint8_t addr[16];

  void setAddr(const SocketAddr& sa);
  SocketAddr getAddr() const;
};
static_assert(sizeof(UniqueIdInternal) <= VCCL_UNIQUE_ID_BYTES,
              "unique id too large");

constexpr uint64_t kVcclMagic = 0x7663636c30312e30ull;  // "vccl01.0"

vcclResult_t uniqueIdCreate(vcclUniqueId* out);

// Rendezvous network rooted at rank 0. Every rank holds a persistent
// connection to root; root holds one per rank. Collective calls must be made
// by all ranks in the same order.
class Bootstrap {
 public:
  static vcclResult_t init(const vcclUniqueId& id, int rank, int nranks,
                           std::unique_ptr<Bootstrap>* out);
  ~Bootstrap();

  // Gather sliceBytes from each rank into all ranks' buf (nranks*sliceBytes,
  // rank-ordered). Each rank's contribution starts at rank*sliceBytes.
  vcclResult_t allGather(void* buf, size_t sliceBytes);
  vcclResult_t barrier();

  int rank() const { return rank_; }
  int nranks() const { return nranks_; }

 private:
  Bootstrap(int rank, int nranks) : rank_(rank), nranks_(nranks) {}

  int rank_;
  int nranks_;
  int rootFd_ = -1;                // non-root: connection to root
  std::vector<int> peerFds_;       // root only: fd per rank (own rank = -1)
};

}  // namespace vccl

#endif  // VCCL_CORE_BOOTSTRAP_H_
