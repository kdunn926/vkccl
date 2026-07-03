/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_CORE_COMM_H_
#define VCCL_CORE_COMM_H_

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/bootstrap.h"
#include "transport/transport.h"
#include "util/log.h"
#include "vccl.h"

struct vcclComm {
  // One user memory registration; `handle` is the public vcclMemHandle_t.
  struct MemReg {
    void* transportHandle = nullptr;
    void* mapped = nullptr;  // non-null when vccl owns a dmabuf mmap
    size_t bytes = 0;
  };

  // PreMulSum op created with vcclRedOpCreatePreMulSum. The public
  // vcclRedOp_t value is vcclNumOps + index; freed slots hold active=false.
  struct CustomOp {
    double scalar = 1.0;
    bool active = false;
  };

  int rank = -1;
  int nranks = 0;
  std::unique_ptr<vccl::Bootstrap> bootstrap;
  std::unique_ptr<vccl::Transport> transport;
  std::vector<MemReg*> registrations;
  std::vector<CustomOp> customOps;
  std::atomic<vcclResult_t> asyncError{vcclSuccess};
  std::atomic<bool> aborted{false};
  std::mutex errMutex;
  // Guards the `registrations` vector against concurrent
  // vcclCommRegister/Deregister. Collectives are NOT serialized by this; the
  // documented contract (see vccl.h) is one collective per comm at a time,
  // with no Register/Deregister concurrent with a running collective.
  std::mutex regMutex;
  std::string lastError;
  // Scratch space for ring reduce steps, grown on demand.
  std::vector<char> scratch;

  int next() const { return (rank + 1) % nranks; }
  int prev() const { return (rank + nranks - 1) % nranks; }

  // Scratch is registered with the transport across growths: it is the
  // hottest buffer in ring reductions, and re-registering per step would
  // dominate small-message latency. The handle's region is torn down with
  // the transport.
  void* scratchRegHandle = nullptr;
  char* scratchAt(size_t bytes) {
    if (scratch.size() < bytes) {
      if (transport && scratchRegHandle != nullptr) {
        transport->deregMr(scratchRegHandle);
        scratchRegHandle = nullptr;
      }
      scratch.resize(bytes);
      if (transport) {
        transport->regMr(scratch.data(), scratch.size(), &scratchRegHandle);
      }
    }
    return scratch.data();
  }

  // ── H1: transient MR cache ────────────────────────────────────────────
  // ScopedReg pins the user buffer for one collective. On RDMA ibv_reg_mr is
  // ~10-50us (comparable to the wire RTT) and the decode loop re-pins the SAME
  // stable tensors every token. This bounded LRU keeps a few most-recent
  // (addr,len) MRs alive so a repeat collective reuses one instead of pinning.
  // SAFETY: only EXACT (addr,len) repeats hit; the whole cache is flushed on
  // any user deregistration/free (flushMrCache) so a freed+remapped range can
  // never leave the NIC pinned to stale pages. VCCL_MR_CACHE=0 disables it;
  // default ON (16); measured 6.8x on the BC-250 cluster (1KB all_reduce).
  // VCCL_MR_CACHE=0 restores pin-per-call. TCP regMr yields a null handle, so
  // the cache is naturally inert there.
  struct MrCacheEntry {
    uintptr_t addr = 0;
    size_t len = 0;
    void* handle = nullptr;   // transport MR handle
    uint64_t seq = 0;         // LRU recency stamp
  };
  std::vector<MrCacheEntry> mrCache;
  uint64_t mrClock = 0;

  int mrCacheCap() {
    static int cap = [] {
      const char* e = std::getenv("VCCL_MR_CACHE");
      if (e == nullptr) return 16;     // DEFAULT ON (16). VCCL_MR_CACHE=0 disables.
      if (e[0] == '0' && e[1] == '\0') return 0;   // explicit kill switch
      int v = atoi(e);
      return v > 0 ? v : 16;           // "1"/non-numeric -> default capacity
    }();
    return cap;
  }

  // Exact (addr,len) hit: bump recency, return the handle via *out.
  bool mrCacheGet(uintptr_t addr, size_t len, void** out) {
    for (auto& e : mrCache) {
      if (e.addr == addr && e.len == len) {
        e.seq = ++mrClock;
        *out = e.handle;
        return true;
      }
    }
    return false;
  }

  // Insert a freshly-registered handle, evicting (and deregistering) the LRU
  // entry when at capacity.
  void mrCachePut(uintptr_t addr, size_t len, void* handle) {
    const int cap = mrCacheCap();
    if (cap <= 0 || handle == nullptr) return;   // disabled / nothing to cache
    if (static_cast<int>(mrCache.size()) >= cap) {
      size_t victim = 0;
      for (size_t i = 1; i < mrCache.size(); i++)
        if (mrCache[i].seq < mrCache[victim].seq) victim = i;
      if (transport && mrCache[victim].handle)
        transport->deregMr(mrCache[victim].handle);
      mrCache.erase(mrCache.begin() + victim);
    }
    mrCache.push_back({addr, len, handle, ++mrClock});
  }

  // Deregister and drop every cached MR. Called on ANY user deregistration or
  // free and on comm destroy so no stale (freed) range stays pinned.
  void flushMrCache() {
    for (auto& e : mrCache)
      if (transport && e.handle) transport->deregMr(e.handle);
    mrCache.clear();
  }

  // PreMulSum scalar for `op`, or 0 if op is not a valid custom op.
  bool premulScalar(vcclRedOp_t op, double* scalar) const {
    size_t idx = static_cast<size_t>(op) - vcclNumOps;
    if (op < vcclNumOps || idx >= customOps.size() || !customOps[idx].active)
      return false;
    *scalar = customOps[idx].scalar;
    return true;
  }

  void noteError(vcclResult_t res) {
    if (res == vcclSuccess) return;
    vcclResult_t expected = vcclSuccess;
    asyncError.compare_exchange_strong(expected, res);
    // Latch the comm dead on transport-fatal codes so later collectives fail
    // cleanly with vcclInvalidUsage (via checkComm) instead of cascading
    // "ibv_post_send failed" on a QP already in ERROR (M3).
    if (res == vcclSystemError || res == vcclRemoteError ||
        res == vcclInternalError || res == vcclUnhandledError)
      aborted.store(true, std::memory_order_relaxed);
    // Copy the failing thread's last-error text into the comm so
    // vcclGetLastError works for grouped ops / vcclCommInitAll worker threads
    // (M10).
    const char* msg = vccl::lastErrorString();
    if (msg != nullptr && msg[0] != '\0') {
      std::lock_guard<std::mutex> lock(errMutex);
      lastError = msg;
    }
  }
};

namespace vccl {

// Group machinery (vcclGroupStart/End). When the calling thread is inside a
// group, enqueueOrRun queues the closure and returns vcclSuccess; otherwise
// it runs the closure immediately. P2p ops carry their P2pOp descriptor so
// consecutive runs can be progressed as one transport batch.
bool inGroup();
vcclResult_t enqueueOrRun(vcclComm_t comm, std::function<vcclResult_t()> fn);
vcclResult_t enqueueOrRunP2p(vcclComm_t comm, const P2pOp& op);

// Registers buffers with the communicator's transport for the duration of
// one collective, so the data path never creates per-step registrations for
// them. Ranges already covered by a user registration just gain a second,
// short-lived MR — harmless. No-op without an RDMA transport.
class ScopedReg {
 public:
  explicit ScopedReg(vcclComm_t comm) : comm_(comm) {}
  ~ScopedReg() {
    for (void* h : handles_) comm_->transport->deregMr(h);
  }
  ScopedReg(const ScopedReg&) = delete;
  ScopedReg& operator=(const ScopedReg&) = delete;

  vcclResult_t add(const void* ptr, size_t bytes) {
    if (!comm_->transport || ptr == nullptr || bytes == 0)
      return vcclSuccess;
    // Already inside a user/dmabuf registration: pinning again would be
    // wasted work, and dmabuf mappings cannot be pinned at all.
    if (comm_->transport->covers(ptr, bytes)) return vcclSuccess;
    const uintptr_t a = reinterpret_cast<uintptr_t>(ptr);
    if (comm_->mrCacheCap() > 0) {
      void* cached = nullptr;
      if (comm_->mrCacheGet(a, bytes, &cached)) return vcclSuccess;  // reuse
      void* handle = nullptr;
      vcclResult_t res =
          comm_->transport->regMr(const_cast<void*>(ptr), bytes, &handle);
      if (res != vcclSuccess) return res;
      comm_->mrCachePut(a, bytes, handle);  // cache owns it; NOT added to handles_
      return vcclSuccess;
    }
    // Cache disabled: original behavior — pin now, dereg at ScopedReg dtor.
    void* handle = nullptr;
    vcclResult_t res =
        comm_->transport->regMr(const_cast<void*>(ptr), bytes, &handle);
    if (res == vcclSuccess && handle != nullptr) handles_.push_back(handle);
    return res;
  }

 private:
  vcclComm_t comm_;
  std::vector<void*> handles_;
};

}  // namespace vccl

#endif  // VCCL_CORE_COMM_H_
