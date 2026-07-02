/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_CORE_COMM_H_
#define VCCL_CORE_COMM_H_

#include <atomic>
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
