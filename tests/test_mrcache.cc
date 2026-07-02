/* SPDX-License-Identifier: Apache-2.0
 *
 * H1(a) unit test: exercises vcclComm's transient MR cache (comm.h) against a
 * fake Transport whose regMr/deregMr are directly observable. On the real
 * (TCP) transport regMr yields a null handle, so the LRU bookkeeping goes
 * untested end-to-end; this test drives vcclComm directly (it is a plain
 * struct) so the eviction/flush logic is deterministically checked without
 * RDMA. No process fork, no network.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/comm.h"
#include "transport/transport.h"

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                   \
  do {                                                      \
    if (!(cond)) {                                          \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);  \
      fprintf(stderr, __VA_ARGS__);                         \
      fprintf(stderr, "\n");                                \
      g_failures++;                                         \
    }                                                        \
  } while (0)

// A Transport whose regMr hands out unique non-null handles and whose
// deregMr counts calls, so the MR cache's eviction/flush logic is directly
// observable without any real memory registration.
class FakeTransport final : public vccl::Transport {
 public:
  vcclResult_t send(int, const void*, size_t) override { return vcclSuccess; }
  vcclResult_t recv(int, void*, size_t) override { return vcclSuccess; }
  vcclResult_t sendRecv(int, const void*, size_t, int, void*,
                        size_t) override {
    return vcclSuccess;
  }
  vcclResult_t batch(const std::vector<vccl::P2pOp>&) override {
    return vcclSuccess;
  }
  const char* name() const override { return "fake"; }

  vcclResult_t regMr(void* buf, size_t bytes, void** handle) override {
    (void)buf;
    (void)bytes;
    regCount++;
    // Non-null, unique per call: use the running counter's address so every
    // handle is distinct and never equals a real pointer by accident.
    handles_.push_back(std::make_unique<int>(regCount));
    *handle = handles_.back().get();
    return vcclSuccess;
  }
  vcclResult_t deregMr(void* handle) override {
    (void)handle;
    deregCount++;
    return vcclSuccess;
  }

  int regCount = 0;
  int deregCount = 0;

 private:
  std::vector<std::unique_ptr<int>> handles_;
};

}  // namespace

int main() {
  setenv("VCCL_MR_CACHE", "4", /*overwrite=*/1);

  vcclComm comm;
  comm.rank = 0;
  comm.nranks = 1;
  auto fake = std::make_unique<FakeTransport>();
  FakeTransport* fakePtr = fake.get();
  comm.transport = std::move(fake);

  CHECK(comm.mrCacheCap() == 4, "expected VCCL_MR_CACHE=4 to set capacity 4");

  // Insert 7 distinct (addr,len) ranges, one per ScopedReg (mimicking one
  // collective call per range). With capacity 4, the 4th..7th inserts each
  // evict one LRU entry -> 3 evictions (deregs), 4 entries remain.
  std::vector<char> ranges(7 * 64, 0);
  for (int i = 0; i < 7; i++) {
    vccl::ScopedReg reg(&comm);
    void* ptr = ranges.data() + i * 64;
    vcclResult_t res = reg.add(ptr, 32);
    CHECK(res == vcclSuccess, "ScopedReg::add failed on range %d", i);
  }
  CHECK(fakePtr->regCount == 7, "expected 7 regMr calls, got %d",
        fakePtr->regCount);
  CHECK(fakePtr->deregCount == 3, "expected 3 evicting deregMr calls, got %d",
        fakePtr->deregCount);
  CHECK(comm.mrCache.size() == 4, "expected 4 entries cached, got %zu",
        comm.mrCache.size());

  // The cache should hold ranges 3..6 (0-indexed), the 4 most recent.
  for (int i = 0; i < 3; i++) {
    void* ptr = ranges.data() + i * 64;
    void* out = nullptr;
    CHECK(!comm.mrCacheGet(reinterpret_cast<uintptr_t>(ptr), 32, &out),
          "range %d should have been evicted", i);
  }

  // Re-touch an early surviving entry (range 3) to bump its recency, then
  // insert one more distinct range (8th). Range 3 should now survive the
  // next eviction instead of range 4 (the next-LRU).
  {
    void* ptr3 = ranges.data() + 3 * 64;
    void* out = nullptr;
    CHECK(comm.mrCacheGet(reinterpret_cast<uintptr_t>(ptr3), 32, &out),
          "range 3 should still be cached before the re-touch");
  }
  {
    vccl::ScopedReg reg(&comm);
    std::vector<char> extra(64, 0);
    vcclResult_t res = reg.add(extra.data(), 32);
    CHECK(res == vcclSuccess, "ScopedReg::add failed on the 8th range");
  }
  CHECK(comm.mrCache.size() == 4,
        "expected the cache to stay at capacity 4, got %zu",
        comm.mrCache.size());
  {
    void* ptr3 = ranges.data() + 3 * 64;
    void* out = nullptr;
    CHECK(comm.mrCacheGet(reinterpret_cast<uintptr_t>(ptr3), 32, &out),
          "range 3 should have survived the eviction after being re-touched");
  }
  {
    void* ptr4 = ranges.data() + 4 * 64;
    void* out = nullptr;
    CHECK(!comm.mrCacheGet(reinterpret_cast<uintptr_t>(ptr4), 32, &out),
          "range 4 (now the LRU) should have been evicted instead");
  }

  const int deregBeforeFlush = fakePtr->deregCount;
  const size_t cachedBeforeFlush = comm.mrCache.size();
  comm.flushMrCache();
  CHECK(comm.mrCache.empty(), "flushMrCache should empty the cache");
  CHECK(fakePtr->deregCount ==
            deregBeforeFlush + static_cast<int>(cachedBeforeFlush),
        "flushMrCache should deregister every remaining cached entry "
        "(expected %d, got %d)",
        deregBeforeFlush + static_cast<int>(cachedBeforeFlush),
        fakePtr->deregCount);

  printf("test_mrcache: %s\n", g_failures ? "FAIL" : "OK");
  return g_failures ? 1 : 0;
}
