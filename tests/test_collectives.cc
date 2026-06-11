/* SPDX-License-Identifier: Apache-2.0
 *
 * Multi-process correctness tests: forks nranks children over the loopback
 * TCP transport and validates every collective against a locally computed
 * reference. Exit code 0 on success.
 */
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vccl.h"

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                          \
  do {                                                            \
    if (!(cond)) {                                                \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);        \
      fprintf(stderr, __VA_ARGS__);                               \
      fprintf(stderr, "\n");                                      \
      g_failures++;                                               \
    }                                                             \
  } while (0)

#define VCHECK(call) CHECK((call) == vcclSuccess, "%s", #call)

// Deterministic per-rank pattern. Small integers so f16 sums stay exact.
float patternValue(int rank, size_t idx) {
  return static_cast<float>((rank * 31 + idx * 7) % 16);
}

void fillBuffer(void* buf, vcclDataType_t dt, int rank, size_t count,
                size_t globalOffset) {
  for (size_t i = 0; i < count; i++) {
    float v = patternValue(rank, globalOffset + i);
    switch (dt) {
      case vcclFloat32: static_cast<float*>(buf)[i] = v; break;
      case vcclFloat64: static_cast<double*>(buf)[i] = v; break;
      case vcclInt32: static_cast<int32_t*>(buf)[i] = static_cast<int32_t>(v); break;
      case vcclFloat16:
      case vcclBfloat16: {
        // Round-trip through the public API is unavailable; emulate bf16/f16
        // by using values exactly representable in both (small integers).
        uint16_t bits;
        if (dt == vcclBfloat16) {
          uint32_t u;
          memcpy(&u, &v, 4);
          bits = static_cast<uint16_t>(u >> 16);
        } else {
          // Small non-negative integers < 2048: exact in f16.
          int iv = static_cast<int>(v);
          if (iv == 0) {
            bits = 0;
          } else {
            int exp = 0, m = iv;
            while (m >= 2) { m >>= 1; exp++; }
            uint32_t mant = (static_cast<uint32_t>(iv) << (10 - exp)) & 0x3ff;
            bits = static_cast<uint16_t>(((exp + 15) << 10) | mant);
          }
        }
        static_cast<uint16_t*>(buf)[i] = bits;
        break;
      }
      default: abort();
    }
  }
}

float readValue(const void* buf, vcclDataType_t dt, size_t i) {
  switch (dt) {
    case vcclFloat32: return static_cast<const float*>(buf)[i];
    case vcclFloat64: return static_cast<float>(static_cast<const double*>(buf)[i]);
    case vcclInt32: return static_cast<float>(static_cast<const int32_t*>(buf)[i]);
    case vcclFloat16: {
      uint16_t h = static_cast<const uint16_t*>(buf)[i];
      if (h == 0) return 0.0f;
      int exp = ((h >> 10) & 0x1f) - 15;
      float mant = 1.0f + (h & 0x3ff) / 1024.0f;
      float v = std::ldexp(mant, exp);
      return (h & 0x8000) ? -v : v;
    }
    case vcclBfloat16: {
      uint32_t u = static_cast<uint32_t>(static_cast<const uint16_t*>(buf)[i]) << 16;
      float f;
      memcpy(&f, &u, 4);
      return f;
    }
    default: abort();
  }
}

float reduceRef(vcclRedOp_t op, int nranks, size_t idx) {
  float acc = patternValue(0, idx);
  for (int r = 1; r < nranks; r++) {
    float v = patternValue(r, idx);
    switch (op) {
      case vcclSum: case vcclAvg: acc += v; break;
      case vcclProd: acc *= v; break;
      case vcclMax: acc = acc > v ? acc : v; break;
      case vcclMin: acc = acc < v ? acc : v; break;
      default: abort();
    }
  }
  if (op == vcclAvg) acc /= nranks;
  return acc;
}

void expectNear(float got, float want, vcclDataType_t dt, const char* what,
                size_t idx) {
  // f32/f64 get an ulp-scale allowance: the library reduces in ring order and
  // averages via multiplication by (float)(1/n), either of which can differ
  // from the straight-line reference by a couple of ulps.
  float tol = std::fabs(want) * 1e-6f + 1e-7f;
  if (dt == vcclFloat16) tol = std::fabs(want) * 1e-2f + 1e-3f;
  if (dt == vcclBfloat16) tol = std::fabs(want) * 4e-2f + 1e-2f;
  CHECK(std::fabs(got - want) <= tol, "%s[%zu]: got %f want %f", what, idx,
        got, want);
}

void testAllGather(vcclComm_t comm, int rank, int nranks, vcclDataType_t dt,
                   size_t count) {
  size_t esize = vcclTypeSize(dt);
  std::vector<char> send(count * esize), recv(nranks * count * esize);
  fillBuffer(send.data(), dt, rank, count, 0);
  VCHECK(vcclAllGather(send.data(), recv.data(), count, dt, comm));
  for (int r = 0; r < nranks; r++) {
    for (size_t i = 0; i < count; i++) {
      expectNear(readValue(recv.data(), dt, r * count + i),
                 patternValue(r, i), dt, "allgather", r * count + i);
    }
  }
  // In-place variant.
  std::vector<char> recv2(nranks * count * esize);
  fillBuffer(recv2.data() + rank * count * esize, dt, rank, count, 0);
  VCHECK(vcclAllGather(recv2.data() + rank * count * esize, recv2.data(),
                       count, dt, comm));
  CHECK(memcmp(recv.data(), recv2.data(), recv.size()) == 0,
        "allgather in-place mismatch");
}

void testReduceScatter(vcclComm_t comm, int rank, int nranks,
                       vcclDataType_t dt, vcclRedOp_t op, size_t recvcount) {
  size_t esize = vcclTypeSize(dt);
  std::vector<char> send(nranks * recvcount * esize), recv(recvcount * esize);
  fillBuffer(send.data(), dt, rank, nranks * recvcount, 0);
  VCHECK(vcclReduceScatter(send.data(), recv.data(), recvcount, dt, op, comm));
  for (size_t i = 0; i < recvcount; i++) {
    expectNear(readValue(recv.data(), dt, i),
               reduceRef(op, nranks, rank * recvcount + i), dt,
               "reducescatter", i);
  }
}

void testAllReduce(vcclComm_t comm, int rank, int nranks, vcclDataType_t dt,
                   vcclRedOp_t op, size_t count) {
  size_t esize = vcclTypeSize(dt);
  std::vector<char> send(count * esize), recv(count * esize);
  fillBuffer(send.data(), dt, rank, count, 0);
  VCHECK(vcclAllReduce(send.data(), recv.data(), count, dt, op, comm));
  for (size_t i = 0; i < count; i++) {
    expectNear(readValue(recv.data(), dt, i), reduceRef(op, nranks, i), dt,
               "allreduce", i);
  }
  // In-place variant.
  VCHECK(vcclAllReduce(send.data(), send.data(), count, dt, op, comm));
  CHECK(memcmp(send.data(), recv.data(), recv.size()) == 0,
        "allreduce in-place mismatch");
}

void testBroadcast(vcclComm_t comm, int rank, int nranks, size_t count) {
  std::vector<float> buf(count);
  for (int root = 0; root < nranks; root++) {
    if (rank == root) {
      fillBuffer(buf.data(), vcclFloat32, root, count, 0);
    } else {
      std::fill(buf.begin(), buf.end(), -1.0f);
    }
    VCHECK(vcclBroadcast(buf.data(), buf.data(), count, vcclFloat32, root,
                         comm));
    for (size_t i = 0; i < count; i++) {
      expectNear(buf[i], patternValue(root, i), vcclFloat32, "broadcast", i);
    }
  }
}

void testSendRecv(vcclComm_t comm, int rank, int nranks, size_t count) {
  if (nranks < 2) return;
  std::vector<float> buf(count);
  // Symmetric pairing: even rank with the next odd rank; with an odd number
  // of ranks the last one sits out.
  int peer = rank ^ 1;
  if (peer >= nranks) return;
  bool first = rank < peer;
  if (first) {
    fillBuffer(buf.data(), vcclFloat32, rank, count, 0);
    VCHECK(vcclSend(buf.data(), count, vcclFloat32, peer, comm));
    VCHECK(vcclRecv(buf.data(), count, vcclFloat32, peer, comm));
  } else {
    std::vector<float> tmp(count);
    fillBuffer(tmp.data(), vcclFloat32, rank, count, 0);
    VCHECK(vcclRecv(buf.data(), count, vcclFloat32, peer, comm));
    VCHECK(vcclSend(tmp.data(), count, vcclFloat32, peer, comm));
  }
  for (size_t i = 0; i < count; i++) {
    expectNear(buf[i], patternValue(peer, i), vcclFloat32, "sendrecv", i);
  }
}

int childMain(int rank, int nranks, const vcclUniqueId& id) {
  vcclComm_t comm = nullptr;
  VCHECK(vcclCommInitRank(&comm, nranks, id, rank));
  if (comm == nullptr) return 1;

  int gotRank = -1, gotCount = -1;
  VCHECK(vcclCommUserRank(comm, &gotRank));
  VCHECK(vcclCommCount(comm, &gotCount));
  CHECK(gotRank == rank && gotCount == nranks, "rank/count mismatch");

  const size_t sizes[] = {1, 7, 1024, (1 << 16) + 3};
  const vcclDataType_t dtypes[] = {vcclFloat32, vcclFloat16, vcclBfloat16,
                                   vcclInt32, vcclFloat64};
  for (size_t count : sizes) {
    for (vcclDataType_t dt : dtypes) {
      testAllGather(comm, rank, nranks, dt, count);
      testReduceScatter(comm, rank, nranks, dt, vcclSum, count);
      testAllReduce(comm, rank, nranks, dt, vcclSum, count);
      testAllReduce(comm, rank, nranks, dt, vcclMax, count);
      testAllReduce(comm, rank, nranks, dt, vcclMin, count);
    }
    testAllReduce(comm, rank, nranks, vcclFloat32, vcclAvg, count);
    testBroadcast(comm, rank, nranks, count);
    testSendRecv(comm, rank, nranks, count);
  }

  VCHECK(vcclCommDestroy(comm));
  if (g_failures > 0) {
    fprintf(stderr, "rank %d/%d: %d failures\n", rank, nranks, g_failures);
    return 1;
  }
  return 0;
}

int runConfig(int nranks) {
  vcclUniqueId id;
  if (vcclGetUniqueId(&id) != vcclSuccess) {
    fprintf(stderr, "vcclGetUniqueId failed\n");
    return 1;
  }
  std::vector<pid_t> pids;
  for (int r = 0; r < nranks; r++) {
    pid_t pid = fork();
    if (pid == 0) {
      _exit(childMain(r, nranks, id));
    }
    pids.push_back(pid);
  }
  int failed = 0;
  for (pid_t pid : pids) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failed++;
  }
  printf("nranks=%d: %s\n", nranks, failed ? "FAIL" : "OK");
  fflush(stdout);
  return failed ? 1 : 0;
}

}  // namespace

int main() {
  setenv("VCCL_SOCKET_ADDR", "127.0.0.1", 0);
  int failures = 0;
  for (int n : {1, 2, 3, 4, 5}) failures += runConfig(n);
  if (failures == 0) printf("all tests passed\n");
  return failures ? 1 : 0;
}
