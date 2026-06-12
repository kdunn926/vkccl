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

void testReduce(vcclComm_t comm, int rank, int nranks, vcclDataType_t dt,
                vcclRedOp_t op, size_t count) {
  size_t esize = vcclTypeSize(dt);
  for (int root = 0; root < nranks; root++) {
    std::vector<char> send(count * esize), recv(count * esize, 0x5a);
    fillBuffer(send.data(), dt, rank, count, 0);
    VCHECK(vcclReduce(send.data(), recv.data(), count, dt, op, root, comm));
    if (rank == root) {
      for (size_t i = 0; i < count; i++) {
        expectNear(readValue(recv.data(), dt, i), reduceRef(op, nranks, i),
                   dt, "reduce", i);
      }
    }
  }
}

void testGatherScatter(vcclComm_t comm, int rank, int nranks, size_t count) {
  // gather
  std::vector<float> send(count), all(count * nranks, -1.0f);
  fillBuffer(send.data(), vcclFloat32, rank, count, 0);
  int root = nranks - 1;
  VCHECK(vcclGather(send.data(), all.data(), count, vcclFloat32, root, comm));
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      for (size_t i = 0; i < count; i++) {
        expectNear(all[r * count + i], patternValue(r, i), vcclFloat32,
                   "gather", r * count + i);
      }
    }
  }
  // scatter the gathered buffer back: every rank gets its own pattern again
  std::vector<float> got(count, -1.0f);
  VCHECK(vcclScatter(all.data(), got.data(), count, vcclFloat32, root, comm));
  for (size_t i = 0; i < count; i++) {
    expectNear(got[i], patternValue(rank, i), vcclFloat32, "scatter", i);
  }
}

void testAllToAll(vcclComm_t comm, int rank, int nranks, size_t count) {
  // block sent to peer p is filled with pattern(rank*nranks + p)
  std::vector<float> send(count * nranks), recv(count * nranks, -1.0f);
  for (int p = 0; p < nranks; p++) {
    for (size_t i = 0; i < count; i++) {
      send[p * count + i] = patternValue(rank * nranks + p, i);
    }
  }
  VCHECK(vcclAllToAll(send.data(), recv.data(), count, vcclFloat32, comm));
  for (int p = 0; p < nranks; p++) {
    for (size_t i = 0; i < count; i++) {
      expectNear(recv[p * count + i], patternValue(p * nranks + rank, i),
                 vcclFloat32, "alltoall", p * count + i);
    }
  }

  // alltoallv with rank-dependent counts: rank r sends r+1 elements to all
  std::vector<size_t> scnt(nranks, rank + 1), sdis(nranks);
  std::vector<size_t> rcnt(nranks), rdis(nranks);
  size_t soff = 0, roff = 0;
  for (int p = 0; p < nranks; p++) {
    sdis[p] = soff;
    soff += scnt[p];
    rcnt[p] = p + 1;
    rdis[p] = roff;
    roff += rcnt[p];
  }
  std::vector<float> vsend(soff), vrecv(roff, -1.0f);
  for (int p = 0; p < nranks; p++) {
    for (size_t i = 0; i < scnt[p]; i++) {
      vsend[sdis[p] + i] = static_cast<float>(rank * 100 + p);
    }
  }
  VCHECK(vcclAllToAllv(vsend.data(), scnt.data(), sdis.data(), vrecv.data(),
                       rcnt.data(), rdis.data(), vcclFloat32, comm));
  for (int p = 0; p < nranks; p++) {
    for (size_t i = 0; i < rcnt[p]; i++) {
      expectNear(vrecv[rdis[p] + i], static_cast<float>(p * 100 + rank),
                 vcclFloat32, "alltoallv", rdis[p] + i);
    }
  }
}

void testGroupSendRecv(vcclComm_t comm, int rank, int nranks, size_t count) {
  if (nranks < 2) return;
  // Symmetric exchange with both neighbours, all four ops in one group —
  // the canonical pattern (pipeline-parallel boundary) that deadlocks
  // without group semantics.
  std::vector<float> toNext(count), toPrev(count), fromNext(count, -1),
      fromPrev(count, -1);
  fillBuffer(toNext.data(), vcclFloat32, rank * 2, count, 0);
  fillBuffer(toPrev.data(), vcclFloat32, rank * 2 + 1, count, 0);
  int next = (rank + 1) % nranks;
  int prev = (rank + nranks - 1) % nranks;
  VCHECK(vcclGroupStart());
  if (next != prev) {
    VCHECK(vcclSend(toNext.data(), count, vcclFloat32, next, comm));
    VCHECK(vcclSend(toPrev.data(), count, vcclFloat32, prev, comm));
    VCHECK(vcclRecv(fromPrev.data(), count, vcclFloat32, prev, comm));
    VCHECK(vcclRecv(fromNext.data(), count, vcclFloat32, next, comm));
  } else {  // nranks == 2: next == prev; one pair each way
    VCHECK(vcclSend(toNext.data(), count, vcclFloat32, next, comm));
    VCHECK(vcclRecv(fromPrev.data(), count, vcclFloat32, prev, comm));
  }
  VCHECK(vcclGroupEnd());
  for (size_t i = 0; i < count; i++) {
    expectNear(fromPrev[i], patternValue(prev * 2, i), vcclFloat32,
               "group fromPrev", i);
    if (next != prev) {
      expectNear(fromNext[i], patternValue(next * 2 + 1, i), vcclFloat32,
                 "group fromNext", i);
    }
  }
}

void testPreMulSum(vcclComm_t comm, int rank, int nranks, size_t count) {
  float scalar = 0.5f;
  vcclRedOp_t op;
  VCHECK(vcclRedOpCreatePreMulSum(&op, &scalar, vcclFloat32,
                                  vcclScalarHostImmediate, comm));
  std::vector<float> send(count), recv(count);
  fillBuffer(send.data(), vcclFloat32, rank, count, 0);
  VCHECK(vcclAllReduce(send.data(), recv.data(), count, vcclFloat32, op,
                       comm));
  for (size_t i = 0; i < count; i++) {
    float want = 0;
    for (int r = 0; r < nranks; r++) want += 0.5f * patternValue(r, i);
    expectNear(recv[i], want, vcclFloat32, "premulsum allreduce", i);
  }
  // reduce-scatter with the same op
  size_t per = count / nranks;
  if (per > 0) {
    std::vector<float> rs(per);
    VCHECK(vcclReduceScatter(send.data(), rs.data(), per, vcclFloat32, op,
                             comm));
    for (size_t i = 0; i < per; i++) {
      float want = 0;
      for (int r = 0; r < nranks; r++)
        want += 0.5f * patternValue(r, rank * per + i);
      expectNear(rs[i], want, vcclFloat32, "premulsum rs", i);
    }
  }
  VCHECK(vcclRedOpDestroy(op, comm));
  // op must be rejected after destroy
  CHECK(vcclAllReduce(send.data(), recv.data(), count, vcclFloat32, op,
                      comm) == vcclInvalidArgument,
        "destroyed op accepted");
  // re-sync ranks: the rejected call was local-only
  VCHECK(vcclAllReduce(send.data(), recv.data(), count, vcclFloat32, vcclSum,
                       comm));
}

void testCommSplit(vcclComm_t comm, int rank, int nranks) {
  // Split into even/odd parent ranks; verify a collective inside the half.
  int color = rank % 2;
  vcclComm_t sub = nullptr;
  VCHECK(vcclCommSplit(comm, color, /*key=*/rank, &sub, nullptr));
  CHECK(sub != nullptr, "split returned null comm");
  if (sub == nullptr) return;
  int subRank = -1, subCount = -1;
  VCHECK(vcclCommUserRank(sub, &subRank));
  VCHECK(vcclCommCount(sub, &subCount));
  int wantCount = (nranks + (color == 0 ? 1 : 0)) / 2;
  CHECK(subCount == wantCount, "split count %d != %d", subCount, wantCount);
  CHECK(subRank == rank / 2, "split rank %d != %d", subRank, rank / 2);

  std::vector<float> v(64);
  fillBuffer(v.data(), vcclFloat32, rank, v.size(), 0);
  VCHECK(vcclAllReduce(v.data(), v.data(), v.size(), vcclFloat32, vcclSum,
                       sub));
  for (size_t i = 0; i < v.size(); i++) {
    float want = 0;
    for (int r = color; r < nranks; r += 2) want += patternValue(r, i);
    expectNear(v[i], want, vcclFloat32, "split allreduce", i);
  }
  VCHECK(vcclCommDestroy(sub));

  // NOCOLOR on the last rank: everyone else splits to one group.
  vcclComm_t sub2 = nullptr;
  int color2 = rank == nranks - 1 ? VCCL_SPLIT_NOCOLOR : 7;
  VCHECK(vcclCommSplit(comm, color2, 0, &sub2, nullptr));
  if (rank == nranks - 1) {
    CHECK(sub2 == nullptr, "NOCOLOR rank got a comm");
  } else {
    CHECK(sub2 != nullptr, "split2 returned null comm");
    if (sub2 != nullptr) VCHECK(vcclCommDestroy(sub2));
  }
}

void testMisc(vcclComm_t comm, int rank, int nranks) {
  int version = 0;
  VCHECK(vcclGetVersion(&version));
  CHECK(version == VCCL_VERSION_CODE, "version mismatch");
  int dev = -1;
  VCHECK(vcclCommDevice(comm, &dev));
  CHECK(dev == 0, "device != 0");
  vcclResult_t async = vcclNumResults;
  VCHECK(vcclCommGetAsyncError(comm, &async));
  CHECK(async == vcclSuccess, "unexpected async error");
  VCHECK(vcclCommFinalize(comm));

  void* mem = nullptr;
  VCHECK(vcclMemAlloc(&mem, 1 << 20));
  CHECK(mem != nullptr, "mem alloc");
  memset(mem, 0xab, 1 << 20);
  // registered collectives from vcclMemAlloc memory
  float* buf = static_cast<float*>(mem);
  vcclMemHandle_t handle = nullptr;
  VCHECK(vcclCommRegister(comm, mem, 1 << 20, &handle));
  fillBuffer(buf, vcclFloat32, rank, 256, 0);
  VCHECK(vcclAllReduce(buf, buf, 256, vcclFloat32, vcclSum, comm));
  for (size_t i = 0; i < 256; i++) {
    expectNear(buf[i], reduceRef(vcclSum, nranks, i), vcclFloat32,
               "memalloc allreduce", i);
  }
  VCHECK(vcclCommDeregister(comm, handle));
  VCHECK(vcclMemFree(mem));
  CHECK(vcclMemFree(buf) == vcclInvalidArgument, "double free accepted");

  // bcast (legacy in-place broadcast)
  std::vector<float> b(33);
  if (rank == 0) fillBuffer(b.data(), vcclFloat32, 0, b.size(), 0);
  VCHECK(vcclBcast(b.data(), b.size(), vcclFloat32, 0, comm));
  for (size_t i = 0; i < b.size(); i++) {
    expectNear(b[i], patternValue(0, i), vcclFloat32, "bcast", i);
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
    testReduce(comm, rank, nranks, vcclFloat32, vcclSum, count);
    testReduce(comm, rank, nranks, vcclFloat32, vcclMax, count);
    testBroadcast(comm, rank, nranks, count);
    testSendRecv(comm, rank, nranks, count);
    testGroupSendRecv(comm, rank, nranks, count);
  }
  testReduce(comm, rank, nranks, vcclFloat16, vcclSum, 511);
  testReduce(comm, rank, nranks, vcclFloat32, vcclAvg, 257);
  testGatherScatter(comm, rank, nranks, 1000);
  testAllToAll(comm, rank, nranks, 333);
  testPreMulSum(comm, rank, nranks, 512);
  testCommSplit(comm, rank, nranks);
  testMisc(comm, rank, nranks);

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

// vcclCommInitAll: several in-process communicators; collectives on them
// must go through group semantics so they progress against each other.
int testInitAll() {
  constexpr int kDev = 3;
  vcclComm_t comms[kDev] = {};
  if (vcclCommInitAll(comms, kDev, nullptr) != vcclSuccess) {
    fprintf(stderr, "vcclCommInitAll failed\n");
    return 1;
  }
  std::vector<std::vector<float>> bufs(kDev, std::vector<float>(128));
  for (int d = 0; d < kDev; d++) {
    fillBuffer(bufs[d].data(), vcclFloat32, d, bufs[d].size(), 0);
  }
  if (vcclGroupStart() != vcclSuccess) return 1;
  for (int d = 0; d < kDev; d++) {
    if (vcclAllReduce(bufs[d].data(), bufs[d].data(), bufs[d].size(),
                      vcclFloat32, vcclSum, comms[d]) != vcclSuccess)
      return 1;
  }
  if (vcclGroupEnd() != vcclSuccess) return 1;
  int failed = 0;
  for (int d = 0; d < kDev; d++) {
    for (size_t i = 0; i < bufs[d].size(); i++) {
      float want = 0;
      for (int r = 0; r < kDev; r++) want += patternValue(r, i);
      if (bufs[d][i] != want) failed++;
    }
    vcclCommDestroy(comms[d]);
  }
  printf("init-all group allreduce: %s\n", failed ? "FAIL" : "OK");
  return failed ? 1 : 0;
}

int main() {
  setenv("VCCL_SOCKET_ADDR", "127.0.0.1", 0);
  int failures = 0;
  for (int n : {1, 2, 3, 4, 5}) failures += runConfig(n);
  failures += testInitAll();
  if (failures == 0) printf("all tests passed\n");
  return failures ? 1 : 0;
}
