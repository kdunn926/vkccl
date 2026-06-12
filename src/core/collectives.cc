/* SPDX-License-Identifier: Apache-2.0
 *
 * Collectives.
 *
 * Topology: all collectives run on a ring (or a chain rooted at one rank),
 * with each step's send and receive progressed concurrently via
 * Transport::sendRecv, so per-step time is max(send, recv) rather than the
 * sum. Data is partitioned into nranks chunks; with rank r and n = nranks:
 *
 *   all-gather    step s: send chunk (r-s) mod n, recv chunk (r-s-1) mod n.
 *   reduce-scatter over a full-size accumulator: step s sends accumulator
 *                 chunk (r-s) mod n, receives into scratch, accumulates into
 *                 chunk (r-s-1) mod n; rank r ends owning chunk (r+1) mod n.
 *   all-reduce  = reduce-scatter + all-gather in place over recvbuff, with a
 *                 chunk table that tolerates non-divisible counts.
 *   reduce      = pipelined chain root+1 -> root+2 -> ... -> root, chunked
 *                 so downstream ranks start work before the full message
 *                 arrives.
 *   broadcast   = the same chain in the opposite direction.
 *
 * PreMulSum ops (vcclRedOpCreatePreMulSum) reduce as sum with each rank's
 * own contribution scaled by the op's scalar exactly once, at the point it
 * enters the accumulation.
 *
 * Inside vcclGroupStart/End every public entry point queues instead of
 * running (see group.cc); point-to-point ops queue their P2pOp descriptor so
 * consecutive runs become one Transport::batch.
 */
#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

#include "core/comm.h"
#include "core/reduce.h"
#include "util/log.h"

namespace {

constexpr size_t kPipeBytes = 1 << 19;

struct Chunk {
  size_t offset;  // bytes from buffer base
  size_t bytes;
};

// Partition count elements into n contiguous chunks (ceil-sized; trailing
// chunks may be smaller or empty).
std::vector<Chunk> makeChunks(size_t count, int n, size_t esize) {
  size_t per = (count + n - 1) / n;
  std::vector<Chunk> chunks(n);
  for (int i = 0; i < n; i++) {
    size_t begin = std::min(per * i, count);
    size_t end = std::min(per * (i + 1), count);
    chunks[i].offset = begin * esize;
    chunks[i].bytes = (end - begin) * esize;
  }
  return chunks;
}

bool isFloatType(vcclDataType_t dt) {
  return dt == vcclFloat16 || dt == vcclBfloat16 || dt == vcclFloat32 ||
         dt == vcclFloat64;
}

vcclResult_t checkComm(vcclComm_t comm) {
  if (comm == nullptr) return vcclInvalidArgument;
  if (comm->aborted.load(std::memory_order_relaxed)) return vcclInvalidUsage;
  return vcclSuccess;
}

vcclResult_t checkArgs(const void* sendbuff, const void* recvbuff,
                       vcclDataType_t dt, vcclComm_t comm) {
  VCCLCHECK(checkComm(comm));
  if (sendbuff == nullptr || recvbuff == nullptr) return vcclInvalidArgument;
  if (vcclTypeSize(dt) == 0) return vcclInvalidArgument;
  return vcclSuccess;
}

// Resolves op: for built-in ops *scalar is unused; for PreMulSum ops the
// scalar is fetched and op degrades to vcclSum with *premul set. Validates
// op/dtype combinations (avg and premul are float-only).
vcclResult_t resolveOp(vcclComm_t comm, vcclDataType_t dt, vcclRedOp_t* op,
                       bool* premul, double* scalar) {
  *premul = false;
  if (*op >= vcclNumOps) {
    if (!comm->premulScalar(*op, scalar)) return vcclInvalidArgument;
    if (!isFloatType(dt)) return vcclInvalidArgument;
    *premul = true;
    *op = vcclSum;
    return vcclSuccess;
  }
  if (*op == vcclAvg && !isFloatType(dt)) return vcclInvalidArgument;
  if (*op < 0) return vcclInvalidArgument;
  return vcclSuccess;
}

// In-place ring reduce-scatter over the full-size buffer `data`, using the
// chunk table. On return, chunk (rank+1)%n of `data` is fully reduced.
vcclResult_t ringReduceScatter(vcclComm_t comm, char* data,
                               const std::vector<Chunk>& chunks,
                               vcclDataType_t dt, vcclRedOp_t op) {
  const int n = comm->nranks;
  const int r = comm->rank;
  size_t maxChunk = 0;
  for (const Chunk& c : chunks) maxChunk = std::max(maxChunk, c.bytes);
  char* scratch = comm->scratchAt(maxChunk);

  for (int s = 0; s < n - 1; s++) {
    const Chunk& sc = chunks[(r - s + n) % n];
    const Chunk& rc = chunks[(r - s - 1 + 2 * n) % n];
    VCCLCHECK(comm->transport->sendRecv(comm->next(), data + sc.offset,
                                        sc.bytes, comm->prev(), scratch,
                                        rc.bytes));
    if (rc.bytes > 0) {
      VCCLCHECK(vccl::cpuReduce(dt, op, data + rc.offset, scratch,
                                rc.bytes / vcclTypeSize(dt)));
    }
  }
  return vcclSuccess;
}

// In-place ring all-gather over `data`. Rank r holds chunk (r+startShift)%n
// at entry: startShift 0 for plain all-gather, 1 after reduce-scatter.
vcclResult_t ringAllGather(vcclComm_t comm, char* data,
                           const std::vector<Chunk>& chunks, int startShift) {
  const int n = comm->nranks;
  const int r = comm->rank;
  for (int s = 0; s < n - 1; s++) {
    const Chunk& sc = chunks[(r + startShift - s + 2 * n) % n];
    const Chunk& rc = chunks[(r + startShift - s - 1 + 2 * n) % n];
    VCCLCHECK(comm->transport->sendRecv(comm->next(), data + sc.offset,
                                        sc.bytes, comm->prev(),
                                        data + rc.offset, rc.bytes));
  }
  return vcclSuccess;
}

vcclResult_t doAllGather(const void* sendbuff, void* recvbuff,
                         size_t sendcount, vcclDataType_t datatype,
                         vcclComm_t comm) {
  const size_t esize = vcclTypeSize(datatype);
  const size_t chunkBytes = sendcount * esize;
  char* recv = static_cast<char*>(recvbuff);
  vccl::ScopedReg reg(comm);
  VCCLCHECK(reg.add(recvbuff, comm->nranks * chunkBytes));

  if (sendbuff != recv + comm->rank * chunkBytes) {
    memcpy(recv + comm->rank * chunkBytes, sendbuff, chunkBytes);
  }
  if (comm->nranks == 1 || sendcount == 0) return vcclSuccess;

  std::vector<Chunk> chunks(comm->nranks);
  for (int i = 0; i < comm->nranks; i++)
    chunks[i] = {i * chunkBytes, chunkBytes};
  return ringAllGather(comm, recv, chunks, 0);
}

vcclResult_t doReduceScatter(const void* sendbuff, void* recvbuff,
                             size_t recvcount, vcclDataType_t datatype,
                             vcclRedOp_t op, bool premul, double scalar,
                             vcclComm_t comm) {
  const size_t esize = vcclTypeSize(datatype);
  const size_t chunkBytes = recvcount * esize;
  const int n = comm->nranks;
  const int r = comm->rank;
  const char* send = static_cast<const char*>(sendbuff);
  vccl::ScopedReg reg(comm);
  VCCLCHECK(reg.add(sendbuff, n * chunkBytes));

  if (n == 1 || recvcount == 0) {
    if (recvcount > 0 && sendbuff != recvbuff)
      memcpy(recvbuff, sendbuff, chunkBytes);
    if (premul)
      VCCLCHECK(vccl::cpuScale(datatype, recvbuff, recvcount, scalar));
    return vcclSuccess;
  }

  // Traveling-partial ring: the partial sum of a chunk hops around the ring,
  // gaining each rank's contribution. Scratch: two chunk buffers to receive
  // into one while sending the other, plus one for the scaled step-0 chunk
  // when the op is a PreMulSum.
  char* scratch = comm->scratchAt((premul ? 3 : 2) * chunkBytes);
  char* bufA = scratch;
  char* bufB = scratch + chunkBytes;

  // Step 0 sends our chunk (r-1)%n raw (scaled for premul); step s receives
  // partial chunk (r-s-2)%n, which after adding our contribution is exactly
  // what step s+1 sends.
  const char* sendPtr = send + ((r - 1 + n) % n) * chunkBytes;
  if (premul) {
    char* scaled = scratch + 2 * chunkBytes;
    VCCLCHECK(vccl::cpuScaleCopy(datatype, scaled, sendPtr, recvcount,
                                 scalar));
    sendPtr = scaled;
  }
  for (int s = 0; s < n - 1; s++) {
    char* recvPtr = (s % 2 == 0) ? bufA : bufB;
    VCCLCHECK(comm->transport->sendRecv(comm->next(), sendPtr, chunkBytes,
                                        comm->prev(), recvPtr, chunkBytes));
    const int rc = (r - s - 2 + 2 * n) % n;
    const char* local = send + rc * chunkBytes;
    if (s == n - 2) {
      // Final hop: rc == r. recvbuff = local chunk r (op) incoming partial.
      if (premul) {
        VCCLCHECK(vccl::cpuScaleCopy(datatype, recvbuff, local, recvcount,
                                     scalar));
      } else if (recvbuff != local) {
        memcpy(recvbuff, local, chunkBytes);
      }
      VCCLCHECK(vccl::cpuReduce(datatype, op, recvbuff, recvPtr, recvcount));
    } else if (premul) {
      VCCLCHECK(vccl::cpuScaledAccumulate(datatype, recvPtr, local,
                                          recvcount, scalar));
      sendPtr = recvPtr;
    } else {
      VCCLCHECK(vccl::cpuReduce(datatype, op, recvPtr, local, recvcount));
      sendPtr = recvPtr;
    }
  }
  if (op == vcclAvg) {
    VCCLCHECK(vccl::cpuScale(datatype, recvbuff, recvcount, 1.0 / n));
  }
  return vcclSuccess;
}

vcclResult_t doAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                         vcclDataType_t datatype, vcclRedOp_t op, bool premul,
                         double scalar, vcclComm_t comm) {
  const size_t esize = vcclTypeSize(datatype);
  char* recv = static_cast<char*>(recvbuff);
  vccl::ScopedReg reg(comm);
  VCCLCHECK(reg.add(recvbuff, count * esize));
  if (sendbuff != recvbuff) memcpy(recv, sendbuff, count * esize);
  // PreMulSum: scale the whole local contribution once up front, then the
  // ring reduces as a plain sum.
  if (premul) VCCLCHECK(vccl::cpuScale(datatype, recv, count, scalar));
  if (comm->nranks == 1 || count == 0) return vcclSuccess;

  std::vector<Chunk> chunks = makeChunks(count, comm->nranks, esize);
  VCCLCHECK(ringReduceScatter(comm, recv, chunks, datatype, op));
  if (op == vcclAvg) {
    const Chunk& mine = chunks[(comm->rank + 1) % comm->nranks];
    if (mine.bytes > 0) {
      VCCLCHECK(vccl::cpuScale(datatype, recv + mine.offset,
                               mine.bytes / esize, 1.0 / comm->nranks));
    }
  }
  return ringAllGather(comm, recv, chunks, 1);
}

vcclResult_t doReduce(const void* sendbuff, void* recvbuff, size_t count,
                      vcclDataType_t datatype, vcclRedOp_t op, bool premul,
                      double scalar, int root, vcclComm_t comm) {
  const size_t esize = vcclTypeSize(datatype);
  const size_t bytes = count * esize;
  const int n = comm->nranks;
  const int r = comm->rank;
  const char* send = static_cast<const char*>(sendbuff);
  char* recv = static_cast<char*>(recvbuff);

  if (n == 1) {
    if (count > 0 && sendbuff != recvbuff) memcpy(recvbuff, sendbuff, bytes);
    if (premul) VCCLCHECK(vccl::cpuScale(datatype, recvbuff, count, scalar));
    return vcclSuccess;
  }
  if (count == 0) return vcclSuccess;

  // Pipelined chain root+1 -> root+2 -> ... -> root: each rank folds its
  // contribution into the partial flowing past, piece by piece, so the whole
  // chain works concurrently after a one-piece ramp-up.
  const bool isRoot = r == root;
  const bool isHead = r == (root + 1) % n;
  char* scratch = comm->scratchAt(kPipeBytes);
  vccl::ScopedReg reg(comm);
  if (isHead && !premul) VCCLCHECK(reg.add(sendbuff, bytes));

  for (size_t off = 0; off < bytes; off += kPipeBytes) {
    const size_t piece = std::min(kPipeBytes, bytes - off);
    const size_t pieceCount = piece / esize;
    if (isHead) {
      const char* out = send + off;
      if (premul) {
        VCCLCHECK(vccl::cpuScaleCopy(datatype, scratch, out, pieceCount,
                                     scalar));
        out = scratch;
      }
      VCCLCHECK(comm->transport->send(comm->next(), out, piece));
    } else if (isRoot) {
      VCCLCHECK(comm->transport->recv(comm->prev(), scratch, piece));
      if (premul) {
        VCCLCHECK(vccl::cpuScaleCopy(datatype, recv + off, send + off,
                                     pieceCount, scalar));
      } else if (recv + off != send + off) {
        memcpy(recv + off, send + off, piece);
      }
      VCCLCHECK(vccl::cpuReduce(datatype, op, recv + off, scratch,
                                pieceCount));
      if (op == vcclAvg) {
        VCCLCHECK(vccl::cpuScale(datatype, recv + off, pieceCount, 1.0 / n));
      }
    } else {
      VCCLCHECK(comm->transport->recv(comm->prev(), scratch, piece));
      if (premul) {
        VCCLCHECK(vccl::cpuScaledAccumulate(datatype, scratch, send + off,
                                            pieceCount, scalar));
      } else {
        VCCLCHECK(vccl::cpuReduce(datatype, op, scratch, send + off,
                                  pieceCount));
      }
      VCCLCHECK(comm->transport->send(comm->next(), scratch, piece));
    }
  }
  return vcclSuccess;
}

vcclResult_t doBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                         vcclDataType_t datatype, int root, vcclComm_t comm) {
  const size_t bytes = count * vcclTypeSize(datatype);
  char* recv = static_cast<char*>(recvbuff);

  if (comm->rank == root && sendbuff != recvbuff)
    memcpy(recv, sendbuff, bytes);
  if (comm->nranks == 1 || count == 0) return vcclSuccess;
  vccl::ScopedReg reg(comm);
  VCCLCHECK(reg.add(recvbuff, bytes));

  // Pipelined ring: root -> root+1 -> ... -> root-1, streamed in pieces so
  // downstream ranks start forwarding before the full message arrives.
  const bool isRoot = comm->rank == root;
  const bool isTail = comm->next() == root;
  for (size_t off = 0; off < bytes; off += kPipeBytes) {
    size_t piece = std::min(kPipeBytes, bytes - off);
    if (isRoot) {
      VCCLCHECK(comm->transport->send(comm->next(), recv + off, piece));
    } else {
      VCCLCHECK(comm->transport->recv(comm->prev(), recv + off, piece));
      if (!isTail) {
        VCCLCHECK(comm->transport->send(comm->next(), recv + off, piece));
      }
    }
  }
  return vcclSuccess;
}

vcclResult_t doGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                      vcclDataType_t datatype, int root, vcclComm_t comm) {
  const size_t chunkBytes = sendcount * vcclTypeSize(datatype);
  if (comm->nranks == 1 || sendcount == 0) {
    if (sendcount > 0 && sendbuff != recvbuff)
      memcpy(recvbuff, sendbuff, chunkBytes);
    return vcclSuccess;
  }
  vccl::ScopedReg reg(comm);
  if (comm->rank != root) {
    VCCLCHECK(reg.add(sendbuff, chunkBytes));
    return comm->transport->send(root, sendbuff, chunkBytes);
  }
  char* recv = static_cast<char*>(recvbuff);
  VCCLCHECK(reg.add(recvbuff, comm->nranks * chunkBytes));
  if (sendbuff != recv + root * chunkBytes) {
    memcpy(recv + root * chunkBytes, sendbuff, chunkBytes);
  }
  // Receive in ring order starting after root: senders are all blocked on
  // their send to root, so any fixed order drains them.
  for (int s = 1; s < comm->nranks; s++) {
    int peer = (root + s) % comm->nranks;
    VCCLCHECK(comm->transport->recv(peer, recv + peer * chunkBytes,
                                    chunkBytes));
  }
  return vcclSuccess;
}

vcclResult_t doScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                       vcclDataType_t datatype, int root, vcclComm_t comm) {
  const size_t chunkBytes = recvcount * vcclTypeSize(datatype);
  if (comm->nranks == 1 || recvcount == 0) {
    if (recvcount > 0 && sendbuff != recvbuff)
      memcpy(recvbuff, sendbuff, chunkBytes);
    return vcclSuccess;
  }
  vccl::ScopedReg reg(comm);
  if (comm->rank != root) {
    VCCLCHECK(reg.add(recvbuff, chunkBytes));
    return comm->transport->recv(root, recvbuff, chunkBytes);
  }
  const char* send = static_cast<const char*>(sendbuff);
  VCCLCHECK(reg.add(sendbuff, comm->nranks * chunkBytes));
  if (recvbuff != send + root * chunkBytes) {
    memcpy(recvbuff, send + root * chunkBytes, chunkBytes);
  }
  for (int s = 1; s < comm->nranks; s++) {
    int peer = (root + s) % comm->nranks;
    VCCLCHECK(comm->transport->send(peer, send + peer * chunkBytes,
                                    chunkBytes));
  }
  return vcclSuccess;
}

// Pairwise-exchange all-to-all: at step s every rank sends its block for
// (r+s)%n while receiving its block from (r-s)%n — globally deadlock-free
// and bandwidth-balanced. Counts/displacements are in elements.
vcclResult_t doAllToAllv(const void* sendbuff, const size_t* sendcounts,
                         const size_t* sdispls, void* recvbuff,
                         const size_t* recvcounts, const size_t* rdispls,
                         vcclDataType_t datatype, vcclComm_t comm) {
  const size_t esize = vcclTypeSize(datatype);
  const int n = comm->nranks;
  const int r = comm->rank;
  const char* send = static_cast<const char*>(sendbuff);
  char* recv = static_cast<char*>(recvbuff);

  vccl::ScopedReg reg(comm);
  size_t sendEnd = 0, recvEnd = 0;
  for (int p = 0; p < n; p++) {
    sendEnd = std::max(sendEnd, (sdispls[p] + sendcounts[p]) * esize);
    recvEnd = std::max(recvEnd, (rdispls[p] + recvcounts[p]) * esize);
  }
  VCCLCHECK(reg.add(sendbuff, sendEnd));
  VCCLCHECK(reg.add(recvbuff, recvEnd));

  if (sendcounts[r] > 0 &&
      recv + rdispls[r] * esize != send + sdispls[r] * esize) {
    memcpy(recv + rdispls[r] * esize, send + sdispls[r] * esize,
           sendcounts[r] * esize);
  }
  for (int s = 1; s < n; s++) {
    const int to = (r + s) % n;
    const int from = (r - s + n) % n;
    VCCLCHECK(comm->transport->sendRecv(
        to, send + sdispls[to] * esize, sendcounts[to] * esize, from,
        recv + rdispls[from] * esize, recvcounts[from] * esize));
  }
  return vcclSuccess;
}

// Queue inside groups, run immediately otherwise; record failures for
// vcclCommGetAsyncError either way.
vcclResult_t runOp(vcclComm_t comm, std::function<vcclResult_t()> fn) {
  return vccl::enqueueOrRun(comm, [comm, fn = std::move(fn)] {
    vcclResult_t res = fn();
    comm->noteError(res);
    return res;
  });
}

}  // namespace

extern "C" {

vcclResult_t vcclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, vcclDataType_t datatype,
                           vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  return runOp(comm, [=] {
    return doAllGather(sendbuff, recvbuff, sendcount, datatype, comm);
  });
}

vcclResult_t vcclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, vcclDataType_t datatype,
                               vcclRedOp_t op, vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  bool premul;
  double scalar = 1.0;
  VCCLCHECK(resolveOp(comm, datatype, &op, &premul, &scalar));
  return runOp(comm, [=] {
    return doReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                           premul, scalar, comm);
  });
}

vcclResult_t vcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           vcclDataType_t datatype, vcclRedOp_t op,
                           vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  bool premul;
  double scalar = 1.0;
  VCCLCHECK(resolveOp(comm, datatype, &op, &premul, &scalar));
  return runOp(comm, [=] {
    return doAllReduce(sendbuff, recvbuff, count, datatype, op, premul,
                       scalar, comm);
  });
}

vcclResult_t vcclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        vcclDataType_t datatype, vcclRedOp_t op, int root,
                        vcclComm_t comm) {
  VCCLCHECK(checkComm(comm));
  if (sendbuff == nullptr || vcclTypeSize(datatype) == 0 || root < 0 ||
      root >= comm->nranks)
    return vcclInvalidArgument;
  if (comm->rank == root && recvbuff == nullptr) return vcclInvalidArgument;
  bool premul;
  double scalar = 1.0;
  VCCLCHECK(resolveOp(comm, datatype, &op, &premul, &scalar));
  return runOp(comm, [=] {
    return doReduce(sendbuff, recvbuff, count, datatype, op, premul, scalar,
                    root, comm);
  });
}

vcclResult_t vcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           vcclDataType_t datatype, int root,
                           vcclComm_t comm) {
  VCCLCHECK(checkComm(comm));
  if (recvbuff == nullptr || vcclTypeSize(datatype) == 0 || root < 0 ||
      root >= comm->nranks)
    return vcclInvalidArgument;
  // sendbuff is only significant at root (NCCL semantics).
  if (comm->rank == root && sendbuff == nullptr) return vcclInvalidArgument;
  return runOp(comm, [=] {
    return doBroadcast(sendbuff, recvbuff, count, datatype, root, comm);
  });
}

vcclResult_t vcclBcast(void* buff, size_t count, vcclDataType_t datatype,
                       int root, vcclComm_t comm) {
  return vcclBroadcast(buff, buff, count, datatype, root, comm);
}

vcclResult_t vcclGather(const void* sendbuff, void* recvbuff,
                        size_t sendcount, vcclDataType_t datatype, int root,
                        vcclComm_t comm) {
  VCCLCHECK(checkComm(comm));
  if (sendbuff == nullptr || vcclTypeSize(datatype) == 0 || root < 0 ||
      root >= comm->nranks)
    return vcclInvalidArgument;
  if (comm->rank == root && recvbuff == nullptr) return vcclInvalidArgument;
  return runOp(comm, [=] {
    return doGather(sendbuff, recvbuff, sendcount, datatype, root, comm);
  });
}

vcclResult_t vcclScatter(const void* sendbuff, void* recvbuff,
                         size_t recvcount, vcclDataType_t datatype, int root,
                         vcclComm_t comm) {
  VCCLCHECK(checkComm(comm));
  if (recvbuff == nullptr || vcclTypeSize(datatype) == 0 || root < 0 ||
      root >= comm->nranks)
    return vcclInvalidArgument;
  if (comm->rank == root && sendbuff == nullptr) return vcclInvalidArgument;
  return runOp(comm, [=] {
    return doScatter(sendbuff, recvbuff, recvcount, datatype, root, comm);
  });
}

vcclResult_t vcclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                          vcclDataType_t datatype, vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  const int n = comm->nranks;
  return runOp(comm, [=] {
    std::vector<size_t> counts(n, count), displs(n);
    for (int i = 0; i < n; i++) displs[i] = i * count;
    return doAllToAllv(sendbuff, counts.data(), displs.data(), recvbuff,
                       counts.data(), displs.data(), datatype, comm);
  });
}

vcclResult_t vcclAllToAllv(const void* sendbuff, const size_t sendcounts[],
                           const size_t sdispls[], void* recvbuff,
                           const size_t recvcounts[], const size_t rdispls[],
                           vcclDataType_t datatype, vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  if (sendcounts == nullptr || sdispls == nullptr || recvcounts == nullptr ||
      rdispls == nullptr)
    return vcclInvalidArgument;
  // Copy count tables: with group semantics the call runs at vcclGroupEnd,
  // possibly after the caller's arrays went out of scope.
  std::vector<size_t> sc(sendcounts, sendcounts + comm->nranks);
  std::vector<size_t> sd(sdispls, sdispls + comm->nranks);
  std::vector<size_t> rc(recvcounts, recvcounts + comm->nranks);
  std::vector<size_t> rd(rdispls, rdispls + comm->nranks);
  return runOp(comm, [=, sc = std::move(sc), sd = std::move(sd),
                      rc = std::move(rc), rd = std::move(rd)] {
    return doAllToAllv(sendbuff, sc.data(), sd.data(), recvbuff, rc.data(),
                       rd.data(), datatype, comm);
  });
}

vcclResult_t vcclSend(const void* sendbuff, size_t count,
                      vcclDataType_t datatype, int peer, vcclComm_t comm) {
  VCCLCHECK(checkComm(comm));
  if (sendbuff == nullptr || vcclTypeSize(datatype) == 0 || peer < 0 ||
      peer >= comm->nranks || peer == comm->rank)
    return vcclInvalidArgument;
  vccl::P2pOp op{true, peer, const_cast<void*>(sendbuff),
                 count * vcclTypeSize(datatype)};
  return vccl::enqueueOrRunP2p(comm, op);
}

vcclResult_t vcclRecv(void* recvbuff, size_t count, vcclDataType_t datatype,
                      int peer, vcclComm_t comm) {
  VCCLCHECK(checkComm(comm));
  if (recvbuff == nullptr || vcclTypeSize(datatype) == 0 || peer < 0 ||
      peer >= comm->nranks || peer == comm->rank)
    return vcclInvalidArgument;
  vccl::P2pOp op{false, peer, recvbuff, count * vcclTypeSize(datatype)};
  return vccl::enqueueOrRunP2p(comm, op);
}

}  // extern "C"
