/* SPDX-License-Identifier: Apache-2.0
 *
 * Ring collectives.
 *
 * Data is partitioned into nranks chunks. With rank r, n = nranks:
 *
 * all-gather (rank r starts owning chunk r): at step s = 0..n-2, send chunk
 * (r-s) mod n to next, receive chunk (r-s-1) mod n from prev.
 *
 * reduce-scatter over a full-size accumulator buffer: at step s, send
 * accumulator chunk (r-s) mod n, receive into scratch, accumulate scratch
 * into chunk (r-s-1) mod n. After n-1 steps rank r holds the fully reduced
 * chunk (r+1) mod n.
 *
 * all-reduce = reduce-scatter + all-gather over recvbuff in place, using a
 * chunk table that tolerates non-divisible counts (last chunks smaller or
 * empty). Each step's send and receive proceed concurrently via
 * Transport::sendRecv.
 */
#include <algorithm>
#include <cstring>
#include <vector>

#include "core/comm.h"
#include "core/reduce.h"
#include "util/log.h"

namespace {

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

vcclResult_t checkArgs(const void* sendbuff, const void* recvbuff,
                       vcclDataType_t dt, vcclComm_t comm) {
  if (comm == nullptr) return vcclInvalidArgument;
  if (sendbuff == nullptr || recvbuff == nullptr) return vcclInvalidArgument;
  if (vcclTypeSize(dt) == 0) return vcclInvalidArgument;
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

// In-place ring all-gather over `data`. startChunk(r) is the chunk rank r
// holds at entry; for plain all-gather that is r, after reduce-scatter it is
// (r+1)%n.
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

}  // namespace

extern "C" {

vcclResult_t vcclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, vcclDataType_t datatype,
                           vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  const size_t esize = vcclTypeSize(datatype);
  const size_t chunkBytes = sendcount * esize;
  char* recv = static_cast<char*>(recvbuff);

  if (sendbuff != recv + comm->rank * chunkBytes) {
    memcpy(recv + comm->rank * chunkBytes, sendbuff, chunkBytes);
  }
  if (comm->nranks == 1 || sendcount == 0) return vcclSuccess;

  std::vector<Chunk> chunks(comm->nranks);
  for (int i = 0; i < comm->nranks; i++)
    chunks[i] = {i * chunkBytes, chunkBytes};
  return ringAllGather(comm, recv, chunks, 0);
}

vcclResult_t vcclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, vcclDataType_t datatype,
                               vcclRedOp_t op, vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  if (op == vcclAvg && vcclTypeSize(datatype) != 0 &&
      datatype != vcclFloat16 && datatype != vcclBfloat16 &&
      datatype != vcclFloat32 && datatype != vcclFloat64) {
    return vcclInvalidArgument;
  }
  const size_t esize = vcclTypeSize(datatype);
  const size_t chunkBytes = recvcount * esize;
  const int n = comm->nranks;
  const int r = comm->rank;
  const char* send = static_cast<const char*>(sendbuff);

  if (n == 1 || recvcount == 0) {
    if (recvcount > 0 && sendbuff != recvbuff)
      memcpy(recvbuff, sendbuff, chunkBytes);
    return vcclSuccess;
  }

  // Traveling-partial ring: the partial sum of a chunk hops around the ring,
  // gaining each rank's contribution. Only 2 chunk-sized scratch buffers are
  // needed (receive into one while sending the other) instead of a full-size
  // accumulator.
  char* scratch = comm->scratchAt(2 * chunkBytes);
  char* bufA = scratch;
  char* bufB = scratch + chunkBytes;

  // Step 0 sends our raw chunk (r-1)%n; step s receives partial chunk
  // (r-s-2)%n, which after accumulating our contribution is exactly what
  // step s+1 sends.
  const char* sendPtr = send + ((r - 1 + n) % n) * chunkBytes;
  for (int s = 0; s < n - 1; s++) {
    char* recvPtr = (s % 2 == 0) ? bufA : bufB;
    VCCLCHECK(comm->transport->sendRecv(comm->next(), sendPtr, chunkBytes,
                                        comm->prev(), recvPtr, chunkBytes));
    const int rc = (r - s - 2 + 2 * n) % n;
    if (s == n - 2) {
      // Final hop: rc == r. recvbuff = local chunk r (op) incoming partial.
      if (recvbuff != send + r * chunkBytes)
        memcpy(recvbuff, send + r * chunkBytes, chunkBytes);
      VCCLCHECK(vccl::cpuReduce(datatype, op, recvbuff, recvPtr, recvcount));
    } else {
      VCCLCHECK(vccl::cpuReduce(datatype, op, recvPtr,
                                send + rc * chunkBytes, recvcount));
      sendPtr = recvPtr;
    }
  }
  if (op == vcclAvg) {
    VCCLCHECK(vccl::cpuScale(datatype, recvbuff, recvcount, 1.0 / n));
  }
  return vcclSuccess;
}

vcclResult_t vcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           vcclDataType_t datatype, vcclRedOp_t op,
                           vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  if (op == vcclAvg && datatype != vcclFloat16 && datatype != vcclBfloat16 &&
      datatype != vcclFloat32 && datatype != vcclFloat64) {
    return vcclInvalidArgument;
  }
  const size_t esize = vcclTypeSize(datatype);
  char* recv = static_cast<char*>(recvbuff);
  if (sendbuff != recvbuff) memcpy(recv, sendbuff, count * esize);
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

vcclResult_t vcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           vcclDataType_t datatype, int root,
                           vcclComm_t comm) {
  VCCLCHECK(checkArgs(sendbuff, recvbuff, datatype, comm));
  if (root < 0 || root >= comm->nranks) return vcclInvalidArgument;
  const size_t bytes = count * vcclTypeSize(datatype);
  char* recv = static_cast<char*>(recvbuff);

  if (comm->rank == root && sendbuff != recvbuff)
    memcpy(recv, sendbuff, bytes);
  if (comm->nranks == 1 || count == 0) return vcclSuccess;

  // Pipelined ring: root -> root+1 -> ... -> root-1, streamed in pieces so
  // downstream ranks start forwarding before the full message arrives.
  constexpr size_t kPipeBytes = 1 << 19;
  const bool isRoot = comm->rank == root;
  const bool isTail = comm->next() == root;
  for (size_t off = 0; off < bytes; off += kPipeBytes) {
    size_t piece = std::min(kPipeBytes, bytes - off);
    if (isRoot) {
      VCCLCHECK(comm->transport->send(comm->next(), recv + off, piece));
    } else if (isTail) {
      VCCLCHECK(comm->transport->recv(comm->prev(), recv + off, piece));
    } else {
      VCCLCHECK(comm->transport->recv(comm->prev(), recv + off, piece));
      VCCLCHECK(comm->transport->send(comm->next(), recv + off, piece));
    }
  }
  return vcclSuccess;
}

vcclResult_t vcclSend(const void* sendbuff, size_t count,
                      vcclDataType_t datatype, int peer, vcclComm_t comm) {
  if (comm == nullptr || sendbuff == nullptr || vcclTypeSize(datatype) == 0 ||
      peer < 0 || peer >= comm->nranks || peer == comm->rank)
    return vcclInvalidArgument;
  return comm->transport->send(peer, sendbuff, count * vcclTypeSize(datatype));
}

vcclResult_t vcclRecv(void* recvbuff, size_t count, vcclDataType_t datatype,
                      int peer, vcclComm_t comm) {
  if (comm == nullptr || recvbuff == nullptr || vcclTypeSize(datatype) == 0 ||
      peer < 0 || peer >= comm->nranks || peer == comm->rank)
    return vcclInvalidArgument;
  return comm->transport->recv(peer, recvbuff, count * vcclTypeSize(datatype));
}

}  // extern "C"
