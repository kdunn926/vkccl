/* SPDX-License-Identifier: Apache-2.0
 *
 * vccl — Vulkan Collective Communications Library.
 *
 * NCCL-style collectives for multi-node, single-device-per-node clusters,
 * over RDMA (RoCE v2 via libibverbs) with a TCP fallback transport.
 *
 * v1 semantics: all calls are host-blocking and operate on host-accessible
 * memory. On unified-memory GPUs (APUs) this includes mapped device-local
 * Vulkan allocations. There is no stream argument yet; async submission
 * arrives with the Vulkan-buffer path.
 */
#ifndef VCCL_H_
#define VCCL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VCCL_MAJOR 0
#define VCCL_MINOR 1
#define VCCL_PATCH 0

#define VCCL_UNIQUE_ID_BYTES 128
typedef struct {
  char internal[VCCL_UNIQUE_ID_BYTES];
} vcclUniqueId;

typedef struct vcclComm* vcclComm_t;

typedef enum {
  vcclSuccess = 0,
  vcclUnhandledError = 1,
  vcclSystemError = 2,
  vcclInternalError = 3,
  vcclInvalidArgument = 4,
  vcclInvalidUsage = 5,
  vcclRemoteError = 6,
  vcclNumResults = 7
} vcclResult_t;

typedef enum {
  vcclInt8 = 0,
  vcclUint8 = 1,
  vcclInt32 = 2,
  vcclUint32 = 3,
  vcclInt64 = 4,
  vcclUint64 = 5,
  vcclFloat16 = 6,
  vcclFloat32 = 7,
  vcclFloat64 = 8,
  vcclBfloat16 = 9,
  vcclNumTypes = 10
} vcclDataType_t;

typedef enum {
  vcclSum = 0,
  vcclProd = 1,
  vcclMax = 2,
  vcclMin = 3,
  vcclAvg = 4,
  vcclNumOps = 5
} vcclRedOp_t;

/* Create a unique id for a new communicator. Must be called by the process
 * that will become rank 0; it binds a listening socket whose address is
 * embedded in the id. Distribute the id to all ranks out-of-band. */
vcclResult_t vcclGetUniqueId(vcclUniqueId* uniqueId);

/* Collectively create a communicator. Blocks until all nranks have joined. */
vcclResult_t vcclCommInitRank(vcclComm_t* comm, int nranks,
                              vcclUniqueId commId, int rank);

vcclResult_t vcclCommDestroy(vcclComm_t comm);
vcclResult_t vcclCommCount(const vcclComm_t comm, int* count);
vcclResult_t vcclCommUserRank(const vcclComm_t comm, int* rank);

/* Gathers sendcount elements from every rank; recvbuff holds
 * nranks*sendcount elements, ordered by rank. In-place is supported when
 * sendbuff == recvbuff + rank*sendcount*elemsize. */
vcclResult_t vcclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, vcclDataType_t datatype,
                           vcclComm_t comm);

/* Reduces nranks*recvcount elements from every rank's sendbuff; rank r
 * receives the r-th recvcount-element block of the result. In-place is
 * supported when recvbuff == sendbuff + rank*recvcount*elemsize. */
vcclResult_t vcclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, vcclDataType_t datatype,
                               vcclRedOp_t op, vcclComm_t comm);

/* Reduces count elements across all ranks into every rank's recvbuff.
 * In-place is supported when sendbuff == recvbuff. */
vcclResult_t vcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           vcclDataType_t datatype, vcclRedOp_t op,
                           vcclComm_t comm);

/* Copies count elements from root's sendbuff to every rank's recvbuff. */
vcclResult_t vcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           vcclDataType_t datatype, int root, vcclComm_t comm);

/* Point-to-point. Send/recv pairs must match; blocking. */
vcclResult_t vcclSend(const void* sendbuff, size_t count,
                      vcclDataType_t datatype, int peer, vcclComm_t comm);
vcclResult_t vcclRecv(void* recvbuff, size_t count, vcclDataType_t datatype,
                      int peer, vcclComm_t comm);

/* ── Memory registration ────────────────────────────────────────────────
 * Optional: collectives work on any host-accessible memory, but registering
 * buffers up front pins them with the RDMA transport once instead of on
 * first use, and is required for imported GPU memory. */

typedef void* vcclMemHandle_t;

/* Pre-register a host-accessible buffer with the communicator's transport. */
vcclResult_t vcclCommRegister(vcclComm_t comm, void* buff, size_t bytes,
                              vcclMemHandle_t* handle);
vcclResult_t vcclCommDeregister(vcclComm_t comm, vcclMemHandle_t handle);

/* Import a dmabuf (e.g. a Vulkan allocation exported with
 * VK_EXT_external_memory_dma_buf via vkGetMemoryFdKHR) and register it.
 * The buffer is mapped into the process; *ptr receives the mapping, which
 * is what you pass to the collectives. The RDMA transport registers the
 * dmabuf directly (ibv_reg_dmabuf_mr) when the NIC supports it, falling
 * back to registering the CPU mapping. vccl does not take ownership of fd;
 * the caller may close it after this returns. Linux only. */
vcclResult_t vcclCommRegisterDmaBuf(vcclComm_t comm, int fd, uint64_t offset,
                                    size_t bytes, void** ptr,
                                    vcclMemHandle_t* handle);

const char* vcclGetErrorString(vcclResult_t result);

/* Size in bytes of one element of datatype, or 0 if invalid. */
size_t vcclTypeSize(vcclDataType_t datatype);

#ifdef __cplusplus
}
#endif

#endif /* VCCL_H_ */
