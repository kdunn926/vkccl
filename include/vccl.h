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
#define VCCL_MINOR 2
#define VCCL_PATCH 0
/* NCCL-style encoded version: MAJOR*10000 + MINOR*100 + PATCH. */
#define VCCL_VERSION_CODE (VCCL_MAJOR * 10000 + VCCL_MINOR * 100 + VCCL_PATCH)

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
  /* Values >= vcclNumOps are communicator-local ops created with
   * vcclRedOpCreatePreMulSum. */
} vcclRedOp_t;

typedef enum {
  vcclScalarDevice = 0,
  vcclScalarHostImmediate = 1
} vcclScalarResidence_t;

/* Communicator configuration (ncclConfig_t analog). Zero/default-initialize
 * with VCCL_CONFIG_INITIALIZER. vccl is always blocking; fields beyond
 * `size` are currently accepted and ignored, reserved for compatibility. */
typedef struct {
  size_t size;
  int blocking;
  int splitShare;
} vcclConfig_t;

#define VCCL_CONFIG_INITIALIZER { sizeof(vcclConfig_t), 1, 0 }

/* Color value for vcclCommSplit ranks that leave the communicator. */
#define VCCL_SPLIT_NOCOLOR (-1)

/* Runtime version, encoded like VCCL_VERSION_CODE. */
vcclResult_t vcclGetVersion(int* version);

/* Create a unique id for a new communicator. Must be called by the process
 * that will become rank 0; it binds a listening socket whose address is
 * embedded in the id. Distribute the id to all ranks out-of-band. */
vcclResult_t vcclGetUniqueId(vcclUniqueId* uniqueId);

/* Collectively create a communicator. Blocks until all nranks have joined. */
vcclResult_t vcclCommInitRank(vcclComm_t* comm, int nranks,
                              vcclUniqueId commId, int rank);

/* As vcclCommInitRank, with a config. vccl is always blocking; the config
 * is validated and otherwise ignored. */
vcclResult_t vcclCommInitRankConfig(vcclComm_t* comm, int nranks,
                                    vcclUniqueId commId, int rank,
                                    vcclConfig_t* config);

/* Create ndev communicators in the calling process, one per "device"
 * (ranks 0..ndev-1). devlist may be NULL. Collectives on these comms must
 * be issued inside vcclGroupStart/vcclGroupEnd, since a plain blocking call
 * on one comm cannot progress its in-process peers. */
vcclResult_t vcclCommInitAll(vcclComm_t* comms, int ndev, const int* devlist);

/* Collectively split a communicator into subgroups by color (equal colors
 * end up in the same new communicator, ordered by key then parent rank).
 * Ranks passing VCCL_SPLIT_NOCOLOR receive *newcomm == NULL. */
vcclResult_t vcclCommSplit(vcclComm_t comm, int color, int key,
                           vcclComm_t* newcomm, vcclConfig_t* config);

/* Finalize outstanding operations. vccl operations are blocking, so this
 * only flushes state; provided for NCCL API parity. */
vcclResult_t vcclCommFinalize(vcclComm_t comm);

vcclResult_t vcclCommDestroy(vcclComm_t comm);

/* Unblock any pending operations with an error and mark the communicator
 * dead. Safe to call from another thread; follow with vcclCommDestroy. */
vcclResult_t vcclCommAbort(vcclComm_t comm);

/* Last error observed by operations on this communicator. */
vcclResult_t vcclCommGetAsyncError(vcclComm_t comm, vcclResult_t* asyncError);

vcclResult_t vcclCommCount(const vcclComm_t comm, int* count);
vcclResult_t vcclCommUserRank(const vcclComm_t comm, int* rank);
/* Device index backing this communicator (always 0: one device per node). */
vcclResult_t vcclCommDevice(const vcclComm_t comm, int* device);

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

/* Reduces count elements across all ranks into root's recvbuff (recvbuff is
 * only significant at root). In-place when sendbuff == recvbuff at root. */
vcclResult_t vcclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        vcclDataType_t datatype, vcclRedOp_t op, int root,
                        vcclComm_t comm);

/* Copies count elements from root's sendbuff to every rank's recvbuff.
 * sendbuff is only significant at root. */
vcclResult_t vcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           vcclDataType_t datatype, int root, vcclComm_t comm);

/* Legacy in-place broadcast (ncclBcast analog). */
vcclResult_t vcclBcast(void* buff, size_t count, vcclDataType_t datatype,
                       int root, vcclComm_t comm);

/* Every rank sends sendcount elements to root; root's recvbuff receives
 * nranks*sendcount elements ordered by rank (recvbuff significant only at
 * root). In-place at root when sendbuff == recvbuff + rank*sendcount*esize. */
vcclResult_t vcclGather(const void* sendbuff, void* recvbuff,
                        size_t sendcount, vcclDataType_t datatype, int root,
                        vcclComm_t comm);

/* Root sends the r-th recvcount-element block of sendbuff to rank r
 * (sendbuff significant only at root). */
vcclResult_t vcclScatter(const void* sendbuff, void* recvbuff,
                         size_t recvcount, vcclDataType_t datatype, int root,
                         vcclComm_t comm);

/* RCCL extension: every rank sends count elements to every other rank.
 * sendbuff/recvbuff hold nranks*count elements, block r for rank r. */
vcclResult_t vcclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                          vcclDataType_t datatype, vcclComm_t comm);

/* RCCL extension: variable all-to-all. counts/displacements are in
 * elements; block sent to (received from) rank r starts at sdispls[r]
 * (rdispls[r]). sendcounts[r] must equal rank r's recvcounts[rank]. */
vcclResult_t vcclAllToAllv(const void* sendbuff, const size_t sendcounts[],
                           const size_t sdispls[], void* recvbuff,
                           const size_t recvcounts[], const size_t rdispls[],
                           vcclDataType_t datatype, vcclComm_t comm);

/* Point-to-point. Send/recv pairs must match; blocking outside groups. A
 * send+recv pair on the same communicator must be issued inside
 * vcclGroupStart/vcclGroupEnd so both directions progress concurrently. */
vcclResult_t vcclSend(const void* sendbuff, size_t count,
                      vcclDataType_t datatype, int peer, vcclComm_t comm);
vcclResult_t vcclRecv(void* recvbuff, size_t count, vcclDataType_t datatype,
                      int peer, vcclComm_t comm);

/* ── Group semantics ────────────────────────────────────────────────────
 * Operations issued between vcclGroupStart and vcclGroupEnd are queued and
 * executed by vcclGroupEnd: per communicator in issue order, with runs of
 * consecutive point-to-point operations progressed as one batch, and
 * different communicators progressed concurrently. Required for: send/recv
 * pairs on one communicator, and any use of multiple in-process
 * communicators (vcclCommInitAll). Groups nest; ops run at outermost end. */
vcclResult_t vcclGroupStart(void);
vcclResult_t vcclGroupEnd(void);

/* ── Custom reduction ops ───────────────────────────────────────────────
 * Creates op computing sum(scalar * input_i). scalar points to a value of
 * `datatype` (host immediate only). The op is local to comm; destroy before
 * the communicator. Float datatypes only. */
vcclResult_t vcclRedOpCreatePreMulSum(vcclRedOp_t* op, void* scalar,
                                      vcclDataType_t datatype,
                                      vcclScalarResidence_t residence,
                                      vcclComm_t comm);
vcclResult_t vcclRedOpDestroy(vcclRedOp_t op, vcclComm_t comm);

/* ── Memory ─────────────────────────────────────────────────────────────
 * Page-aligned allocation suitable for registration (ncclMemAlloc analog). */
vcclResult_t vcclMemAlloc(void** ptr, size_t size);
vcclResult_t vcclMemFree(void* ptr);

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

/* Human-readable detail of the calling thread's most recent vccl error.
 * The comm argument is accepted for NCCL parity and may be NULL. */
const char* vcclGetLastError(vcclComm_t comm);

/* Size in bytes of one element of datatype, or 0 if invalid. */
size_t vcclTypeSize(vcclDataType_t datatype);

#ifdef __cplusplus
}
#endif

#endif /* VCCL_H_ */
