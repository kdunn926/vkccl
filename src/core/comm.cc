/* SPDX-License-Identifier: Apache-2.0 */
#include "core/comm.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "util/log.h"

namespace {

// vcclMemAlloc bookkeeping so vcclMemFree doesn't need a size.
std::mutex g_memMutex;
std::map<void*, size_t> g_memAllocs;

vcclResult_t validateConfig(const vcclConfig_t* config) {
  if (config == nullptr) return vcclSuccess;
  if (config->size < sizeof(size_t)) return vcclInvalidArgument;
  return vcclSuccess;
}

}  // namespace

extern "C" {

vcclResult_t vcclGetVersion(int* version) {
  if (version == nullptr) return vcclInvalidArgument;
  *version = VCCL_VERSION_CODE;
  return vcclSuccess;
}

const char* vcclGetLastError(vcclComm_t comm) {
  if (comm != nullptr) {
    std::lock_guard<std::mutex> lock(comm->errMutex);
    if (!comm->lastError.empty()) return comm->lastError.c_str();
  }
  return vccl::lastErrorString();
}

vcclResult_t vcclGetUniqueId(vcclUniqueId* uniqueId) {
  if (uniqueId == nullptr) return vcclInvalidArgument;
  return vccl::uniqueIdCreate(uniqueId);
}

vcclResult_t vcclCommInitRank(vcclComm_t* comm, int nranks,
                              vcclUniqueId commId, int rank) {
  if (comm == nullptr || nranks < 1 || rank < 0 || rank >= nranks)
    return vcclInvalidArgument;

  auto c = std::make_unique<vcclComm>();
  c->rank = rank;
  c->nranks = nranks;
  VCCLCHECK(vccl::Bootstrap::init(commId, rank, nranks, &c->bootstrap));
  if (nranks > 1) {
    VCCLCHECK(vccl::transportCreate(c->bootstrap.get(), &c->transport));
  }
  VCCL_INFO("comm initialized: rank %d/%d, transport %s", rank, nranks,
            c->transport ? c->transport->name() : "none");
  *comm = c.release();
  return vcclSuccess;
}

vcclResult_t vcclCommInitRankConfig(vcclComm_t* comm, int nranks,
                                    vcclUniqueId commId, int rank,
                                    vcclConfig_t* config) {
  VCCLCHECK(validateConfig(config));
  // vccl is always blocking; all other config fields are parity-only.
  return vcclCommInitRank(comm, nranks, commId, rank);
}

vcclResult_t vcclCommInitAll(vcclComm_t* comms, int ndev,
                             const int* devlist) {
  (void)devlist;  // one device per node; indices are informational
  if (comms == nullptr || ndev < 1) return vcclInvalidArgument;
  vcclUniqueId id;
  VCCLCHECK(vcclGetUniqueId(&id));
  // Init is collective: every rank's bootstrap must progress at once.
  std::vector<std::thread> threads;
  std::vector<vcclResult_t> results(ndev, vcclSuccess);
  for (int i = 0; i < ndev; i++) {
    threads.emplace_back([&, i] {
      results[i] = vcclCommInitRank(&comms[i], ndev, id, i);
    });
  }
  for (auto& t : threads) t.join();
  for (int i = 0; i < ndev; i++) {
    if (results[i] != vcclSuccess) return results[i];
  }
  return vcclSuccess;
}

vcclResult_t vcclCommSplit(vcclComm_t comm, int color, int key,
                           vcclComm_t* newcomm, vcclConfig_t* config) {
  if (comm == nullptr || newcomm == nullptr) return vcclInvalidArgument;
  VCCLCHECK(validateConfig(config));
  *newcomm = nullptr;

  // Exchange (color, key) and, from each subgroup's leader, a fresh unique
  // id. Two bootstrap allgathers: membership first (so leaders know they
  // lead), then ids.
  struct SplitInfo {
    int32_t color;
    int32_t key;
  };
  std::vector<SplitInfo> info(comm->nranks);
  info[comm->rank] = {color, key};
  VCCLCHECK(comm->bootstrap->allGather(info.data(), sizeof(SplitInfo)));

  // Members of my color ordered by (key, parent rank): that order is the
  // new rank order, and member 0 creates the new unique id. Negative color
  // (VCCL_SPLIT_NOCOLOR) opts out but still participates in the exchanges.
  std::vector<std::pair<int32_t, int>> members;  // (key, parent rank)
  if (color >= 0) {
    for (int r = 0; r < comm->nranks; r++) {
      if (info[r].color == color) members.emplace_back(info[r].key, r);
    }
    std::sort(members.begin(), members.end());
  }
  int newRank = -1;
  for (size_t i = 0; i < members.size(); i++) {
    if (members[i].second == comm->rank) newRank = static_cast<int>(i);
  }

  std::vector<vcclUniqueId> ids(comm->nranks);
  memset(ids.data(), 0, sizeof(vcclUniqueId) * comm->nranks);
  if (newRank == 0) {
    VCCLCHECK(vcclGetUniqueId(&ids[comm->rank]));
  }
  VCCLCHECK(comm->bootstrap->allGather(ids.data(), sizeof(vcclUniqueId)));

  if (color < 0) return vcclSuccess;
  return vcclCommInitRank(newcomm, static_cast<int>(members.size()),
                          ids[members[0].second], newRank);
}

vcclResult_t vcclCommFinalize(vcclComm_t comm) {
  // All vccl operations complete before returning; nothing outstanding.
  return comm == nullptr ? vcclInvalidArgument : vcclSuccess;
}

vcclResult_t vcclCommAbort(vcclComm_t comm) {
  if (comm == nullptr) return vcclInvalidArgument;
  comm->aborted.store(true, std::memory_order_relaxed);
  comm->noteError(vcclUnhandledError);
  if (comm->transport) comm->transport->abort();
  return vcclSuccess;
}

vcclResult_t vcclCommGetAsyncError(vcclComm_t comm,
                                   vcclResult_t* asyncError) {
  if (comm == nullptr || asyncError == nullptr) return vcclInvalidArgument;
  *asyncError = comm->asyncError.load(std::memory_order_relaxed);
  return vcclSuccess;
}

vcclResult_t vcclCommDestroy(vcclComm_t comm) {
  if (comm != nullptr) {
    std::lock_guard<std::mutex> lock(comm->regMutex);
    for (vcclComm::MemReg* reg : comm->registrations) {
      if (comm->transport) comm->transport->deregMr(reg->transportHandle);
      if (reg->mapped != nullptr) munmap(reg->mapped, reg->bytes);
      delete reg;
    }
  }
  delete comm;
  return vcclSuccess;
}

vcclResult_t vcclCommRegister(vcclComm_t comm, void* buff, size_t bytes,
                              vcclMemHandle_t* handle) {
  if (comm == nullptr || buff == nullptr || bytes == 0 || handle == nullptr)
    return vcclInvalidArgument;
  auto reg = std::make_unique<vcclComm::MemReg>();
  reg->bytes = bytes;
  std::lock_guard<std::mutex> lock(comm->regMutex);
  if (comm->transport) {
    VCCLCHECK(comm->transport->regMr(buff, bytes, &reg->transportHandle));
  }
  comm->registrations.push_back(reg.get());
  *handle = reg.release();
  return vcclSuccess;
}

vcclResult_t vcclCommRegisterDmaBuf(vcclComm_t comm, int fd, uint64_t offset,
                                    size_t bytes, void** ptr,
                                    vcclMemHandle_t* handle) {
  if (comm == nullptr || fd < 0 || bytes == 0 || ptr == nullptr ||
      handle == nullptr)
    return vcclInvalidArgument;
  // Collectives need CPU access for reduction kernels, so the dmabuf must be
  // mappable (true for APU/GTT and udmabuf memory; discrete VRAM needs the
  // GPU-direct path planned for M4).
  void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                      static_cast<off_t>(offset));
  if (mapped == MAP_FAILED) {
    VCCL_ERR("mmap of dmabuf fd %d failed: %s", fd, strerror(errno));
    return vcclSystemError;
  }
  auto reg = std::make_unique<vcclComm::MemReg>();
  reg->mapped = mapped;
  reg->bytes = bytes;
  std::lock_guard<std::mutex> lock(comm->regMutex);
  if (comm->transport) {
    vcclResult_t res = comm->transport->regDmaBuf(fd, offset, mapped, bytes,
                                                  &reg->transportHandle);
    if (res != vcclSuccess) {
      munmap(mapped, bytes);
      return res;
    }
  }
  comm->registrations.push_back(reg.get());
  *ptr = mapped;
  *handle = reg.release();
  return vcclSuccess;
}

vcclResult_t vcclCommDeregister(vcclComm_t comm, vcclMemHandle_t handle) {
  if (comm == nullptr || handle == nullptr) return vcclInvalidArgument;
  auto* reg = static_cast<vcclComm::MemReg*>(handle);
  std::lock_guard<std::mutex> lock(comm->regMutex);
  auto it = std::find(comm->registrations.begin(), comm->registrations.end(),
                      reg);
  if (it == comm->registrations.end()) return vcclInvalidArgument;
  comm->registrations.erase(it);
  if (comm->transport) comm->transport->deregMr(reg->transportHandle);
  if (reg->mapped != nullptr) munmap(reg->mapped, reg->bytes);
  delete reg;
  return vcclSuccess;
}

vcclResult_t vcclCommCount(const vcclComm_t comm, int* count) {
  if (comm == nullptr || count == nullptr) return vcclInvalidArgument;
  *count = comm->nranks;
  return vcclSuccess;
}

vcclResult_t vcclCommUserRank(const vcclComm_t comm, int* rank) {
  if (comm == nullptr || rank == nullptr) return vcclInvalidArgument;
  *rank = comm->rank;
  return vcclSuccess;
}

vcclResult_t vcclCommDevice(const vcclComm_t comm, int* device) {
  if (comm == nullptr || device == nullptr) return vcclInvalidArgument;
  *device = 0;  // one device per node
  return vcclSuccess;
}

vcclResult_t vcclRedOpCreatePreMulSum(vcclRedOp_t* op, void* scalar,
                                      vcclDataType_t datatype,
                                      vcclScalarResidence_t residence,
                                      vcclComm_t comm) {
  if (op == nullptr || scalar == nullptr || comm == nullptr)
    return vcclInvalidArgument;
  if (residence != vcclScalarHostImmediate) {
    VCCL_ERR("only vcclScalarHostImmediate scalars are supported");
    return vcclInvalidArgument;
  }
  double value;
  switch (datatype) {
    case vcclFloat32: value = *static_cast<float*>(scalar); break;
    case vcclFloat64: value = *static_cast<double*>(scalar); break;
    case vcclFloat16:
    case vcclBfloat16: {
      // Scalar arrives encoded as the op's datatype; widen to double.
      uint16_t bits = *static_cast<uint16_t*>(scalar);
      float f;
      if (datatype == vcclFloat16) {
        int exp = ((bits >> 10) & 0x1f) - 15;
        float mant = (bits & 0x3ff) / 1024.0f;
        f = (bits & 0x8000 ? -1.0f : 1.0f) *
            ((exp == -15) ? std::ldexp(mant, -14)
                          : std::ldexp(1.0f + mant, exp));
      } else {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        memcpy(&f, &u, sizeof(f));
      }
      value = f;
      break;
    }
    default:
      return vcclInvalidArgument;  // PreMulSum is float-only
  }
  // Reuse a freed slot if one exists.
  for (size_t i = 0; i < comm->customOps.size(); i++) {
    if (!comm->customOps[i].active) {
      comm->customOps[i] = {value, true};
      *op = static_cast<vcclRedOp_t>(vcclNumOps + i);
      return vcclSuccess;
    }
  }
  comm->customOps.push_back({value, true});
  *op = static_cast<vcclRedOp_t>(vcclNumOps + comm->customOps.size() - 1);
  return vcclSuccess;
}

vcclResult_t vcclRedOpDestroy(vcclRedOp_t op, vcclComm_t comm) {
  if (comm == nullptr || op < vcclNumOps) return vcclInvalidArgument;
  size_t idx = static_cast<size_t>(op) - vcclNumOps;
  if (idx >= comm->customOps.size() || !comm->customOps[idx].active)
    return vcclInvalidArgument;
  comm->customOps[idx].active = false;
  return vcclSuccess;
}

vcclResult_t vcclMemAlloc(void** ptr, size_t size) {
  if (ptr == nullptr || size == 0) return vcclInvalidArgument;
  long page = sysconf(_SC_PAGESIZE);
  size_t aligned = (size + page - 1) / page * page;
  void* p = mmap(nullptr, aligned, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return vcclSystemError;
  {
    std::lock_guard<std::mutex> lock(g_memMutex);
    g_memAllocs[p] = aligned;
  }
  *ptr = p;
  return vcclSuccess;
}

vcclResult_t vcclMemFree(void* ptr) {
  if (ptr == nullptr) return vcclSuccess;
  size_t size;
  {
    std::lock_guard<std::mutex> lock(g_memMutex);
    auto it = g_memAllocs.find(ptr);
    if (it == g_memAllocs.end()) return vcclInvalidArgument;
    size = it->second;
    g_memAllocs.erase(it);
  }
  munmap(ptr, size);
  return vcclSuccess;
}

}  // extern "C"
