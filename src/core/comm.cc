/* SPDX-License-Identifier: Apache-2.0 */
#include "core/comm.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>

#include <algorithm>
#include <memory>

#include "util/log.h"

extern "C" {

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

vcclResult_t vcclCommDestroy(vcclComm_t comm) {
  if (comm != nullptr) {
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

}  // extern "C"
