/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_CORE_COMM_H_
#define VCCL_CORE_COMM_H_

#include <memory>
#include <vector>

#include "core/bootstrap.h"
#include "transport/transport.h"
#include "vccl.h"

struct vcclComm {
  // One user memory registration; `handle` is the public vcclMemHandle_t.
  struct MemReg {
    void* transportHandle = nullptr;
    void* mapped = nullptr;  // non-null when vccl owns a dmabuf mmap
    size_t bytes = 0;
  };

  int rank = -1;
  int nranks = 0;
  std::unique_ptr<vccl::Bootstrap> bootstrap;
  std::unique_ptr<vccl::Transport> transport;
  std::vector<MemReg*> registrations;
  // Scratch space for ring reduce steps, grown on demand.
  std::vector<char> scratch;

  int next() const { return (rank + 1) % nranks; }
  int prev() const { return (rank + nranks - 1) % nranks; }
  char* scratchAt(size_t bytes) {
    if (scratch.size() < bytes) scratch.resize(bytes);
    return scratch.data();
  }
};

#endif  // VCCL_CORE_COMM_H_
