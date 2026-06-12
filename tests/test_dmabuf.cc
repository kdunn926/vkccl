/* SPDX-License-Identifier: Apache-2.0
 *
 * dmabuf registration test (Linux only). Uses /dev/udmabuf to mint a real
 * dmabuf from a sealed memfd — exercising exactly the path a Vulkan
 * allocation exported with VK_EXT_external_memory_dma_buf takes — then runs
 * collectives in the mapped dmabuf memory across forked ranks.
 *
 * Exit codes: 0 = pass, 1 = fail, 77 = skipped (no /dev/udmabuf).
 */
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vccl.h"

namespace {

constexpr size_t kPages = 64;
constexpr size_t kBytes = kPages * 4096;

int makeDmaBuf() {
  int memfd = memfd_create("vccl-test", MFD_ALLOW_SEALING);
  if (memfd < 0) {
    perror("memfd_create");
    return -1;
  }
  if (ftruncate(memfd, kBytes) != 0 ||
      fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK) != 0) {
    perror("memfd setup");
    close(memfd);
    return -1;
  }
  int dev = open("/dev/udmabuf", O_RDWR);
  if (dev < 0) {
    close(memfd);
    return -2;  // skip: no udmabuf support
  }
  udmabuf_create create{};
  create.memfd = memfd;
  create.offset = 0;
  create.size = kBytes;
  int buf = ioctl(dev, UDMABUF_CREATE, &create);
  if (buf < 0) perror("UDMABUF_CREATE");
  close(dev);
  close(memfd);
  return buf;
}

int childMain(int rank, int nranks, const vcclUniqueId& id) {
  vcclComm_t comm = nullptr;
  if (vcclCommInitRank(&comm, nranks, id, rank) != vcclSuccess) return 1;

  int fd = makeDmaBuf();
  if (fd == -2) {
    fprintf(stderr, "rank %d: /dev/udmabuf not available, skipping\n", rank);
    vcclCommDestroy(comm);
    return 77;
  }
  if (fd < 0) return 1;

  void* ptr = nullptr;
  vcclMemHandle_t handle = nullptr;
  if (vcclCommRegisterDmaBuf(comm, fd, 0, kBytes, &ptr, &handle) !=
      vcclSuccess) {
    fprintf(stderr, "rank %d: vcclCommRegisterDmaBuf failed\n", rank);
    return 1;
  }
  close(fd);  // vccl does not take ownership; mapping must stay valid

  // All-reduce a float buffer living in the dmabuf, in place.
  size_t count = kBytes / sizeof(float) / 2;
  float* data = static_cast<float*>(ptr);
  for (size_t i = 0; i < count; i++)
    data[i] = static_cast<float>(rank + 1 + (i % 5));
  if (vcclAllReduce(data, data, count, vcclFloat32, vcclSum, comm) !=
      vcclSuccess)
    return 1;

  int failures = 0;
  for (size_t i = 0; i < count; i++) {
    float want = 0;
    for (int r = 0; r < nranks; r++) want += static_cast<float>(r + 1 + (i % 5));
    if (data[i] != want) {
      if (failures++ < 3)
        fprintf(stderr, "rank %d: allreduce[%zu] got %f want %f\n", rank, i,
                data[i], want);
    }
  }

  // All-gather between two halves of the same dmabuf.
  size_t per = count / (2 * nranks);
  float* agSend = data;
  float* agRecv = data + count;
  for (size_t i = 0; i < per; i++)
    agSend[i] = static_cast<float>(rank * 1000 + static_cast<int>(i % 100));
  if (vcclAllGather(agSend, agRecv, per, vcclFloat32, comm) != vcclSuccess)
    return 1;
  for (int r = 0; r < nranks; r++) {
    for (size_t i = 0; i < per; i++) {
      float want = static_cast<float>(r * 1000 + static_cast<int>(i % 100));
      if (agRecv[r * per + i] != want) {
        if (failures++ < 6)
          fprintf(stderr, "rank %d: allgather[%d][%zu] got %f want %f\n", rank,
                  r, i, agRecv[r * per + i], want);
      }
    }
  }

  if (vcclCommDeregister(comm, handle) != vcclSuccess) return 1;
  vcclCommDestroy(comm);
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main() {
  setenv("VCCL_SOCKET_ADDR", "127.0.0.1", 0);
  const int nranks = 2;
  vcclUniqueId id;
  if (vcclGetUniqueId(&id) != vcclSuccess) return 1;
  std::vector<pid_t> pids;
  for (int r = 0; r < nranks; r++) {
    pid_t pid = fork();
    if (pid == 0) _exit(childMain(r, nranks, id));
    pids.push_back(pid);
  }
  bool fail = false, skip = false;
  for (pid_t pid : pids) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status)) fail = true;
    else if (WEXITSTATUS(status) == 77) skip = true;
    else if (WEXITSTATUS(status) != 0) fail = true;
  }
  if (skip && !fail) {
    printf("dmabuf test skipped (no /dev/udmabuf)\n");
    return 77;
  }
  printf("dmabuf test: %s\n", fail ? "FAIL" : "OK");
  return fail ? 1 : 0;
}
