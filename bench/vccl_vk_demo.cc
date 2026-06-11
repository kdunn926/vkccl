/* SPDX-License-Identifier: Apache-2.0
 *
 * Vulkan ↔ vccl interop demo (Linux only).
 *
 * Each rank allocates a Vulkan device memory block (preferring
 * DEVICE_LOCAL|HOST_VISIBLE — i.e. unified memory on APUs), exports it as a
 * dmabuf via VK_EXT_external_memory_dma_buf, registers it with vccl, and
 * runs an all-gather directly in the exported GPU memory. This is the M3
 * data path for vllm-vulkan on BC-250-class hardware: collectives operate
 * on GPU allocations with zero copies on unified-memory systems.
 *
 *   ./vccl-vk-demo --launch 2          # fork 2 ranks on this machine
 *   VCCL_RANK/VCCL_NRANKS/VCCL_ID_FILE # multi-node, as vccl-bench
 */
#include <sys/wait.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vccl.h"

namespace {

#define VKCHECK(call)                                              \
  do {                                                             \
    VkResult r_ = (call);                                          \
    if (r_ != VK_SUCCESS) {                                        \
      fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #call, \
              static_cast<int>(r_));                               \
      return 1;                                                    \
    }                                                              \
  } while (0)

struct Vk {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
};

int createExportableMemory(Vk* vk, VkDeviceSize bytes, int* fdOut) {
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "vccl-vk-demo";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;
  VKCHECK(vkCreateInstance(&ici, nullptr, &vk->instance));

  uint32_t ndev = 0;
  VKCHECK(vkEnumeratePhysicalDevices(vk->instance, &ndev, nullptr));
  if (ndev == 0) {
    fprintf(stderr, "no Vulkan devices\n");
    return 1;
  }
  std::vector<VkPhysicalDevice> devs(ndev);
  VKCHECK(vkEnumeratePhysicalDevices(vk->instance, &ndev, devs.data()));

  // Pick the first device exposing dmabuf export.
  const char* wantExts[] = {
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
      VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
  };
  for (VkPhysicalDevice d : devs) {
    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(d, nullptr, &next, nullptr);
    std::vector<VkExtensionProperties> exts(next);
    vkEnumerateDeviceExtensionProperties(d, nullptr, &next, exts.data());
    int found = 0;
    for (const auto& e : exts)
      for (const char* w : wantExts)
        if (strcmp(e.extensionName, w) == 0) found++;
    if (found == 2) {
      vk->phys = d;
      break;
    }
  }
  if (vk->phys == VK_NULL_HANDLE) {
    fprintf(stderr, "no Vulkan device with dmabuf export support\n");
    return 1;
  }
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(vk->phys, &props);

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 2;
  dci.ppEnabledExtensionNames = wantExts;
  VKCHECK(vkCreateDevice(vk->phys, &dci, nullptr, &vk->device));

  VkExternalMemoryBufferCreateInfo embci{};
  embci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  embci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.pNext = &embci;
  bci.size = bytes;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VKCHECK(vkCreateBuffer(vk->device, &bci, nullptr, &vk->buffer));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(vk->device, vk->buffer, &req);
  VkPhysicalDeviceMemoryProperties mem;
  vkGetPhysicalDeviceMemoryProperties(vk->phys, &mem);

  // Prefer unified memory (DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT), the
  // BC-250/APU case; fall back to plain HOST_VISIBLE.
  auto findType = [&](VkMemoryPropertyFlags want) -> int {
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
      if ((req.memoryTypeBits & (1u << i)) &&
          (mem.memoryTypes[i].propertyFlags & want) == want)
        return static_cast<int>(i);
    }
    return -1;
  };
  int type = findType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  bool unified = type >= 0;
  if (type < 0)
    type = findType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (type < 0) {
    fprintf(stderr, "no host-visible exportable memory type\n");
    return 1;
  }

  VkExportMemoryAllocateInfo emai{};
  emai.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.pNext = &emai;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = static_cast<uint32_t>(type);
  VKCHECK(vkAllocateMemory(vk->device, &mai, nullptr, &vk->memory));
  VKCHECK(vkBindBufferMemory(vk->device, vk->buffer, vk->memory, 0));

  auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
      vkGetDeviceProcAddr(vk->device, "vkGetMemoryFdKHR"));
  if (getFd == nullptr) {
    fprintf(stderr, "vkGetMemoryFdKHR not available\n");
    return 1;
  }
  VkMemoryGetFdInfoKHR gfi{};
  gfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  gfi.memory = vk->memory;
  gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  VKCHECK(getFd(vk->device, &gfi, fdOut));

  printf("rank: %s, memory type %d (%s), dmabuf fd %d\n", props.deviceName,
         type, unified ? "unified" : "host-visible", *fdOut);
  return 0;
}

int runRank(int rank, int nranks, const vcclUniqueId& id) {
  constexpr size_t kCount = 1 << 20;  // floats per rank
  const VkDeviceSize bytes = kCount * sizeof(float) * (nranks + 1);

  Vk vk;
  int fd = -1;
  if (createExportableMemory(&vk, bytes, &fd) != 0) return 1;

  vcclComm_t comm = nullptr;
  if (vcclCommInitRank(&comm, nranks, id, rank) != vcclSuccess) return 1;

  void* ptr = nullptr;
  vcclMemHandle_t handle = nullptr;
  if (vcclCommRegisterDmaBuf(comm, fd, 0, bytes, &ptr, &handle) !=
      vcclSuccess) {
    fprintf(stderr, "rank %d: dmabuf registration failed\n", rank);
    return 1;
  }
  close(fd);

  // sendbuf = first slot, recvbuf = remaining nranks slots, all in the
  // exported Vulkan memory.
  float* send = static_cast<float*>(ptr);
  float* recv = send + kCount;
  for (size_t i = 0; i < kCount; i++)
    send[i] = static_cast<float>(rank * 7 + static_cast<int>(i % 13));

  if (vcclAllGather(send, recv, kCount, vcclFloat32, comm) != vcclSuccess)
    return 1;

  size_t bad = 0;
  for (int r = 0; r < nranks; r++)
    for (size_t i = 0; i < kCount; i++)
      if (recv[r * kCount + i] !=
          static_cast<float>(r * 7 + static_cast<int>(i % 13)))
        bad++;

  vcclCommDeregister(comm, handle);
  vcclCommDestroy(comm);
  vkDestroyBuffer(vk.device, vk.buffer, nullptr);
  vkFreeMemory(vk.device, vk.memory, nullptr);
  vkDestroyDevice(vk.device, nullptr);
  vkDestroyInstance(vk.instance, nullptr);

  printf("rank %d: all-gather of %zu MiB in exported Vulkan memory: %s\n",
         rank, kCount * sizeof(float) * nranks >> 20, bad ? "FAIL" : "OK");
  return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int launch = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--launch") == 0 && i + 1 < argc)
      launch = atoi(argv[++i]);
  }

  if (launch > 0) {
    setenv("VCCL_SOCKET_ADDR", "127.0.0.1", 0);
    vcclUniqueId id;
    if (vcclGetUniqueId(&id) != vcclSuccess) return 1;
    std::vector<pid_t> pids;
    for (int r = 0; r < launch; r++) {
      pid_t pid = fork();
      if (pid == 0) _exit(runRank(r, launch, id));
      pids.push_back(pid);
    }
    int failed = 0;
    for (pid_t pid : pids) {
      int status = 0;
      waitpid(pid, &status, 0);
      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failed++;
    }
    printf("vccl-vk-demo: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
  }

  const char* rankEnv = getenv("VCCL_RANK");
  const char* nranksEnv = getenv("VCCL_NRANKS");
  const char* idFile = getenv("VCCL_ID_FILE");
  if (rankEnv == nullptr || nranksEnv == nullptr || idFile == nullptr) {
    fprintf(stderr,
            "usage: vccl-vk-demo --launch N | "
            "VCCL_RANK/VCCL_NRANKS/VCCL_ID_FILE envs\n");
    return 2;
  }
  // Reuse the id-file rendezvous from vccl-bench.
  vcclUniqueId id;
  int rank = atoi(rankEnv);
  if (rank == 0) {
    if (vcclGetUniqueId(&id) != vcclSuccess) return 1;
    FILE* f = fopen(idFile, "w");
    if (f == nullptr) return 1;
    for (int i = 0; i < VCCL_UNIQUE_ID_BYTES; i++)
      fprintf(f, "%02x", static_cast<unsigned char>(id.internal[i]));
    fclose(f);
  } else {
    for (int tries = 0; tries < 1200; tries++) {
      FILE* f = fopen(idFile, "r");
      if (f != nullptr) {
        char buf[2 * VCCL_UNIQUE_ID_BYTES + 2];
        if (fgets(buf, sizeof(buf), f) != nullptr &&
            strlen(buf) >= 2 * VCCL_UNIQUE_ID_BYTES) {
          for (int i = 0; i < VCCL_UNIQUE_ID_BYTES; i++) {
            unsigned v = 0;
            sscanf(buf + 2 * i, "%02x", &v);
            id.internal[i] = static_cast<char>(v);
          }
          fclose(f);
          goto have_id;
        }
        fclose(f);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    fprintf(stderr, "timed out waiting for %s\n", idFile);
    return 1;
  }
have_id:
  return runRank(rank, atoi(nranksEnv), id);
}
