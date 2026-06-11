/* SPDX-License-Identifier: Apache-2.0
 *
 * vccl-bench — nccl-tests-style bandwidth/latency sweep.
 *
 * Local mode (single machine, forks ranks):
 *   vccl-bench --launch 4 --op allreduce -b 1K -e 64M -f 2
 *
 * Multi-node mode (one process per node, shared file for id exchange):
 *   node0$ VCCL_ID_FILE=/shared/id VCCL_RANK=0 VCCL_NRANKS=2 vccl-bench ...
 *   node1$ VCCL_ID_FILE=/shared/id VCCL_RANK=1 VCCL_NRANKS=2 vccl-bench ...
 * Rank 0 writes the unique id to VCCL_ID_FILE (hex); other ranks poll-read
 * it. The file can also be copied between nodes (scp) before launch.
 */
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vccl.h"

namespace {

struct Options {
  std::string op = "allreduce";
  size_t minBytes = 1024;
  size_t maxBytes = 64ull << 20;
  double factor = 2.0;
  int iters = 20;
  int warmup = 3;
  int launch = 0;   // >0: fork this many local ranks
  bool check = false;  // verify results before timing
  vcclDataType_t dtype = vcclFloat32;
};

// One correctness pass with rank-determined f32 patterns; returns 0 on match.
int verifyOnce(const Options& opt, vcclComm_t comm, int rank, int nranks,
               float* send, float* recv, size_t count) {
  size_t per = count / nranks;
  size_t bad = 0;
  if (opt.op == "allreduce") {
    for (size_t i = 0; i < count; i++)
      send[i] = static_cast<float>((rank + i) % 8);
    if (vcclAllReduce(send, recv, count, vcclFloat32, vcclSum, comm) !=
        vcclSuccess)
      return 1;
    for (size_t i = 0; i < count; i++) {
      float want = 0;
      for (int r = 0; r < nranks; r++)
        want += static_cast<float>((r + i) % 8);
      if (recv[i] != want) bad++;
    }
  } else if (opt.op == "allgather") {
    for (size_t i = 0; i < per; i++)
      send[i] = static_cast<float>(rank * 100 + static_cast<int>(i % 50));
    if (vcclAllGather(send, recv, per, vcclFloat32, comm) != vcclSuccess)
      return 1;
    for (int r = 0; r < nranks; r++)
      for (size_t i = 0; i < per; i++)
        if (recv[r * per + i] !=
            static_cast<float>(r * 100 + static_cast<int>(i % 50)))
          bad++;
  } else if (opt.op == "reducescatter") {
    for (size_t i = 0; i < per * nranks; i++)
      send[i] = static_cast<float>((rank + i) % 8);
    if (vcclReduceScatter(send, recv, per, vcclFloat32, vcclSum, comm) !=
        vcclSuccess)
      return 1;
    for (size_t i = 0; i < per; i++) {
      float want = 0;
      for (int r = 0; r < nranks; r++)
        want += static_cast<float>((r + rank * per + i) % 8);
      if (recv[i] != want) bad++;
    }
  } else {
    return 0;  // broadcast: nothing rank-asymmetric to verify cheaply
  }
  if (bad > 0) {
    fprintf(stderr, "rank %d: %s check FAILED (%zu mismatches)\n", rank,
            opt.op.c_str(), bad);
    return 1;
  }
  printf("# rank %d: %s correctness check passed\n", rank, opt.op.c_str());
  return 0;
}

size_t parseBytes(const char* s) {
  char* end = nullptr;
  double v = strtod(s, &end);
  if (end != nullptr) {
    if (*end == 'K' || *end == 'k') v *= 1 << 10;
    if (*end == 'M' || *end == 'm') v *= 1 << 20;
    if (*end == 'G' || *end == 'g') v *= 1 << 30;
  }
  return static_cast<size_t>(v);
}

[[noreturn]] void usage() {
  fprintf(stderr,
          "usage: vccl-bench [--launch N] [--op "
          "allreduce|allgather|reducescatter|broadcast]\n"
          "                  [-b minbytes] [-e maxbytes] [-f factor]\n"
          "                  [--iters N] [--warmup N] [--dtype f32|f16|bf16]\n"
          "                  [--check]\n");
  exit(2);
}

bool writeIdFile(const char* path, const vcclUniqueId& id) {
  std::string tmp = std::string(path) + ".tmp";
  FILE* f = fopen(tmp.c_str(), "w");
  if (f == nullptr) return false;
  for (int i = 0; i < VCCL_UNIQUE_ID_BYTES; i++)
    fprintf(f, "%02x", static_cast<unsigned char>(id.internal[i]));
  fprintf(f, "\n");
  fclose(f);
  return rename(tmp.c_str(), path) == 0;
}

bool readIdFile(const char* path, vcclUniqueId* id, int timeoutSec) {
  for (int waited = 0; waited < timeoutSec * 10; waited++) {
    FILE* f = fopen(path, "r");
    if (f != nullptr) {
      char buf[2 * VCCL_UNIQUE_ID_BYTES + 2];
      if (fgets(buf, sizeof(buf), f) != nullptr &&
          strlen(buf) >= 2 * VCCL_UNIQUE_ID_BYTES) {
        for (int i = 0; i < VCCL_UNIQUE_ID_BYTES; i++) {
          unsigned v = 0;
          sscanf(buf + 2 * i, "%02x", &v);
          id->internal[i] = static_cast<char>(v);
        }
        fclose(f);
        return true;
      }
      fclose(f);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return false;
}

double busBwFactor(const std::string& op, int n) {
  if (op == "allreduce") return 2.0 * (n - 1) / n;
  if (op == "allgather" || op == "reducescatter")
    return static_cast<double>(n - 1) / n;
  return 1.0;  // broadcast
}

int runBench(const Options& opt, int rank, int nranks,
             const vcclUniqueId& id) {
  // Line-buffer stdout: results must survive the forked child's _exit and
  // stream as they are produced.
  setvbuf(stdout, nullptr, _IOLBF, 0);
  vcclComm_t comm = nullptr;
  if (vcclCommInitRank(&comm, nranks, id, rank) != vcclSuccess) {
    fprintf(stderr, "rank %d: comm init failed\n", rank);
    return 1;
  }

  const size_t esize = vcclTypeSize(opt.dtype);
  std::vector<char> sendbuf(opt.maxBytes), recvbuf(opt.maxBytes);
  memset(sendbuf.data(), 1, sendbuf.size());

  if (opt.check) {
    size_t n = opt.maxBytes / sizeof(float);
    if (verifyOnce(opt, comm, rank, nranks,
                   reinterpret_cast<float*>(sendbuf.data()),
                   reinterpret_cast<float*>(recvbuf.data()), n) != 0) {
      vcclCommDestroy(comm);
      return 1;
    }
  }

  if (rank == 0) {
    printf("# op=%s dtype_size=%zu nranks=%d iters=%d\n", opt.op.c_str(),
           esize, nranks, opt.iters);
    printf("%12s %12s %12s %12s\n", "bytes", "time(us)", "algbw(GB/s)",
           "busbw(GB/s)");
  }

  for (size_t bytes = opt.minBytes; bytes <= opt.maxBytes;
       bytes = std::max(bytes + 1, static_cast<size_t>(bytes * opt.factor))) {
    // Counts per op so that `bytes` is the in/out buffer footprint.
    size_t count = bytes / esize;
    if (count == 0) continue;
    size_t perRank = count / nranks;
    if ((opt.op == "allgather" || opt.op == "reducescatter") && perRank == 0)
      continue;

    auto run = [&]() -> vcclResult_t {
      if (opt.op == "allreduce")
        return vcclAllReduce(sendbuf.data(), recvbuf.data(), count, opt.dtype,
                             vcclSum, comm);
      if (opt.op == "allgather")
        return vcclAllGather(sendbuf.data(), recvbuf.data(), perRank,
                             opt.dtype, comm);
      if (opt.op == "reducescatter")
        return vcclReduceScatter(sendbuf.data(), recvbuf.data(), perRank,
                                 opt.dtype, vcclSum, comm);
      if (opt.op == "broadcast")
        return vcclBroadcast(sendbuf.data(), recvbuf.data(), count, opt.dtype,
                             0, comm);
      return vcclInvalidArgument;
    };

    for (int i = 0; i < opt.warmup; i++) {
      if (run() != vcclSuccess) return 1;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.iters; i++) {
      if (run() != vcclSuccess) return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / opt.iters;

    if (rank == 0) {
      double effBytes =
          (opt.op == "allgather" || opt.op == "reducescatter")
              ? static_cast<double>(perRank * nranks * esize)
              : static_cast<double>(count * esize);
      double algbw = effBytes / (us * 1e3);  // GB/s
      printf("%12zu %12.1f %12.3f %12.3f\n",
             static_cast<size_t>(effBytes), us, algbw,
             algbw * busBwFactor(opt.op, nranks));
    }
  }

  vcclCommDestroy(comm);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--op") opt.op = next();
    else if (a == "-b") opt.minBytes = parseBytes(next());
    else if (a == "-e") opt.maxBytes = parseBytes(next());
    else if (a == "-f") opt.factor = atof(next());
    else if (a == "--iters") opt.iters = atoi(next());
    else if (a == "--warmup") opt.warmup = atoi(next());
    else if (a == "--launch") opt.launch = atoi(next());
    else if (a == "--check") opt.check = true;
    else if (a == "--dtype") {
      std::string d = next();
      if (d == "f32") opt.dtype = vcclFloat32;
      else if (d == "f16") opt.dtype = vcclFloat16;
      else if (d == "bf16") opt.dtype = vcclBfloat16;
      else if (d == "f64") opt.dtype = vcclFloat64;
      else usage();
    } else usage();
  }
  if (opt.op != "allreduce" && opt.op != "allgather" &&
      opt.op != "reducescatter" && opt.op != "broadcast") {
    usage();
  }

  if (opt.launch > 0) {
    setenv("VCCL_SOCKET_ADDR", "127.0.0.1", 0);
    vcclUniqueId id;
    if (vcclGetUniqueId(&id) != vcclSuccess) return 1;
    std::vector<pid_t> pids;
    for (int r = 0; r < opt.launch; r++) {
      pid_t pid = fork();
      if (pid == 0) _exit(runBench(opt, r, opt.launch, id));
      pids.push_back(pid);
    }
    int failed = 0;
    for (pid_t pid : pids) {
      int status = 0;
      waitpid(pid, &status, 0);
      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failed++;
    }
    return failed ? 1 : 0;
  }

  const char* rankEnv = getenv("VCCL_RANK");
  const char* nranksEnv = getenv("VCCL_NRANKS");
  const char* idFile = getenv("VCCL_ID_FILE");
  if (rankEnv == nullptr || nranksEnv == nullptr || idFile == nullptr) {
    fprintf(stderr,
            "set VCCL_RANK, VCCL_NRANKS and VCCL_ID_FILE, or use --launch N\n");
    return 2;
  }
  int rank = atoi(rankEnv);
  int nranks = atoi(nranksEnv);
  vcclUniqueId id;
  if (rank == 0) {
    if (vcclGetUniqueId(&id) != vcclSuccess) return 1;
    if (!writeIdFile(idFile, id)) {
      fprintf(stderr, "cannot write id file %s\n", idFile);
      return 1;
    }
  } else if (!readIdFile(idFile, &id, 120)) {
    fprintf(stderr, "timed out waiting for id file %s\n", idFile);
    return 1;
  }
  return runBench(opt, rank, nranks, id);
}
