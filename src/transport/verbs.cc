/* SPDX-License-Identifier: Apache-2.0
 *
 * ibverbs RC transport for RoCE v2 (and plain InfiniBand).
 *
 * One RC queue pair per peer, connected eagerly at communicator init; QP
 * parameters are exchanged through the bootstrap network. Data moves with
 * two-sided send/recv: the receiver pre-posts before the sender posts, and
 * RNR retries absorb any remaining race.
 *
 * Memory registration: persistent MRs come from vcclCommRegister /
 * vcclCommRegisterDmaBuf and from the per-collective ScopedReg done in the
 * collectives layer; anything else gets a temporary MR released when its
 * work completes. On-demand MRs are deliberately NOT cached across calls: a
 * freed allocation's virtual range can be remapped to new physical pages,
 * and a cached MR would keep the NIC pointed at the old ones. Sends small
 * enough for IBV_SEND_INLINE skip registration entirely. On unified-memory
 * APUs this path is already zero-copy for GPU data, since mapped
 * device-local Vulkan allocations are ordinary host pages.
 *
 * Env knobs:
 *   VCCL_IB_HCA       device name (default: first device with an active port)
 *   VCCL_GID_INDEX    GID table index (default: first "RoCE v2" entry, else
 *                     first non-zero GID)
 */
#include <infiniband/verbs.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/bootstrap.h"
#include "transport/transport.h"
#include "util/deadline.h"
#include "util/log.h"

namespace vccl {
namespace {

constexpr size_t kMaxSegBytes = 1ull << 30;
constexpr int kQueueDepth = 128;
// Inline-send threshold to request at QP creation: payloads this small ride
// inside the WQE (lower latency, no memory registration involved).
constexpr uint32_t kWantInline = 220;

struct QpInfo {
  uint32_t qpn;
  uint32_t psn;
  uint16_t lid;
  uint8_t mtu;
  uint8_t pad;
  uint8_t gid[16];
};

struct Device {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  uint8_t port = 1;
  ibv_port_attr portAttr{};
  int gidIndex = 0;
  ibv_gid gid{};
  bool roce = false;
};

bool gidIsZero(const ibv_gid& g) {
  for (unsigned char b : g.raw)
    if (b != 0) return false;
  return true;
}

// Look up the GID type via sysfs; rdma-core exposes "RoCE v2" / "IB/RoCE v1".
bool gidIsRoceV2(const char* devName, uint8_t port, int idx) {
  char path[256];
  snprintf(path, sizeof(path),
           "/sys/class/infiniband/%s/ports/%u/gid_attrs/types/%d", devName,
           port, idx);
  FILE* f = fopen(path, "r");
  if (f == nullptr) return false;
  char buf[32] = {0};
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  (void)n;
  return strstr(buf, "RoCE v2") != nullptr;
}

bool gidIsIpv4Mapped(const ibv_gid& g) {
  for (int i = 0; i < 10; i++)
    if (g.raw[i] != 0) return false;
  return g.raw[10] == 0xff && g.raw[11] == 0xff;
}

bool gidIsLinkLocal(const ibv_gid& g) {
  return g.raw[0] == 0xfe && (g.raw[1] & 0xc0) == 0x80;
}

vcclResult_t pickGid(Device* dev) {
  if (const char* env = std::getenv("VCCL_GID_INDEX")) {
    dev->gidIndex = atoi(env);
    if (ibv_query_gid(dev->ctx, dev->port, dev->gidIndex, &dev->gid) != 0) {
      VCCL_ERR("VCCL_GID_INDEX=%d: gid query failed", dev->gidIndex);
      return vcclSystemError;
    }
    return vcclSuccess;
  }
  // Preference order: RoCE v2 + IPv4-mapped (routable, and link-local IPv6
  // GIDs often fail neighbor resolution), then RoCE v2 non-link-local, then
  // RoCE v2, then any non-zero GID.
  const char* name = ibv_get_device_name(dev->ctx->device);
  int best = -1, bestScore = -1;
  ibv_gid bestGid{};
  for (int i = 0; i < dev->portAttr.gid_tbl_len; i++) {
    ibv_gid g{};
    if (ibv_query_gid(dev->ctx, dev->port, i, &g) != 0) continue;
    if (gidIsZero(g)) continue;
    int score = 0;
    if (gidIsRoceV2(name, dev->port, i)) {
      score = gidIsIpv4Mapped(g) ? 3 : gidIsLinkLocal(g) ? 1 : 2;
    }
    if (score > bestScore) {
      bestScore = score;
      best = i;
      bestGid = g;
    }
  }
  if (best < 0) {
    VCCL_ERR("no usable GID on %s port %u", name, dev->port);
    return vcclSystemError;
  }
  dev->gidIndex = best;
  dev->gid = bestGid;
  return vcclSuccess;
}

vcclResult_t openDevice(Device* dev) {
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (list == nullptr || num == 0) {
    if (list != nullptr) ibv_free_device_list(list);
    VCCL_INFO("no RDMA devices found");
    return vcclSystemError;
  }
  const char* want = std::getenv("VCCL_IB_HCA");

  vcclResult_t result = vcclSystemError;
  for (int i = 0; i < num; i++) {
    const char* name = ibv_get_device_name(list[i]);
    if (want != nullptr && strcmp(name, want) != 0) continue;
    ibv_context* ctx = ibv_open_device(list[i]);
    if (ctx == nullptr) continue;

    ibv_device_attr devAttr{};
    if (ibv_query_device(ctx, &devAttr) != 0) {
      ibv_close_device(ctx);
      continue;
    }
    bool found = false;
    for (uint8_t p = 1; p <= devAttr.phys_port_cnt; p++) {
      ibv_port_attr pa{};
      if (ibv_query_port(ctx, p, &pa) != 0) continue;
      if (pa.state != IBV_PORT_ACTIVE) continue;
      dev->ctx = ctx;
      dev->port = p;
      dev->portAttr = pa;
      dev->roce = pa.link_layer == IBV_LINK_LAYER_ETHERNET;
      found = true;
      break;
    }
    if (!found) {
      ibv_close_device(ctx);
      continue;
    }
    if (dev->roce && pickGid(dev) != vcclSuccess) {
      ibv_close_device(ctx);
      dev->ctx = nullptr;
      continue;
    }
    dev->pd = ibv_alloc_pd(dev->ctx);
    if (dev->pd == nullptr) {
      ibv_close_device(ctx);
      dev->ctx = nullptr;
      continue;
    }
    VCCL_INFO("verbs: using %s port %u (%s, gid index %d)", name, dev->port,
              dev->roce ? "RoCE" : "IB", dev->gidIndex);
    result = vcclSuccess;
    break;
  }
  ibv_free_device_list(list);
  return result;
}

class VerbsTransport final : public Transport {
 public:
  static vcclResult_t create(Bootstrap* bs, std::unique_ptr<Transport>* out) {
    std::unique_ptr<VerbsTransport> t(new VerbsTransport());
    VCCLCHECK(openDevice(&t->dev_));

    const int rank = bs->rank();
    const int nranks = bs->nranks();
    t->cqs_.assign(nranks, nullptr);
    t->qps_.assign(nranks, nullptr);
    t->maxInline_.assign(nranks, 0);

    std::mt19937 rng(std::random_device{}());
    std::vector<QpInfo> local(nranks);
    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      VCCLCHECK(t->createQp(p));
      QpInfo& info = local[p];
      info.qpn = t->qps_[p]->qp_num;
      info.psn = rng() & 0xffffff;
      info.lid = t->dev_.portAttr.lid;
      info.mtu = static_cast<uint8_t>(t->dev_.portAttr.active_mtu);
      memcpy(info.gid, t->dev_.gid.raw, 16);
    }

    // all[r*nranks + p] is rank r's QP info for talking to p.
    std::vector<QpInfo> all(static_cast<size_t>(nranks) * nranks);
    memcpy(&all[static_cast<size_t>(rank) * nranks], local.data(),
           sizeof(QpInfo) * nranks);
    VCCLCHECK(bs->allGather(all.data(), sizeof(QpInfo) * nranks));

    for (int p = 0; p < nranks; p++) {
      if (p == rank) continue;
      const QpInfo& mine = all[static_cast<size_t>(rank) * nranks + p];
      const QpInfo& theirs = all[static_cast<size_t>(p) * nranks + rank];
      VCCLCHECK(t->connectQp(p, mine, theirs));
    }
    // Ensure every QP is in RTS before any peer starts sending.
    VCCLCHECK(bs->barrier());
    VCCL_INFO("verbs transport connected (%d ranks)", nranks);
    *out = std::move(t);
    return vcclSuccess;
  }

  vcclResult_t regMr(void* buf, size_t bytes, void** handle) override {
    ibv_mr* mr = ibv_reg_mr(dev_.pd, buf, bytes, IBV_ACCESS_LOCAL_WRITE);
    if (mr == nullptr) {
      VCCL_ERR("ibv_reg_mr(%p, %zu) failed: %s", buf, bytes, strerror(errno));
      return vcclSystemError;
    }
    regions_.push_back(
        Region{reinterpret_cast<uintptr_t>(buf), bytes, mr, nullptr});
    *handle = &regions_.back();
    return vcclSuccess;
  }

  vcclResult_t regDmaBuf(int fd, uint64_t offset, void* ptr, size_t bytes,
                         void** handle) override {
    // Prefer a true dmabuf MR: the NIC then reads/writes the buffer without
    // going through the CPU mapping (required for non-host-visible VRAM,
    // free on APUs). iova = the CPU mapping address so SGEs keep using
    // ordinary pointers.
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    ibv_mr* mr =
        ibv_reg_dmabuf_mr(dev_.pd, offset, bytes,
                          reinterpret_cast<uint64_t>(ptr), fd,
                          IBV_ACCESS_LOCAL_WRITE);
    if (mr != nullptr) {
      VCCL_INFO("dmabuf MR registered (fd %d, %zu bytes)", fd, bytes);
      regions_.push_back(Region{start, bytes, mr, nullptr});
      *handle = &regions_.back();
      return vcclSuccess;
    }
    // No provider dmabuf support (e.g. rxe). The CPU mapping of a dmabuf
    // cannot be pinned either (get_user_pages fails on such VMAs), so the
    // data path bounces through a registered staging buffer instead.
    char* staging = static_cast<char*>(malloc(bytes));
    if (staging == nullptr) return vcclSystemError;
    ibv_mr* smr = ibv_reg_mr(dev_.pd, staging, bytes, IBV_ACCESS_LOCAL_WRITE);
    if (smr == nullptr) {
      VCCL_ERR("ibv_reg_mr(staging, %zu) failed: %s", bytes, strerror(errno));
      free(staging);
      return vcclSystemError;
    }
    VCCL_INFO(
        "ibv_reg_dmabuf_mr unsupported (%s); staging dmabuf region through "
        "a bounce buffer",
        strerror(errno));
    regions_.push_back(Region{start, bytes, smr, staging});
    *handle = &regions_.back();
    return vcclSuccess;
  }

  vcclResult_t deregMr(void* handle) override {
    if (handle == nullptr) return vcclSuccess;
    for (auto it = regions_.begin(); it != regions_.end(); ++it) {
      if (&*it == handle) {
        ibv_dereg_mr(it->mr);
        free(it->staging);
        regions_.erase(it);
        return vcclSuccess;
      }
    }
    return vcclInvalidArgument;
  }

  bool covers(const void* buf, size_t bytes) const override {
    uintptr_t addr = reinterpret_cast<uintptr_t>(buf);
    for (const Region& region : regions_) {
      if (addr >= region.start && addr + bytes <= region.start + region.len)
        return true;
    }
    return false;
  }

  ~VerbsTransport() override {
    for (auto& region : regions_) {
      ibv_dereg_mr(region.mr);
      free(region.staging);
    }
    for (ibv_qp* qp : qps_)
      if (qp != nullptr) ibv_destroy_qp(qp);
    for (ibv_cq* cq : cqs_)
      if (cq != nullptr) ibv_destroy_cq(cq);
    if (dev_.pd != nullptr) ibv_dealloc_pd(dev_.pd);
    if (dev_.ctx != nullptr) ibv_close_device(dev_.ctx);
  }

  vcclResult_t send(int peer, const void* buf, size_t bytes) override {
    if (bytes == 0) return vcclSuccess;
    int wrs = 0;
    VCCLCHECK(postSend(peer, buf, bytes, &wrs));
    return pollPeer(peer, wrs, 0);
  }

  vcclResult_t recv(int peer, void* buf, size_t bytes) override {
    if (bytes == 0) return vcclSuccess;
    int wrs = 0;
    VCCLCHECK(postRecv(peer, buf, bytes, &wrs));
    VCCLCHECK(pollPeer(peer, 0, wrs));
    drainCompleted();
    return vcclSuccess;
  }

  vcclResult_t sendRecv(int sendPeer, const void* sbuf, size_t sbytes,
                        int recvPeer, void* rbuf, size_t rbytes) override {
    int sendWrs = 0, recvWrs = 0;
    if (rbytes > 0) VCCLCHECK(postRecv(recvPeer, rbuf, rbytes, &recvWrs));
    if (sbytes > 0) VCCLCHECK(postSend(sendPeer, sbuf, sbytes, &sendWrs));
    if (sendPeer == recvPeer) {
      VCCLCHECK(pollPeer(sendPeer, sendWrs, recvWrs));
    } else {
      // Poll both CQs; either side may finish first.
      Deadline deadline;
      int sendDone = 0, recvDone = 0;
      while (sendDone < sendWrs || recvDone < recvWrs) {
        if (sendDone < sendWrs)
          VCCLCHECK(pollOnce(sendPeer, &sendDone, nullptr));
        if (recvDone < recvWrs)
          VCCLCHECK(pollOnce(recvPeer, nullptr, &recvDone));
        if (deadline.expired()) {
          VCCL_ERR("sendRecv completion wait timed out (VCCL_TIMEOUT)");
          return vcclSystemError;
        }
      }
    }
    drainCompleted();
    return vcclSuccess;
  }

  vcclResult_t batch(const std::vector<P2pOp>& ops) override {
    // Post in waves so no QP exceeds its work-queue depth, polling each
    // wave to completion. Within a wave, all transfers progress together
    // (RC QPs are independent; send/recv on one QP are independent queues).
    size_t idx = 0;
    while (idx < ops.size()) {
      std::map<int, std::pair<int, int>> waveWrs;  // peer -> (send, recv)
      std::map<int, std::pair<int, int>> posted;
      size_t waveEnd = idx;
      for (; waveEnd < ops.size(); waveEnd++) {
        const P2pOp& op = ops[waveEnd];
        if (op.bytes == 0) continue;
        int needed =
            static_cast<int>((op.bytes + kMaxSegBytes - 1) / kMaxSegBytes);
        auto& counts = waveWrs[op.peer];
        int& dirCount = op.isSend ? counts.first : counts.second;
        if (dirCount + needed > kQueueDepth) break;
        dirCount += needed;
      }
      if (waveEnd == idx) {  // single oversized op: segments fit by design
        waveEnd = idx + 1;
      }
      for (size_t i = idx; i < waveEnd; i++) {
        const P2pOp& op = ops[i];
        if (op.bytes == 0) continue;
        int wrs = 0;
        if (op.isSend) {
          VCCLCHECK(postSend(op.peer, op.buf, op.bytes, &wrs));
          posted[op.peer].first += wrs;
        } else {
          VCCLCHECK(postRecv(op.peer, op.buf, op.bytes, &wrs));
          posted[op.peer].second += wrs;
        }
      }
      Deadline deadline;
      bool pending = true;
      std::map<int, std::pair<int, int>> done;
      while (pending) {
        pending = false;
        for (auto& entry : posted) {
          auto& d = done[entry.first];
          if (d.first < entry.second.first ||
              d.second < entry.second.second) {
            VCCLCHECK(pollOnce(entry.first, &d.first, &d.second));
            if (d.first < entry.second.first ||
                d.second < entry.second.second) {
              pending = true;
            }
          }
        }
        if (isAborted()) {
          VCCL_ERR("batch aborted");
          return vcclSystemError;
        }
        if (pending && deadline.expired()) {
          VCCL_ERR("batch completion wait timed out (VCCL_TIMEOUT)");
          return vcclSystemError;
        }
      }
      drainCompleted();
      idx = waveEnd;
    }
    return vcclSuccess;
  }

  const char* name() const override { return "verbs"; }

 private:
  // One registered range. staging != nullptr marks a bounce region: the NIC
  // only ever touches `staging`, and the data path copies to/from the real
  // buffer around each transfer.
  struct Region {
    uintptr_t start = 0;
    size_t len = 0;
    ibv_mr* mr = nullptr;
    char* staging = nullptr;
  };
  struct Resolved {
    ibv_mr* mr = nullptr;
    uintptr_t nicAddr = 0;
    bool bounced = false;
  };
  struct Copyback {
    void* dst;
    const void* src;
    size_t bytes;
  };

  VerbsTransport() = default;

  vcclResult_t createQp(int peer) {
    ibv_cq* cq = ibv_create_cq(dev_.ctx, 2 * kQueueDepth, nullptr, nullptr, 0);
    if (cq == nullptr) {
      VCCL_ERR("ibv_create_cq failed");
      return vcclSystemError;
    }
    cqs_[peer] = cq;

    ibv_qp_init_attr attr{};
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = kQueueDepth;
    attr.cap.max_recv_wr = kQueueDepth;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = kWantInline;
    ibv_qp* qp = ibv_create_qp(dev_.pd, &attr);
    if (qp == nullptr) {
      // Provider may not support the requested inline size; retry without.
      attr.cap.max_inline_data = 0;
      qp = ibv_create_qp(dev_.pd, &attr);
    }
    if (qp == nullptr) {
      VCCL_ERR("ibv_create_qp failed");
      return vcclSystemError;
    }
    qps_[peer] = qp;
    maxInline_[peer] = attr.cap.max_inline_data;

    ibv_qp_attr qpa{};
    qpa.qp_state = IBV_QPS_INIT;
    qpa.pkey_index = 0;
    qpa.port_num = dev_.port;
    qpa.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp, &qpa,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
      VCCL_ERR("modify_qp -> INIT failed: %s", strerror(errno));
      return vcclSystemError;
    }
    return vcclSuccess;
  }

  vcclResult_t connectQp(int peer, const QpInfo& mine, const QpInfo& theirs) {
    ibv_qp* qp = qps_[peer];

    ibv_qp_attr rtr{};
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = static_cast<ibv_mtu>(
        std::min<uint8_t>(mine.mtu, theirs.mtu));
    rtr.dest_qp_num = theirs.qpn;
    rtr.rq_psn = theirs.psn;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.port_num = dev_.port;
    if (dev_.roce) {
      rtr.ah_attr.is_global = 1;
      memcpy(rtr.ah_attr.grh.dgid.raw, theirs.gid, 16);
      rtr.ah_attr.grh.sgid_index = dev_.gidIndex;
      rtr.ah_attr.grh.hop_limit = 64;
    } else {
      rtr.ah_attr.dlid = theirs.lid;
    }
    if (ibv_modify_qp(qp, &rtr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC |
                          IBV_QP_MIN_RNR_TIMER) != 0) {
      VCCL_ERR("modify_qp -> RTR failed: %s", strerror(errno));
      return vcclSystemError;
    }

    ibv_qp_attr rts{};
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;  // infinite: receiver may post late
    rts.sq_psn = mine.psn;
    rts.max_rd_atomic = 1;
    if (ibv_modify_qp(qp, &rts,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
      VCCL_ERR("modify_qp -> RTS failed: %s", strerror(errno));
      return vcclSystemError;
    }
    return vcclSuccess;
  }

  // Resolve [buf, buf+bytes) to its MR and the address the NIC should use.
  // For bounce (staging) regions the NIC address lives in the staging copy.
  // Ranges with no covering registration get a temporary MR, deregistered
  // once the posted work completes (see drainTempMrs) — caching it would go
  // stale when the allocation is freed and its pages returned to the kernel.
  vcclResult_t resolve(const void* buf, size_t bytes, Resolved* out) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(buf);
    for (const Region& region : regions_) {
      if (addr >= region.start && addr + bytes <= region.start + region.len) {
        out->mr = region.mr;
        out->nicAddr = region.staging != nullptr
                           ? reinterpret_cast<uintptr_t>(region.staging) +
                                 (addr - region.start)
                           : addr;
        out->bounced = region.staging != nullptr;
        return vcclSuccess;
      }
    }
    ibv_mr* mr = ibv_reg_mr(dev_.pd, const_cast<void*>(buf), bytes,
                            IBV_ACCESS_LOCAL_WRITE);
    if (mr == nullptr) {
      VCCL_ERR("ibv_reg_mr(%p, %zu) failed: %s", buf, bytes, strerror(errno));
      return vcclSystemError;
    }
    tempMrs_.push_back(mr);
    out->mr = mr;
    out->nicAddr = addr;
    out->bounced = false;
    return vcclSuccess;
  }

  vcclResult_t postSend(int peer, const void* buf, size_t bytes, int* wrs) {
    if (bytes <= maxInline_[peer]) {
      // Inline: the CPU copies the payload into the WQE; no MR, and the
      // buffer (even a dmabuf mapping) is reusable immediately.
      ibv_sge sge{};
      sge.addr = reinterpret_cast<uintptr_t>(buf);
      sge.length = static_cast<uint32_t>(bytes);
      ibv_send_wr wr{}, *bad = nullptr;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_SEND;
      wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
      if (ibv_post_send(qps_[peer], &wr, &bad) != 0) {
        VCCL_ERR("ibv_post_send(inline) failed: %s", strerror(errno));
        return vcclSystemError;
      }
      (*wrs)++;
      return vcclSuccess;
    }
    Resolved r;
    VCCLCHECK(resolve(buf, bytes, &r));
    if (r.bounced) {
      memcpy(reinterpret_cast<void*>(r.nicAddr), buf, bytes);
    }
    for (size_t off = 0; off < bytes; off += kMaxSegBytes) {
      size_t seg = std::min(kMaxSegBytes, bytes - off);
      ibv_sge sge{};
      sge.addr = r.nicAddr + off;
      sge.length = static_cast<uint32_t>(seg);
      sge.lkey = r.mr->lkey;
      ibv_send_wr wr{}, *bad = nullptr;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_SEND;
      wr.send_flags = IBV_SEND_SIGNALED;
      if (ibv_post_send(qps_[peer], &wr, &bad) != 0) {
        VCCL_ERR("ibv_post_send failed: %s", strerror(errno));
        return vcclSystemError;
      }
      (*wrs)++;
    }
    return vcclSuccess;
  }

  vcclResult_t postRecv(int peer, void* buf, size_t bytes, int* wrs) {
    Resolved r;
    VCCLCHECK(resolve(buf, bytes, &r));
    if (r.bounced) {
      // Data lands in staging; copy out after the completion is polled.
      pendingCopyback_.push_back(
          {buf, reinterpret_cast<const void*>(r.nicAddr), bytes});
    }
    for (size_t off = 0; off < bytes; off += kMaxSegBytes) {
      size_t seg = std::min(kMaxSegBytes, bytes - off);
      ibv_sge sge{};
      sge.addr = r.nicAddr + off;
      sge.length = static_cast<uint32_t>(seg);
      sge.lkey = r.mr->lkey;
      ibv_recv_wr wr{}, *bad = nullptr;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      if (ibv_post_recv(qps_[peer], &wr, &bad) != 0) {
        VCCL_ERR("ibv_post_recv failed: %s", strerror(errno));
        return vcclSystemError;
      }
      (*wrs)++;
    }
    return vcclSuccess;
  }

  // Run after every completion wait: flush staged receives back to their
  // dmabuf mappings, and release temporary MRs whose work has finished.
  void drainCompleted() {
    for (const Copyback& cb : pendingCopyback_)
      memcpy(cb.dst, cb.src, cb.bytes);
    pendingCopyback_.clear();
    for (ibv_mr* mr : tempMrs_) ibv_dereg_mr(mr);
    tempMrs_.clear();
  }

  vcclResult_t pollOnce(int peer, int* sendDone, int* recvDone) {
    ibv_wc wc[16];
    int n = ibv_poll_cq(cqs_[peer], 16, wc);
    if (n < 0) {
      VCCL_ERR("ibv_poll_cq failed");
      return vcclSystemError;
    }
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        VCCL_ERR("work completion failed: %s (peer %d)",
                 ibv_wc_status_str(wc[i].status), peer);
        return vcclRemoteError;
      }
      if (wc[i].opcode & IBV_WC_RECV) {
        if (recvDone != nullptr) (*recvDone)++;
      } else if (sendDone != nullptr) {
        (*sendDone)++;
      }
    }
    return vcclSuccess;
  }

  vcclResult_t pollPeer(int peer, int sendWrs, int recvWrs) {
    Deadline deadline;
    int sendDone = 0, recvDone = 0;
    while (sendDone < sendWrs || recvDone < recvWrs) {
      VCCLCHECK(pollOnce(peer, &sendDone, &recvDone));
      if (isAborted()) {
        VCCL_ERR("operation aborted (peer %d)", peer);
        return vcclSystemError;
      }
      if (deadline.expired()) {
        VCCL_ERR("completion wait timed out (peer %d, VCCL_TIMEOUT)", peer);
        return vcclSystemError;
      }
    }
    return vcclSuccess;
  }

  Device dev_;
  std::vector<ibv_cq*> cqs_;
  std::vector<ibv_qp*> qps_;
  std::vector<uint32_t> maxInline_;
  // Persistent registrations (vcclCommRegister / dmabuf imports). std::list:
  // Region addresses are handed out as handles and must stay stable.
  std::list<Region> regions_;
  std::vector<Copyback> pendingCopyback_;
  // MRs created on the fly for unregistered ranges, freed by drainCompleted.
  std::vector<ibv_mr*> tempMrs_;
};

}  // namespace

bool verbsAvailable() {
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (list != nullptr) ibv_free_device_list(list);
  return num > 0;
}

vcclResult_t verbsTransportCreate(Bootstrap* bootstrap,
                                  std::unique_ptr<Transport>* out) {
  return VerbsTransport::create(bootstrap, out);
}

}  // namespace vccl
