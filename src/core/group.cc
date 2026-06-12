/* SPDX-License-Identifier: Apache-2.0
 *
 * vcclGroupStart/vcclGroupEnd.
 *
 * Operations issued inside a group are queued per communicator and executed
 * by the outermost vcclGroupEnd:
 *   - each communicator's queue runs on its own thread (required for
 *     multiple in-process communicators, e.g. vcclCommInitAll, whose
 *     collectives must progress against each other);
 *   - within one communicator, maximal runs of consecutive point-to-point
 *     operations are handed to Transport::batch so a send+recv pair with
 *     the same peer progresses concurrently; everything else runs in issue
 *     order.
 */
#include <map>
#include <thread>
#include <utility>
#include <vector>

#include "core/comm.h"

namespace vccl {
namespace {

struct QueuedOp {
  vcclComm_t comm;
  std::function<vcclResult_t()> fn;  // empty for p2p ops
  P2pOp p2p{};
  bool isP2p = false;
};

thread_local int t_groupDepth = 0;
thread_local std::vector<QueuedOp> t_groupOps;

vcclResult_t runCommOps(vcclComm_t comm, const std::vector<QueuedOp>& ops) {
  size_t i = 0;
  while (i < ops.size()) {
    if (ops[i].isP2p) {
      std::vector<P2pOp> run;
      while (i < ops.size() && ops[i].isP2p) run.push_back(ops[i++].p2p);
      if (comm->transport == nullptr) return vcclInvalidUsage;
      vcclResult_t res = comm->transport->batch(run);
      if (res != vcclSuccess) {
        comm->noteError(res);
        return res;
      }
    } else {
      vcclResult_t res = ops[i++].fn();
      if (res != vcclSuccess) {
        comm->noteError(res);
        return res;
      }
    }
  }
  return vcclSuccess;
}

}  // namespace

bool inGroup() { return t_groupDepth > 0; }

vcclResult_t enqueueOrRun(vcclComm_t comm, std::function<vcclResult_t()> fn) {
  if (!inGroup()) return fn();
  QueuedOp op;
  op.comm = comm;
  op.fn = std::move(fn);
  t_groupOps.push_back(std::move(op));
  return vcclSuccess;
}

vcclResult_t enqueueOrRunP2p(vcclComm_t comm, const P2pOp& p2p) {
  if (!inGroup()) {
    if (comm->transport == nullptr) return vcclInvalidUsage;
    if (p2p.isSend) return comm->transport->send(p2p.peer, p2p.buf, p2p.bytes);
    return comm->transport->recv(p2p.peer, p2p.buf, p2p.bytes);
  }
  QueuedOp op;
  op.comm = comm;
  op.p2p = p2p;
  op.isP2p = true;
  t_groupOps.push_back(std::move(op));
  return vcclSuccess;
}

}  // namespace vccl

extern "C" {

vcclResult_t vcclGroupStart(void) {
  vccl::t_groupDepth++;
  return vcclSuccess;
}

vcclResult_t vcclGroupEnd(void) {
  using vccl::t_groupDepth;
  using vccl::t_groupOps;
  if (t_groupDepth == 0) return vcclInvalidUsage;
  if (--t_groupDepth > 0) return vcclSuccess;

  // Partition by communicator, preserving per-comm issue order.
  std::vector<vcclComm_t> order;
  std::map<vcclComm_t, std::vector<vccl::QueuedOp>> byComm;
  for (auto& op : t_groupOps) {
    if (byComm.find(op.comm) == byComm.end()) order.push_back(op.comm);
    byComm[op.comm].push_back(std::move(op));
  }
  t_groupOps.clear();

  if (order.empty()) return vcclSuccess;
  if (order.size() == 1) {
    return vccl::runCommOps(order[0], byComm[order[0]]);
  }
  std::vector<std::thread> threads;
  std::vector<vcclResult_t> results(order.size(), vcclSuccess);
  for (size_t i = 0; i < order.size(); i++) {
    threads.emplace_back([&, i] {
      results[i] = vccl::runCommOps(order[i], byComm[order[i]]);
    });
  }
  for (auto& t : threads) t.join();
  for (vcclResult_t res : results) {
    if (res != vcclSuccess) return res;
  }
  return vcclSuccess;
}

}  // extern "C"
