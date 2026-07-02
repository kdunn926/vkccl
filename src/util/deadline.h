/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_UTIL_DEADLINE_H_
#define VCCL_UTIL_DEADLINE_H_

#include <chrono>
#include <cstdlib>

namespace vccl {

// Collective/transport watchdog from VCCL_TIMEOUT (seconds). Default 600;
// 0 disables. A dead or wedged peer then fails the operation with an error
// instead of hanging the job forever.
inline int timeoutSeconds() {
  static int secs = [] {
    const char* env = std::getenv("VCCL_TIMEOUT");
    return env != nullptr ? atoi(env) : 600;
  }();
  return secs;
}

// Timeout for a single blocking connect(), in ms: VCCL_TIMEOUT capped at
// 120 s, or 120 s when the watchdog is disabled (VCCL_TIMEOUT=0).
inline int connectTimeoutMs() {
  int secs = timeoutSeconds();
  if (secs <= 0 || secs > 120) return 120000;
  return secs * 1000;
}

class Deadline {
 public:
  Deadline() {
    int secs = timeoutSeconds();
    enabled_ = secs > 0;
    if (enabled_) {
      end_ = std::chrono::steady_clock::now() + std::chrono::seconds(secs);
    }
  }

  bool expired() const {
    return enabled_ && std::chrono::steady_clock::now() >= end_;
  }

  // Milliseconds left, clamped for poll(); -1 (infinite) when disabled.
  int pollMs() const {
    if (!enabled_) return -1;
    auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_ - std::chrono::steady_clock::now())
                    .count();
    if (left <= 0) return 0;
    return left > 60000 ? 60000 : static_cast<int>(left);
  }

 private:
  bool enabled_ = false;
  std::chrono::steady_clock::time_point end_;
};

}  // namespace vccl

#endif  // VCCL_UTIL_DEADLINE_H_
