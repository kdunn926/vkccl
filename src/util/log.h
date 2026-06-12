/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_UTIL_LOG_H_
#define VCCL_UTIL_LOG_H_

#include <cstdarg>

#include "vccl.h"

namespace vccl {

enum class LogLevel { kNone = 0, kError = 1, kWarn = 2, kInfo = 3, kTrace = 4 };

// Controlled by VCCL_DEBUG = NONE|ERROR|WARN|INFO|TRACE (default WARN).
LogLevel logLevel();

void logMsg(LogLevel level, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

// Most recent error message logged by this thread (for vcclGetLastError).
const char* lastErrorString();

}  // namespace vccl

#define VCCL_LOG(level, ...)                                  \
  do {                                                        \
    if (::vccl::logLevel() >= ::vccl::LogLevel::level)        \
      ::vccl::logMsg(::vccl::LogLevel::level, __VA_ARGS__);   \
  } while (0)

#define VCCL_ERR(...) VCCL_LOG(kError, __VA_ARGS__)
#define VCCL_WARN(...) VCCL_LOG(kWarn, __VA_ARGS__)
#define VCCL_INFO(...) VCCL_LOG(kInfo, __VA_ARGS__)
#define VCCL_TRACE(...) VCCL_LOG(kTrace, __VA_ARGS__)

// Propagate a vcclResult_t failure out of the current function.
#define VCCLCHECK(call)                                                     \
  do {                                                                      \
    vcclResult_t res_ = (call);                                             \
    if (res_ != vcclSuccess) {                                              \
      VCCL_ERR("%s:%d -> %s", __FILE__, __LINE__, vcclGetErrorString(res_)); \
      return res_;                                                          \
    }                                                                       \
  } while (0)

#endif  // VCCL_UTIL_LOG_H_
