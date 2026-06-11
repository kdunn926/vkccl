/* SPDX-License-Identifier: Apache-2.0 */
#include "util/log.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vccl {

LogLevel logLevel() {
  static LogLevel level = [] {
    const char* env = std::getenv("VCCL_DEBUG");
    if (env == nullptr) return LogLevel::kWarn;
    if (strcasecmp(env, "NONE") == 0) return LogLevel::kNone;
    if (strcasecmp(env, "ERROR") == 0) return LogLevel::kError;
    if (strcasecmp(env, "WARN") == 0) return LogLevel::kWarn;
    if (strcasecmp(env, "INFO") == 0) return LogLevel::kInfo;
    if (strcasecmp(env, "TRACE") == 0) return LogLevel::kTrace;
    return LogLevel::kWarn;
  }();
  return level;
}

void logMsg(LogLevel level, const char* fmt, ...) {
  static const char* names[] = {"NONE", "ERROR", "WARN", "INFO", "TRACE"};
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  fprintf(stderr, "vccl [%d] %s: %s\n", static_cast<int>(getpid()),
          names[static_cast<int>(level)], buf);
}

}  // namespace vccl
