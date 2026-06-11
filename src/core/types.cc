/* SPDX-License-Identifier: Apache-2.0 */
#include "vccl.h"

extern "C" {

const char* vcclGetErrorString(vcclResult_t result) {
  switch (result) {
    case vcclSuccess: return "no error";
    case vcclUnhandledError: return "unhandled error";
    case vcclSystemError: return "system error";
    case vcclInternalError: return "internal error";
    case vcclInvalidArgument: return "invalid argument";
    case vcclInvalidUsage: return "invalid usage";
    case vcclRemoteError: return "remote process error";
    default: return "unknown result code";
  }
}

size_t vcclTypeSize(vcclDataType_t datatype) {
  switch (datatype) {
    case vcclInt8:
    case vcclUint8: return 1;
    case vcclFloat16:
    case vcclBfloat16: return 2;
    case vcclInt32:
    case vcclUint32:
    case vcclFloat32: return 4;
    case vcclInt64:
    case vcclUint64:
    case vcclFloat64: return 8;
    default: return 0;
  }
}

}  // extern "C"
