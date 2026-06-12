/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VCCL_CORE_REDUCE_H_
#define VCCL_CORE_REDUCE_H_

#include <cstddef>

#include "vccl.h"

namespace vccl {

// dst[i] = dst[i] (op) src[i] for count elements. vcclAvg accumulates as sum;
// the caller applies the final 1/nranks scale via cpuScale.
vcclResult_t cpuReduce(vcclDataType_t dt, vcclRedOp_t op, void* dst,
                       const void* src, size_t count);

// dst[i] = dst[i] + preScale*src[i] (the PreMulSum accumulation step).
// Float datatypes only.
vcclResult_t cpuScaledAccumulate(vcclDataType_t dt, void* dst,
                                 const void* src, size_t count,
                                 double preScale);

// dst[i] = factor*src[i] (PreMulSum's scaling of the local contribution).
// dst may equal src. Float datatypes only.
vcclResult_t cpuScaleCopy(vcclDataType_t dt, void* dst, const void* src,
                          size_t count, double factor);

// dst[i] *= factor, for the final averaging step. Float types only.
vcclResult_t cpuScale(vcclDataType_t dt, void* dst, size_t count,
                      double factor);

}  // namespace vccl

#endif  // VCCL_CORE_REDUCE_H_
