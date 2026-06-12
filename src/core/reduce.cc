/* SPDX-License-Identifier: Apache-2.0 */
#include "core/reduce.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace vccl {
namespace {

float halfToFloat(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal: normalize.
      int shift = 0;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        shift++;
      }
      mant &= 0x3ffu;
      bits = sign | ((127 - 15 - shift + 1) << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t floatToHalf(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;
  if (((bits >> 23) & 0xff) == 0xff) {  // Inf/NaN
    return sign | 0x7c00u | (mant != 0 ? 0x200u : 0);
  }
  if (exp >= 0x1f) return sign | 0x7c00u;  // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) return sign;  // underflow -> zero
    // Subnormal half.
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half & 1))) half++;
    return sign | static_cast<uint16_t>(half);
  }
  uint16_t half = sign | static_cast<uint16_t>(exp << 10) |
                  static_cast<uint16_t>(mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) half++;
  return half;
}

float bf16ToFloat(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t floatToBf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  if ((bits & 0x7fffffffu) > 0x7f800000u) {  // NaN: keep quiet
    return static_cast<uint16_t>((bits >> 16) | 0x40u);
  }
  uint32_t rounding = 0x7fffu + ((bits >> 16) & 1);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

template <typename T>
T applyOp(vcclRedOp_t op, T a, T b) {
  switch (op) {
    case vcclProd: return a * b;
    case vcclMax: return a > b ? a : b;
    case vcclMin: return a < b ? a : b;
    case vcclSum:
    case vcclAvg:
    default: return a + b;
  }
}

template <typename T>
void reduceNative(vcclRedOp_t op, T* dst, const T* src, size_t count) {
  for (size_t i = 0; i < count; i++) dst[i] = applyOp(op, dst[i], src[i]);
}

template <float (*ToFloat)(uint16_t), uint16_t (*FromFloat)(float)>
void reduce16(vcclRedOp_t op, uint16_t* dst, const uint16_t* src,
              size_t count) {
  for (size_t i = 0; i < count; i++)
    dst[i] = FromFloat(applyOp(op, ToFloat(dst[i]), ToFloat(src[i])));
}

}  // namespace

vcclResult_t cpuReduce(vcclDataType_t dt, vcclRedOp_t op, void* dst,
                       const void* src, size_t count) {
  if (op < 0 || op >= vcclNumOps) return vcclInvalidArgument;
  switch (dt) {
    case vcclInt8:
      reduceNative(op, static_cast<int8_t*>(dst),
                   static_cast<const int8_t*>(src), count);
      return vcclSuccess;
    case vcclUint8:
      reduceNative(op, static_cast<uint8_t*>(dst),
                   static_cast<const uint8_t*>(src), count);
      return vcclSuccess;
    case vcclInt32:
      reduceNative(op, static_cast<int32_t*>(dst),
                   static_cast<const int32_t*>(src), count);
      return vcclSuccess;
    case vcclUint32:
      reduceNative(op, static_cast<uint32_t*>(dst),
                   static_cast<const uint32_t*>(src), count);
      return vcclSuccess;
    case vcclInt64:
      reduceNative(op, static_cast<int64_t*>(dst),
                   static_cast<const int64_t*>(src), count);
      return vcclSuccess;
    case vcclUint64:
      reduceNative(op, static_cast<uint64_t*>(dst),
                   static_cast<const uint64_t*>(src), count);
      return vcclSuccess;
    case vcclFloat32:
      reduceNative(op, static_cast<float*>(dst),
                   static_cast<const float*>(src), count);
      return vcclSuccess;
    case vcclFloat64:
      reduceNative(op, static_cast<double*>(dst),
                   static_cast<const double*>(src), count);
      return vcclSuccess;
    case vcclFloat16:
      reduce16<halfToFloat, floatToHalf>(op, static_cast<uint16_t*>(dst),
                                         static_cast<const uint16_t*>(src),
                                         count);
      return vcclSuccess;
    case vcclBfloat16:
      reduce16<bf16ToFloat, floatToBf16>(op, static_cast<uint16_t*>(dst),
                                         static_cast<const uint16_t*>(src),
                                         count);
      return vcclSuccess;
    default:
      return vcclInvalidArgument;
  }
}

vcclResult_t cpuScaledAccumulate(vcclDataType_t dt, void* dst,
                                 const void* src, size_t count,
                                 double preScale) {
  switch (dt) {
    case vcclFloat32: {
      float* d = static_cast<float*>(dst);
      const float* s = static_cast<const float*>(src);
      const float k = static_cast<float>(preScale);
      for (size_t i = 0; i < count; i++) d[i] += k * s[i];
      return vcclSuccess;
    }
    case vcclFloat64: {
      double* d = static_cast<double*>(dst);
      const double* s = static_cast<const double*>(src);
      for (size_t i = 0; i < count; i++) d[i] += preScale * s[i];
      return vcclSuccess;
    }
    case vcclFloat16: {
      uint16_t* d = static_cast<uint16_t*>(dst);
      const uint16_t* s = static_cast<const uint16_t*>(src);
      const float k = static_cast<float>(preScale);
      for (size_t i = 0; i < count; i++)
        d[i] = floatToHalf(halfToFloat(d[i]) + k * halfToFloat(s[i]));
      return vcclSuccess;
    }
    case vcclBfloat16: {
      uint16_t* d = static_cast<uint16_t*>(dst);
      const uint16_t* s = static_cast<const uint16_t*>(src);
      const float k = static_cast<float>(preScale);
      for (size_t i = 0; i < count; i++)
        d[i] = floatToBf16(bf16ToFloat(d[i]) + k * bf16ToFloat(s[i]));
      return vcclSuccess;
    }
    default:
      return vcclInvalidArgument;
  }
}

vcclResult_t cpuScaleCopy(vcclDataType_t dt, void* dst, const void* src,
                          size_t count, double factor) {
  switch (dt) {
    case vcclFloat32: {
      float* d = static_cast<float*>(dst);
      const float* s = static_cast<const float*>(src);
      const float k = static_cast<float>(factor);
      for (size_t i = 0; i < count; i++) d[i] = k * s[i];
      return vcclSuccess;
    }
    case vcclFloat64: {
      double* d = static_cast<double*>(dst);
      const double* s = static_cast<const double*>(src);
      for (size_t i = 0; i < count; i++) d[i] = factor * s[i];
      return vcclSuccess;
    }
    case vcclFloat16: {
      uint16_t* d = static_cast<uint16_t*>(dst);
      const uint16_t* s = static_cast<const uint16_t*>(src);
      const float k = static_cast<float>(factor);
      for (size_t i = 0; i < count; i++)
        d[i] = floatToHalf(k * halfToFloat(s[i]));
      return vcclSuccess;
    }
    case vcclBfloat16: {
      uint16_t* d = static_cast<uint16_t*>(dst);
      const uint16_t* s = static_cast<const uint16_t*>(src);
      const float k = static_cast<float>(factor);
      for (size_t i = 0; i < count; i++)
        d[i] = floatToBf16(k * bf16ToFloat(s[i]));
      return vcclSuccess;
    }
    default:
      return vcclInvalidArgument;
  }
}

vcclResult_t cpuScale(vcclDataType_t dt, void* dst, size_t count,
                      double factor) {
  switch (dt) {
    case vcclFloat32: {
      float* p = static_cast<float*>(dst);
      for (size_t i = 0; i < count; i++) p[i] *= static_cast<float>(factor);
      return vcclSuccess;
    }
    case vcclFloat64: {
      double* p = static_cast<double*>(dst);
      for (size_t i = 0; i < count; i++) p[i] *= factor;
      return vcclSuccess;
    }
    case vcclFloat16: {
      uint16_t* p = static_cast<uint16_t*>(dst);
      for (size_t i = 0; i < count; i++)
        p[i] = floatToHalf(halfToFloat(p[i]) * static_cast<float>(factor));
      return vcclSuccess;
    }
    case vcclBfloat16: {
      uint16_t* p = static_cast<uint16_t*>(dst);
      for (size_t i = 0; i < count; i++)
        p[i] = floatToBf16(bf16ToFloat(p[i]) * static_cast<float>(factor));
      return vcclSuccess;
    }
    default:
      // vcclAvg is only defined for floating-point types.
      return vcclInvalidArgument;
  }
}

}  // namespace vccl
