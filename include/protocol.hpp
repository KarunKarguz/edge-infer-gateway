// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 edgeAdaptics
// Author: Karguz

#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace proto {

// Wire types
enum class DType : uint8_t { FP32=0, FP16=1, INT8=2, INT32=3 };

#pragma pack(push,1)
struct MsgHdr {
  char     magic[4];     // "TRT\1"
  uint16_t version;      // 1
  uint16_t flags;        // reserved
  uint32_t model_len;    // bytes of model_id ASCII
  uint32_t n_inputs;     // number of inputs
  uint32_t payload_len;  // bytes after header
};
#pragma pack(pop)

struct TensorDesc {
  DType dtype{DType::FP32};
  std::vector<int32_t> shape;   // NCHW, etc.
  uint32_t byte_len{0};
};

inline size_t dtype_size(DType t){
  switch(t){
    case DType::FP32: return 4;
    case DType::FP16: return 2;
    case DType::INT8: return 1;
    case DType::INT32:return 4;
  }
  return 4;
}

} // namespace proto
