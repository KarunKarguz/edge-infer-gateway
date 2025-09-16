// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 edgeAdaptics
// Author: Karguz
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>

struct Binding {
  std::string name;
  bool isInput{false};
  nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
  nvinfer1::Dims dims{};
  size_t bytes{0};
  void*  dptr{nullptr};
};

class TRTRunner {
public:
  TRTRunner(const std::string& engine_path, int concurrency);
  ~TRTRunner();

  const std::vector<Binding>& inputs()  const { return inputs_; }
  const std::vector<Binding>& outputs() const { return outputs_; }

  // Validates shapes/bytes, runs inference using a checked-out context.
  void infer(const std::vector<const void*>& host_inputs,
             const std::vector<void*>&       host_outputs);

private:
  struct Ctx {
    nvinfer1::IExecutionContext* ctx{nullptr};
    cudaStream_t stream{};
  };

  std::string engine_path_;
  nvinfer1::IRuntime*    runtime_{nullptr};
  nvinfer1::ICudaEngine* engine_{nullptr};

  // device bindings (one set per engine binding; reused by all contexts)
  std::vector<Binding> inputs_;
  std::vector<Binding> outputs_;

  // pool of execution contexts
  std::vector<Ctx> pool_;
  std::mutex m_;
  std::condition_variable cv_;
  std::vector<bool> in_use_;

  void load_engine(int concurrency);
  static size_t vol(const nvinfer1::Dims& d);
  static size_t dtype_bytes(nvinfer1::DataType t);

  int checkout();
  void checkin(int idx);

  void validate_host_io(const std::vector<const void*>& host_inputs,
                        const std::vector<void*>&       host_outputs);
};
