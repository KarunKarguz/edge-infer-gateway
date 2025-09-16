// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

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
  explicit TRTRunner(const std::string& engine_path);
  ~TRTRunner();

  const std::vector<Binding>& inputs()  const { return inputs_; }
  const std::vector<Binding>& outputs() const { return outputs_; }

  // H2D/D2H inside; host buffers must match sizes()
  void infer(const std::vector<const void*>& host_inputs,
             const std::vector<void*>&       host_outputs);

private:
  std::string engine_path_;
  nvinfer1::IRuntime*           runtime_{nullptr};
  nvinfer1::ICudaEngine*        engine_{nullptr};
  nvinfer1::IExecutionContext*  ctx_{nullptr};
  cudaStream_t                  stream_{};

  std::vector<Binding> inputs_;
  std::vector<Binding> outputs_;

  void load_engine();
  static size_t vol(const nvinfer1::Dims& d);
  static size_t dtype_bytes(nvinfer1::DataType t);
};
