// SPDX-License-Identifier: Apache-2.0
#include "trt_runner.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>
using namespace nvinfer1;

// Minimal TensorRT logger
class Logger : public ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << "\n";
    }
  }
};
static Logger gLogger;

static std::vector<char> read_file(const std::string& p){
  std::ifstream f(p, std::ios::binary);
  if(!f) throw std::runtime_error("engine file not found: " + p);
  f.seekg(0, std::ios::end); size_t n=f.tellg(); f.seekg(0);
  std::vector<char> b(n); f.read(b.data(), n); return b;
}

size_t TRTRunner::vol(const Dims& d){ size_t v=1; for(int i=0;i<d.nbDims;i++) v*=d.d[i]; return v; }
size_t TRTRunner::dtype_bytes(DataType t){
  switch(t){
    case DataType::kFLOAT: return 4;
    case DataType::kHALF:  return 2;
    case DataType::kINT8:  return 1;
    case DataType::kINT32: return 4;
    default: return 4;
  }
}

TRTRunner::TRTRunner(const std::string& engine_path, int concurrency)
: engine_path_(engine_path) { load_engine(concurrency); }

void TRTRunner::load_engine(int concurrency){
  auto blob = read_file(engine_path_);
  runtime_ = createInferRuntime(gLogger);
  if(!runtime_) throw std::runtime_error("createInferRuntime failed");
  engine_  = runtime_->deserializeCudaEngine(blob.data(), blob.size());
  if(!engine_) throw std::runtime_error("deserializeCudaEngine failed");

  // bindings (one set of device buffers shared across contexts)
  int nb = engine_->getNbBindings();
  for(int i=0;i<nb;i++){
    Binding b;
    b.name   = engine_->getBindingName(i);
    b.isInput= engine_->bindingIsInput(i);
    b.dtype  = engine_->getBindingDataType(i);
    b.dims   = engine_->getBindingDimensions(i);
    b.bytes  = vol(b.dims) * dtype_bytes(b.dtype);
    cudaMalloc(&b.dptr, b.bytes);
    (b.isInput ? inputs_ : outputs_).push_back(b);
  }

  // create execution contexts
  pool_.resize(concurrency);
  in_use_.assign(concurrency, false);
  for(int i=0;i<concurrency;i++){
    auto* ctx = engine_->createExecutionContext();
    if(!ctx) throw std::runtime_error("createExecutionContext failed");
    pool_[i].ctx = ctx;
    cudaStreamCreate(&pool_[i].stream);
  }
}

TRTRunner::~TRTRunner(){
  for(auto& b: inputs_)  if(b.dptr) cudaFree(b.dptr);
  for(auto& b: outputs_) if(b.dptr) cudaFree(b.dptr);
  for(auto& c: pool_){
    if(c.stream) cudaStreamDestroy(c.stream);
    if(c.ctx)    c.ctx->destroy();
  }
  if(engine_)  engine_->destroy();
  if(runtime_) runtime_->destroy();
}

int TRTRunner::checkout(){
  std::unique_lock<std::mutex> lk(m_);
  cv_.wait(lk, [&]{ for(bool u: in_use_) if(!u) return true; return false; });
  for(size_t i=0;i<in_use_.size();++i) if(!in_use_[i]) { in_use_[i]=true; return (int)i; }
  return -1;
}
void TRTRunner::checkin(int idx){
  {
    std::lock_guard<std::mutex> lk(m_);
    in_use_[idx]=false;
  }
  cv_.notify_one();
}

void TRTRunner::validate_host_io(const std::vector<const void*>& in,
                                 const std::vector<void*>& out){
  if(in.size()!=inputs_.size() || out.size()!=outputs_.size())
    throw std::runtime_error("io count mismatch");
  // (We rely on engine static dims; for dynamic profiles you’d set them here.)
}

void TRTRunner::infer(const std::vector<const void*>& h_in,
                      const std::vector<void*>&       h_out){
  validate_host_io(h_in, h_out);
  int idx = checkout();
  auto& ctx = pool_[idx];
  // H2D
  for(size_t i=0;i<inputs_.size();++i)
    cudaMemcpyAsync(inputs_[i].dptr, h_in[i], inputs_[i].bytes, cudaMemcpyHostToDevice, ctx.stream);
  // bindings array
  std::vector<void*> bindings(engine_->getNbBindings(), nullptr);
  int in_ord=0, out_ord=0;
  for(int i=0;i<engine_->getNbBindings();++i){
    if(engine_->bindingIsInput(i))  bindings[i] = inputs_[in_ord++].dptr;
    else                            bindings[i] = outputs_[out_ord++].dptr;
  }
  if(!ctx.ctx->enqueueV2(bindings.data(), ctx.stream, nullptr)){
    checkin(idx);
    throw std::runtime_error("enqueueV2 failed");
  }
  // D2H
  for(size_t i=0;i<outputs_.size();++i)
    cudaMemcpyAsync(h_out[i], outputs_[i].dptr, outputs_[i].bytes, cudaMemcpyDeviceToHost, ctx.stream);
  cudaStreamSynchronize(ctx.stream);
  checkin(idx);
}
