// SPDX-License-Identifier: Apache-2.0
#include "model_manager.hpp"
#include "trt_runner.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

ModelManager::ModelManager(const std::string& yaml_path):yaml_path_(yaml_path){
  YAML::Node n = YAML::LoadFile(yaml_path_);
  if(!n["models"]) throw std::runtime_error("models.yaml missing 'models'");
  for(auto m : n["models"]){
    ModelCfg c;
    c.id        = m["id"].as<std::string>();
    c.engine    = m["engine"].as<std::string>();
    c.concurrency = m["concurrency"] ? m["concurrency"].as<int>() : 1;
    cfgs_.push_back(std::move(c));
  }
}

TRTRunner& ModelManager::get_or_load(const std::string& id){
  if(auto it=runners_.find(id); it!=runners_.end()) return *it->second;
  for(auto& c: cfgs_) if(c.id==id){
    auto r = std::make_unique<TRTRunner>(c.engine, c.concurrency);
    auto* p=r.get(); runners_[id]=std::move(r); return *p;
  }
  throw std::runtime_error("unknown model id: " + id);
}
