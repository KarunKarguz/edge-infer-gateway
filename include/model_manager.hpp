// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class TRTRunner;

struct ModelCfg {
  std::string id;
  std::string engine;
  int concurrency{1};
};

class ModelManager {
public:
  explicit ModelManager(const std::string& yaml_path);
  TRTRunner& get_or_load(const std::string& id);

private:
  std::string yaml_path_;
  std::vector<ModelCfg> cfgs_;
  std::unordered_map<std::string, std::unique_ptr<TRTRunner>> runners_;
};
