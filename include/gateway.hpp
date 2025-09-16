// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>

class ModelManager;

struct GatewayOpts {
  std::string config_yaml{"config/models.yaml"};
  std::string host{"0.0.0.0"};
  int port{8008};
};

int run_gateway(const GatewayOpts& opt);
