// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>

struct GatewayOpts {
  std::string config_yaml{"config/models.yaml"};
  std::string host{"0.0.0.0"};
  int  port{8008};
  int  http_port{8080};
  int  max_clients{256};
  int  read_timeout_ms{30000};
  int  write_timeout_ms{30000};
  int  queue_depth{1024};
};

int run_gateway(const GatewayOpts& opt);
