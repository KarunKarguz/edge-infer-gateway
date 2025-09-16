// SPDX-License-Identifier: Apache-2.0
#include "gateway.hpp"
#include <iostream>
#include <cstring>

int main(int argc, char** argv){
  GatewayOpts opt;
  for(int i=1;i<argc;i++){
    std::string a = argv[i];
    if((a=="-c"||a=="--config") && i+1<argc) opt.config_yaml = argv[++i];
    else if((a=="-p"||a=="--port") && i+1<argc) opt.port = std::stoi(argv[++i]);
    else if(a=="-h"||a=="--help"){
      std::cout << "Usage: " << argv[0] << " [-c config/models.yaml] [-p 8008]\n";
      return 0;
    }
  }
  return run_gateway(opt);
}
