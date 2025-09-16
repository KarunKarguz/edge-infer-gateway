// SPDX-License-Identifier: Apache-2.0
#include "gateway.hpp"
#include <iostream>

static const char* kVersion = "1.0.0";

int main(int argc, char** argv){
  GatewayOpts opt;
  for(int i=1;i<argc;i++){
    std::string a=argv[i];
    if((a=="-c"||a=="--config") && i+1<argc) opt.config_yaml=argv[++i];
    else if((a=="-p"||a=="--port") && i+1<argc) opt.port=std::stoi(argv[++i]);
    else if(a=="--http-port" && i+1<argc) opt.http_port=std::stoi(argv[++i]);
    else if(a=="--version"){ std::cout << "edge-infer-gateway " << kVersion << "\n"; return 0; }
    else if(a=="-h"||a=="--help"){
      std::cout << "Usage: " << argv[0] << " [-c config.yaml] [-p 8008] [--http-port 8080] [--version]\n";
      return 0;
    }
  }
  return run_gateway(opt);
}
