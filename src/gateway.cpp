// SPDX-License-Identifier: Apache-2.0
#include "gateway.hpp"
#include "model_manager.hpp"
#include "protocol.hpp"
#include "trt_runner.hpp"

#include <yaml-cpp/yaml.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#include <atomic>
#include <iostream>
#include <chrono>
#include <queue>
#include <thread>

using namespace std::chrono;
static std::atomic<bool> gStop{false};
static void on_sig(int){ gStop=true; }

static uint64_t now_ms(){
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static void log_json(const char* lvl, const std::string& msg, uint32_t req=0){
  std::cout << "{"
            << "\"ts\":" << now_ms() << ","
            << "\"level\":\"" << lvl << "\","
            << "\"msg\":\"" << msg << "\","
            << "\"req_id\":" << req
            << "}\n";
}

static int make_tcp_listener(const char* ip, int port, int backlog){
  int fd = ::socket(AF_INET, SOCK_STREAM|SOCK_NONBLOCK, 0);
  int one=1; setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
  sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(port);
  a.sin_addr.s_addr = (ip && std::strcmp(ip,"0.0.0.0")!=0) ? inet_addr(ip) : INADDR_ANY;
  if(::bind(fd,(sockaddr*)&a,sizeof(a))<0){ perror("bind"); std::exit(2); }
  if(::listen(fd, backlog)<0){ perror("listen"); std::exit(2); }
  return fd;
}

// tiny HTTP server for health/metrics (best-effort)
static void http_thread(int port, std::atomic<uint64_t>& ok, std::atomic<uint64_t>& errs){
  int sfd = make_tcp_listener("0.0.0.0", port, 16);
  char buf[1024];
  while(!gStop){
    int cfd = ::accept4(sfd,nullptr,nullptr,SOCK_NONBLOCK);
    if(cfd<0){ std::this_thread::sleep_for(50ms); continue; }
    ssize_t n = ::recv(cfd, buf, sizeof(buf), 0);
    std::string resp;
    if(n>0){
      std::string req(buf, buf+n);
      if(req.find("GET /healthz")!=std::string::npos){
        resp = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nok\n";
      }else if(req.find("GET /readyz")!=std::string::npos){
        resp = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nready\n";
      }else if(req.find("GET /metrics")!=std::string::npos){
        char m[256];
        std::snprintf(m,sizeof(m),
          "eig_requests_total %llu\n"
          "eig_errors_total %llu\n",
          (unsigned long long)ok.load(), (unsigned long long)errs.load());
        resp = std::string("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n")+m;
      }else{
        resp = "HTTP/1.1 404 Not Found\r\n\r\n";
      }
    }
    ::send(cfd, resp.data(), resp.size(), MSG_NOSIGNAL);
    ::shutdown(cfd, SHUT_RDWR); ::close(cfd);
  }
  ::close(sfd);
}

static bool recvn(int fd, void* buf, size_t n, int timeout_ms){
  char* p=(char*)buf; size_t left=n; auto t0=steady_clock::now();
  while(left){
    ssize_t k=::recv(fd,p,left,0);
    if(k==0) return false;
    if(k<0){
      if(errno==EINTR) continue;
      if(errno==EAGAIN||errno==EWOULDBLOCK){
        if(duration_cast<milliseconds>(steady_clock::now()-t0).count() > timeout_ms) return false;
        std::this_thread::sleep_for(1ms); continue;
      }
      return false;
    }
    p+=k; left-=k;
  }
  return true;
}
static bool sendall(int fd, const void* buf, size_t n, int timeout_ms){
  const char* p=(const char*)buf; size_t left=n; auto t0=steady_clock::now();
  while(left){
    ssize_t k=::send(fd,p,left,MSG_NOSIGNAL);
    if(k<=0){
      if(errno==EINTR) continue;
      if(errno==EAGAIN||errno==EWOULDBLOCK){
        if(duration_cast<milliseconds>(steady_clock::now()-t0).count() > timeout_ms) return false;
        std::this_thread::sleep_for(1ms); continue;
      }
      return false;
    }
    p+=k; left-=k;
  }
  return true;
}

int run_gateway(const GatewayOpts& opt){
  // Load server opts from YAML + env overrides
  YAML::Node conf = YAML::LoadFile(opt.config_yaml);
  GatewayOpts s = opt;
  if(auto sN=conf["server"]){
    if(sN["port"]) s.port = sN["port"].as<int>();
    if(sN["http_port"]) s.http_port = sN["http_port"].as<int>();
    if(sN["max_clients"]) s.max_clients = sN["max_clients"].as<int>();
    if(sN["read_timeout_ms"]) s.read_timeout_ms = sN["read_timeout_ms"].as<int>();
    if(sN["write_timeout_ms"]) s.write_timeout_ms = sN["write_timeout_ms"].as<int>();
    if(sN["queue_depth"]) s.queue_depth = sN["queue_depth"].as<int>();
  }
  if(const char* p=getenv("EIG_PORT")) s.port = std::atoi(p);
  if(const char* p=getenv("EIG_HTTP_PORT")) s.http_port = std::atoi(p);

  std::signal(SIGINT,  on_sig);
  std::signal(SIGTERM, on_sig);

  ModelManager mm(opt.config_yaml);
  int sfd = make_tcp_listener(s.host.c_str(), s.port, s.max_clients);
  int ep  = ::epoll_create1(0);
  epoll_event ev{}; ev.events=EPOLLIN; ev.data.fd=sfd;
  epoll_ctl(ep, EPOLL_CTL_ADD, sfd, &ev);

  std::atomic<uint64_t> ok{0}, errs{0};
  std::thread httpd(http_thread, s.http_port, std::ref(ok), std::ref(errs));

  auto send_status = [&](int fd, uint32_t req_id, uint32_t status)->bool{
    uint32_t nout = 0;
    uint32_t payload = 12; // req_id + status + nout
    std::vector<char> resp(4 + payload);
    char* rp = resp.data();
    std::memcpy(rp, &payload,4); rp+=4;
    std::memcpy(rp, &req_id,4); rp+=4;
    std::memcpy(rp, &status,4); rp+=4;
    std::memcpy(rp, &nout,4);
    return sendall(fd, resp.data(), resp.size(), s.write_timeout_ms);
  };

  auto handle_request = [&](int fd)->bool {
    uint32_t frame_len=0;
    if(!recvn(fd,&frame_len,4,s.read_timeout_ms)) return false;
    std::vector<char> frame(frame_len);
    if(!recvn(fd,frame.data(),frame.size(),s.read_timeout_ms)) return false;

    const char* p = frame.data();
    const char* end = p + frame.size();
    if((end - p) < (int)sizeof(proto::MsgHdr)){ errs++; log_json("WARN","bad header"); return false; }
    const proto::MsgHdr* H = reinterpret_cast<const proto::MsgHdr*>(p);
    p += sizeof(proto::MsgHdr);
    if(std::memcmp(H->magic,"TRT\1",4)!=0 || H->version!=1){ errs++; log_json("WARN","bad magic/version", H->n_inputs); return false; }
    uint32_t req_id = 0; // extend protocol when ready

    if((end - p) < (int)H->model_len){ errs++; log_json("WARN","short model_id",req_id); return false; }
    std::string model_id(p, p+H->model_len); p += H->model_len;

    struct InDesc{ uint8_t dt, nd; std::vector<int32_t> dims; uint32_t blen; };
    std::vector<InDesc> ind; ind.reserve(H->n_inputs);
    for(uint32_t j=0;j<H->n_inputs;j++){
      if((end - p) < 2){ errs++; log_json("WARN","short tensor desc",req_id); return false; }
      uint8_t dt=*reinterpret_cast<const uint8_t*>(p); p++;
      uint8_t nd=*reinterpret_cast<const uint8_t*>(p); p++;
      if(nd>8){ errs++; log_json("WARN","ndims>8",req_id); return false; }
      if((end - p) < nd*4){ errs++; log_json("WARN","short dims",req_id); return false; }
      std::vector<int32_t> dims(nd);
      std::memcpy(dims.data(), p, nd*4); p += nd*4;
      if((end - p) < 4){ errs++; log_json("WARN","short blen",req_id); return false; }
      uint32_t blen=*reinterpret_cast<const uint32_t*>(p); p+=4;
      ind.push_back({dt,nd,std::move(dims),blen});
    }
    size_t payload_left = end - p;
    size_t want=0; for(auto& d:ind) want += d.blen;
    if(payload_left < want){ errs++; log_json("WARN","short payload",req_id); return false; }

    std::vector<const void*> h_in; h_in.reserve(ind.size());
    std::vector<std::vector<char>> in_slices; in_slices.reserve(ind.size());
    for(auto& d: ind){
      in_slices.emplace_back(d.blen);
      std::memcpy(in_slices.back().data(), p, d.blen);
      p += d.blen;
      h_in.push_back(in_slices.back().data());
    }

    TRTRunner* runner=nullptr;
    try { runner = &mm.get_or_load(model_id); }
    catch(...){
      if(!send_status(fd, req_id, 2)) return false;
      return true;
    }

    std::vector<std::vector<char>> out_host;
    std::vector<void*> h_out;
    for(auto& b: runner->outputs()){
      out_host.emplace_back(b.bytes);
      h_out.push_back(out_host.back().data());
    }

    auto t0 = steady_clock::now();
    try { runner->infer(h_in, h_out); }
    catch(...){
      errs++;
      if(!send_status(fd, req_id, 4)) return false;
      return true;
    }
    double ms = duration<double,std::milli>(steady_clock::now()-t0).count();
    ok++; log_json("INFO", "infer_ok ms="+std::to_string(ms), req_id);

    uint32_t nout = (uint32_t)out_host.size();
    uint32_t payload_bytes = 12 + 4*nout; // req_id+status+nout + lens
    for(auto& v: out_host) payload_bytes += v.size();
    uint32_t resp_frame_len = payload_bytes;
    std::vector<char> R(4 + resp_frame_len);
    char* rp = R.data();
    std::memcpy(rp, &resp_frame_len,4); rp+=4;
    uint32_t status=0;
    std::memcpy(rp,&req_id,4); rp+=4;
    std::memcpy(rp,&status,4); rp+=4;
    std::memcpy(rp,&nout,4);   rp+=4;
    for(auto& v: out_host){ uint32_t L=(uint32_t)v.size(); std::memcpy(rp,&L,4); rp+=4; }
    for(auto& v: out_host){ std::memcpy(rp, v.data(), v.size()); rp+=v.size(); }
    if(!sendall(fd,R.data(),R.size(),s.write_timeout_ms)) return false;
    return true;
  };

  log_json("INFO", "edge-infer-gateway started");
  while(!gStop){
    epoll_event events[128];
    int n = ::epoll_wait(ep, events, 128, 500);
    if(n<0){ if(errno==EINTR) continue; perror("epoll"); break; }
    for(int i=0;i<n;i++){
      int fd = events[i].data.fd;
      if(fd==sfd){
        int cfd = ::accept4(sfd,nullptr,nullptr,0);
        if(cfd<0) continue;
        epoll_event cev{}; cev.events=EPOLLIN; cev.data.fd=cfd;
        epoll_ctl(ep, EPOLL_CTL_ADD, cfd, &cev);
        continue;
      }
      // ===== per-connection: loop until client closes =====
      while(true){
        if(!handle_request(fd)){ ::shutdown(fd,SHUT_RDWR); ::close(fd); break; }
      }
    }
  }

  ::close(ep); ::close(sfd);
  gStop=true; httpd.join();
  log_json("INFO","edge-infer-gateway stopped");
  return 0;
}
