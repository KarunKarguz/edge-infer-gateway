// SPDX-License-Identifier: Apache-2.0
#include "gateway.hpp"
#include "model_manager.hpp"
#include "protocol.hpp"

#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>

using namespace std::chrono;

static int64_t ms_since(high_resolution_clock::time_point t0){
  return duration_cast<milliseconds>(high_resolution_clock::now()-t0).count();
}

static ssize_t read_full(int fd, void* buf, size_t n){
  char* p = (char*)buf; size_t left = n;
  while(left){
    ssize_t k = ::recv(fd, p, left, 0);
    if(k==0) return -1;
    if(k<0){ if(errno==EINTR) continue; return -1; }
    p += k; left -= (size_t)k;
  }
  return 0;
}
static ssize_t write_all(int fd, const void* buf, size_t n){
  const char* p=(const char*)buf; size_t left=n;
  while(left){
    ssize_t k = ::send(fd, p, left, MSG_NOSIGNAL);
    if(k<=0){ if(errno==EINTR) continue; return -1; }
    p += k; left -= (size_t)k;
  }
  return 0;
}

int run_gateway(const GatewayOpts& opt){
  ModelManager mm(opt.config_yaml);

  int sfd = ::socket(AF_INET, SOCK_STREAM|SOCK_NONBLOCK, 0);
  int one=1; setsockopt(sfd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
  sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_addr.s_addr=INADDR_ANY; addr.sin_port=htons(opt.port);
  if(::bind(sfd,(sockaddr*)&addr,sizeof(addr))<0){ perror("bind"); return 1; }
  if(::listen(sfd, 128)<0){ perror("listen"); return 1; }

  int ep = ::epoll_create1(0);
  epoll_event ev{}; ev.events = EPOLLIN; ev.data.fd = sfd;
  epoll_ctl(ep, EPOLL_CTL_ADD, sfd, &ev);

  std::cout << "edge-infer-gateway listening on 0.0.0.0:" << opt.port << "\n";

  uint64_t nreq=0; double sum=0, sumsq=0; double minv=1e9, maxv=0;

  while(true){
    epoll_event events[64];
    int n = ::epoll_wait(ep, events, 64, 500);
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
      // handle request (single request per connection for MVP)
      proto::MsgHdr hdr{};
      if(read_full(fd,&hdr,sizeof(hdr))<0){ ::close(fd); continue; }
      if(std::memcmp(hdr.magic,"TRT\1",4)!=0 || hdr.version!=1){ ::close(fd); continue; }

      std::string model_id(hdr.model_len,'\0');
      if(read_full(fd,model_id.data(),hdr.model_len)<0){ ::close(fd); continue; }

      std::vector<proto::TensorDesc> in_descs;
      in_descs.reserve(hdr.n_inputs);
      for(uint32_t j=0;j<hdr.n_inputs;++j){
        uint8_t dt=0, nd=0; if(read_full(fd,&dt,1)<0 || read_full(fd,&nd,1)<0){ ::close(fd); goto next; }
        std::vector<int32_t> dims(nd);
        if(read_full(fd,dims.data(),nd*sizeof(int32_t))<0){ ::close(fd); goto next; }
        uint32_t blen=0; if(read_full(fd,&blen,4)<0){ ::close(fd); goto next; }
        in_descs.push_back({(proto::DType)dt, std::move(dims), blen});
      }

      // read payload
      size_t tot_in_bytes=0; for(auto& d:in_descs) tot_in_bytes+=d.byte_len;
      std::vector<char> payload(tot_in_bytes);
      if(read_full(fd,payload.data(),payload.size())<0){ ::close(fd); continue; }

      // route to model
      auto t0 = high_resolution_clock::now();
      TRTRunner& r = mm.get_or_load(model_id);

      // map host ptrs
      std::vector<const void*> h_in; h_in.reserve(in_descs.size());
      size_t off=0;
      for(auto& d: in_descs){ h_in.push_back(payload.data()+off); off+=d.byte_len; }

      std::vector<std::vector<char>> out_host;
      std::vector<void*> h_out;
      for(auto& b: r.outputs()){
        out_host.emplace_back(b.bytes);
        h_out.push_back(out_host.back().data());
      }

      try { r.infer(h_in, h_out); }
      catch(const std::exception& e){ ::close(fd); continue; }

      // response: simple header (status + n_outputs + each output byte_len) then raw bytes
      uint32_t status=0, nout=(uint32_t)out_host.size();
      if(write_all(fd,&status,4)<0 || write_all(fd,&nout,4)<0){ ::close(fd); continue; }
      for(auto& b: r.outputs()){
        uint32_t blen = (uint32_t)b.bytes;
        if(write_all(fd,&blen,4)<0){ ::close(fd); continue; }
      }
      for(auto& v: out_host){
        if(write_all(fd,v.data(),v.size())<0){ ::close(fd); continue; }
      }
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);

      // metrics
      double ms = (double)ms_since(t0);
      nreq++; sum+=ms; sumsq+=ms*ms; if(ms<minv)minv=ms; if(ms>maxv)maxv=ms;
      if(nreq%100==0){
        double mean=sum/nreq; double var=(sumsq/nreq)-mean*mean; double stdv=var>0?std::sqrt(var):0;
        std::cout << "[metrics] n="<<nreq<<" mean="<<mean<<"ms std="<<stdv<<" min="<<minv<<" max="<<maxv
                  <<" qps~="<<(mean>0?1000.0/mean:0) << "\n";
      }
      next: ;
    }
  }

  ::close(ep); ::close(sfd);
  return 0;
}
