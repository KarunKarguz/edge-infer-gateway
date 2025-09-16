// SPDX-License-Identifier: Apache-2.0
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>

#ifdef USE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

static void die(const char* m){ perror(m); std::exit(1); }

static bool sendall(int fd, const void* buf, size_t n){
  const char* p=(const char*)buf; size_t left=n;
  while(left){
    ssize_t k=::send(fd,p,left,MSG_NOSIGNAL);
    if(k<=0){ if(errno==EINTR) continue; return false; }
    p+=k; left-=k;
  }
  return true;
}
static bool recvn(int fd, void* buf, size_t n){
  char* p=(char*)buf; size_t left=n;
  while(left){
    ssize_t k=::recv(fd,p,left,0);
    if(k<=0){ if(errno==EINTR) continue; return false; }
    p+=k; left-=k;
  }
  return true;
}

static std::vector<float> load_input_1x3x224x224(const std::string& imgPath){
  const int H=224, W=224;
  std::vector<float> nchw(1*3*H*W);
#ifdef USE_OPENCV
  cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
  if(img.empty()) throw std::runtime_error("failed to read image: "+imgPath);
  cv::resize(img, img, cv::Size(W,H), 0,0, cv::INTER_LINEAR);
  img.convertTo(img, CV_32FC3, 1.0/255.0);
  // BGR->RGB, HWC->CHW
  std::vector<cv::Mat> ch; cv::split(img, ch); // B,G,R
  // normalize (ImageNet)
  const float mean[3]={0.485f,0.456f,0.406f};
  const float stdv[3]={0.229f,0.224f,0.225f};
  for(int c=0;c<3;c++){
    for(int y=0;y<H;y++){
      for(int x=0;x<W;x++){
        float v = ch[2-c].at<float>(y,x); // swap to RGB
        v = (v - mean[c]) / stdv[c];
        nchw[c*H*W + y*W + x] = v;
      }
    }
  }
#else
  // fallback: random input
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> U(0,1);
  for(auto& v:nchw) v=U(rng);
#endif
  return nchw;
}

int main(int argc, char** argv){
  if(argc<5){
    std::cerr << "Usage: " << argv[0] << " --host 127.0.0.1 --port 8008 --model mobilenet_v2_cls [--image path]\n";
    return 1;
  }
  std::string host="127.0.0.1"; int port=8008; std::string model="mobilenet_v2_cls"; std::string image="";
  for(int i=1;i<argc;i++){
    std::string a=argv[i];
    if(a=="--host" && i+1<argc) host=argv[++i];
    else if(a=="--port" && i+1<argc) port=std::atoi(argv[++i]);
    else if(a=="--model" && i+1<argc) model=argv[++i];
    else if(a=="--image" && i+1<argc) image=argv[++i];
  }

  // Prepare input tensor (1x3x224x224 FP32)
  std::vector<float> input = load_input_1x3x224x224(image);

  // Build protocol frame (MAGIC "TRT\1", version 1)
  auto pack_u32=[&](uint32_t v, std::vector<char>& b){
    char x[4]; std::memcpy(x,&v,4); b.insert(b.end(),x,x+4);
  };
  std::vector<char> body;
  // header: magic(4) ver(2) flags(2) model_len(4) n_inputs(4) reserved(4)
  const char magic[4]={'T','R','T','\x01'};
  body.insert(body.end(), magic, magic+4);
  uint16_t ver=1, flags=0;
  body.insert(body.end(), (char*)&ver, (char*)&ver+2);
  body.insert(body.end(), (char*)&flags, (char*)&flags+2);
  pack_u32((uint32_t)model.size(), body);
  pack_u32(1, body);            // n_inputs
  pack_u32(0, body);            // reserved (req_id could live elsewhere if you extend)

  // model id
  body.insert(body.end(), model.begin(), model.end());

  // input desc
  uint8_t dtype=0; // fp32
  uint8_t nd=4;    // NCHW
  body.push_back((char)dtype);
  body.push_back((char)nd);
  int32_t dims[4]={1,3,224,224};
  body.insert(body.end(), (char*)dims, (char*)dims + 4*4);
  uint32_t blen = (uint32_t)(input.size()*sizeof(float));
  pack_u32(blen, body);
  // data
  body.insert(body.end(), (char*)input.data(), (char*)input.data()+blen);

  // frame: len + body
  uint32_t frame_len = (uint32_t)body.size();
  std::vector<char> frame; frame.reserve(4+frame_len);
  pack_u32(frame_len, frame);
  frame.insert(frame.end(), body.begin(), body.end());

  // connect
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if(fd<0) die("socket");
  sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(port);
  if(::inet_pton(AF_INET, host.c_str(), &a.sin_addr)!=1) die("inet_pton");
  if(::connect(fd, (sockaddr*)&a, sizeof(a))<0) die("connect");

  auto t0=std::chrono::high_resolution_clock::now();
  if(!sendall(fd, frame.data(), frame.size())) die("send");
  uint32_t rlen=0; if(!recvn(fd,&rlen,4)) die("read len");
  std::vector<char> resp(rlen);
  if(!recvn(fd, resp.data(), resp.size())) die("read payload");
  auto ms = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();

  const char* p=resp.data();
  uint32_t req_id=0, status=0, nout=0;
  std::memcpy(&req_id,p,4); p+=4;
  std::memcpy(&status,p,4); p+=4;
  std::memcpy(&nout,p,4);   p+=4;
  std::vector<uint32_t> lens(nout);
  for(uint32_t i=0;i<nout;i++){ std::memcpy(&lens[i],p,4); p+=4; }
  std::vector<float> out;
  if(nout>0){
    out.resize(lens[0]/sizeof(float));
    std::memcpy(out.data(), p, lens[0]);
  }
  ::close(fd);

  std::cout << "status="<<status<<" time_ms="<<ms<<" out0[0]="<<(out.empty()?0.0f:out[0])<<"\n";
  // print top-5 if looks like logits(1x1000)
  if(out.size()==1000){
    std::vector<int> idx(1000); for(int i=0;i<1000;i++) idx[i]=i;
    std::partial_sort(idx.begin(), idx.begin()+5, idx.end(), [&](int a,int b){return out[a]>out[b];});
    for(int k=0;k<5;k++) std::cout << k << ": id="<<idx[k]<<" score="<<out[idx[k]]<<"\n";
  }
  return 0;
}
