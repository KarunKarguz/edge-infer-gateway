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
#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

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

struct Args{
  std::string host = "127.0.0.1";
  int port = 8008;
  std::string model = "yolov5n_coco";
  std::string mode  = "yolo"; // yolo|ssd
  std::string source= "0";    // camera index or path
  float conf = 0.25f;
  bool show = true;
};

static Args parse(int argc, char** argv){
  Args a; for(int i=1;i<argc;i++){
    std::string s=argv[i];
    if(s=="--host"&&i+1<argc) a.host=argv[++i];
    else if(s=="--port"&&i+1<argc) a.port=std::atoi(argv[++i]);
    else if(s=="--model"&&i+1<argc) a.model=argv[++i];
    else if(s=="--mode"&&i+1<argc) a.mode=argv[++i];
    else if(s=="--source"&&i+1<argc) a.source=argv[++i];
    else if(s=="--conf"&&i+1<argc) a.conf=std::atof(argv[++i]);
    else if(s=="--no-show") a.show=false;
  } return a;
}

static void pack_u32(uint32_t v, std::vector<char>& b){ char x[4]; std::memcpy(x,&v,4); b.insert(b.end(),x,x+4); }

static cv::Mat letterbox(const cv::Mat& img, int W, int H, int pad=114){
  int ih=img.rows, iw=img.cols; double r=std::min(double(W)/iw, double(H)/ih);
  int nw=int(iw*r+0.5), nh=int(ih*r+0.5);
  cv::Mat resized; cv::resize(img,resized,cv::Size(nw,nh),0,0,cv::INTER_LINEAR);
  int dw=(W-nw)/2, dh=(H-nh)/2;
  cv::Mat out(H,W,img.type(), cv::Scalar(pad,pad,pad));
  resized.copyTo(out(cv::Rect(dw,dh,nw,nh)));
  return out;
}

static std::vector<float> preprocess_yolo(const cv::Mat& bgr){
  cv::Mat lb = letterbox(bgr, 640, 640, 114);
  cv::Mat rgb; cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0/255.0);
  std::vector<float> nchw(1*3*640*640);
  int H=640,W=640; std::vector<cv::Mat> ch(3);
  cv::split(rgb, ch);
  for(int c=0;c<3;c++){
    for(int y=0;y<H;y++){
      const float* row = ch[c].ptr<float>(y);
      for(int x=0;x<W;x++) nchw[c*H*W + y*W + x] = row[x];
    }
  }
  return nchw;
}

static std::vector<float> preprocess_ssd(const cv::Mat& bgr){
  cv::Mat img; cv::resize(bgr,img,cv::Size(300,300));
  img.convertTo(img, CV_32FC3, 1.0/255.0);
  cv::Mat rgb; cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
  int H=300,W=300; std::vector<cv::Mat> ch(3); cv::split(rgb, ch);
  std::vector<float> nchw(1*3*H*W);
  for(int c=0;c<3;c++){
    for(int y=0;y<H;y++){
      const float* row = ch[c].ptr<float>(y);
      for(int x=0;x<W;x++) nchw[c*H*W + y*W + x] = row[x];
    }
  }
  return nchw;
}

static std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, float iouTh){
  std::vector<int> idx(boxes.size()); std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a,int b){return scores[a]>scores[b];});
  std::vector<int> keep;
  std::vector<char> removed(boxes.size(),0);
  for(size_t _i=0; _i<idx.size(); ++_i){ int i=idx[_i]; if(removed[i]) continue; keep.push_back(i);
    for(size_t _j=_i+1; _j<idx.size(); ++_j){ int j=idx[_j]; if(removed[j]) continue;
      float inter = (boxes[i] & boxes[j]).area();
      float uni = boxes[i].area() + boxes[j].area() - inter + 1e-6f;
      if(inter/uni > iouTh) removed[j]=1;
    }
  }
  return keep;
}

int main(int argc, char** argv){
  auto args = parse(argc, argv);
  // open source
  cv::VideoCapture cap;
  if(args.source.size()==1 && std::isdigit(args.source[0])) cap.open(int(args.source[0]-'0'));
  else cap.open(args.source);
  if(!cap.isOpened()){ std::cerr << "failed to open source: "<<args.source<<"\n"; return 2; }

  // connect persistent socket
  int fd = ::socket(AF_INET, SOCK_STREAM, 0); if(fd<0) die("socket");
  sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(args.port);
  if(::inet_pton(AF_INET, args.host.c_str(), &a.sin_addr)!=1) die("inet_pton");
  if(::connect(fd, (sockaddr*)&a, sizeof(a))<0) die("connect");

  const char magic[4]={'T','R','T','\x01'}; uint16_t ver=1, flags=0;

  cv::Mat frame; int frame_count=0; auto t0=std::chrono::high_resolution_clock::now();
  while(true){
    if(!cap.read(frame)) break;
    int iw=frame.cols, ih=frame.rows;
    std::vector<float> input = (args.mode=="ssd") ? preprocess_ssd(frame) : preprocess_yolo(frame);
    bool fp16 = (args.mode!="ssd");

    // Build frame
    std::vector<char> body;
    body.insert(body.end(), magic, magic+4);
    body.insert(body.end(), (char*)&ver, (char*)&ver+2);
    body.insert(body.end(), (char*)&flags, (char*)&flags+2);
    pack_u32((uint32_t)args.model.size(), body);
    pack_u32(1, body); // n_inputs
    pack_u32(0, body); // reserved
    body.insert(body.end(), args.model.begin(), args.model.end());
    uint8_t dtype = fp16 ? 1 : 0; // 1=fp16,0=fp32
    if(args.mode=="ssd"){
      body.push_back((char)dtype); body.push_back(4);
      int32_t dims[4]={1,3,300,300}; body.insert(body.end(), (char*)dims, (char*)dims+16);
    } else {
      body.push_back((char)dtype); body.push_back(4);
      int32_t dims[4]={1,3,640,640}; body.insert(body.end(), (char*)dims, (char*)dims+16);
    }
    if(fp16){
      std::vector<uint16_t> half(input.size());
      // naive float32->float16 (round-to-nearest via cv::Mat)
      cv::Mat f32(1,(int)input.size(),CV_32F,input.data());
      cv::Mat f16; f32.convertTo(f16, CV_16F);
      std::memcpy(half.data(), f16.ptr(), half.size()*sizeof(uint16_t));
      uint32_t blen = (uint32_t)(half.size()*sizeof(uint16_t)); pack_u32(blen, body);
      body.insert(body.end(), (char*)half.data(), (char*)half.data()+blen);
    } else {
      uint32_t blen = (uint32_t)(input.size()*sizeof(float)); pack_u32(blen, body);
      body.insert(body.end(), (char*)input.data(), (char*)input.data()+blen);
    }
    uint32_t frame_len=(uint32_t)body.size(); std::vector<char> frame_buf; frame_buf.reserve(4+frame_len);
    pack_u32(frame_len, frame_buf); frame_buf.insert(frame_buf.end(), body.begin(), body.end());

    auto t_send = std::chrono::high_resolution_clock::now();
    if(!sendall(fd, frame_buf.data(), frame_buf.size())) break;
    uint32_t rlen=0; if(!recvn(fd,&rlen,4)) break; std::vector<char> resp(rlen); if(!recvn(fd, resp.data(), resp.size())) break;
    auto t_recv = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t_recv - t_send).count();

    const char* p=resp.data(); uint32_t req_id=0,status=0,nout=0; std::memcpy(&req_id,p,4); p+=4; std::memcpy(&status,p,4); p+=4; std::memcpy(&nout,p,4); p+=4;
    std::vector<uint32_t> lens(nout); for(uint32_t i=0;i<nout;i++){ std::memcpy(&lens[i],p,4); p+=4; }

    if(status!=0 || nout==0){ std::cerr << "infer error status="<<status<<"\n"; continue; }

    if(args.mode=="ssd"){
      // expect [1,1,200,7] float32
      std::vector<float> dets(lens[0]/sizeof(float)); std::memcpy(dets.data(), p, lens[0]);
      int N = dets.size()/7; for(int i=0;i<N;i++){
        float image_id=dets[i*7+0]; if(image_id<0) continue; // unused
        int label=(int)dets[i*7+1]; float conf=dets[i*7+2]; if(conf<args.conf) continue;
        float x1=dets[i*7+3]*iw, y1=dets[i*7+4]*ih, x2=dets[i*7+5]*iw, y2=dets[i*7+6]*ih;
        cv::rectangle(frame, cv::Rect(cv::Point(int(x1),int(y1)), cv::Point(int(x2),int(y2))), {0,255,0}, 2);
        char txt[64]; std::snprintf(txt,sizeof(txt),"id%d:%.2f",label,conf);
        cv::putText(frame, txt, {int(x1), std::max(0,int(y1)-5)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 2);
      }
    } else {
      // YOLOv5: [1,25200,85] fp16 buffer -> fp32
      std::vector<uint16_t> half(lens[0]/2); std::memcpy(half.data(), p, lens[0]);
      cv::Mat f16(1,(int)half.size(),CV_16U,half.data()); cv::Mat f32; f16.convertTo(f32, CV_32F);
      std::vector<float> pred(f32.total()); std::memcpy(pred.data(), f32.ptr(), pred.size()*4);
      int stride=85; int M = pred.size()/stride;
      std::vector<cv::Rect2f> boxes; boxes.reserve(M);
      std::vector<float> scores; scores.reserve(M);
      for(int i=0;i<M;i++){
        float x=pred[i*stride+0], y=pred[i*stride+1], w=pred[i*stride+2], h=pred[i*stride+3];
        float obj=1.f/(1.f+std::exp(-pred[i*stride+4]));
        // pick best class
        int best=-1; float bests=0.f;
        for(int c=5;c<85;c++){ float s=1.f/(1.f+std::exp(-pred[i*stride+c])); if(s>bests){bests=s; best=c-5;} }
        float conf = obj*bests; if(conf<args.conf) continue;
        float x1=x-w/2, y1=y-h/2, x2=x+w/2, y2=y+h/2;
        // map back from 640 padded to orig
        double gain = std::min(640.0/ih, 640.0/iw); double pad_x=(640 - iw*gain)/2, pad_y=(640 - ih*gain)/2;
        x1 = (x1 - pad_x)/gain; y1 = (y1 - pad_y)/gain; x2 = (x2 - pad_x)/gain; y2 = (y2 - pad_y)/gain;
        x1 = std::clamp(x1, 0.f, (float)iw); y1 = std::clamp(y1, 0.f, (float)ih);
        x2 = std::clamp(x2, 0.f, (float)iw); y2 = std::clamp(y2, 0.f, (float)ih);
        boxes.emplace_back(cv::Rect2f(cv::Point2f(x1,y1), cv::Point2f(x2,y2)));
        scores.push_back(conf);
      }
      auto keep = nms(boxes, scores, 0.45f);
      for(int i: keep){ auto r=boxes[i]; cv::rectangle(frame, r, {0,255,0}, 2);
        char txt[64]; std::snprintf(txt,sizeof(txt),"%.2f", scores[i]);
        cv::putText(frame, txt, {int(r.x), std::max(0,int(r.y)-5)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 2);
      }
    }

    frame_count++;
    if(args.show){
      cv::putText(frame, (args.mode+" "+std::to_string(ms)+" ms").c_str(), {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,255}, 2);
      cv::imshow("eig-stream", frame);
      if((cv::waitKey(1)&0xFF)==27) break;
    }
  }

  ::shutdown(fd, SHUT_RDWR); ::close(fd);
  return 0;
}

