FROM nvcr.io/nvidia/tensorrt:22.12-py3

WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq build-essential cmake libyaml-cpp-dev curl && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN bash tools/download_assets.sh && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && \
    bash tools/build_engine.sh assets/mobilenetv2.onnx

EXPOSE 8008

CMD ["./build/edge-infer-gateway","--config","config/models.yaml","--port","8008"]
