FROM nvcr.io/nvidia/tensorrt:22.12-py3
WORKDIR /app
RUN apt-get update -qq && apt-get install -y -qq build-essential cmake libyaml-cpp-dev curl && rm -rf /var/lib/apt/lists/*
COPY . /app

ENV EIG_FETCH_MOBILENET=0
ENV EIG_AUTO_ASSET_DOWNLOAD=1
ENV EIG_REUSE_ENGINES=0
ENV EIG_EXPORT_SSD=1

RUN bash tools/download_assets.sh && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && \
    chmod +x docker/entrypoint.sh
# drop privs
RUN useradd -u 10001 -m eig
RUN chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8008 8080
HEALTHCHECK --interval=30s --timeout=2s --start-period=10s CMD curl -sf http://127.0.0.1:8080/healthz || exit 1
ENV EIG_PORT=8008 EIG_HTTP_PORT=8080
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["./build/edge-infer-gateway","--config","config/models.yaml","--port","8008","--http-port","8080"]
