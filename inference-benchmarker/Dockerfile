FROM rust:1-bullseye AS builder
LABEL org.opencontainers.image.source=https://github.com/huggingface/inference-benchmarker
LABEL org.opencontainers.image.description="A benchmark tool for LLM inference engines"
LABEL org.opencontainers.image.licenses="Apache-2.0"
ARG GIT_SHA
WORKDIR /usr/src/inference-benchmarker
COPY . .
RUN cargo install --path .
FROM debian:bullseye-slim
RUN apt update && apt install -y ca-certificates && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/inference-benchmarker/results
WORKDIR /opt/inference-benchmarker
COPY --from=builder /usr/local/cargo/bin/inference-benchmarker /usr/local/bin/inference-benchmarker
CMD ["inference-benchmarker"]