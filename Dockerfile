FROM python:3.11

# install git, pip, wget dependencies
RUN apt-get update -y && apt-get install -y git python3-pip wget tmux && rm -rf /var/lib/apt/lists/*

# Rust Setup
RUN apt-get update && apt-get install -y \
    curl \
    build-essential ca-certificates \
    libhwloc-dev libudev-dev pkg-config libclang-dev \
    protobuf-compiler python3-dev cmake \
 && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustc --version && cargo --version

WORKDIR /app/repo

RUN pip install matplotlib maturin
RUN pip install git+https://github.com/ai-dynamo/aiperf.git

COPY . /app/repo
ENV DYNAMO_HOME=/app/repo

RUN pip install -e ./benchmarks

# Compile rust
ENV PYTHONPATH=/app/repo/components/src
RUN uv venv dynamo && \
    . dynamo/bin/activate && \
    cd lib/bindings/python && \
    maturin develop --release --strip && \
    cd /app/repo && \
    uv pip install -e .
