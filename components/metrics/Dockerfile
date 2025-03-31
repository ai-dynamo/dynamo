# ---- Build Stage ----
FROM rust:1.85 AS builder

RUN apt-get update && apt-get install -y protobuf-compiler

WORKDIR /workspace
    
# Copy only essential files
COPY ../../Cargo.toml ../../Cargo.lock ./
COPY ../../components/metrics ./components/metrics
COPY ../../README.md ./README.md
COPY ../../launch ./launch
COPY ../../lib ./lib
    
# Set the working directory to `components/metrics`
WORKDIR /workspace/components/metrics

# Cache dependencies
RUN cargo fetch

# Build the `metrics` package in release mode
RUN cargo build --release --package metrics

# ---- Runtime Stage ----
FROM ubuntu:22.04
WORKDIR /app
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the built binary from the builder stage
COPY --from=builder /workspace/target/release/metrics /app/

# Run the binary
CMD ["./metrics"]
    