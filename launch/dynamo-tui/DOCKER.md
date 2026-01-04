# Docker Build & Test Guide for macOS

This guide helps you build and test `dynamo-tui` on macOS using Docker, since the native build requires Linux-only dependencies (`inotify`).

## Quick Start

### 1. Build the Docker Image

```bash
./launch/dynamo-tui/docker-build.sh
```

This will:
- Build a Docker image with Rust 1.90.0
- Compile `dynamo-tui` in release mode
- Tag the image as `dynamo-tui:dev`

### 2. Run Tests

```bash
./launch/dynamo-tui/docker-test.sh
```

Or run a specific test:
```bash
./launch/dynamo-tui/docker-test.sh discovery_snapshot_populates_tree
```

### 3. Run the TUI

First, make sure ETCD and NATS are running:

```bash
# Start dependencies
docker compose -f deploy/docker-compose.yml up -d
```

Then run the TUI:

```bash
./launch/dynamo-tui/docker-run.sh
```

With metrics:
```bash
./launch/dynamo-tui/docker-run.sh --metrics-url http://host.docker.internal:9100/metrics
```

## Manual Docker Commands

If you prefer to run Docker commands directly:

### Build
```bash
cd /path/to/dynamo
docker build -f launch/dynamo-tui/Dockerfile.dev -t dynamo-tui:dev .
```

### Test
```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  rust:1.90.0-slim bash -c "
    apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev build-essential >/dev/null 2>&1
    cargo test -p dynamo-tui
  "
```

### Run
```bash
docker run --rm -it \
  -e ETCD_ENDPOINTS=http://host.docker.internal:2379 \
  -e NATS_SERVER=nats://host.docker.internal:4222 \
  dynamo-tui:dev
```

## Troubleshooting

### ETCD/NATS Connection Issues

If the TUI can't connect to ETCD/NATS running on your Mac:

1. Make sure services are running:
   ```bash
   docker compose -f deploy/docker-compose.yml ps
   ```

2. Test connectivity from Docker:
   ```bash
   docker run --rm curlimages/curl:latest \
     curl http://host.docker.internal:2379/health
   ```

3. If using custom ports, set environment variables:
   ```bash
   export ETCD_ENDPOINTS=http://host.docker.internal:2379
   export NATS_SERVER=nats://host.docker.internal:4222
   ./launch/dynamo-tui/docker-run.sh
   ```

### Build Failures

If the build fails:

1. Check Docker has enough resources (memory/CPU)
2. Try building without cache:
   ```bash
   docker build --no-cache -f launch/dynamo-tui/Dockerfile.dev -t dynamo-tui:dev .
   ```

3. Check Rust version matches:
   ```bash
   cat rust-toolchain.toml
   ```

## Notes

- The Docker image uses `rust:1.90.0-slim` which matches the project's `rust-toolchain.toml`
- On macOS, Docker containers use `host.docker.internal` to access host services
- The build process compiles the entire `dynamo-runtime` dependency, which may take several minutes on first build
- Subsequent builds will be faster due to Docker layer caching

