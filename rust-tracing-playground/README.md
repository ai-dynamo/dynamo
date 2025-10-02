# Rust Tracing with OpenTelemetry, Tempo & Grafana

A complete observability stack demonstrating distributed tracing in Rust with OpenTelemetry export to Tempo and visualization in Grafana.

## 🚀 Quick Start

1. **Start the observability stack:**
   ```bash
   docker-compose up -d
   ```

2. **Run the Rust application on your host:**
   ```bash
   cargo run
   ```

3. **Access Grafana:**
   - URL: http://localhost:3000
   - Username: `admin`
   - Password: `admin`

4. **View traces:**
   - Go to "Explore" in Grafana
   - Select "Tempo" datasource
   - Use TraceQL queries or browse traces

## 📊 What You'll See

The Rust application generates realistic traces with:
- **Two separate requests** with distinct trace IDs
- **Nested spans** (database queries, API calls, processing)
- **Structured logging** with trace correlation
- **Error simulation** for realistic scenarios
- **Service metadata** (name, version)

## 🔍 Example TraceQL Queries

```traceql
# Find traces with warnings
{ status = error }

# Find traces for specific service
{ service.name = "rust-tracing-playground" }

# Find slow traces (>1s duration)
{ duration > 1s }

# Find traces with database operations
{ span.name =~ "database.*" }
```

## 🏗️ Architecture

```
┌─────────────┐    OTLP/gRPC    ┌───────────┐    HTTP    ┌─────────────┐
│ Rust App    │ ──────────────> │  Tempo    │ <────────> │  Grafana    │
│ (Host)      │   localhost:4317│(Container)│   :3200    │ (Container) │
└─────────────┘                 └───────────┘            └─────────────┘
```

## 🛠️ Components

- **Rust App**: Runs on host, generates OpenTelemetry traces via OTLP
- **Tempo**: Docker container, receives/stores traces on port 4317
- **Grafana**: Docker container, provides trace visualization

## 🔧 Development Workflow

1. **Start infrastructure:** `docker-compose up -d` (runs in background)
2. **Develop and test:** `cargo run` (run app multiple times)
3. **View results:** Check traces in Grafana at http://localhost:3000
4. **Iterate:** Make code changes and re-run `cargo run`

## ⏹️ Stopping

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```
