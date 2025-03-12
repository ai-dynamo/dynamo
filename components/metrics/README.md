# Metrics

## Quickstart

To start the `metrics` component, simply point it at the `namespace/component/endpoint` trio that
you're interested in observing metrics from.

This will:
1. Scrape statistics from the services associated with that `endpoint`, do some postprocessing, and aggregate them.
2. Listen for `KvHitRateEvent`s on `namespace/kv-hit-rate`, and aggregate them.

For example:
```bash
# For more details, try DYN_LOG=debug
DYN_LOG=info metrics --namespace dynamo --component backend --endpoint generate

# 2025-02-26T18:45:05.467026Z  INFO metrics: Creating unique instance of Metrics at dynamo/components/metrics/instance
# 2025-02-26T18:45:05.472146Z  INFO metrics: Scraping service dynamo_backend_720278f8 and filtering on subject dynamo_backend_720278f8.generate
# ...
```

With no matching endpoints running to collect stats from, you should see warnings in the logs:
```bash
2025-02-26T18:45:06.474161Z  WARN metrics: No endpoints found matching subject dynamo_backend_720278f8.generate
```

After a matching endpoint gets started, you should see the warnings stop
when the endpoint gets automatically discovered.

## Building/Running from Source

For easy iteration while making edits to the metrics component, you can use `cargo run`
to build and run with your local changes:

```bash
DYN_LOG=info cargo run --bin metrics -- --namespace dynamo --component backend --endpoint generate
```

## Metrics Collection Modes

The metrics component supports two modes for exposing metrics in a Prometheus format:

### Pull Mode (Default)

When running in pull mode (the default), the metrics component will expose a Prometheus metrics endpoint on the specified host and port that a Prometheus server or curl client can pull from:

```bash
# Start metrics server on default host (0.0.0.0) and port (9091)
DYN_LOG=info metrics --component backend --endpoint generate

# Or specify a custom port
DYN_LOG=info metrics --component backend --endpoint generate --port 9092

# Or specify a custom host and port
DYN_LOG=info metrics --component backend --endpoint generate --host 127.0.0.1 --port 9092
```

In pull mode:
- The `--host` parameter must be a valid IPv4 or IPv6 address (e.g., "0.0.0.0", "127.0.0.1")
- The `--port` parameter specifies which port the HTTP server will listen on

You can then query the metrics using:
```bash
curl localhost:9091/metrics

# # HELP llm_kv_blocks_active Active KV cache blocks
# # TYPE llm_kv_blocks_active gauge
# llm_kv_blocks_active{component="backend",endpoint="generate",worker_id="7587884888253033398"} 40
# llm_kv_blocks_active{component="backend",endpoint="generate",worker_id="7587884888253033401"} 2
# # HELP llm_kv_blocks_total Total KV cache blocks
# # TYPE llm_kv_blocks_total gauge
# llm_kv_blocks_total{component="backend",endpoint="generate",worker_id="7587884888253033398"} 100
# llm_kv_blocks_total{component="backend",endpoint="generate",worker_id="7587884888253033401"} 100
```

### Push Mode

For ephemeral or batch jobs, or when metrics need to be pushed through a firewall, you can use Push mode. In this mode, the metrics component will periodically push metrics to an externally hosted Prometheus PushGateway:

Start a prometheus push gateway service via docker:
```bash
docker run --rm -d -p 9091:9091 --name pushgateway prom/pushgateway
```

Start the metrics component in `--push` mode, specifying the host and port of your PushGateway:
```bash
# Push metrics to a Prometheus PushGateway every --push-interval seconds
DYN_LOG=info metrics \
    --component backend \
    --endpoint generate \
    --host 127.0.0.1 \
    --port 9091 \
    --push
```

When using Push mode:
- The `--host` parameter specifies be the IP address of the PushGateway
- The `--port` parameter specifies the port of the PushGateway
- The push interval can be configured with `--push-interval` (default: 2 seconds)
- A default job name of "dynamo_metrics" is used for the Prometheus job label
- Metrics persist in the PushGateway until explicitly deleted
- Prometheus should be configured to scrape the PushGateway with `honor_labels: true`

To view the metrics hosted on the PushGateway:
```bash
# View all metrics
# curl http://<pushgateway_ip>:<pushgateway_port>/metrics
curl 127.0.0.1:9091/metrics
```

## Workers

### Mock Worker

For convenience and debugging, there is a mock worker that registers a mock `StatsHandler`
with the `endpoint` and publishes mock `KvHitRateEvent`s on `namespace/kv-hit-rate`.

```bash
# Can run multiple workers in separate shells to see aggregation as well.
DYN_LOG=info cargo run --bin mock_worker
```

**NOTE**: When using the mock worker, the data from the stats handler and the
events will be random and shouldn't be expected to correlate with each other.

### Real Worker

See the KV Routing example in `examples/python_rs/llm/vllm`.

Start the `metrics` component with the corresponding namespace/component/endpoint that the
KV Routing example is using (NOTE: `load_metrics` endpoint is currently a hard-coded value
internally for the ForwardPassMetrics StatsHandler), for example:
```
DYN_LOG=info metrics --namespace dynamo --component vllm --endpoint load_metrics
```
