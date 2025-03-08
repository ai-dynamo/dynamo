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
DYN_LOG=info cargo run --bin metrics -- --namespace dynamo --component backend --endpoint generate

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

## Metrics Collection Modes

The metrics component supports two modes for exposing metrics:

### Server Mode (Default)

When running in server mode, the metrics component will expose a Prometheus metrics endpoint on the specified port (default: 9091):

```bash
# Start metrics server on port 9091 (default)
DYN_LOG=info cargo run --bin metrics -- --component backend --endpoint generate

# Or specify a custom port
DYN_LOG=info cargo run --bin metrics -- --component backend --endpoint generate --metrics-port 9092
```

You can then scrape the metrics using:
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

### Pushgateway Mode

For ephemeral or batch jobs, or when metrics need to be pushed through a firewall, you can use Pushgateway mode. In this mode, the metrics component will periodically push metrics to a Prometheus Pushgateway:

Start a prometheus push gateway service via docker:
```bash
docker run --rm -d -p 9091:9091 --name pushgateway prom/pushgateway
```

Start the metrics component in `--push` mode, pointing at the URL for the
prometheus push gateway service that's been started:
```bash
# Push metrics to a Prometheus Pushgateway every 2 seconds
DYN_LOG=info cargo run --bin metrics -- \
    --component backend \
    --endpoint generate \
    --push \
    --push-url http://localhost:9091 \
    --push-interval 2
```

When using Pushgateway mode:
- Metrics are pushed to the specified URL with the job label
- The push interval can be configured (default: 15 seconds)
- Metrics persist in the Pushgateway until explicitly deleted
- Prometheus should be configured to scrape the Pushgateway with `honor_labels: true`

To view the metrics in the Pushgateway:
```bash
# View all metrics
curl http://localhost:9091/metrics
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
KV Routing example is using, for example:
```
DYN_LOG=info cargo run --bin metrics -- --namespace dynamo --component vllm --endpoint load_metrics
```
