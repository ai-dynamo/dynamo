# Metrics

## Quickstart

To start `metrics`, simply point it at the namespace/component/endpoint trio that
you're interested in observing metrics from. This will scrape statistics from
the services associated with that endpoint, do some postprocessing on them,
and then publish an event with the postprocessed data.

```bash
# For more details, try DYN_LOG=debug
DYN_LOG=info cargo run --bin metrics -- --namespace dynemo --component backend --endpoint generate

# 2025-02-26T18:45:05.467026Z  INFO metrics: Creating unique instance of Metrics at dynemo/components/metrics/instance
# 2025-02-26T18:45:05.472146Z  INFO metrics: Scraping service dynemo_backend_720278f8 and filtering on subject dynemo_backend_720278f8.generate
# ...
```

With no matching endpoints running, you should see warnings in the logs:
```bash
2025-02-26T18:45:06.474161Z  WARN metrics: No endpoints found matching subject dynemo_backend_720278f8.generate
```

To see metrics published to a matching endpoint, you can use the
[mock_worker example](src/bin/mock_worker.rs) in this directory to launch
1 or more workers that publish LLM Metrics:
```bash
# Can run multiple workers in separate shells
cargo run --bin mock_worker
```

After a matching endpoint gets started, you should see the warnings go away
since the endpoint will automatically get discovered.

When stats are found from target endpoints, the metrics component will
aggregate and publish metrics as both events and as updates to a prometheus server:
```
2025-02-28T04:05:58.077901Z  INFO metrics: Aggregated metrics: ProcessedEndpoints { endpoints: [Endpoint { name: "worker-7587884888253033398", subject: "dynemo_backend_720278f8.generate-694d951a80e06bb6", data: ForwardPassMetrics { request_active_slots: 58, request_total_slots: 100, kv_active_blocks: 77, kv_total_blocks: 100 } }, Endpoint { name: "worker-7587884888253033401", subject: "dynemo_backend_720278f8.generate-694d951a80e06bb9", data: ForwardPassMetrics { request_active_slots: 71, request_total_slots: 100, kv_active_blocks: 29, kv_total_blocks: 100 } }], worker_ids: [7587884888253033398, 7587884888253033401], load_avg: 53.0, load_std: 24.0 }
```

To see the metrics being published in prometheus format, you can run:
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
