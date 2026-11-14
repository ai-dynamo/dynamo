// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { ObservabilityConfig } from "./config";
import { metricNamespace } from "./collector";
export function dashboardBody(c: ObservabilityConfig, global: boolean) {
 const regions = global ? c.dashboardRegions : c.dashboardRegions.filter(r => r.region === c.cluster.region);
 const widgets: Record<string, unknown>[] = [];
 let row = 0;
 const text = (markdown: string) => { widgets.push({ type: "text", x: 0, y: row, width: 24, height: 3, properties: { markdown } }); row += 3; };
 const chart = (title: string, metrics: unknown[], stat = "Sum", width = 12) => {
   const x = widgets.length && widgets[widgets.length - 1].y === row && widgets[widgets.length - 1].x === 0 && widgets[widgets.length - 1].width === 12 ? 12 : 0;
   widgets.push({ type: "metric", x, y: row, width, height: 6, properties: { title, region: c.cluster.region, period: 60, stat, view: "timeSeries", stacked: false, metrics } });
   if (x === 12 || width === 24) row += 6;
 };
 const app = (metric: string, component: string, stat = "Sum", extra: string[] = []) => regions.map(r => [metricNamespace, metric, "ClusterName", r.clusterName, "Component", component, ...extra, { region: r.region, stat, label: r.region }]);
 const ratio = (sum: string, count: string, component: string, multiplier = 1) => regions.flatMap((r, i) => [
   [metricNamespace, sum, "ClusterName", r.clusterName, "Component", component, { region: r.region, id: `s${i}`, visible: false }],
   [metricNamespace, count, "ClusterName", r.clusterName, "Component", component, { region: r.region, id: `n${i}`, visible: false }],
   [{ expression: `IF(n${i}>0,${multiplier}*s${i}/n${i})`, label: r.region, id: `e${i}` }],
 ]);
 // SEARCH discovers newly deployed pools/replicas. Gauge searches retain target dimensions.
 const search = (metric: string, dimensions: string, component: string, stat = "Sum", reduce?: string, filter = "") => regions.map(r => {
   const query = `SEARCH('{${metricNamespace},ClusterName,Component,${dimensions}} MetricName="${metric}" ClusterName="${r.clusterName}" Component="${component}" ${filter}', '${stat}', 60)`;
   return [{ expression: reduce ? `${reduce}(${query})` : query, region: r.region, ...(reduce ? { label: r.region } : {}) }];
 });
 const frontendGauges = "Pool,ServingRegion,PodName,model";
 const workerGauges = "Pool,ServingRegion,PodName,worker_id,model,engine";
 text(`# ${c.fleet} — ${global ? "global on-call" : c.cluster.region}\n60-second scrapes. Traffic graphs show interval increments, latency graphs show weighted means (not p95/p99). Tokens are counted only at serving pool frontends. Missing telemetry is not zero. Current pools use CPU mock workers. Gauge totals sum per-target averages; missing targets can undercount and 60-second sampling can miss brief spikes. Worker occupancy is simulated KV capacity, not a cache-hit percentage.`);
 chart("Routed requests / minute — ingress region", app("dynamo_global_frontend_requests_total", "global-router"));
 chart("Pool requests / minute — serving region", app("dynamo_frontend_requests_total", "pool-frontend"));
 chart("Router mean request duration (seconds)", ratio("dynamo_global_frontend_request_duration_seconds_sum", "dynamo_global_frontend_request_duration_seconds_count", "global-router"));
 chart("Pool mean TTFT (seconds)", ratio("dynamo_frontend_time_to_first_token_seconds_sum", "dynamo_frontend_time_to_first_token_seconds_count", "pool-frontend"));
 chart("Requests / minute by destination pool — all discovered pools", search("dynamo_global_frontend_requests_total", "pool,region", "global-router"));
 chart("Cross-region request percentage — ingress region", regions.flatMap((r, i) => [
   [metricNamespace, "dynamo_global_frontend_requests_total", "ClusterName", r.clusterName, "Component", "global-router", "local", "false", { region: r.region, id: `remote${i}`, visible: false }],
   [metricNamespace, "dynamo_global_frontend_requests_total", "ClusterName", r.clusterName, "Component", "global-router", { region: r.region, id: `all${i}`, visible: false }],
   [{ expression: `IF(all${i}>0,100*FILL(remote${i},0)/all${i})`, label: r.region }],
 ]));
 chart("Pool active requests — sum of sampled frontend/model gauges", search("dynamo_frontend_active_requests", frontendGauges, "pool-frontend", "Average", "SUM"), "Average");
 chart("Awaiting first token — sum of sampled frontend/model gauges", search("dynamo_frontend_awaiting_first_token_requests", frontendGauges, "pool-frontend", "Average", "SUM"), "Average");
 chart("Queued requests by frontend/model — sampled gauges", search("dynamo_frontend_queued_requests", frontendGauges, "pool-frontend", "Average"), "Average");
 chart("Global-router in-flight requests — sum of sampled target/destination gauges", search("dynamo_global_frontend_inflight_requests", "PodName,pool,region", "global-router", "Average", "SUM"), "Average");
 chart("Outstanding input tokens by frontend/model/state — sampled gauges", search("dynamo_frontend_inflight_input_tokens", `${frontendGauges},state`, "pool-frontend", "Average"), "Average");
 chart("Available workers by discovered pool — frontend average", search("dynamo_frontend_router_workers", "Pool,ServingRegion,state", "pool-frontend", "Average", undefined, 'state="available"'), "Average");
 chart("Worker scheduler waiting requests — sum of sampled engine gauges", search("vllm:num_requests_waiting", workerGauges, "pool-worker", "Average", "SUM"), "Average");
 chart("Worker scheduler running requests — sum of sampled engine gauges", search("vllm:num_requests_running", workerGauges, "pool-worker", "Average", "SUM"), "Average");
 chart("Worker KV-cache occupancy fraction — sampled per engine, 1 = full", search("vllm:kv_cache_usage_perc", workerGauges, "pool-worker", "Maximum"), "Maximum");
 chart("Worker preemptions / minute by discovered pool", search("vllm:num_preemptions_total", "Pool,ServingRegion", "pool-worker"));
 chart("Global-router mean time to first response byte (seconds)", ratio("dynamo_global_frontend_time_to_first_byte_seconds_sum", "dynamo_global_frontend_time_to_first_byte_seconds_count", "global-router"));
 chart("Pool mean request duration (seconds)", ratio("dynamo_frontend_request_duration_seconds_sum", "dynamo_frontend_request_duration_seconds_count", "pool-frontend"));
 chart("Pool mean inter-token latency (seconds)", ratio("dynamo_frontend_inter_token_latency_seconds_sum", "dynamo_frontend_inter_token_latency_seconds_count", "pool-frontend"));
 chart("Pool arrivals and completions / minute — separate lifecycle stages", regions.flatMap(r => [
   [metricNamespace, "dynamo_frontend_requests_started_total", "ClusterName", r.clusterName, "Component", "pool-frontend", { region: r.region, label: `${r.region} started` }],
   [metricNamespace, "dynamo_frontend_requests_total", "ClusterName", r.clusterName, "Component", "pool-frontend", { region: r.region, label: `${r.region} completed` }],
 ]));
 chart("Affinity storage mean overhead per session request (ms)", ratio("dynamo_global_frontend_overhead_affinity_store_ms_sum", "dynamo_global_frontend_overhead_affinity_store_ms_count", "global-router"));
 chart("Affinity cache outcomes / minute — hits, misses, invalidations", search("dynamo_global_frontend_affinity_cache_total", "outcome", "global-router"));
 chart("Affinity store writes / minute by reason — create, move, refresh, repair", search("dynamo_global_frontend_affinity_store_writes_total", "reason", "global-router"));
 chart("Affinity storage errors / minute by kind", search("dynamo_global_frontend_affinity_store_errors_total", "kind", "global-router"));
 chart("Mean input sequence length (tokens)", ratio("dynamo_frontend_input_sequence_tokens_sum", "dynamo_frontend_input_sequence_tokens_count", "pool-frontend"));
 chart("Mean output sequence length (tokens)", ratio("dynamo_frontend_output_sequence_tokens_sum", "dynamo_frontend_output_sequence_tokens_count", "pool-frontend"));
 chart("Tokenizer mean latency by operation (ms)", regions.flatMap((r, i) => ["tokenize", "detokenize"].flatMap((operation, j) => [
   [metricNamespace, "dynamo_frontend_tokenizer_latency_ms_sum", "ClusterName", r.clusterName, "Component", "pool-frontend", "Pool", r.pool, "ServingRegion", r.region, "operation", operation, { region: r.region, id: `ts${i}${j}`, visible: false }],
   [metricNamespace, "dynamo_frontend_tokenizer_latency_ms_count", "ClusterName", r.clusterName, "Component", "pool-frontend", "Pool", r.pool, "ServingRegion", r.region, "operation", operation, { region: r.region, id: `tn${i}${j}`, visible: false }],
   [{ expression: `IF(tn${i}${j}>0,ts${i}${j}/tn${i}${j})`, label: `${r.region}/${r.pool} ${operation}` }],
 ])));
 chart("Scrape health — minimum across worker targets", app("up", "pool-worker", "Minimum"), "Minimum");
 chart("Affinity storage failures / minute", app("dynamo_global_frontend_affinity_store_errors_total", "global-router"));
 chart("Affinity hit percentage — session pin, not KV reuse", regions.flatMap((r, i) => [
   [metricNamespace, "dynamo_global_frontend_affinity_lookups_total", "ClusterName", r.clusterName, "Component", "global-router", "outcome", "hit", { region: r.region, id: `h${i}`, visible: false }],
   [metricNamespace, "dynamo_global_frontend_affinity_lookups_total", "ClusterName", r.clusterName, "Component", "global-router", { region: r.region, id: `t${i}`, visible: false }],
   [{ expression: `IF(t${i}>0,100*FILL(h${i},0)/t${i})`, label: r.region, id: `a${i}` }],
 ]));
 chart("Rejected requests / minute", app("dynamo_global_frontend_rejected_requests_total", "global-router"));
 chart("Interrupted response bodies / minute", app("dynamo_global_frontend_response_errors_total", "global-router"));
 chart("Cached input tokens / minute — serving pool", regions.map(r => [metricNamespace, "dynamo_frontend_cached_tokens_sum", "ClusterName", r.clusterName, "Component", "pool-frontend", "Pool", r.pool, "ServingRegion", r.region, { region: r.region, label: `${r.region}/${r.pool}` }]));
 // Input is counted during preprocessing; cached usage arrives with a later response.
 // Their interval deltas cover different requests and cannot define a hit percentage.
 chart("Input and cached tokens / minute — unpaired request stages", regions.flatMap(r => [
   [metricNamespace, "dynamo_frontend_input_tokens_total", "ClusterName", r.clusterName, "Component", "pool-frontend", { region: r.region, label: `${r.region} input (preprocessing)` }],
   [metricNamespace, "dynamo_frontend_cached_tokens_sum", "ClusterName", r.clusterName, "Component", "pool-frontend", { region: r.region, label: `${r.region} cached (response report)` }],
 ]));
 chart("Input tokens / minute", app("dynamo_frontend_input_tokens_total", "pool-frontend"));
 chart("Output tokens / minute", app("dynamo_frontend_output_tokens_total", "pool-frontend"));
 if (global) {
   const totals = regions.map((r, i) => [metricNamespace, "dynamo_frontend_cached_tokens_sum", "ClusterName", r.clusterName, "Component", "pool-frontend", { region: r.region, id: `c${i}`, visible: false }]);
   chart("Global cached input tokens / minute — sum of serving regions", [...totals, [{ expression: `SUM([${regions.map((_, i) => `c${i}`).join(",")}])`, label: "Global cached input tokens" }]]);
   chart("Cached-token usage reports / minute — coverage", app("dynamo_frontend_cached_tokens_count", "pool-frontend"));
 }
 chart("Scrape health — minimum across router targets", app("up", "global-router", "Minimum"), "Minimum");
 chart("Scrape health — minimum across pool targets", app("up", "pool-frontend", "Minimum"), "Minimum");
 chart("Pool model readiness", app("dynamo_frontend_model_ready", "pool-frontend", "Minimum"), "Minimum");
 chart("Available workers — serving pool", regions.map(r => [metricNamespace, "dynamo_frontend_router_workers", "ClusterName", r.clusterName, "Component", "pool-frontend", "Pool", r.pool, "ServingRegion", r.region, "state", "available", { region: r.region, stat: "Average", label: `${r.region}/${r.pool}` }]), "Average");
 chart("Router HTTP status / minute", regions.map(r => [{ expression: `SEARCH('{${metricNamespace},ClusterName,Component,status} MetricName="dynamo_global_frontend_requests_total" ClusterName="${r.clusterName}"', 'Sum', 60)`, region: r.region }]), "Sum");
 chart("Pool request status / minute", regions.map(r => [{ expression: `SEARCH('{${metricNamespace},ClusterName,Component,status} MetricName="dynamo_frontend_requests_total" ClusterName="${r.clusterName}"', 'Sum', 60)`, region: r.region }]), "Sum");
 for (const name of ["node_cpu_utilization", "node_memory_utilization", "pod_cpu_utilization", "pod_memory_utilization", "pod_number_of_container_restarts", "cluster_failed_node_count"]) chart(name, regions.map(r => ["ContainerInsights", name, "ClusterName", r.clusterName, { region: r.region, stat: "Maximum", label: r.region }]), "Maximum");
 chart("Service running pods — router / discovery / frontend / relay", regions.flatMap(r => c.monitoredServices.map(service => ["ContainerInsights", "service_number_of_running_pods", "ClusterName", r.clusterName, "Namespace", service.namespace, "Service", service.name, { region: r.region, stat: "Minimum", label: `${r.region} ${service.name}` }])), "Minimum", 24);
 for (const table of [c.stores.affinityTableName, c.stores.discoveryTableName]) {
   chart(`${table} throttling`, regions.flatMap(r => ["ReadThrottleEvents", "WriteThrottleEvents"].map(name => ["AWS/DynamoDB", name, "TableName", table, { region: r.region, label: `${r.region} ${name}` }])));
   chart(`${table} request latency (ms)`, regions.flatMap(r => ["GetItem", "PutItem", "Query", "UpdateItem"].map(operation => ["AWS/DynamoDB", "SuccessfulRequestLatency", "TableName", table, "Operation", operation, { region: r.region, stat: "Average", label: `${r.region} ${operation}` }])), "Average");
   chart(`${table} replication latency (ms)`, regions.flatMap(r => c.dashboardRegions.filter(dest => dest.region !== r.region).map(dest => ["AWS/DynamoDB", "ReplicationLatency", "TableName", table, "ReceivingRegion", dest.region, { region: r.region, stat: "Maximum", label: `${r.region} → ${dest.region}` }])), "Maximum", 24);
 }
 text("## Triage\nInput and cached token increments cover different request stages; dividing them per minute is NOT a KV-cache hit percentage and can exceed 100%. Exact cache reuse needs paired prompt/cache usage from the same requests. Start with regional alarms, scrape health, request errors and latency. Correlate pod restarts/CPU/memory with DynamoDB throttling and affinity failures. Discovery and relay details are in application logs; no discovery-membership metric is claimed. Collectors discard their initial cumulative baseline on restart. No alarm notification recipients are configured.");
 for (const r of regions) {
   widgets.push({ type: "log", x: 0, y: row, width: 24, height: 6, properties: { region: r.region, title: `${r.region} — recent global / pool application warnings and errors`, query: `SOURCE '/aws/containerinsights/${r.clusterName}/application' | fields @timestamp, kubernetes.namespace_name, kubernetes.pod_name, log | filter kubernetes.namespace_name in ['${c.globalNamespace}', '${c.poolNamespace}'] | filter log like /(?i)(error|warn|timeout|fail)/ | sort @timestamp desc | limit 40`, view: "table" } }); row += 6;
 }
 return { start: "-PT1H", periodOverride: "inherit", widgets };
}
