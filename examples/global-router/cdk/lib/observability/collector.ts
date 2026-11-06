// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { ObservabilityConfig } from "./config";
export const metricNamespace = "Dynamo/GlobalRouter";
export function collectorConfig(c: ObservabilityConfig) {
  const labels = [{ target_label: "ClusterName", replacement: c.cluster.name }];
  const job = (component: string, namespace: string, selector: string, pattern: string, port = "http") => ({
    job_name: component, scrape_interval: "60s", scrape_timeout: "10s",
    kubernetes_sd_configs: [{ role: "pod", namespaces: { names: [namespace] } }],
    relabel_configs: [
      { source_labels: [selector], action: "keep", regex: pattern },
      { source_labels: ["__meta_kubernetes_pod_container_port_name"], action: "keep", regex: port },
      { source_labels: ["__meta_kubernetes_pod_phase"], action: "keep", regex: "Running" },
      ...labels, { source_labels: ["__meta_kubernetes_pod_name"], target_label: "PodName" }, { target_label: "Component", replacement: component },
      { target_label: "IngressRegion", replacement: c.cluster.region },
      ...(component !== "global-router" ? [
        { target_label: "ServingRegion", replacement: c.cluster.region },
        { source_labels: ["__meta_kubernetes_pod_label_nvidia_com_dynamo_graph_deployment_name"], target_label: "Pool" },
      ] : []),
    ],
    metric_relabel_configs: [{ source_labels: ["__name__"], action: "keep", regex:
      component === "global-router" ? "dynamo_global_frontend_.*" :
      component === "pool-worker" ? "vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)" :
        "dynamo_frontend_(requests_total|requests_started_total|input_tokens_total|output_tokens_total|cached_tokens.*|request_duration_seconds.*|time_to_first_token_seconds.*|inter_token_latency_seconds.*|model_ready|router_workers|active_requests|queued_requests|awaiting_first_token_requests|inflight_input_tokens|input_sequence_tokens.*|output_sequence_tokens.*|tokenizer_latency_ms.*)" }],
  });
  return {
    extensions: { health_check: { endpoint: "0.0.0.0:13133" } },
    receivers: { prometheus: { config: { scrape_configs: [
      job("global-router", c.globalNamespace, "__meta_kubernetes_pod_label_app_kubernetes_io_name", c.routerName),
      job("pool-frontend", c.poolNamespace, "__meta_kubernetes_pod_label_nvidia_com_dynamo_component_type", "frontend"),
      job("pool-worker", c.poolNamespace, "__meta_kubernetes_pod_label_nvidia_com_dynamo_component_type", "decode|prefill|worker", "system"),
    ] } } },
    processors: {
      memory_limiter: { check_interval: "5s", limit_mib: 384, spike_limit_mib: 96 },
      "transform/histograms": { error_mode: "propagate", metric_statements: [
        'extract_sum_metric(true, "_sum") where metric.type == METRIC_DATA_TYPE_HISTOGRAM',
        'extract_count_metric(true, "_count") where metric.type == METRIC_DATA_TYPE_HISTOGRAM',
      ] },
      "filter/histograms": { error_mode: "propagate", metrics: { metric: ['type == METRIC_DATA_TYPE_HISTOGRAM'] } },
      // Preserve target identity until reset-aware delta conversion. Never difference aggregated counters.
      cumulativetodelta: { initial_value: "drop", max_staleness: "10m" },
      "transform/status": { error_mode: "propagate", metric_statements: [
        { context: "datapoint", statements: ['set(attributes["ResponseClass"], "server_error") where metric.name == "dynamo_global_frontend_requests_total" and IsMatch(attributes["status"], "^(5[0-9][0-9]|no_response)$")'] },
      ] },
      batch: { timeout: "10s", send_batch_size: 1024 },
    },
    exporters: { awsemf: {
      region: c.cluster.region, namespace: metricNamespace,
      log_group_name: `/dynamo/${c.fleet}/prometheus`, log_stream_name: c.cluster.name,
      dimension_rollup_option: "NoDimensionRollup", retain_initial_value_of_delta_metric: true,
      metric_declarations: [
        { dimensions: [["ClusterName", "Component"]], metric_name_selectors: ["^dynamo_.*", "^vllm:.*", "^up$"] },
        { dimensions: [["ClusterName", "Component", "Pool", "ServingRegion"]], metric_name_selectors: ["^dynamo_frontend_.*", "^vllm:.*", "^up$"] },
        { dimensions: [["ClusterName", "Component", "pool", "region"]], metric_name_selectors: ["^dynamo_global_frontend_(requests_total|request_duration_seconds_.*|time_to_first_byte_seconds_.*)$"] },
        // Per-target gauges: dashboards sum target averages, never temporal sample sums.
        { dimensions: [["ClusterName", "Component", "Pool", "ServingRegion", "PodName", "model"]], metric_name_selectors: ["^dynamo_frontend_(active_requests|queued_requests|awaiting_first_token_requests)$"] },
        { dimensions: [["ClusterName", "Component", "Pool", "ServingRegion", "PodName", "model", "state"]], metric_name_selectors: ["^dynamo_frontend_inflight_input_tokens$"] },
        { dimensions: [["ClusterName", "Component", "PodName", "pool", "region"]], metric_name_selectors: ["^dynamo_global_frontend_inflight_requests$"] },
        { dimensions: [["ClusterName", "Component", "Pool", "ServingRegion", "PodName", "worker_id", "model", "engine"]], metric_name_selectors: ["^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)$"] },
        { dimensions: [["ClusterName", "Component", "local"]], metric_name_selectors: ["^dynamo_global_frontend_requests_total$"] },
        { dimensions: [["ClusterName", "Component", "reason"]], metric_name_selectors: ["^dynamo_global_frontend_(affinity_store_writes_total|rejected_requests_total|response_errors_total)$"] },
        { dimensions: [["ClusterName", "Component", "Pool", "ServingRegion", "error_type"]], metric_name_selectors: ["^dynamo_frontend_requests_total$"] },
        { dimensions: [["ClusterName", "Component", "Pool", "ServingRegion", "operation"]], metric_name_selectors: ["^dynamo_frontend_tokenizer_latency_ms_(sum|count)$"] },
        { dimensions: [["ClusterName", "Component", "ResponseClass"]], metric_name_selectors: ["^dynamo_global_frontend_requests_total$"] },
        { dimensions: [["ClusterName", "Component", "status"]], metric_name_selectors: [".*requests_total$"] },
        { dimensions: [["ClusterName", "Component", "outcome"]], metric_name_selectors: [".*affinity_(lookups|cache)_total$"] },
        { dimensions: [["ClusterName", "Component", "kind"]], metric_name_selectors: [".*affinity_store_errors_total$"] },
        { dimensions: [["ClusterName", "Component", "Pool", "ServingRegion", "state"]], metric_name_selectors: ["^dynamo_frontend_router_workers$"] },
      ],
    } },
    service: { extensions: ["health_check"], pipelines: { metrics: {
      receivers: ["prometheus"], processors: ["memory_limiter", "transform/histograms", "filter/histograms", "cumulativetodelta", "transform/status", "batch"], exporters: ["awsemf"],
    } }, telemetry: { logs: { level: "info" } } },
  };
}
