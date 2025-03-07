// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Metrics is a metrics aggregator designed to operate within a namespace and collect
//! metrics from all workers.
//!
//! Metrics will collect for now:
//!
//! - LLM Worker Load:Capacity
//!   - These metrics will be scraped by the LLM NATS Service API's stats request
//!   - Request Slots: [Active, Total]
//!   - KV Cache Blocks: [Active, Total]
//! - KV Hit Rate:
//!   - These metrics will be collected from KV hit rate events published by the KV router
//!   - ISL Blocks: Cumulative count of total blocks in all KV hit rate events
//!   - Overlap Blocks: Cumulative count of blocks that were already in the KV cache
use clap::Parser;
use dynamo_llm::kv_router::scheduler::KVHitRateEvent;
use dynamo_llm::kv_router::KV_HIT_RATE_SUBJECT;
use dynamo_runtime::{
    error, logging,
    traits::events::{EventPublisher, EventSubscriber},
    utils::{Duration, Instant},
    DistributedRuntime, ErrorContext, Result, Runtime, Worker,
};
use futures::stream::StreamExt;
use std::sync::Arc;

// Import from our library
use metrics::{
    collect_endpoints, extract_metrics, postprocess_metrics, LLMWorkerLoadCapacityConfig,
    PrometheusMetricsServer,
};

/// CLI arguments for the metrics application
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Component to scrape metrics from
    #[arg(long)]
    component: String,

    /// Endpoint to scrape metrics from
    #[arg(long)]
    endpoint: String,

    /// Namespace to operate in
    #[arg(long, env = "DYN_NAMESPACE", default_value = "dynamo")]
    namespace: String,

    /// Polling interval in seconds (minimum 1 second)
    #[arg(long, default_value = "2")]
    poll_interval: u64,
}

fn get_config(args: &Args) -> Result<LLMWorkerLoadCapacityConfig> {
    if args.component.is_empty() {
        return Err(error!("Component name cannot be empty"));
    }

    if args.endpoint.is_empty() {
        return Err(error!("Endpoint name cannot be empty"));
    }

    if args.poll_interval < 1 {
        return Err(error!("Polling interval must be at least 1 second"));
    }

    Ok(LLMWorkerLoadCapacityConfig {
        component_name: args.component.clone(),
        endpoint_name: args.endpoint.clone(),
    })
}

async fn app(runtime: Runtime) -> Result<()> {
    let args = Args::parse();
    let config = get_config(&args)?;
    tracing::info!("Config: {config:?}");

    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    let namespace = drt.namespace(args.namespace)?;
    let component = namespace.component("count")?;

    // Create unique instance of Count
    let key = format!("{}/instance", component.etcd_path());
    tracing::info!("Creating unique instance of Count at {key}");
    drt.etcd_client()
        .kv_create(
            key,
            serde_json::to_vec_pretty(&config)?,
            Some(drt.primary_lease().id()),
        )
        .await
        .context("Unable to create unique instance of Count; possibly one already exists")?;

    let target_component = namespace.component(&config.component_name)?;
    let target_endpoint = target_component.endpoint(&config.endpoint_name);

    let service_name = target_component.service_name();
    let service_subject = target_endpoint.subject();
    tracing::info!("Scraping service {service_name} and filtering on subject {service_subject}");

    let token = drt.primary_lease().child_token();
    let event_name = format!("l2c.{}.{}", config.component_name, config.endpoint_name);

    // TODO: Make metrics host/port configurable
    // Initialize Prometheus metrics and start server
    let metrics_server = PrometheusMetricsServer::new()?;
    // Metrics will be updated concurrently, so protect it with a mutex:
    // - Main loop: Collect and process ForwardPassMetrics at an interval from endpoint stats handlers
    // - Subscription task: Collect and process KVHitRateEvent metrics from the KV router as they are published
    let metrics_server = Arc::new(tokio::sync::Mutex::new(metrics_server));
    metrics_server.lock().await.start(9091);

    // Subscribe to KV hit rate events
    let kv_hit_rate_subject = KV_HIT_RATE_SUBJECT;
    tracing::info!("Subscribing to KV hit rate events on subject: {kv_hit_rate_subject}");

    // Clone the metrics server and config for the subscription task
    let metrics_server_clone = metrics_server.clone();
    let config_clone = config.clone();
    // Clone the namespace for the subscription task
    let namespace_clone = namespace.clone();

    // Spawn a task to handle KV hit rate events
    tokio::spawn(async move {
        match namespace_clone.subscribe(kv_hit_rate_subject).await {
            Ok(mut subscriber) => {
                tracing::info!("Successfully subscribed to KV hit rate events");

                while let Some(msg) = subscriber.next().await {
                    match serde_json::from_slice::<KVHitRateEvent>(&msg.payload) {
                        Ok(event) => {
                            // TODO: Lower to debug
                            let cache_hit_pct =
                                (event.overlap_blocks as f64 / event.isl_blocks as f64) * 100.0;
                            tracing::info!(
                                "Received KV hit rate event: worker_id={}, isl_blocks={}, overlap_blocks={}, cache_hit_pct={:.2}%",
                                event.worker_id,
                                event.isl_blocks,
                                event.overlap_blocks,
                                cache_hit_pct
                            );

                            // Update metrics with the event data
                            let mut metrics = metrics_server_clone.lock().await;
                            metrics.update_kv_hit_rate(
                                &config_clone,
                                event.worker_id,
                                event.isl_blocks,
                                event.overlap_blocks,
                            );
                        }
                        Err(e) => {
                            tracing::warn!("Failed to deserialize KV hit rate event: {e}");
                        }
                    }
                }

                tracing::warn!("KV hit rate event subscription stream ended");
            }
            Err(e) => {
                tracing::error!("Failed to subscribe to KV hit rate events: {:?}", e);
            }
        }
    });

    loop {
        let next = Instant::now() + Duration::from_secs(args.poll_interval);

        // Collect and process metrics
        let scrape_timeout = Duration::from_secs(1);
        let endpoints =
            collect_endpoints(&target_component, &service_subject, scrape_timeout).await?;
        let metrics = extract_metrics(&endpoints);
        let processed = postprocess_metrics(&metrics, &endpoints);
        tracing::debug!("Aggregated metrics: {processed:?}");

        // Update Prometheus metrics
        metrics_server.lock().await.update(&config, &processed);

        // TODO: Enable KV Routers to subscribe to metrics events published here
        // for a single view of the aggregated metrics, as opposed to the current
        // approach where each KV Router computes and published its own metrics.
        // Publish metrics event
        namespace.publish(&event_name, &processed).await?;

        // Wait until cancelled or the next tick
        match tokio::time::timeout_at(next, token.cancelled()).await {
            Ok(_) => break,
            Err(_) => continue,
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_namespace_from_env() {
        env::set_var("DYN_NAMESPACE", "test-namespace");
        let args = Args::parse_from(["count", "--component", "comp", "--endpoint", "end"]);
        assert_eq!(args.namespace, "test-namespace");
    }
}
