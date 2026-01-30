// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::config::environment_names::zmq_broker as env;
use dynamo_runtime::discovery::{DiscoverySpec, EventTransport};
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::runtime::Runtime;
use std::env as std_env;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Parse configuration from env vars
    let xsub_bind = std_env::var(env::ZMQ_BROKER_XSUB_BIND)
        .unwrap_or_else(|_| "tcp://0.0.0.0:5555".to_string());
    let xpub_bind = std_env::var(env::ZMQ_BROKER_XPUB_BIND)
        .unwrap_or_else(|_| "tcp://0.0.0.0:5556".to_string());
    let namespace =
        std_env::var(env::ZMQ_BROKER_NAMESPACE).unwrap_or_else(|_| "dynamo".to_string());

    tracing::info!(
        xsub_bind = %xsub_bind,
        xpub_bind = %xpub_bind,
        namespace = %namespace,
        "Starting ZMQ broker"
    );

    // Create ZMQ context (shared across sockets)
    let ctx = Arc::new(zmq::Context::new());

    // Resolve actual endpoints (handles port 0 -> random port)
    let (xsub_endpoint, xpub_endpoint) = {
        let ctx = Arc::clone(&ctx);
        let xsub_bind = xsub_bind.clone();
        let xpub_bind = xpub_bind.clone();

        tokio::task::spawn_blocking(move || -> Result<(String, String)> {
            // Bind XSUB socket
            let xsub = ctx.socket(zmq::XSUB)?;
            xsub.bind(&xsub_bind)?;
            let xsub_ep = xsub
                .get_last_endpoint()?
                .map_err(|e| anyhow::anyhow!("Invalid XSUB endpoint: {:?}", e))?;

            // Bind XPUB socket
            let xpub = ctx.socket(zmq::XPUB)?;
            xpub.bind(&xpub_bind)?;
            let xpub_ep = xpub
                .get_last_endpoint()?
                .map_err(|e| anyhow::anyhow!("Invalid XPUB endpoint: {:?}", e))?;

            // Drop sockets so we can rebind in the proxy thread
            drop(xsub);
            drop(xpub);

            Ok((xsub_ep, xpub_ep))
        })
        .await??
    };

    tracing::info!(
        xsub_endpoint = %xsub_endpoint,
        xpub_endpoint = %xpub_endpoint,
        "Resolved broker endpoints"
    );

    // Start the proxy in a blocking thread
    let proxy_xsub_bind = xsub_bind.clone();
    let proxy_xpub_bind = xpub_bind.clone();
    let proxy_handle = tokio::task::spawn_blocking(move || -> Result<()> {
        let ctx = zmq::Context::new();

        // XSUB socket for publishers to connect
        let xsub = ctx.socket(zmq::XSUB)?;
        xsub.bind(&proxy_xsub_bind)?;
        tracing::info!(endpoint = %proxy_xsub_bind, "XSUB socket bound");

        // XPUB socket for subscribers to connect
        let xpub = ctx.socket(zmq::XPUB)?;
        xpub.bind(&proxy_xpub_bind)?;
        tracing::info!(endpoint = %proxy_xpub_bind, "XPUB socket bound");

        tracing::info!("Starting ZMQ proxy (XSUB <-> XPUB)");

        // Run proxy (blocking - forwards all messages bidirectionally)
        // XSUB receives from publishers, XPUB sends to subscribers
        // Subscription messages flow XPUB -> XSUB
        zmq::proxy(&xsub, &xpub)?;

        Ok(())
    });

    let _discovery_handle = if std_env::var("ETCD_ENDPOINTS").is_ok() {
        tracing::info!("Registering broker with discovery plane");

        let rt = Runtime::from_handle(tokio::runtime::Handle::current())?;
        let config = DistributedConfig::from_settings();
        let drt = DistributedRuntime::new(rt, config).await?;
        let _ns = drt.namespace(&namespace)?;

        // Convert local bind to public endpoint
        // Replace 0.0.0.0 with actual IP for discovery
        let public_xsub = make_public_endpoint(&xsub_endpoint)?;
        let public_xpub = make_public_endpoint(&xpub_endpoint)?;

        let spec = DiscoverySpec::EventChannel {
            namespace: namespace.clone(),
            component: "zmq_broker".to_string(),
            topic: "broker".to_string(),
            transport: EventTransport::zmq_broker(vec![public_xsub], vec![public_xpub]),
        };

        let instance = drt.discovery().register(spec).await?;
        tracing::info!(
            instance_id = %instance.instance_id(),
            "Broker registered with discovery plane"
        );

        Some((drt, instance))
    } else {
        tracing::warn!("ETCD_ENDPOINTS not set, skipping discovery registration");
        tracing::info!("Clients should use DYN_ZMQ_BROKER_URL to connect directly");
        None
    };

    // Setup signal handler for graceful shutdown
    let shutdown_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        tracing::info!("Received shutdown signal");
    };

    // Wait for shutdown signal or proxy error
    tokio::select! {
        result = proxy_handle => {
            match result {
                Ok(Ok(())) => {
                    tracing::info!("Proxy thread completed normally");
                }
                Ok(Err(e)) => {
                    tracing::error!(error = %e, "Proxy thread failed");
                    return Err(e);
                }
                Err(e) => {
                    tracing::error!(error = %e, "Proxy thread panicked");
                    return Err(anyhow::anyhow!("Proxy thread panicked: {}", e));
                }
            }
        }
        _ = shutdown_signal => {
            tracing::info!("Shutting down broker");
        }
    }

    Ok(())
}

/// Convert a local bind address to a public endpoint for discovery.
/// Replaces 0.0.0.0 with the local IP address.
fn make_public_endpoint(endpoint: &str) -> Result<String> {
    if endpoint.contains("0.0.0.0") {
        let local_ip = dynamo_runtime::utils::ip_resolver::get_local_ip_for_advertise();
        Ok(endpoint.replace("0.0.0.0", &local_ip.to_string()))
    } else {
        Ok(endpoint.to_string())
    }
}
