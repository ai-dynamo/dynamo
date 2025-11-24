// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "integration")]

use dynamo_runtime::{
    DistributedRuntime, Result, Runtime, config::environment_names::runtime::system as env_system,
    pipeline::PushRouter, protocols::annotated::Annotated,
};
use futures::StreamExt;
use rand::Rng;
use reqwest;
use std::env;
use system_metrics::{DEFAULT_COMPONENT, DEFAULT_ENDPOINT, DEFAULT_NAMESPACE, backend};
use tokio::time::{Duration, sleep};

#[tokio::test]
async fn test_backend_with_metrics() -> Result<()> {
    // Set environment variable for dynamic port allocation (0 = auto-assign)
    env::set_var(env_system::DYN_SYSTEM_PORT, "0");

    // Generate a random endpoint name to avoid collisions
    let random_suffix = rand::rng().random_range(1000..9999);
    let test_endpoint = format!("{}{}", DEFAULT_ENDPOINT, random_suffix);

    // Initialize logging
    dynamo_runtime::logging::init();

    // Create a runtime and distributed runtime for the backend
    let runtime = Runtime::from_current()?;
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    // Get the System status server info to find the actual port
    let system_status_info = distributed.system_status_server_info();
    let system_status_port = match system_status_info {
        Some(info) => {
            println!("System status server running on: {}", info.address());
            info.port()
        }
        None => {
            panic!("System status server not started - check DYN_SYSTEM_PORT environment variable");
        }
    };

    // Start the backend in a separate task with custom endpoint
    let test_endpoint_clone = test_endpoint.clone();
    let backend_handle =
        tokio::spawn(async move { backend(distributed, Some(&test_endpoint_clone)).await });

    // Give the backend some time to start up
    sleep(Duration::from_millis(1000)).await;

    // Create a client runtime to connect to the backend
    let client_runtime = Runtime::from_current()?;
    let client_distributed = DistributedRuntime::from_settings(client_runtime.clone()).await?;

    // Connect to the backend similar to system_client.rs
    let namespace = client_distributed.namespace(DEFAULT_NAMESPACE)?;
    let component = namespace.component(DEFAULT_COMPONENT)?;
    let client = component.endpoint(&test_endpoint).client().await?;

    // Wait for backend instances to be available
    client.wait_for_instances().await?;

    // Create a router and send some requests to generate metrics
    let router =
        PushRouter::<String, Annotated<String>>::from_client(client, Default::default()).await?;

    // Send a few test requests to generate metrics
    for i in 0..3 {
        let test_message = format!("test message {}", i);
        let mut stream = router.random(test_message.clone().into()).await?;

        // Process the response stream
        while let Some(resp) = stream.next().await {
            println!("Response {}: {:?}", i, resp);
        }

        // Small delay between requests
        sleep(Duration::from_millis(100)).await;
    }

    // Give some time for metrics to be updated
    sleep(Duration::from_millis(500)).await;

    // Now fetch the HTTP metrics endpoint using the dynamic port
    let metrics_url = format!("http://localhost:{}/metrics", system_status_port);

    println!("Fetching metrics from: {}", metrics_url);

    // Make HTTP request to get metrics
    let client = reqwest::Client::new();
    let response = client.get(&metrics_url).send().await;

    match response {
        Ok(response) => {
            if response.status().is_success() {
                let metrics_content = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Failed to read response body".to_string());

                println!("=== METRICS CONTENT ===");
                println!("{}", metrics_content);
                println!("=== END METRICS CONTENT ===");

                // Parse and verify ingress metrics are greater than 0 (except inflight_requests)
                verify_ingress_metrics_greater_than_0(&metrics_content);

                println!("Successfully retrieved and verified metrics!");
            } else {
                println!("HTTP request failed with status: {}", response.status());
                panic!("Failed to get metrics: HTTP {}", response.status());
            }
        }
        Err(e) => {
            println!("Failed to connect to metrics endpoint: {}", e);
            panic!("Failed to connect to metrics endpoint: {}", e);
        }
    }

    // Shutdown the runtime
    client_runtime.shutdown();

    // Cancel the backend task
    backend_handle.abort();

    Ok(())
}

fn verify_ingress_metrics_greater_than_0(metrics_content: &str) {
    // Define the work handler metrics we want to verify (excluding inflight_requests which can be 0)
    let metrics_to_verify = [
        "my_custom_bytes_processed_total",
        "requests_total",
        "request_bytes_total",
        "response_bytes_total",
        "request_duration_seconds_count",
        "request_duration_seconds_sum",
    ];

    for metric_name in &metrics_to_verify {
        let line = metrics_content
            .lines()
            .find(|l| l.contains(metric_name) && !l.contains("#"))
            .unwrap_or_else(|| panic!("{} metric not found", metric_name));

        let value = extract_metric_value(line);
        assert!(
            value > 0.0,
            "{} should be greater than 0, got: {}",
            metric_name,
            value
        );
        println!("{}: {}", metric_name, value);
    }

    println!("All work handler metrics verified successfully!");
}

fn extract_metric_value(line: &str) -> f64 {
    // Extract the numeric value from a Prometheus metric line
    // Format: metric_name{labels} value
    line.split_whitespace()
        .last()
        .expect("Metric line should have a value")
        .parse::<f64>()
        .expect("Metric value should be a valid number")
}
