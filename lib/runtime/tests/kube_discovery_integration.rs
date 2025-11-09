// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Kubernetes discovery client
//! 
//! These tests require:
//! 1. Access to a Kubernetes cluster (kubectl configured)
//! 2. Test resources deployed (run k8s-test/deploy.sh)
//! 
//! Run with: cargo test --test kube_discovery_integration -- --nocapture

use futures::StreamExt;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use kube::runtime::{watcher, watcher::Config};

/// Test that we can successfully create a Kubernetes client
#[tokio::test]
#[ignore] // Run manually with: cargo test --test kube_discovery_integration test_kube_client_connection -- --ignored
async fn test_kube_client_connection() {
    println!("🔌 Testing Kubernetes client connection...");
    
    let client = Client::try_default()
        .await
        .expect("Failed to create Kubernetes client - is kubectl configured?");
    
    println!("✅ Successfully connected to Kubernetes cluster");
    
    // Try to list namespaces as a connectivity test
    let namespaces: Api<k8s_openapi::api::core::v1::Namespace> = Api::all(client);
    let ns_list = namespaces.list(&Default::default()).await
        .expect("Failed to list namespaces");
    
    println!("📋 Found {} namespaces", ns_list.items.len());
    println!("✅ Kubernetes API is accessible");
}

/// Test listing EndpointSlices
#[tokio::test]
#[ignore] // Run manually with: cargo test --test kube_discovery_integration test_list_endpointslices -- --ignored
async fn test_list_endpointslices() {
    println!("📋 Testing EndpointSlice listing...");
    
    let client = Client::try_default()
        .await
        .expect("Failed to create Kubernetes client");
    
    let endpoint_slices: Api<EndpointSlice> = Api::namespaced(client, "default");
    
    // List all EndpointSlices in default namespace
    let list_params = kube::api::ListParams::default();
    let slices = endpoint_slices.list(&list_params).await
        .expect("Failed to list EndpointSlices");
    
    println!("📊 Found {} EndpointSlices in default namespace", slices.items.len());
    
    for slice in &slices.items {
        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
        let service = slice.metadata.labels.as_ref()
            .and_then(|l| l.get("kubernetes.io/service-name"))
            .map(|s| s.as_str())
            .unwrap_or("<none>");
        
        let endpoint_count = slice.endpoints.len();
        
        println!("  • {} (service: {}, endpoints: {})", name, service, endpoint_count);
        
        // Show endpoint details
        for (i, endpoint) in slice.endpoints.iter().enumerate() {
            let ready = endpoint.conditions.as_ref()
                .and_then(|c| c.ready)
                .unwrap_or(false);
            let addresses = &endpoint.addresses;
            let pod_name = endpoint.target_ref.as_ref()
                .and_then(|t| t.name.as_ref())
                .map(|n| n.as_str())
                .unwrap_or("<unknown>");
            
            println!("    [{}] pod={}, ready={}, addresses={:?}", 
                     i, pod_name, ready, addresses);
        }
    }
    
    println!("✅ EndpointSlice listing test completed");
}

/// Test listing EndpointSlices with label selector (like our discovery client does)
#[tokio::test]
#[ignore] // Run manually with: cargo test --test kube_discovery_integration test_list_with_labels -- --ignored
async fn test_list_with_labels() {
    println!("🏷️  Testing EndpointSlice listing with label selector...");
    
    let client = Client::try_default()
        .await
        .expect("Failed to create Kubernetes client");
    
    let endpoint_slices: Api<EndpointSlice> = Api::all(client);
    
    // Test the label selector we use in our discovery client
    let label_selector = "dynamo.nvidia.com/namespace=test-namespace,dynamo.nvidia.com/component=test-component";
    println!("Using label selector: {}", label_selector);
    
    let list_params = kube::api::ListParams::default()
        .labels(label_selector);
    
    let slices = endpoint_slices.list(&list_params).await
        .expect("Failed to list EndpointSlices with labels");
    
    println!("📊 Found {} EndpointSlices matching labels", slices.items.len());
    
    if slices.items.is_empty() {
        println!("⚠️  No EndpointSlices found with Dynamo labels.");
        println!("   Make sure test resources are deployed: ./k8s-test/deploy.sh");
        println!("   Note: Kubernetes creates EndpointSlices automatically,");
        println!("   but pod labels don't flow to EndpointSlices by default.");
    }
    
    for slice in &slices.items {
        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
        let endpoint_count = slice.endpoints.len();
        println!("  • {} (endpoints: {})", name, endpoint_count);
    }
    
    println!("✅ Label selector test completed");
}

/// Test watching EndpointSlices for changes
#[tokio::test]
#[ignore] // Run manually with: cargo test --test kube_discovery_integration test_watch_endpointslices -- --ignored
async fn test_watch_endpointslices() {
    println!("👀 Testing EndpointSlice watching...");
    println!("   This test will watch for 10 seconds or 5 events, whichever comes first");
    
    let client = Client::try_default()
        .await
        .expect("Failed to create Kubernetes client");
    
    let endpoint_slices: Api<EndpointSlice> = Api::namespaced(client, "default");
    
    // Create watcher
    let watch_config = Config::default();
    let mut watch_stream = Box::pin(watcher(endpoint_slices, watch_config));
    
    println!("📡 Watch stream started...");
    
    let mut event_count = 0;
    let max_events = 5;
    let timeout = tokio::time::Duration::from_secs(10);
    let deadline = tokio::time::Instant::now() + timeout;
    
    loop {
        tokio::select! {
            Some(event) = watch_stream.next() => {
                event_count += 1;
                match event {
                    Ok(watcher::Event::Apply(slice)) => {
                        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
                        let endpoint_count = slice.endpoints.len();
                        println!("  [{}] ✅ Apply: {} (endpoints: {})", event_count, name, endpoint_count);
                    }
                    Ok(watcher::Event::InitApply(slice)) => {
                        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
                        let endpoint_count = slice.endpoints.len();
                        println!("  [{}] 🔄 InitApply: {} (endpoints: {})", event_count, name, endpoint_count);
                    }
                    Ok(watcher::Event::Delete(slice)) => {
                        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
                        println!("  [{}] ❌ Delete: {}", event_count, name);
                    }
                    Ok(watcher::Event::Init) => {
                        println!("  [{}] 🚀 Init - watch stream starting", event_count);
                    }
                    Ok(watcher::Event::InitDone) => {
                        println!("  [{}] ✅ InitDone - initial list complete", event_count);
                    }
                    Err(e) => {
                        println!("  [{}] ⚠️  Error: {}", event_count, e);
                    }
                }
                
                if event_count >= max_events {
                    println!("📊 Reached max events ({}), stopping watch", max_events);
                    break;
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                println!("⏰ Timeout reached ({}s), stopping watch", timeout.as_secs());
                break;
            }
        }
    }
    
    println!("✅ Watch test completed ({} events received)", event_count);
}

/// Test watching EndpointSlices with label selector
#[tokio::test]
#[ignore] // Run manually with: cargo test --test kube_discovery_integration test_watch_with_labels -- --ignored
async fn test_watch_with_labels() {
    println!("👀 Testing EndpointSlice watching with label selector...");
    println!("   This test will watch for 5 seconds or until InitDone");
    
    let client = Client::try_default()
        .await
        .expect("Failed to create Kubernetes client");
    
    let endpoint_slices: Api<EndpointSlice> = Api::all(client);
    
    // Watch with our discovery labels
    let label_selector = "kubernetes.io/service-name=dynamo-test-service";
    println!("Using label selector: {}", label_selector);
    
    let watch_config = Config::default()
        .labels(label_selector);
    let mut watch_stream = Box::pin(watcher(endpoint_slices, watch_config));
    
    println!("📡 Watch stream started...");
    
    let mut event_count = 0;
    let timeout = tokio::time::Duration::from_secs(5);
    let deadline = tokio::time::Instant::now() + timeout;
    let mut init_done = false;
    
    loop {
        tokio::select! {
            Some(event) = watch_stream.next() => {
                event_count += 1;
                match event {
                    Ok(watcher::Event::Apply(slice)) => {
                        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
                        let endpoint_count = slice.endpoints.len();
                        println!("  [{}] ✅ Apply: {} (endpoints: {})", event_count, name, endpoint_count);
                    }
                    Ok(watcher::Event::InitApply(slice)) => {
                        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
                        let endpoint_count = slice.endpoints.len();
                        println!("  [{}] 🔄 InitApply: {} (endpoints: {})", event_count, name, endpoint_count);
                    }
                    Ok(watcher::Event::Delete(slice)) => {
                        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
                        println!("  [{}] ❌ Delete: {}", event_count, name);
                    }
                    Ok(watcher::Event::Init) => {
                        println!("  [{}] 🚀 Init - watch stream starting", event_count);
                    }
                    Ok(watcher::Event::InitDone) => {
                        println!("  [{}] ✅ InitDone - initial list complete", event_count);
                        init_done = true;
                    }
                    Err(e) => {
                        println!("  [{}] ⚠️  Error: {}", event_count, e);
                    }
                }
                
                if init_done {
                    println!("📊 InitDone received, stopping watch");
                    break;
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                println!("⏰ Timeout reached ({}s), stopping watch", timeout.as_secs());
                break;
            }
        }
    }
    
    println!("✅ Watch with labels test completed ({} events received)", event_count);
}

/// Comprehensive test that simulates our discovery client behavior
#[tokio::test]
#[ignore] // Run manually with: cargo test --test kube_discovery_integration test_discovery_simulation -- --ignored
async fn test_discovery_simulation() {
    println!("🔍 Testing discovery client simulation...");
    println!("   This simulates how our KubeDiscoveryClient list_and_watch works");
    
    let client = Client::try_default()
        .await
        .expect("Failed to create Kubernetes client");
    
    let endpoint_slices: Api<EndpointSlice> = Api::all(client);
    
    // Use service name label (EndpointSlices automatically get this label)
    let label_selector = "kubernetes.io/service-name=dynamo-test-service";
    println!("Label selector: {}", label_selector);
    
    let watch_config = Config::default()
        .labels(label_selector);
    let mut watch_stream = Box::pin(watcher(endpoint_slices, watch_config));
    
    println!("📡 Starting watch stream...");
    
    let mut seen_endpoints = std::collections::HashSet::new();
    let timeout = tokio::time::Duration::from_secs(10);
    let deadline = tokio::time::Instant::now() + timeout;
    
    loop {
        tokio::select! {
            Some(event) = watch_stream.next() => {
                match event {
                    Ok(watcher::Event::Apply(slice)) | Ok(watcher::Event::InitApply(slice)) => {
                        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
                        println!("  📦 Processing EndpointSlice: {}", name);
                        
                        // Extract endpoints (simulate our discovery logic)
                        for endpoint in &slice.endpoints {
                            let ready = endpoint.conditions.as_ref()
                                .and_then(|c| c.ready)
                                .unwrap_or(false);
                            
                            if !ready {
                                continue;
                            }
                            
                            let pod_name = endpoint.target_ref.as_ref()
                                .and_then(|t| t.name.as_ref())
                                .map(|n| n.as_str())
                                .unwrap_or_default();
                            
                            if pod_name.is_empty() {
                                continue;
                            }
                            
                            // Hash the pod name (simulate instance_id generation)
                            use std::collections::hash_map::DefaultHasher;
                            use std::hash::{Hash, Hasher};
                            let mut hasher = DefaultHasher::new();
                            pod_name.hash(&mut hasher);
                            let instance_id = hasher.finish();
                            
                            if seen_endpoints.insert(instance_id) {
                                let addresses = &endpoint.addresses;
                                println!("    ✅ New endpoint: pod={}, instance_id={:x}, addresses={:?}", 
                                         pod_name, instance_id, addresses);
                            }
                        }
                    }
                    Ok(watcher::Event::Delete(slice)) => {
                        let name = slice.metadata.name.as_deref().unwrap_or("<unknown>");
                        println!("  ❌ EndpointSlice deleted: {}", name);
                    }
                    Ok(watcher::Event::Init) => {
                        println!("  🚀 Watch stream initialized");
                    }
                    Ok(watcher::Event::InitDone) => {
                        println!("  ✅ Initial sync complete");
                        println!("  📊 Discovered {} unique endpoints", seen_endpoints.len());
                        break;
                    }
                    Err(e) => {
                        eprintln!("  ⚠️  Watch error: {}", e);
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                println!("⏰ Timeout reached");
                break;
            }
        }
    }
    
    println!("✅ Discovery simulation completed");
    println!("📊 Total unique endpoints discovered: {}", seen_endpoints.len());
    
    assert!(seen_endpoints.len() > 0, "Should have discovered at least one endpoint");
}

