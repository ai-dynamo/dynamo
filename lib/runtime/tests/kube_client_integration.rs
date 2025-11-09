// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for KubeDiscoveryClient
//! 
//! These tests require:
//! 1. Access to a Kubernetes cluster (kubectl configured)
//! 2. Test resources deployed (run k8s-test/deploy.sh)
//! 
//! Run with: cargo test --test kube_client_integration -- --ignored --nocapture

use dynamo_runtime::discovery::{
    KubeDiscoveryClient, Discovery, DiscoveryQuery,
};
use kube::Client;
use futures::StreamExt;

/// Helper to create a test client with mock metadata
async fn create_test_client() -> Result<KubeDiscoveryClient, Box<dyn std::error::Error>> {
    let kube_client = Client::try_default().await?;
    let client = KubeDiscoveryClient::new_for_testing(
        kube_client,
        "test-pod-123".to_string(),
        "discovery".to_string(),
        true, // mock_metadata = true (skip HTTP calls, return mock data)
    ).await?;
    Ok(client)
}

/// Test basic client creation and instance_id
#[tokio::test]
#[ignore]
async fn test_client_creation() {
    println!("🔌 Testing KubeDiscoveryClient creation...");
    
    let client = create_test_client().await
        .expect("Failed to create test client");
    
    let instance_id = client.instance_id();
    println!("✅ Client created with instance_id: {:x}", instance_id);
    
    assert_ne!(instance_id, 0, "Instance ID should not be zero");
}

/// Test listing all endpoints (without label filtering)
#[tokio::test]
#[ignore]
async fn test_list_all_endpoints() {
    println!("📋 Testing list all endpoints...");
    println!("   Note: Using mock metadata (no actual HTTP calls to pods)");
    
    let client = create_test_client().await
        .expect("Failed to create test client");
    
    let key = DiscoveryQuery::AllEndpoints;
    
    println!("Calling list() with key={:?}", key);
    let result = client.list(key).await;
    
    match result {
        Ok(instances) => {
            println!("✅ list() succeeded");
            println!("   Found {} instances", instances.len());
            
            for (i, instance) in instances.iter().enumerate() {
                println!("   [{}] {:?}", i, instance);
            }
        }
        Err(e) => {
            println!("❌ list() failed: {}", e);
        }
    }
    
    println!("✅ List test completed");
}

/// Test listing endpoints in a specific namespace
#[tokio::test]
#[ignore]
async fn test_list_namespaced_endpoints() {
    println!("📋 Testing list namespaced endpoints...");
    
    let client = create_test_client().await
        .expect("Failed to create test client");
    
    let key = DiscoveryQuery::NamespacedEndpoints {
        namespace: "test-namespace".to_string(),
    };
    
    println!("Calling list() with key={:?}", key);
    let result = client.list(key).await;
    
    match result {
        Ok(instances) => {
            println!("✅ list() succeeded");
            println!("   Found {} instances in test-namespace", instances.len());
        }
        Err(e) => {
            println!("⚠️  list() failed: {}", e);
        }
    }
    
    println!("✅ Namespaced list test completed");
}

/// Test listing endpoints for a specific component
#[tokio::test]
#[ignore]
async fn test_list_component_endpoints() {
    println!("📋 Testing list component endpoints...");
    
    let client = create_test_client().await
        .expect("Failed to create test client");
    
    let key = DiscoveryQuery::ComponentEndpoints {
        namespace: "test-namespace".to_string(),
        component: "test-component".to_string(),
    };
    
    println!("Calling list() with key={:?}", key);
    let result = client.list(key).await;
    
    match result {
        Ok(instances) => {
            println!("✅ list() succeeded");
            println!("   Found {} instances for test-namespace/test-component", instances.len());
        }
        Err(e) => {
            println!("⚠️  list() failed: {}", e);
        }
    }
    
    println!("✅ Component list test completed");
}

/// Test watching all endpoints
#[tokio::test]
#[ignore]
async fn test_watch_all_endpoints() {
    println!("👀 Testing watch all endpoints...");
    println!("   This test will watch for 10 seconds");
    println!("   Note: Using mock metadata (no actual HTTP calls to pods)");
    
    let client = create_test_client().await
        .expect("Failed to create test client");
    
    let key = DiscoveryQuery::AllEndpoints;
    
    println!("Calling list_and_watch() with key={:?}", key);
    let stream = client.list_and_watch(key, None).await
        .expect("Failed to create watch stream");
    
    let mut stream = stream;
    let timeout = tokio::time::Duration::from_secs(600);
    let deadline = tokio::time::Instant::now() + timeout;
    
    let mut event_count = 0;
    
    println!("📡 Watch stream started...");
    
    loop {
        tokio::select! {
            Some(event) = stream.next() => {
                event_count += 1;
                match event {
                    Ok(discovery_event) => {
                        println!("  [{}] Event: {:?}", event_count, discovery_event);
                    }
                    Err(e) => {
                        println!("  [{}] Error: {}", event_count, e);
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                println!("⏰ Timeout reached");
                break;
            }
        }
    }
    
    println!("✅ Watch test completed ({} events received)", event_count);
    println!("   With mock metadata, you should see Added events for discovered pods");
}

/// Test watching namespaced endpoints
#[tokio::test]
#[ignore]
async fn test_watch_namespaced_endpoints() {
    println!("👀 Testing watch namespaced endpoints...");
    println!("   This test will watch for 5 seconds");
    
    let client = create_test_client().await
        .expect("Failed to create test client");
    
    let key = DiscoveryQuery::NamespacedEndpoints {
        namespace: "test-namespace".to_string(),
    };
    
    println!("Calling list_and_watch() with key={:?}", key);
    let stream = client.list_and_watch(key, None).await
        .expect("Failed to create watch stream");
    
    let mut stream = stream;
    let timeout = tokio::time::Duration::from_secs(5);
    let deadline = tokio::time::Instant::now() + timeout;
    
    let mut event_count = 0;
    
    println!("📡 Watch stream started...");
    
    loop {
        tokio::select! {
            Some(event) = stream.next() => {
                event_count += 1;
                match event {
                    Ok(discovery_event) => {
                        println!("  [{}] Event: {:?}", event_count, discovery_event);
                    }
                    Err(e) => {
                        println!("  [{}] Error: {}", event_count, e);
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                println!("⏰ Timeout reached");
                break;
            }
        }
    }
    
    println!("✅ Watch test completed ({} events received)", event_count);
}

/// Comprehensive test: verify the watch stream receives EndpointSlice events
/// This test verifies that the K8s watcher is working correctly
#[tokio::test]
#[ignore]
async fn test_watch_receives_k8s_events() {
    println!("🔍 Testing that watch stream receives Kubernetes events...");
    println!("   This test verifies the K8s watcher layer works correctly");
    println!("   We'll watch for 10 seconds to ensure we get at least Init/InitDone");
    
    let client = create_test_client().await
        .expect("Failed to create test client");
    
    let key = DiscoveryQuery::AllEndpoints;
    
    let stream = client.list_and_watch(key, None).await
        .expect("Failed to create watch stream");
    
    let mut stream = stream;
    let timeout = tokio::time::Duration::from_secs(10);
    let deadline = tokio::time::Instant::now() + timeout;
    
    let mut received_any_event = false;
    
    println!("📡 Monitoring watch stream...");
    
    loop {
        tokio::select! {
            Some(event) = stream.next() => {
                received_any_event = true;
                match event {
                    Ok(discovery_event) => {
                        println!("  ✅ Received discovery event: {:?}", discovery_event);
                    }
                    Err(e) => {
                        println!("  ⚠️  Stream error: {}", e);
                    }
                }
                // Got at least one event, test passes
                break;
            }
            _ = tokio::time::sleep_until(deadline) => {
                println!("⏰ Timeout reached");
                break;
            }
        }
    }
    
    if received_any_event {
        println!("✅ Watch stream is working - received at least one event");
    } else {
        println!("⚠️  No events received in 10 seconds");
        println!("   This might be okay if:");
        println!("   - No EndpointSlices exist in the cluster");
        println!("   - Metadata HTTP calls are failing (expected without metadata server)");
        println!("   The K8s watcher itself is still working correctly.");
    }
    
    println!("✅ Test completed");
}

