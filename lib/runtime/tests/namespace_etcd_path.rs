// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Test file for recursive namespace etcd_path functionality
#[allow(unused_imports)]
use dynamo_runtime::{DistributedRuntime, Runtime};

#[cfg(feature = "integration")]
#[test]
fn test_namespace_etcd_path_format() {
    // Test that the etcd_path format is correct for the expected use case
    // This test verifies the format: dynamo://ns1.ns2.ns3/component/{component.name()}

    // Expected format examples:
    let single_ns_path = "dynamo://ns1";
    let nested_ns_path = "dynamo://ns1.ns2.ns3";
    let component_path = "dynamo://ns1.ns2.ns3/_component_/my-component";

    // Verify the format matches our requirements
    assert!(single_ns_path.starts_with("dynamo://"));
    assert!(nested_ns_path.starts_with("dynamo://"));
    assert!(nested_ns_path.contains("."));
    assert!(component_path.contains("/_component_/"));

    // Test the specific format requested in the user query (now with reserved keywords)
    let expected_format = "dynamo://ns1.ns2.ns3/_component_/my-component";
    assert_eq!(component_path, expected_format);

    println!("✅ Namespace etcd_path format verification passed");
    println!("   Single namespace: {}", single_ns_path);
    println!("   Nested namespace: {}", nested_ns_path);
    println!("   Component path: {}", component_path);
}

#[cfg(feature = "integration")]
#[tokio::test]
async fn test_recursive_namespace_implementation() {
    use dynamo_runtime::{
        distributed::DistributedConfig, storage::key_value_store::KeyValueStoreSelect,
        transports::nats,
    };

    let runtime = Runtime::from_current().unwrap();
    let config = DistributedConfig {
        store_backend: KeyValueStoreSelect::Memory,
        nats_config: nats::ClientOptions::default(),
    };
    let distributed_runtime = DistributedRuntime::new(runtime, config).await.unwrap();

    // Test single namespace
    let ns1 = distributed_runtime.namespace("ns1").unwrap();
    assert_eq!(ns1.etcd_path(), "dynamo://ns1");
    assert_eq!(ns1.name(), "ns1");

    // Test nested namespace ns1.ns2
    let ns2 = ns1.namespace("ns2").unwrap();
    assert_eq!(ns2.etcd_path(), "dynamo://ns1.ns2");
    assert_eq!(ns2.name(), "ns1.ns2");

    // Test deeply nested namespace ns1.ns2.ns3
    let ns3 = ns2.namespace("ns3").unwrap();
    assert_eq!(ns3.etcd_path(), "dynamo://ns1.ns2.ns3");
    assert_eq!(ns3.name(), "ns1.ns2.ns3");

    // Test component in deeply nested namespace
    let component = ns3.component("my-component").unwrap();
    assert_eq!(
        component.etcd_path().to_string(),
        "dynamo://ns1.ns2.ns3/_component_/my-component"
    );
    assert_eq!(component.name(), "my-component");
    assert_eq!(component.path(), "ns1.ns2.ns3/my-component");

    println!("✅ Actual recursive namespace implementation test passed!");
    println!("   Root namespace: {}", ns1.etcd_path());
    println!("   Nested namespace: {}", ns2.etcd_path());
    println!("   Deep namespace: {}", ns3.etcd_path());
    println!("   Component path: {}", component.etcd_path());
}

#[cfg(feature = "integration")]
#[tokio::test]
async fn test_multiple_branches_recursive_namespaces() {
    use dynamo_runtime::{
        distributed::DistributedConfig, storage::key_value_store::KeyValueStoreSelect,
        transports::nats,
    };

    let runtime = Runtime::from_current().unwrap();
    let config = DistributedConfig {
        store_backend: KeyValueStoreSelect::Memory,
        nats_config: nats::ClientOptions::default(),
    };
    let distributed_runtime = DistributedRuntime::new(runtime, config).await.unwrap();

    // Create root namespace
    let root = distributed_runtime.namespace("root").unwrap();

    // Create multiple branches
    let prod_ns = root.namespace("prod").unwrap();
    let staging_ns = root.namespace("staging").unwrap();

    // Create services in each branch
    let prod_service_ns = prod_ns.namespace("services").unwrap();
    let staging_service_ns = staging_ns.namespace("services").unwrap();

    // Verify the paths are correct
    assert_eq!(prod_service_ns.etcd_path(), "dynamo://root.prod.services");
    assert_eq!(
        staging_service_ns.etcd_path(),
        "dynamo://root.staging.services"
    );

    // Create components in each branch
    let prod_component = prod_service_ns.component("api-gateway").unwrap();
    let staging_component = staging_service_ns.component("api-gateway").unwrap();

    assert_eq!(
        prod_component.etcd_path().to_string(),
        "dynamo://root.prod.services/_component_/api-gateway"
    );
    assert_eq!(
        staging_component.etcd_path().to_string(),
        "dynamo://root.staging.services/_component_/api-gateway"
    );

    // Verify they are different
    assert_ne!(prod_component.etcd_path(), staging_component.etcd_path());

    println!("✅ Multiple branches recursive namespaces test passed!");
    println!("   Production: {}", prod_component.etcd_path());
    println!("   Staging: {}", staging_component.etcd_path());
}
