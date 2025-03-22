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

use dynamo_runtime::{worker::Worker, Result, Runtime, DistributedRuntime};
use tokio::time::sleep;
use std::time::Duration;

async fn hello_world(_runtime: Runtime) -> Result<()> {
    Ok(())
}

#[test]
fn test_lifecycle() {
    let worker = Worker::from_settings().unwrap();
    worker.execute(hello_world).unwrap();
}

#[cfg(feature = "integration")]
#[tokio::test]
async fn test_etcd_key_cleanup() -> Result<()> {
    use dynamo_runtime::transports::etcd::Client as EtcdClient;
    
    // Create runtime and distributed runtime
    let runtime = Runtime::from_settings()?;
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;
    
    // Create namespace and component
    let namespace = drt.namespace("test_cleanup")?;
    let component = namespace.component("test_component")?;
    
    // Create service and endpoint
    let service = component.service_builder().create().await?;
    let endpoint = component.endpoint("test_endpoint");
    
    // Set up a simple echo handler
    let handler = endpoint
        .endpoint_builder()
        .handler(|_ctx, _input: Vec<u8>| async move { Ok(vec![]) })
        .stats_handler(|_| serde_json::Value::Null);
    
    // Start the endpoint
    handler.start().await?;
    
    // Verify component key exists in etcd
    let component_path = component.etcd_path();
    let component_keys = drt.etcd_client().kv_get_prefix(&component_path).await?;
    assert!(!component_keys.is_empty(), "Component key should exist in etcd");
    
    // Now shutdown the runtime
    drt.shutdown();
    
    // Wait a bit for leases to be revoked
    sleep(Duration::from_secs(2)).await;
    
    // Create a new client to check if keys still exist
    let etcd_client = EtcdClient::new(Default::default(), Runtime::from_settings()?).await?;
    
    // Check for component key - should be gone
    let component_keys = etcd_client.kv_get_prefix(&component_path).await?;
    assert!(component_keys.is_empty(), "Component key should be cleaned up after shutdown");
    
    Ok(())
}

// async fn discoverable(runtime: Runtime) -> Result<()> {
//     let config = DiscoveryConfig {
//         etcd_url: vec!["http://localhost:2379".to_string()],
//         etcd_connect_options: None,
//     };

//     let client = DiscoveryClient::new(config, runtime.clone()).await?;
//     println!("Primary lease id: {:x}", client.lease_id());

//     let lease = client.create_lease(60).await?;

//     // Keys and values
//     let lock_key = "lock_key"; // Key for the lock
//     let object_key = "object_key"; // Key for the object
//     let object_value = "This is the object value"; // Value for the object
//     let lock_value = "locked"; // Value indicating a lock

//     let put_options = Some(PutOptions::new().with_lease(lease.id()));

//     // Build the transaction
//     let txn = Txn::new()
//         .when(vec![Compare::version(lock_key, CompareOp::Equal, 0)]) // Ensure the lock does not exist
//         .and_then(vec![
//             TxnOp::put(object_key, object_value, put_options.clone()), // Create the object
//             TxnOp::put(lock_key, lock_value, put_options),             // Set the lock
//         ]);

//     // Execute the transaction
//     let txn_response = client.etc_client().kv_client().txn(txn).await?;

//     tokio::spawn(async move {
//         println!("custom lease id: {:x}", lease.id());
//         lease.cancel_token().cancelled().await;
//         println!("custom lease revoked");
//     });

//     runtime.child_token().cancelled().await;

//     Ok(())
// }

// #[test]
// fn test_discovery_client() {
//     let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
//     runtime.execute(discoverable).unwrap();
// }
