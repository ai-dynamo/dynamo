// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{Result, Runtime, worker::Worker};

async fn hello_world(_runtime: Runtime) -> Result<()> {
    Ok(())
}

#[test]
fn test_lifecycle() {
    let worker = Worker::from_settings().unwrap();
    worker.execute(hello_world).unwrap();
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
