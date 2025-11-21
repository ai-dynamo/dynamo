// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for discovery system backends.

#![cfg(test)]

use anyhow::{Context, Result, anyhow};
use std::future::Future;
use std::sync::Arc;
use uuid::Uuid;

use crate::peer::{InstanceId, WorkerAddress};

use super::{DiscoverySystem, peer_discovery_handle};

fn make_test_address() -> WorkerAddress {
    WorkerAddress::from_bytes(b"127.0.0.1:8080".as_slice())
}

fn unique_cluster_id(suffix: &str) -> String {
    format!("test-{suffix}-{}", Uuid::new_v4())
}

pub(crate) async fn register_and_discover_by_worker_id<F, Fut>(mut factory: F) -> Result<()>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<Arc<dyn DiscoverySystem>>>,
{
    let cluster_id = unique_cluster_id("worker-id");
    let system = factory(cluster_id).await?;
    let peer_discovery = peer_discovery_handle(Arc::clone(&system))
        .ok_or_else(|| anyhow!("Peer discovery should be available"))?;

    let instance_id = InstanceId::new_v4();
    let address = make_test_address();
    let worker_id = instance_id.worker_id();

    peer_discovery
        .register_instance(instance_id, address.clone())
        .await
        .context("Failed to register instance")?;

    let found = peer_discovery
        .discover_by_worker_id(worker_id)
        .await
        .context("Failed to discover by worker_id")?;
    assert_eq!(found.instance_id(), instance_id);
    assert_eq!(&found.worker_address, &address);

    peer_discovery
        .unregister_instance(instance_id)
        .await
        .context("Failed to unregister instance")?;

    system.shutdown();
    Ok(())
}

pub(crate) async fn register_and_discover_by_instance_id<F, Fut>(mut factory: F) -> Result<()>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<Arc<dyn DiscoverySystem>>>,
{
    let cluster_id = unique_cluster_id("instance-id");
    let system = factory(cluster_id).await?;
    let peer_discovery = peer_discovery_handle(Arc::clone(&system))
        .ok_or_else(|| anyhow!("Peer discovery should be available"))?;

    let instance_id = InstanceId::new_v4();
    let address = make_test_address();

    peer_discovery
        .register_instance(instance_id, address.clone())
        .await
        .context("Failed to register instance")?;

    let found = peer_discovery
        .discover_by_instance_id(instance_id)
        .await
        .context("Failed to discover by instance_id")?;
    assert_eq!(found.instance_id(), instance_id);
    assert_eq!(&found.worker_address, &address);

    peer_discovery
        .unregister_instance(instance_id)
        .await
        .context("Failed to unregister instance")?;

    system.shutdown();
    Ok(())
}

pub(crate) async fn collision_detection<F, Fut>(mut factory: F) -> Result<()>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<Arc<dyn DiscoverySystem>>>,
{
    let cluster_id = unique_cluster_id("collision");
    let system = factory(cluster_id).await?;
    let peer_discovery = peer_discovery_handle(Arc::clone(&system))
        .ok_or_else(|| anyhow!("Peer discovery should be available"))?;

    let instance_id = InstanceId::new_v4();
    let address = make_test_address();

    peer_discovery
        .register_instance(instance_id, address.clone())
        .await
        .context("Failed to register instance")?;

    let result = peer_discovery
        .register_instance(instance_id, address.clone())
        .await;
    assert!(
        result.is_err(),
        "Re-registration should be rejected to prevent collisions"
    );

    peer_discovery
        .unregister_instance(instance_id)
        .await
        .context("Failed to unregister instance")?;

    system.shutdown();
    Ok(())
}

pub(crate) async fn checksum_validation<F, Fut>(mut factory: F) -> Result<()>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<Arc<dyn DiscoverySystem>>>,
{
    let cluster_id = unique_cluster_id("checksum");
    let system = factory(cluster_id).await?;
    let peer_discovery = peer_discovery_handle(Arc::clone(&system))
        .ok_or_else(|| anyhow!("Peer discovery should be available"))?;

    let instance_id = InstanceId::new_v4();
    let address1 = WorkerAddress::from_bytes(&b"tcp://127.0.0.1:5555"[..]);
    let address2 = WorkerAddress::from_bytes(&b"tcp://127.0.0.1:6666"[..]);

    peer_discovery
        .register_instance(instance_id, address1)
        .await
        .context("Failed to register instance")?;

    let result = peer_discovery
        .register_instance(instance_id, address2)
        .await;
    assert!(result.is_err(), "Checksum mismatch should be rejected");

    peer_discovery
        .unregister_instance(instance_id)
        .await
        .context("Failed to unregister instance")?;

    system.shutdown();
    Ok(())
}

pub(crate) async fn not_found_errors<F, Fut>(mut factory: F) -> Result<()>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<Arc<dyn DiscoverySystem>>>,
{
    let cluster_id = unique_cluster_id("not-found");
    let system = factory(cluster_id).await?;
    let peer_discovery = peer_discovery_handle(Arc::clone(&system))
        .ok_or_else(|| anyhow!("Peer discovery should be available"))?;

    let fake_worker_id = InstanceId::new_v4().worker_id();
    let fake_instance_id = InstanceId::new_v4();

    let worker_result = peer_discovery.discover_by_worker_id(fake_worker_id).await;
    assert!(
        worker_result.is_err(),
        "Discover by worker_id should return an error for missing entry"
    );

    let instance_result = peer_discovery
        .discover_by_instance_id(fake_instance_id)
        .await;
    assert!(
        instance_result.is_err(),
        "Discover by instance_id should return an error for missing entry"
    );

    system.shutdown();
    Ok(())
}
