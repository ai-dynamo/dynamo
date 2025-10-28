// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo

#![allow(dead_code)]
#![allow(unused_imports)]

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
};

pub use anyhow::{
    Context as ErrorContext, Error, Ok as OK, Result, anyhow as error, bail as raise,
};

use async_once_cell::OnceCell;

mod config;
pub use config::RuntimeConfig;

pub mod component;
pub mod compute;
pub mod engine;
pub mod health_check;
pub mod system_status_server;
pub use system_status_server::SystemStatusServerInfo;
pub mod instances;
pub mod logging;
pub mod metrics;
pub mod pipeline;
pub mod prelude;
pub mod protocols;
pub mod runnable;
pub mod runtime;
pub mod service;
pub mod slug;
pub mod storage;
pub mod system_health;
pub mod traits;
pub mod transports;
pub mod utils;
pub mod worker;

pub mod distributed;
pub use distributed::distributed_test_utils;
pub use futures::stream;
pub use metrics::MetricsRegistry;
pub use system_health::{HealthCheckTarget, SystemHealth};
pub use tokio_util::sync::CancellationToken;
pub use worker::Worker;

use crate::{
    metrics::prometheus_names::distributed_runtime,
    storage::key_value_store::{KeyValueStore, KeyValueStoreManager},
};

use component::{Endpoint, InstanceSource};
use utils::GracefulShutdownTracker;

use config::HealthStatus;

/// Types of Tokio runtimes that can be used to construct a Dynamo [Runtime].
#[derive(Clone)]
enum RuntimeType {
    Shared(Arc<tokio::runtime::Runtime>),
    External(tokio::runtime::Handle),
}

/// Local [Runtime] which provides access to shared resources local to the physical node/machine.
#[derive(Debug, Clone)]
pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,
    secondary: RuntimeType,
    cancellation_token: CancellationToken,
    endpoint_shutdown_token: CancellationToken,
    graceful_shutdown_tracker: Arc<GracefulShutdownTracker>,
    compute_pool: Option<Arc<compute::ComputePool>>,
    block_in_place_permits: Option<Arc<tokio::sync::Semaphore>>,
}

/// Distributed [Runtime] which provides access to shared resources across the cluster, this includes
/// communication protocols and transports.
#[derive(Clone)]
pub struct DistributedRuntime {
    // local runtime
    runtime: Runtime,

    // we might consider a unifed transport manager here
    etcd_client: Option<transports::etcd::Client>,
    nats_client: Option<transports::nats::Client>,
    store: KeyValueStoreManager,
    tcp_server: Arc<OnceCell<Arc<transports::tcp::server::TcpStreamServer>>>,
    system_status_server: Arc<OnceLock<Arc<system_status_server::SystemStatusServerInfo>>>,

    // local registry for components
    // the registry allows us to use share runtime resources across instances of the same component object.
    // take for example two instances of a client to the same remote component. The registry allows us to use
    // a single endpoint watcher for both clients, this keeps the number background tasking watching specific
    // paths in etcd to a minimum.
    component_registry: component::Registry,

    // Will only have static components that are not discoverable via etcd, they must be know at
    // startup. Will not start etcd.
    is_static: bool,

    instance_sources: Arc<tokio::sync::Mutex<HashMap<Endpoint, Weak<InstanceSource>>>>,

    // Health Status
    system_health: Arc<parking_lot::Mutex<SystemHealth>>,

    // This hierarchy's own metrics registry
    metrics_registry: MetricsRegistry,
}
