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

pub mod config;
pub use config::RuntimeConfig;

pub mod component;
pub mod compute;
pub mod discovery;
pub mod engine;
pub mod engine_routes;
pub mod health_check;
pub mod local_endpoint_registry;
pub mod system_status_server;
pub use system_status_server::SystemStatusServerInfo;
pub mod distributed;
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

pub use distributed::{DistributedRuntime, distributed_test_utils};
pub use futures::stream;
pub use metrics::MetricsRegistry;
pub use runtime::Runtime;
pub use system_health::{HealthCheckTarget, SystemHealth};
pub use tokio_util::sync::CancellationToken;
pub use worker::Worker;

use component::Endpoint;
use utils::GracefulShutdownTracker;

use config::HealthStatus;
