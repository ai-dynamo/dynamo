// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KvbmHub - Distributed KV cache coordination system.
//!
//! This module provides infrastructure for coordinating block locations across
//! a fleet of workers. It uses a sparse radix tree approach where blocks at
//! power-of-2 positions are tracked to enable efficient distributed lookup.
//!
//! # Architecture
//!
//! - **EventsManager**: Hooks into BlockRegistry to emit Create/Remove events
//! - **EventEmissionPolicy**: Filters which blocks trigger events (e.g., power-of-2 positions)
//! - **SparseRadixTree**: Maintains worker→block mappings at power-of-2 positions
//! - **KvbmHub**: Central server that processes events and answers queries
//!
//! # Example
//!
//! ```ignore
//! use kvbm::v2::hub::*;
//!
//! // Create policy and events manager
//! let policy = Arc::new(PowerOfTwoPolicy::new());
//! let (manager, event_rx) = EventsManager::new(policy, worker_id, cluster_id);
//!
//! // Create hub
//! let hub = KvbmHub::new(event_rx);
//! tokio::spawn(async move { hub.run().await });
//!
//! // Integrate with BlockRegistry
//! let registry = BlockRegistry::with_event_manager(Arc::new(manager));
//! ```

pub mod radix;
pub mod server;

use crate::v2::logical;
pub use logical::events::{EventReleaseHandle, InstanceId, KvCacheEvent};
pub use manager::EventsManager;
pub use policy::{EventEmissionPolicy, PowerOfTwoPolicy};
pub use radix::PositionalRadixTree;
pub use server::KvbmHub;
