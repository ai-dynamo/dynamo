// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-worker cohort coordination infrastructure.
//!
//! This module provides the building blocks for coordinating a distributed
//! cohort of workers led by a single leader node. The architecture supports:
//!
//! - **Discovery**: Finding the leader node via pluggable discovery mechanisms
//! - **Cohort Formation**: Workers joining the leader's cohort with rank validation
//! - **Layout Coordination**: Leader-driven layout allocation across all workers
//! - **Transfer Coordination**: Executing transfers on specific ranks or broadcasting
//! - **Descriptor Collection**: Gathering layout metadata for remote access
//! - **Barriers**: Named synchronization points for coordinating phases
//!
//! # Architecture
//!
//! ```text
//!                         Leader
//!                    - create_cohort
//!                    - create_layouts
//!                    - execute_transfers
//!                    - collect_metadata
//!                           |
//!           +---------------+---------------+
//!           |               |               |
//!           v               v               v
//!         Worker          Worker          Worker
//!       - join           - join          - join
//!       - create_layout  - create_layout - create_layout
//!       - execute_xfer   - execute_xfer  - execute_xfer
//! ```
//!
//! # World Size Semantics
//!
//! World size N means N workers + 1 leader (not N total including leader).
//! For example, world_size=4 means 4 workers and 1 leader.
//!
//! # Typical Flow
//!
//! 1. **Leader starts**: Creates ActiveMessageServer and cohort
//! 2. **Workers discover**: Use discovery to find leader address
//! 3. **Workers join**: Connect and send `CreateCohortRequest` with rank/world_size
//! 4. **Leader validates**: Checks ranks form contiguous [0, N) sequence
//! 5. **Cohort complete**: Leader waits until all workers have joined
//! 6. **Layout broadcast**: Leader broadcasts `CreateLayoutRequest` to all workers
//! 7. **Workers create**: Allocate layouts locally (Device/Host/Disk)
//! 8. **Barrier sync**: Workers signal "kvbm.layouts.all-registered"
//! 9. **Metadata export**: Leader gathers serialized layouts from all workers
//! 10. **Transfer coordination**: Leader orchestrates block transfers across workers

pub mod discovery;
pub mod leader;
pub mod messages;
pub mod worker;

// Re-export commonly used types
pub use discovery::{LeaderDiscovery, StaticLeaderDiscovery};
pub use leader::Leader;
pub use messages::*;
pub use worker::Worker;
