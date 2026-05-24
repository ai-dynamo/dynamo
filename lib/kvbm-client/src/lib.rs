// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed gRPC client for `kvbm-service`.
//!
//! Connect to a running `kvbm-service` over a unix domain socket, call
//! `Register`, and receive an owned [`RegistrationHandle`] that streams
//! server events. Dropping the handle ungracefully detaches.
//!
//! # Example
//!
//! ```no_run
//! use kvbm_client::{KvbmServiceClient, proto};
//!
//! # async fn run() -> kvbm_client::error::ClientResult<()> {
//! let mut client = KvbmServiceClient::connect_uds("/run/kvbm.sock").await?;
//! let instance = proto::RegistrationInstance {
//!     kind: Some(proto::registration_instance::Kind::Kvbm(proto::KvbmInstance {
//!         model_name: "llm".into(),
//!         layout_mode: Some(proto::LayoutMode {
//!             kind: Some(proto::layout_mode::Kind::UniversalTp1Canonical(vec![0, 1])),
//!         }),
//!         tp_size: 2,
//!         block_size: 64,
//!         mode: proto::ServiceMode::Kvbm as i32,
//!     })),
//! };
//! let handle = client.register("worker-0", instance).await?;
//! println!("id={} slots={}", handle.registration_id(), handle.reserved_slots());
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod error;
pub mod handle;
pub mod proto;

pub use client::KvbmServiceClient;
pub use error::ClientError;
pub use handle::RegistrationHandle;
