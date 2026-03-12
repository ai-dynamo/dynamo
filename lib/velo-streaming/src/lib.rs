// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed exclusive-attachment streaming abstraction over the velo transport.
//!
//! Core wire types:
//! - [`handle::StreamAnchorHandle`]: compact u128 encoding WorkerId + local anchor ID
//! - [`frame::StreamFrame`]: six-variant enum representing all frame types on the wire
//!
//! Transport abstraction:
//! - [`transport::FrameTransport`]: pluggable ordered-delivery transport trait
//! - [`transport::FrameReader`]: frame receive half of the transport channel
//! - [`transport::FrameWriter`]: frame send half of the transport channel

pub mod frame;
pub mod handle;
pub mod transport;

pub use transport::{FrameReader, FrameTransport, FrameWriter};
