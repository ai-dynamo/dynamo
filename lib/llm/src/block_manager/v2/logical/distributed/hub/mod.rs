// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry hub implementations.
//!
//! The hub is the server-side component that stores the registry and handles
//! queries from clients.

mod zmq;

pub use zmq::ZmqRegistryHub;

