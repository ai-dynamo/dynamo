// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// This module provides a high-performance interface that bridges Python
/// applications with the Rust-powered Dynamo LLM runtime.
///
/// It is organized into several specialized sub-modules, each responsible for a particular aspect of the system:
///
/// - `disagg_route`:
///   Implements distributed routing of inference requests with dynamic
///   load balancing and efficient resource allocation across clusters.
/// - `kv`:
///   Implements a high-performance key-value caching system that stores
///   intermediate computations and maintains model state for rapid data access.
/// - `model_card`:
///   Manages model deployment cards containing detailed metadata, configuration
///   settings, and versioning information to ensure consistent deployments.
/// - `preprocessor`:
///   Provides utilities for transforming raw LLM requests—including tokenization,
///   prompt formatting, and validation—into a format required by the Dynamo runtime.
///
/// Each sub-module is designed to encapsulate its functionality for clean
/// integration between Python tools and the Dynamo runtime.
use super::*;

pub mod entrypoint;
pub mod kv;
pub mod local_model;
pub mod lora;
pub mod model_card;
pub mod preprocessor;
