// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend: Dynamo's OpenAI-compatible HTTP service serving one
//! engine, [`PoolForwardingEngine`], that forwards each request to a pool
//! frontend chosen by its pool selector.
//!
//! The Global Frontend does not tokenize. The pool frontend preprocesses a
//! forwarded request exactly as it would one sent to it directly.
//! [`ModelSync`] keeps the HTTP service's models in step with the pool
//! catalog, so a model is served while at least one pool serves it.

mod forward;
mod model_sync;

pub use forward::{ForwardedRequest, PoolForwardingEngine};
pub use model_sync::ModelSync;

mod catalog;
pub use catalog::{PoolCatalog, PoolSnapshot, StaticPoolCatalog};
pub use dynamo_global_common::Pool;
pub use dynamo_global_router::{PoolSelector, PoolTarget, RoundRobinSelector, RoutingRequest};
