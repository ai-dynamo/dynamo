// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common entry point for unified backends.
//!
//! Each backend's `main.rs` parses CLI args, constructs its `LLMEngine`, and
//! hands the pair off to [`run`]:
//!
//! ```ignore
//! use std::sync::Arc;
//!
//! fn main() -> anyhow::Result<()> {
//!     let (engine, config) = MyEngine::from_args(None)?;
//!     dynamo_backend_common::run(Arc::new(engine), config)
//! }
//! ```

use std::sync::Arc;

use dynamo_runtime::{Worker as DynamoWorker, logging};

use crate::engine::LLMEngine;
use crate::worker::{Worker, WorkerConfig};

/// Drive the full lifecycle for an already-constructed engine.
pub fn run(engine: Arc<dyn LLMEngine>, config: WorkerConfig) -> anyhow::Result<()> {
    logging::init();
    let worker = DynamoWorker::from_settings()?;
    worker.execute(|runtime| async move {
        Worker::new(engine, config)
            .run(runtime)
            .await
            .map_err(anyhow::Error::from)
    })
}
