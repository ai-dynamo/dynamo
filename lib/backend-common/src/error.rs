// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend error taxonomy with typed variants for predictable downstream
//! HTTP mapping:
//!
//! | Rust variant      | Intended HTTP |
//! |-------------------|---------------|
//! | `CannotConnect`   | 503           |
//! | `EngineInit`      | 503           |
//! | `EngineShutdown`  | 503           |
//! | `Engine`          | 500           |
//! | `InvalidArgument` | 400           |
//! | `Other`           | 500           |

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("engine initialization failed: {0}")]
    EngineInit(String),

    #[error("engine shutdown: {0}")]
    EngineShutdown(String),

    #[error("cannot connect to runtime transport: {0}")]
    CannotConnect(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("engine error: {0}")]
    Engine(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl BackendError {
    pub fn engine<S: Into<String>>(msg: S) -> Self {
        BackendError::Engine(msg.into())
    }

    pub fn invalid<S: Into<String>>(msg: S) -> Self {
        BackendError::InvalidArgument(msg.into())
    }
}
