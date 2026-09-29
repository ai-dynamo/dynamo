// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Constructors shared by the standalone and Python launchers.

use dynamo_backend_common::{DynamoError, WorkerConfig};
use dynamo_sidecar_common::SidecarStartupError;

use crate::args::Args;
use crate::engine::TrtllmSidecarEngine;

impl TrtllmSidecarEngine {
    pub fn from_env() -> Result<(Self, WorkerConfig), DynamoError> {
        Self::from_parsed(<Args as clap::Parser>::parse())
    }

    pub fn from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        Self::try_from_args(argv).map_err(SidecarStartupError::into_dynamo)
    }

    /// Parse injected arguments while retaining Clap's structured exit error.
    ///
    /// Embedded callers use this to distinguish help and version output from
    /// Dynamo startup failures without changing `from_args`'s error contract.
    pub fn try_from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), SidecarStartupError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)?;
        Self::from_parsed(args).map_err(Into::into)
    }
}
