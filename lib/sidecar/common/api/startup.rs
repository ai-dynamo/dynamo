// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use dynamo_backend_common::DynamoError;

use crate::invalid_argument;

#[derive(Debug)]
pub enum SidecarStartupError {
    Cli(clap::Error),
    Dynamo(DynamoError),
}

impl SidecarStartupError {
    pub fn into_dynamo(self) -> DynamoError {
        match self {
            Self::Cli(error) => invalid_argument(error.to_string()),
            Self::Dynamo(error) => error,
        }
    }
}

impl fmt::Display for SidecarStartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(error) => error.fmt(formatter),
            Self::Dynamo(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SidecarStartupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cli(error) => Some(error),
            Self::Dynamo(error) => Some(error),
        }
    }
}

impl From<clap::Error> for SidecarStartupError {
    fn from(error: clap::Error) -> Self {
        Self::Cli(error)
    }
}

impl From<DynamoError> for SidecarStartupError {
    fn from(error: DynamoError) -> Self {
        Self::Dynamo(error)
    }
}
