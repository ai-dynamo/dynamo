// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Dynamo Logging
//!
//! `dynamo_logging` is a Rust library that provides structured logging functionality for Dynamo applications.
//!
//! It supports two logging formats: readable text and JSON, and can be configured through environment
//! variables or configuration files.

mod config;
mod logging;

pub use logging::{init, log_message};
// Re-export the configuration functions that may be useful externally
pub use config::{
    disable_ansi_logging, env_is_truthy, is_truthy, jsonl_logging_enabled, use_local_timezone,
};
