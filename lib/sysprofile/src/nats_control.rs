// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS control-plane message types for `dynamo.sysprofile.{start,stop,status}`.
//!
//! Components subscribe to these subjects when `DYN_SYSPROFILE_ENABLE=1`.
//! The CLI or operator publishes `StartRequest` / `StopRequest`; each
//! component replies with `StatusReply` on request/reply.

use serde::{Deserialize, Serialize};

pub const SUBJECT_START: &str = "dynamo.sysprofile.start";
pub const SUBJECT_STOP: &str = "dynamo.sysprofile.stop";
pub const SUBJECT_STATUS: &str = "dynamo.sysprofile.status";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartRequest {
    pub run_id: String,
    pub duration_s: u64,
    pub sampling: f64,
    pub backends: Vec<String>,
    pub output_dir: String,
    pub cupti: bool,
    pub nsys: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopRequest {
    pub run_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusReply {
    pub run_id: String,
    pub state: CaptureState,
    pub component: String,
    pub host: String,
    pub files_written: u32,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureState {
    Idle,
    Capturing,
    Flushing,
    Complete,
    Failed,
}
