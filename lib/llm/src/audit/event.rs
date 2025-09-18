// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};
use std::sync::Arc;

#[derive(Clone)]
pub enum AuditEvent {
    Request {
        id: String,
        req: Arc<NvCreateChatCompletionRequest>,
    },
    Response {
        id: String,
        resp: Arc<NvCreateChatCompletionResponse>,
    },
}
