// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Message Common

use bytes::Bytes;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::ResponseId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlMetadata {}

impl ControlMetadata {
    pub fn fire_and_forget() -> Self {
        unimplemented!()
    }

    pub fn expect_response(_response_id: Uuid) -> Self {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Builder)]
pub struct ActiveMessage {
    pub handler_name: String,
    pub payload: Bytes,
    #[builder(default = Uuid::new_v4())]
    pub message_id: Uuid,
    pub response_id: ResponseId,
    pub control: ControlMetadata,
}

impl ActiveMessage {
    pub fn builder() -> ActiveMessageBuilder {
        ActiveMessageBuilder::default()
    }
}
