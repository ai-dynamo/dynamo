// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Message Common

pub(crate) mod events;
pub(crate) mod message_id;
pub(crate) mod messages;
pub(crate) mod responses;

pub(crate) use messages::{ActiveMessage, MessageMetadata};

// Re-export MessageId as public since it's part of the handler API
pub use message_id::MessageId;
