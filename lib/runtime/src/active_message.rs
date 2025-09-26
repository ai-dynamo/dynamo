// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../docs/active_message.md")]

pub mod builder;
pub mod client;
pub mod handler;
pub mod manager;
pub mod response;
pub mod status;
pub mod transport;

pub use builder::MessageBuilder;
pub use client::ActiveMessageClient;
pub use handler::{ActiveMessage, ActiveMessageHandler};
pub use manager::ActiveMessageManager;
pub use response::{ResponseContext, SingleResponseSender};
pub use status::{MessageStatus, SendAndConfirm, DetachedConfirm, WithResponse};
pub use transport::Transport;

pub mod zmq;
