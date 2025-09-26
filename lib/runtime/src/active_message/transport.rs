// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;

use super::client::Endpoint;
use super::handler::ActiveMessage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    Tcp,
    Ipc,
}

#[async_trait]
pub trait Transport: Send + Sync + std::fmt::Debug {
    async fn bind(&mut self, address: &str) -> Result<Endpoint>;

    async fn connect(&mut self, endpoint: &Endpoint) -> Result<()>;

    async fn disconnect(&mut self, endpoint: &Endpoint) -> Result<()>;

    async fn send(&mut self, message: &ActiveMessage) -> Result<()>;

    async fn receive(&mut self) -> Result<ActiveMessage>;

    fn transport_type(&self) -> TransportType;

    fn local_endpoint(&self) -> Option<&Endpoint>;
}
