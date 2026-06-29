// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level EPP router mode wrapper.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;

use crate::epp;
use crate::picker::Endpoint;
use crate::picker::EndpointPicker;
use crate::picker::PickError;
use crate::picker::PickResult;
use crate::picker::RequestInfo;
use crate::selection_router::SelectionRouter;
use crate::selection_router::SelectionRouterConfig;

pub enum Router {
    Runtime(epp::Router),
    Selection(SelectionRouter),
}

impl Router {
    pub async fn from_discovery(
        namespace: &str,
        component: &str,
        enforce_disagg: bool,
    ) -> Result<Self> {
        Ok(Self::Runtime(
            epp::Router::from_discovery(namespace, component, enforce_disagg).await?,
        ))
    }

    pub async fn from_selection_config(config: SelectionRouterConfig) -> Result<Self> {
        Ok(Self::Selection(SelectionRouter::new(config).await?))
    }

    pub fn pod_store_ready(&self) -> Arc<AtomicBool> {
        match self {
            Self::Runtime(router) => router.pod_store_ready(),
            Self::Selection(router) => router.pod_store_ready(),
        }
    }
}

#[tonic::async_trait]
impl EndpointPicker for Router {
    async fn pick(
        &self,
        req: &RequestInfo,
        endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        match self {
            Self::Runtime(router) => router.pick(req, endpoints).await,
            Self::Selection(router) => router.pick(req, endpoints).await,
        }
    }

    async fn on_prefill_complete(&self, request_id: &str) {
        match self {
            Self::Runtime(router) => router.on_prefill_complete(request_id).await,
            Self::Selection(router) => router.on_prefill_complete(request_id).await,
        }
    }

    async fn on_request_complete(&self, request_id: &str) {
        match self {
            Self::Runtime(router) => router.on_request_complete(request_id).await,
            Self::Selection(router) => router.on_request_complete(request_id).await,
        }
    }
}
