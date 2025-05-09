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

use super::metrics;
use super::ModelManager;
use super::RouteDoc;
use crate::request_template::RequestTemplate;
use anyhow::Result;
use derive_builder::Builder;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub struct HttpService {
    models: ModelManager,
    router: axum::Router,
    port: u16,
    host: String,
    route_docs: Vec<RouteDoc>,
}

#[derive(Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct HttpServiceConfig {
    #[builder(default = "8787")]
    port: u16,

    #[builder(setter(into), default = "String::from(\"0.0.0.0\")")]
    host: String,

    // #[builder(default)]
    // custom: Vec<axum::Router>
    #[builder(default = "true")]
    enable_chat_endpoints: bool,

    #[builder(default = "true")]
    enable_cmpl_endpoints: bool,

    #[builder(default = "None")]
    request_template: Option<RequestTemplate>,
}

impl HttpService {
    pub fn builder() -> HttpServiceConfigBuilder {
        HttpServiceConfigBuilder::default()
    }

    pub fn model_manager(&self) -> &ModelManager {
        &self.models
    }

    pub async fn spawn(&self, cancel_token: CancellationToken) -> JoinHandle<Result<()>> {
        let this = self.clone();
        tokio::spawn(async move { this.run(cancel_token).await })
    }

    pub async fn run(&self, cancel_token: CancellationToken) -> Result<()> {
        let address = format!("{}:{}", self.host, self.port);
        tracing::info!(address, "Starting HTTP service on: {address}");

        let listener = tokio::net::TcpListener::bind(address.as_str())
            .await
            .unwrap_or_else(|_| panic!("could not bind to address: {address}"));

        let router = self.router.clone();
        let observer = cancel_token.child_token();

        axum::serve(listener, router)
            .with_graceful_shutdown(observer.cancelled_owned())
            .await
            .inspect_err(|_| cancel_token.cancel())?;

        Ok(())
    }

    /// Documentation of exposed HTTP endpoints
    pub fn route_docs(&self) -> &[RouteDoc] {
        &self.route_docs
    }
}

impl HttpServiceConfigBuilder {
    pub fn build(self) -> Result<HttpService, anyhow::Error> {
        let config = self.build_internal()?;

        let model_manager = ModelManager::new();

        // enable prometheus metrics
        let registry = metrics::Registry::new();
        model_manager.metrics().register(&registry)?;

        let mut router = axum::Router::new();

        let mut all_docs = Vec::new();

        let mut routes = vec![
            metrics::router(registry, None),
            super::openai::list_models_router(model_manager.state(), None),
        ];

        if config.enable_chat_endpoints {
            routes.push(super::openai::chat_completions_router(
                model_manager.state(),
                config.request_template,
                None,
            ));
        }

        if config.enable_cmpl_endpoints {
            routes.push(super::openai::completions_router(
                model_manager.state(),
                None,
            ));
        }

        // for (route_docs, route) in routes.into_iter().chain(self.routes.into_iter()) {
        //     router = router.merge(route);
        //     all_docs.extend(route_docs);
        // }

        for (route_docs, route) in routes.into_iter() {
            router = router.merge(route);
            all_docs.extend(route_docs);
        }

        Ok(HttpService {
            models: model_manager,
            router,
            port: config.port,
            host: config.host,
            route_docs: all_docs,
        })
    }

    pub fn with_request_template(mut self, request_template: Option<RequestTemplate>) -> Self {
        self.request_template = Some(request_template);
        self
    }
}
