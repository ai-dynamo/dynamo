// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::Error;

impl<Req: PipelineIO, Resp: PipelineIO> ServiceBackend<Req, Resp> {
    pub fn from_engine(engine: ServiceEngine<Req, Resp>) -> Arc<Self> {
        Arc::new(Self {
            engine,
            inner: SinkEdge::default(),
        })
    }
}

#[async_trait]
impl<Req: PipelineIO + Sync, Resp: PipelineIO> Sink<Req> for ServiceBackend<Req, Resp> {
    async fn on_data(&self, data: Req, _: Token) -> Result<(), Error> {
        let stream = self.engine.generate(data).await?;
        self.on_next(stream, Token).await
    }
}

#[async_trait]
impl<Req: PipelineIO, Resp: PipelineIO> Source<Resp> for ServiceBackend<Req, Resp> {
    async fn on_next(&self, data: Resp, _: Token) -> Result<(), Error> {
        self.inner.on_next(data, Token).await
    }

    fn set_edge(&self, edge: Edge<Resp>, _: Token) -> Result<(), PipelineError> {
        self.inner.set_edge(edge, Token)
    }
}
