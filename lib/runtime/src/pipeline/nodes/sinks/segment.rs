// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use anyhow::Error;

impl<Req: PipelineIO, Resp: PipelineIO> SegmentSink<Req, Resp> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn attach(&self, engine: ServiceEngine<Req, Resp>) -> Result<(), PipelineError> {
        self.engine
            .set(engine)
            .map_err(|_| PipelineError::EdgeAlreadySet)
    }
}

impl<Req: PipelineIO, Resp: PipelineIO> Default for SegmentSink<Req, Resp> {
    fn default() -> Self {
        Self {
            engine: OnceLock::new(),
            inner: SinkEdge::default(),
        }
    }
}

#[async_trait]
impl<Req: PipelineIO + Sync, Resp: PipelineIO> Sink<Req> for SegmentSink<Req, Resp> {
    async fn on_data(&self, data: Req, _: Token) -> Result<(), Error> {
        let stream = self
            .engine
            .get()
            .ok_or(PipelineError::NoNetworkEdge)?
            .generate(data)
            .await?;
        self.on_next(stream, Token).await
    }
}

#[async_trait]
impl<Req: PipelineIO, Resp: PipelineIO> Source<Resp> for SegmentSink<Req, Resp> {
    async fn on_next(&self, data: Resp, _: Token) -> Result<(), Error> {
        self.inner.on_next(data, Token).await
    }

    fn set_edge(&self, edge: Edge<Resp>, _: Token) -> Result<(), PipelineError> {
        self.inner.set_edge(edge, Token)
    }
}
