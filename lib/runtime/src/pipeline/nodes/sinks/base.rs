// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use anyhow::Error;

impl<Resp: PipelineIO> Default for SinkEdge<Resp> {
    fn default() -> Self {
        Self {
            edge: OnceLock::new(),
        }
    }
}

#[async_trait]
impl<Resp: PipelineIO> Source<Resp> for SinkEdge<Resp> {
    async fn on_next(&self, data: Resp, _: Token) -> Result<(), Error> {
        self.edge
            .get()
            .ok_or(PipelineError::NoEdge)?
            .write(data)
            .await
    }

    fn set_edge(&self, edge: Edge<Resp>, _: Token) -> Result<(), PipelineError> {
        self.edge
            .set(edge)
            .map_err(|_| PipelineError::EdgeAlreadySet)?;
        Ok(())
    }
}
