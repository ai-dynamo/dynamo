// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::block_manager::pool::{AsyncBlockPoolController, BlockPoolStatus};
use futures::stream;
use serde_json::Value;

impl<Locality: LocalityProvider, Metadata: BlockMetadata> ControllerHandler<Locality, Metadata> {
    pub fn new(block_manager: KvBlockManager<Locality, Metadata>) -> Arc<Self> {
        Arc::new(Self { block_manager })
    }

    fn get_pool_controller(
        &self,
        cache_level: &CacheLevel,
    ) -> Result<&dyn AsyncBlockPoolController> {
        match cache_level {
            CacheLevel::G1 => Ok(self
                .block_manager
                .device()
                .ok_or_else(|| anyhow::anyhow!("Device pool not found"))?),
            CacheLevel::G2 => Ok(self
                .block_manager
                .host()
                .ok_or_else(|| anyhow::anyhow!("Host pool not found"))?),
            CacheLevel::G3 => Ok(self
                .block_manager
                .disk()
                .ok_or_else(|| anyhow::anyhow!("Disk pool not found"))?),
        }
    }

    async fn reset_pool(&self, cache_level: &CacheLevel) -> Result<()> {
        Ok(self.get_pool_controller(cache_level)?.reset().await?)
    }

    async fn handle_status(&self, cache_level: &CacheLevel) -> Result<BlockPoolStatus> {
        let pool_controller = self.get_pool_controller(cache_level)?;
        Ok(pool_controller.status().await?)
    }

    async fn handle_pool_reset(&self, cache_level: &CacheLevel) -> Result<()> {
        self.reset_pool(cache_level).await
    }

    async fn handle_blocks_reset(
        &self,
        cache_level: &CacheLevel,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<ResetBlocksResponse> {
        let pool_controller = self.get_pool_controller(cache_level)?;
        Ok(pool_controller.reset_blocks(&sequence_hashes).await?)
    }

    async fn handle_reset_all(&self) -> Result<()> {
        for cache_level in &[CacheLevel::G1, CacheLevel::G2, CacheLevel::G3] {
            if let Ok(pool_controller) = self.get_pool_controller(cache_level) {
                pool_controller.reset().await?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl<Locality: LocalityProvider, Metadata: BlockMetadata>
    AsyncEngine<HandlerInput, HandlerOutput, Error> for ControllerHandler<Locality, Metadata>
{
    async fn generate(&self, input: HandlerInput) -> Result<HandlerOutput> {
        let (data, ctx) = input.into_parts();

        let annotated = match data {
            ControlMessage::Status(cache_level) => {
                // handle status
                make_response(self.handle_status(&cache_level).await)
            }

            ControlMessage::ResetPool(cache_level) => {
                // handle reset
                make_unit_response(self.handle_pool_reset(&cache_level).await)
            }

            ControlMessage::ResetBlocks(request) => {
                // handle reset blocks
                make_response(
                    self.handle_blocks_reset(&request.cache_level, request.sequence_hashes)
                        .await,
                )
            }

            ControlMessage::ResetAll => {
                // hadnle reset all
                make_unit_response(self.handle_reset_all().await)
            }
        };

        let stream = stream::once(async move { annotated });
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

fn make_unit_response(response: Result<()>) -> Annotated<Value> {
    match response {
        Ok(()) => Annotated::from_data(serde_json::Value::Null),
        Err(e) => Annotated::from_error(e.to_string()),
    }
}

fn make_response<T: Serialize>(response: Result<T>) -> Annotated<Value> {
    match response {
        Ok(response) => match serde_json::to_value(response) {
            Ok(values) => Annotated::from_data(values),
            Err(e) => Annotated::from_error(e.to_string()),
        },
        Err(e) => Annotated::from_error(e.to_string()),
    }
}
