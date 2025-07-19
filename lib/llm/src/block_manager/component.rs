// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::pool::BlockPoolError;

use super::*;

use dynamo_runtime::{
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
    traits::DistributedRuntimeProvider,
    utils::task::CriticalTaskExecutionHandle,
};

use futures::stream;

pub struct DynamoKvbmController<Locality: LocalityProvider, Metadata: BlockMetadata> {
    _locality: std::marker::PhantomData<Locality>,
    _metadata: std::marker::PhantomData<Metadata>,
}

impl<Locality: LocalityProvider, Metadata: BlockMetadata> DynamoKvbmController<Locality, Metadata> {
    pub async fn new(
        block_manager: KvBlockManager<Locality, Metadata>,
        component: dynamo_runtime::component::Component,
    ) -> anyhow::Result<Self> {
        let service = component.service_builder().create().await?;

        let reset_handler = ResetHandler::new(block_manager.clone());
        let reset_handler = Ingress::for_engine(reset_handler)?;

        let reset_task = CriticalTaskExecutionHandle::new(
            |_cancel_token| async move {
                service
                    .endpoint("reset_cache_level")
                    .endpoint_builder()
                    .handler(reset_handler)
                    .start()
                    .await
            },
            component.drt().primary_token(),
            "reset_cache_level",
        )?;

        reset_task.detach();

        Ok(Self {
            _locality: std::marker::PhantomData,
            _metadata: std::marker::PhantomData,
        })
    }
}

struct ResetHandler<Locality: LocalityProvider, Metadata: BlockMetadata> {
    block_manager: KvBlockManager<Locality, Metadata>,
}

impl<Locality: LocalityProvider, Metadata: BlockMetadata> ResetHandler<Locality, Metadata> {
    fn new(block_manager: KvBlockManager<Locality, Metadata>) -> Arc<Self> {
        Arc::new(Self { block_manager })
    }
}

#[async_trait]
impl<Locality: LocalityProvider, Metadata: BlockMetadata>
    AsyncEngine<SingleIn<CacheLevel>, ManyOut<Annotated<()>>, Error>
    for ResetHandler<Locality, Metadata>
{
    async fn generate(&self, input: SingleIn<CacheLevel>) -> Result<ManyOut<Annotated<()>>> {
        let (data, ctx) = input.into_parts();

        let result: anyhow::Result<()> = match data {
            CacheLevel::G1 => {
                self.block_manager
                    .device()
                    .ok_or(anyhow::anyhow!("Device pool not found"))?
                    .reset()
                    .await?;

                Ok(())
            }
            CacheLevel::G2 => Ok(self
                .block_manager
                .host()
                .ok_or(anyhow::anyhow!("Host pool not found"))?
                .reset_blocking()?),
            CacheLevel::G3 => Ok(self
                .block_manager
                .disk()
                .ok_or(anyhow::anyhow!("Disk pool not found"))?
                .reset_blocking()?),
            CacheLevel::G4 => Err(BlockPoolError::UnsupportedCacheLevel(data))?,
        };

        let annotated = match result {
            Ok(()) => Annotated::from_data(()),
            Err(e) => Annotated::from_error(e.to_string()),
        };

        let stream = stream::once(async move { annotated });

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

#[cfg(all(test, feature = "testing-etcd"))]
mod tests {
    use super::super::tests::create_reference_block_manager;
    use super::*;

    use dynamo_runtime::{pipeline::PushRouter, protocols::annotated::Annotated};
    use futures::StreamExt;

    #[tokio::test]
    async fn test_reset_cache_level() {
        dynamo_runtime::logging::init();

        let rt = dynamo_runtime::Runtime::from_current().unwrap();
        let drt = dynamo_runtime::DistributedRuntime::from_settings(rt)
            .await
            .unwrap();

        let worker_id = drt.primary_lease().unwrap().id();

        let block_manager = create_reference_block_manager().await;

        let component = drt
            .namespace("test-kvbm")
            .unwrap()
            .component("kvbm")
            .unwrap();

        let _controller =
            component::DynamoKvbmController::new(block_manager.clone(), component.clone())
                .await
                .unwrap();

        let client = component
            .endpoint("reset_cache_level")
            .client()
            .await
            .unwrap();

        client.wait_for_instances().await.unwrap();

        let client =
            PushRouter::<CacheLevel, Annotated<()>>::from_client(client, Default::default())
                .await
                .unwrap();

        let mut stream = client
            .direct(CacheLevel::G1.into(), worker_id)
            .await
            .unwrap();
        while let Some(resp) = stream.next().await {
            assert!(resp.is_ok());
        }

        let device_block = block_manager
            .device()
            .unwrap()
            .allocate_blocks(1)
            .await
            .unwrap();
        assert_eq!(device_block.len(), 1);

        let stream = client.direct(CacheLevel::G1.into(), worker_id).await;
        assert!(stream.is_err());
        // let resp = stream.next().await.unwrap();
        // assert!(resp.is_err());

        drop(device_block);

        let mut stream = client
            .direct(CacheLevel::G1.into(), worker_id)
            .await
            .unwrap();
        while let Some(resp) = stream.next().await {
            assert!(resp.is_ok());
        }
    }
}
