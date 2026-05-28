// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::engine::AsyncEngineContextProvider;

use super::*;

macro_rules! impl_frontend {
    ($type:ident) => {
        impl<In: PipelineIO, Out: PipelineIO> $type<In, Out> {
            pub fn new() -> Arc<Self> {
                Arc::new(Self {
                    inner: Frontend::default(),
                })
            }
        }

        #[async_trait]
        impl<In: PipelineIO, Out: PipelineIO> Source<In> for $type<In, Out> {
            async fn on_next(&self, data: In, token: private::Token) -> Result<(), Error> {
                self.inner.on_next(data, token).await
            }

            fn set_edge(&self, edge: Edge<In>, token: private::Token) -> Result<(), PipelineError> {
                self.inner.set_edge(edge, token)
            }
        }

        #[async_trait]
        impl<In: PipelineIO, Out: PipelineIO + AsyncEngineContextProvider> Sink<Out>
            for $type<In, Out>
        {
            async fn on_data(&self, data: Out, token: private::Token) -> Result<(), Error> {
                self.inner.on_data(data, token).await
            }
        }

        #[async_trait]
        impl<In: PipelineIO + Sync, Out: PipelineIO> AsyncEngine<In, Out, Error>
            for $type<In, Out>
        {
            async fn generate(&self, request: In) -> Result<Out, Error> {
                self.inner.generate(request).await
            }
        }
    };
}

impl_frontend!(ServiceFrontend);
impl_frontend!(SegmentSource);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{ManyOut, PipelineErrorExt, SingleIn};

    #[tokio::test]
    async fn test_pipeline_source_no_edge() {
        let source = Frontend::<SingleIn<()>, ManyOut<()>>::default();
        let stream = source
            .generate(().into())
            .await
            .unwrap_err()
            .try_into_pipeline_error()
            .unwrap();

        match stream {
            PipelineError::NoEdge => (),
            _ => panic!("Expected NoEdge error"),
        }
    }
}
