use std::sync::Arc;

use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::{
    engine::AsyncEngineContext, pipeline::{ManyOut, Operator, ServerStreamingEngine, SingleIn}, protocols::annotated::Annotated
};

pub struct DecodeDisagger {


}

impl DecodeDisagger {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for DecodeDisagger
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>>,
    ) -> Result<ManyOut<Annotated<BackendOutput>>> {
        // Just forward requests for now
        let response_stream = next.generate(request).await?;
        Ok(response_stream)
    }
}

pub struct TransferManager {
    context: Arc<dyn AsyncEngineContext>,
    request: PreprocessedRequest,
    next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>>,
    next_stream: Option<ManyOut<Annotated<BackendOutput>>>,
    retries_left: u32,
    model_name: Arc<String>,
}
