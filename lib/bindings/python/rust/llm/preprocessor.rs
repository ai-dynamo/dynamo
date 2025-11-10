// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::llm::model_card::ModelDeploymentCard;
use std::time::Duration;

use llm_rs::{
    preprocessor::OpenAIPreprocessor,
    preprocessor::media::{MediaDecoder as RsMediaDecoder, MediaFetcher as RsMediaFetcher},
    protocols::common::llm_backend::{BackendOutput, PreprocessedRequest},
    types::{
        Annotated,
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
    },
};

use dynamo_runtime::pipeline::{
    ManyOut, Operator, PushRouter, SegmentSink, ServiceFrontend, SingleIn, Source,
};

#[pyclass]
pub(crate) struct OAIChatPreprocessor {
    inner: Arc<llm_rs::preprocessor::OpenAIPreprocessor>,
    current: Endpoint,
    next: Endpoint,
}

#[pymethods]
impl OAIChatPreprocessor {
    #[new]
    fn new(mdc: ModelDeploymentCard, current: Endpoint, next: Endpoint) -> PyResult<Self> {
        let preprocessor = OpenAIPreprocessor::new(mdc.inner.clone()).map_err(to_pyerr)?;
        Ok(Self {
            inner: preprocessor,
            current,
            next,
        })
    }

    fn start<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let frontend = ServiceFrontend::<
            SingleIn<NvCreateChatCompletionRequest>,
            ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        >::new();

        let network =
            SegmentSink::<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>>::new();

        let preprocessor = self.inner.into_operator();
        let pipeline = frontend
            .link(preprocessor.forward_edge())
            .map_err(to_pyerr)?
            .link(network.clone())
            .map_err(to_pyerr)?
            .link(preprocessor.backward_edge())
            .map_err(to_pyerr)?
            .link(frontend)
            .map_err(to_pyerr)?;
        let ingress = Ingress::for_engine(pipeline).map_err(to_pyerr)?;
        let builder = self.current.inner.endpoint_builder().handler(ingress);
        let endpoint = Arc::new(self.next.inner.clone());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = endpoint.client().await.map_err(to_pyerr)?;
            let router = PushRouter::<PreprocessedRequest, Annotated<BackendOutput>>::from_client(
                client,
                Default::default(),
            )
            .await
            .map_err(to_pyerr)?;
            network.attach(Arc::new(router)).map_err(to_pyerr)?;
            builder.start().await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MediaDecoder {
    pub(crate) inner: RsMediaDecoder,
}

#[pymethods]
impl MediaDecoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsMediaDecoder::default(),
        }
    }

    fn image_decoder(&mut self, image_decoder: &Bound<'_, PyDict>) -> PyResult<()> {
        let image_decoder = pythonize::depythonize(image_decoder).map_err(|err| {
            PyErr::new::<PyException, _>(format!("Failed to parse image_decoder: {}", err))
        })?;
        self.inner.image_decoder = image_decoder;
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MediaFetcher {
    pub(crate) inner: RsMediaFetcher,
}

#[pymethods]
impl MediaFetcher {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsMediaFetcher::default(),
        }
    }
    fn user_agent(&mut self, user_agent: String) {
        self.inner.user_agent = user_agent;
    }

    fn allow_direct_ip(&mut self, allow: bool) {
        self.inner.allow_direct_ip = allow;
    }

    fn allow_direct_port(&mut self, allow: bool) {
        self.inner.allow_direct_port = allow;
    }

    fn allowed_media_domains(&mut self, domains: Vec<String>) {
        self.inner.allowed_media_domains = Some(domains.into_iter().collect());
    }

    fn timeout_ms(&mut self, timeout_ms: u64) {
        self.inner.timeout = Some(Duration::from_millis(timeout_ms));
    }
}
