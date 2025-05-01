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

use super::*;
use crate::llm::model_card::ModelDeploymentCard;

use llm_rs::{
    preprocessor::OpenAIPreprocessor,
    protocols::common::llm_backend::{BackendInput, BackendOutput},
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        Annotated,
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
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let preprocessor = runtime
            .block_on(OpenAIPreprocessor::new(mdc.inner.clone()))
            .map_err(to_pyerr)?;
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
            SegmentSink::<SingleIn<BackendInput>, ManyOut<Annotated<BackendOutput>>>::new();

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
            let router = PushRouter::<BackendInput, Annotated<BackendOutput>>::from_client(
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
