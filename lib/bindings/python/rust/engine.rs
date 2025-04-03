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

use std::sync::Arc;

use dynamo_engine_python::PythonServerStreamingEngine;
use dynamo_runtime::CancellationToken;
pub use dynamo_runtime::{
    pipeline::{async_trait, AsyncEngine, Data, ManyOut, SingleIn},
    protocols::annotated::Annotated,
    Error, Result,
};
pub use serde::{Deserialize, Serialize};

use pyo3::prelude::*;

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonAsyncEngine>()?;
    Ok(())
}

// todos:
// - [ ] enable context cancellation
//   - this will likely require a change to the function signature python calling arguments
// - [ ] other `AsyncEngine` implementations will have a similar pattern, i.e. one AsyncEngine
//       implementation per struct

/// Rust/Python bridge that maps to the [`AsyncEngine`] trait
///
/// Currently this is only implemented for the [`SingleIn`] and [`ManyOut`] types; however,
/// more [`AsyncEngine`] implementations can be added in the future.
///
/// For the [`SingleIn`] and [`ManyOut`] case, this implementation will take a Python async
/// generator and convert it to a Rust async stream.
///
/// ```python
/// class ComputeEngine:
///     def __init__(self):
///         self.compute_engine = make_compute_engine()
///
///     def generate(self, request):
///         async generator():
///            async for output in self.compute_engine.generate(request):
///                yield output
///         return generator()
///
/// def main():
///     loop = asyncio.create_event_loop()
///     compute_engine = ComputeEngine()
///     engine = PythonAsyncEngine(compute_engine.generate, loop)
///     service = RustService()
///     service.add_engine("model_name", engine)
///     loop.run_until_complete(service.run())
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PythonAsyncEngine(PythonServerStreamingEngine);

#[pymethods]
impl PythonAsyncEngine {
    /// Create a new instance of the PythonAsyncEngine
    ///
    /// # Arguments
    /// - `generator`: a Python async generator that will be used to generate responses
    /// - `event_loop`: the Python event loop that will be used to run the generator
    ///
    /// Note: In Rust land, the request and the response are both concrete; however, in
    /// Python land, the request and response not strongly typed, meaning the generator
    /// could accept a different type of request or return a different type of response
    /// and we would not know until runtime.
    #[new]
    pub fn new(generator: PyObject, event_loop: PyObject) -> PyResult<Self> {
        let cancel_token = CancellationToken::new();
        Ok(PythonAsyncEngine(PythonServerStreamingEngine::new(
            cancel_token,
            Arc::new(generator),
            Arc::new(event_loop),
        )))
    }
}

#[async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for PythonAsyncEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        self.0.generate(request).await
    }
}
