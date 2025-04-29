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

use std::ffi::CStr;
use std::{env, path::Path, sync::Arc};

use anyhow::Context;
use dynamo_runtime::pipeline::error as pipeline_error;
pub use dynamo_runtime::{
    error,
    pipeline::{
        async_trait, AsyncEngine, AsyncEngineContextProvider, Data, ManyOut, ResponseStream,
        SingleIn,
    },
    protocols::annotated::Annotated,
    CancellationToken, Error, Result,
};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use pyo3_async_runtimes::TaskLocals;
use pythonize::{depythonize, pythonize};
pub use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::sync::oneshot::Sender;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use dynamo_llm::backend::ExecutionContext;
use dynamo_llm::engines::{EngineDispatcher, StreamingEngine};

/// Python snippet to import a file as a module
const PY_IMPORT: &CStr = cr#"
import runpy
import sys
import os
import functools
import types

module_dir = os.path.dirname(file_path)
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

sys.argv = sys_argv
module_dict = runpy.run_path(file_path, run_name='__main__')

# Create a module class with the generate function
class Module:
    def __init__(self, module_dict):
        self.__dict__.update(module_dict)
        self._generate_func = module_dict['generate']

    async def generate(self, request):
        async for response in self._generate_func(request):
            yield response

# Create module instance and store it in globals
module = Module(module_dict)
globals()['module'] = module
"#;

/// An engine that takes and returns strings, feeding them to a python written engine
pub async fn make_string_engine(
    cancel_token: CancellationToken,
    py_file: &Path,
    py_args: Vec<String>,
) -> pipeline_error::Result<Arc<dyn StreamingEngine>> {
    pyo3::prepare_freethreaded_python();
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        Python::with_gil(|py| {
            if let Err(e) = fix_venv(venv, py) {
                tracing::warn!("failed to fix venv: {}", e);
            }
        });
    }

    let engine = new_engine(cancel_token, py_file, py_args).await?;
    let engine: Arc<dyn StreamingEngine> = Arc::new(EngineDispatcher::new(engine));
    Ok(engine)
}

/// An engine that takes and returns tokens.
pub async fn make_token_engine(
    cancel_token: CancellationToken,
    py_file: &Path,
    py_args: Vec<String>,
) -> pipeline_error::Result<ExecutionContext> {
    pyo3::prepare_freethreaded_python();
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        Python::with_gil(|py| {
            if let Err(e) = fix_venv(venv, py) {
                tracing::warn!("failed to fix venv: {}", e);
            }
        });
    }

    let engine = new_engine(cancel_token, py_file, py_args).await?;
    let engine: ExecutionContext = Arc::new(engine);
    Ok(engine)
}

#[derive(Clone)]
pub struct PythonServerStreamingEngine {
    _cancel_token: CancellationToken,
    generator: Arc<PyObject>,
    event_loop: Arc<PyObject>,
}

async fn new_engine(
    cancel_token: CancellationToken,
    py_file: &Path,
    py_args: Vec<String>,
) -> anyhow::Result<PythonServerStreamingEngine> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::task::spawn_blocking(move || run_asyncio(tx));
    let event_loop = rx.await?;

    let user_module =
        python_file_to_module(py_file, py_args).with_context(|| py_file.display().to_string())?;
    let generator = Python::with_gil(|py| {
        /* Leave commented, `initialize` may be needed to match Triton
        if let Ok(initialize) = user_module.getattr(py, "initialize") {
            initialize
                .call1(py, (py_args,))
                .inspect_err(|err| {
                    println!();
                    err.display(py);
                })
                .with_context(|| "Failed calling python engine's initialize(args)")?;
        };
        */
        user_module
            .getattr(py, "generate")
            .with_context(|| "generate")
    })?;
    Ok(PythonServerStreamingEngine::new(
        cancel_token,
        Arc::new(generator),
        event_loop,
    ))
}

impl PythonServerStreamingEngine {
    pub fn new(
        cancel_token: CancellationToken,
        generator: Arc<PyObject>,
        event_loop: Arc<PyObject>,
    ) -> Self {
        PythonServerStreamingEngine {
            _cancel_token: cancel_token,
            generator,
            event_loop,
        }
    }
}

/// Start asyncio event loop and block on it forever
fn run_asyncio(tx: Sender<Arc<PyObject>>) {
    let event_loop: PyObject = Python::with_gil(|py| {
        let aio: PyObject = py.import("asyncio").unwrap().into();
        aio.call_method0(py, "new_event_loop").unwrap()
    });
    let event_loop = Arc::new(event_loop);
    let _ = tx.send(event_loop.clone());
    Python::with_gil(|py| {
        let _ = event_loop.call_method0(py, "run_forever");
    });
}

fn python_file_to_module(p: &Path, mut py_args: Vec<String>) -> Result<PyObject> {
    if let Some(filename) = p.file_name() {
        py_args.insert(0, filename.to_string_lossy().to_string());
    };
    let module: PyObject = Python::with_gil(|py| {
        let py_file_path: PyObject = p.display().to_string().into_pyobject(py).unwrap().into();
        let py_sys_argv: PyObject = py_args.into_pyobject(py).unwrap().into();
        let globals = [("file_path", py_file_path), ("sys_argv", py_sys_argv)]
            .into_py_dict(py)
            .context("into_py_dict")?;
        let locals = PyDict::new(py);
        py.run(PY_IMPORT, Some(&globals), Some(&locals))
            .context("PY_IMPORT")?;
        let module = locals
            .get_item("module")
            .unwrap()
            .context("get module after import")?;
        module.extract().context("extract")
    })?;
    Ok(module)
}

#[derive(Debug, thiserror::Error)]
enum ResponseProcessingError {
    #[error("python exception: {0}")]
    PythonException(String),

    #[error("deserialize error: {0}")]
    DeserializeError(String),

    #[error("gil offload error: {0}")]
    OffloadError(String),
}

#[async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error>
    for PythonServerStreamingEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        // Create a context
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let id = context.id().to_string();
        tracing::trace!("processing request: {}", id);

        // Clone the PyObject to move into the thread

        // Create a channel to communicate between the Python thread and the Rust async context
        let (tx, rx) = mpsc::channel::<Annotated<Resp>>(128);

        let generator = self.generator.clone();
        let event_loop = self.event_loop.clone();

        // Acquiring the GIL is similar to acquiring a standard lock/mutex
        // Performing this in an tokio async task could block the thread for an undefined amount of time
        // To avoid this, we spawn a blocking task to acquire the GIL and perform the operations needed
        // while holding the GIL.
        //
        // Under low GIL contention, we wouldn't need to do this.
        // However, under high GIL contention, this can lead to significant performance degradation.
        //
        // Since we cannot predict the GIL contention, we will always use the blocking task and pay the
        // cost. The Python GIL is the gift that keeps on giving -- performance hits...
        let stream = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let py_request = pythonize(py, &request)?;
                let gen = generator.call1(py, (py_request,))?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::tokio::into_stream_with_locals_v1(locals, gen.into_bound(py))
            })
        })
        .await??;

        let stream = Box::pin(stream);

        // process the stream
        // any error thrown in the stream will be caught and complete the processing task
        // errors are captured by a task that is watching the processing task
        // the error will be emitted as an annotated error
        let request_id = id.clone();

        tokio::spawn(async move {
            tracing::debug!(
                request_id,
                "starting task to process python async generator stream"
            );

            let mut stream = stream;
            let mut count = 0;

            while let Some(item) = stream.next().await {
                count += 1;
                tracing::trace!(
                    request_id,
                    "processing the {}th item from python async generator",
                    count
                );

                let mut done = false;

                let response = match process_item::<Resp>(item).await {
                    Ok(response) => response,
                    Err(e) => {
                        done = true;

                        let msg = match &e {
                            ResponseProcessingError::DeserializeError(e) => {
                                // tell the python async generator to stop generating
                                // right now, this is impossible as we are not passing the context to the python async generator
                                // todo: add task-local context to the python async generator
                                ctx.stop_generating();
                                let msg = format!("critical error: invalid response object from python async generator; application-logic-mismatch: {}", e);
                                msg
                            }
                            ResponseProcessingError::PythonException(e) => {
                                let msg = format!("a python exception was caught while processing the async generator: {}", e);
                                msg
                            }
                            ResponseProcessingError::OffloadError(e) => {
                                let msg = format!("critical error: failed to offload the python async generator to a new thread: {}", e);
                                msg
                            }
                        };

                        Annotated::from_error(msg)
                    }
                };

                if tx.send(response).await.is_err() {
                    tracing::trace!(
                        request_id,
                        "error forwarding annotated response to channel; channel is closed"
                    );
                    break;
                }

                if done {
                    tracing::debug!(
                        request_id,
                        "early termination of python async generator stream task"
                    );
                    break;
                }
            }

            tracing::debug!(
                request_id,
                "finished processing python async generator stream"
            );
        });

        let stream = ReceiverStream::new(rx);

        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

async fn process_item<Resp>(
    item: Result<Py<PyAny>, PyErr>,
) -> Result<Annotated<Resp>, ResponseProcessingError>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let item = item.map_err(|e| {
        println!();
        Python::with_gil(|py| e.display(py));
        ResponseProcessingError::PythonException(e.to_string())
    })?;
    let response = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| depythonize::<Resp>(&item.into_bound(py)))
    })
    .await
    .map_err(|e| ResponseProcessingError::OffloadError(e.to_string()))?
    .map_err(|e| ResponseProcessingError::DeserializeError(e.to_string()))?;

    let response = Annotated::from_data(response);

    Ok(response)
}

/// On Mac embedded Python interpreters do not pick up the virtual env.
#[cfg(target_os = "macos")]
fn fix_venv(venv: String, py: Python<'_>) -> anyhow::Result<()> {
    let version_info = py.version_info();
    let sys: PyObject = py.import("sys")?.into();
    let sys_path = sys.getattr(py, "path")?;
    let venv_path = format!(
        "{venv}/lib/python{}.{}/site-packages",
        version_info.major, version_info.minor
    );
    // TODO: This should go _before_ the site-packages
    sys_path.call_method1(py, "append", (venv_path,))?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn fix_venv(_venv: String, _py: Python<'_>) -> anyhow::Result<()> {
    Ok(())
}
