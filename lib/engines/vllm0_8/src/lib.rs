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

use std::ffi::CString;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use std::{path::Path, sync::Arc};

use async_stream::stream;
use dynamo_llm::engines::MultiNodeConfig;
use dynamo_llm::protocols::common::FinishReason;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::error as pipeline_error;
use dynamo_runtime::pipeline::{async_trait, Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{CancellationToken, Result};
use pyo3_async_runtimes::TaskLocals;
use serde::Deserialize;
use tokio::sync::oneshot::Sender;

use dynamo_llm::backend::ExecutionContext;
use dynamo_llm::protocols::common::llm_backend::{BackendInput, LLMEngineOutput};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyString, PyTuple};
use pyo3::PyObject;
use pyo3::Python;
use pythonize::pythonize;
use tokio_stream::StreamExt;

// The minor revision version of vllm that this engine supports
const VLLM_VERSION: &str = "0.8";

const PY_MAIN_MOD: &str = include_str!("vllm_inc.py");

const VLLM_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

pub async fn make_engine(
    cancel_token: CancellationToken,
    // Full path to the model, either a GGUF file or an HF repo dir
    model_path: &Path,
    // Multi node settings
    node_conf: MultiNodeConfig,
    // How many GPUs to use
    tensor_parallel_size: u32,
    // Path to extra engine args file
    extra_engine_args: Option<PathBuf>,
) -> pipeline_error::Result<ExecutionContext> {
    let engine = VllmEngine::new(
        cancel_token,
        model_path,
        node_conf,
        tensor_parallel_size,
        extra_engine_args,
    )
    .await?;
    let engine: ExecutionContext = Arc::new(engine);
    Ok(engine)
}

struct VllmEngine {
    cancel_token: CancellationToken,
    // How we send requests to Python / vllm
    request_queue: Arc<PyObject>,
    // asyncio event loop to run all our python futures. vllm is async.
    event_loop: Arc<PyObject>,
    // The python module from vllm_inc.py
    py_main_mod: Arc<PyObject>,
    // vllm.SamplingParams
    sampling_params: PyObject,
}

impl VllmEngine {
    pub async fn new(
        cancel_token: CancellationToken,
        model_path: &Path,
        node_conf: MultiNodeConfig,
        tensor_parallel_size: u32,
        extra_engine_args: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        pyo3::prepare_freethreaded_python();

        // Safety: CString::new will only return an error if the string contains an null byte.
        let py_main_mod: PyObject = Python::with_gil(|py| -> PyResult<PyObject> {
            PyModule::from_code(
                py,
                &CString::new(PY_MAIN_MOD).expect("vllm_inc.py contains a null byte!"),
                &CString::new("_synthetic/dynamo_engine_vllm.py").unwrap(),
                &CString::new("dynamo_engine_vllm").unwrap(),
            )
            .map(|p| p.into())
        })?;
        let py_main_mod = Arc::new(py_main_mod);

        let sampling_params = sampling_params_type();
        let (request_queue_rs, request_queue_py) = make_python_queues(64)?;
        let (ready_event_rs, ready_event_py) = make_python_event()?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let model_path_buf = PathBuf::from(model_path);
        let cancel_token_worker = cancel_token.clone();
        let py_main_mod_worker = py_main_mod.clone();
        tokio::task::spawn(async move {
            if let Err(err) = run_vllm_worker(
                cancel_token_worker,
                tx,
                py_main_mod_worker,
                request_queue_py,
                ready_event_py,
                &model_path_buf,
                node_conf,
                tensor_parallel_size,
                extra_engine_args,
            )
            .await
            {
                tracing::error!(%err, "run_vllm_worker error");
            }
        });
        let event_loop = tokio::select! {
            ev = rx => ev,
            _ = cancel_token.cancelled() => {
                anyhow::bail!("VllmEngine create cancelled");
            }
        };
        let event_loop = event_loop?;

        // Wait for vllm to start accepting requests
        tokio::select! {
            _ = wait_for_vllm(event_loop.clone(), ready_event_rs) => {
                tracing::trace!("vllm worker is ready");
            }
            _ = cancel_token.cancelled() => {
                anyhow::bail!("VllmEngine cancelled waiting for vllm to start");
            }
        };

        let engine = VllmEngine {
            cancel_token,
            request_queue: Arc::new(request_queue_rs),
            event_loop,
            py_main_mod,
            sampling_params,
        };
        Ok(engine)
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<BackendInput>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for VllmEngine
{
    async fn generate(
        &self,
        request: SingleIn<BackendInput>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, context) = request.into_parts();
        let ctx = context.context();
        let request_id = ctx.id().to_string();

        let temperature: f64 = request.sampling_options.temperature.unwrap_or(0.0).into();

        // Send request
        let (response_queue_1, response_queue_2) = make_python_queues(16)?;
        let queue_fut = Python::with_gil(|py| {
            let py_temp: PyObject = temperature.into_pyobject(py).unwrap().into();
            let mut sp_kwargs = vec![("temperature", py_temp)];
            if let Some(max_tokens) = request.stop_conditions.max_tokens {
                let py_max_tokens: PyObject = max_tokens.into_pyobject(py).unwrap().into();
                // vllm defaults this to 16
                sp_kwargs.push(("max_tokens", py_max_tokens));
            }
            let sp_kwargs = sp_kwargs.into_py_dict(py).unwrap();
            let sampling_params = self.sampling_params.call(py, (), Some(&sp_kwargs)).unwrap();

            let py_request = pythonize(py, &request)?;
            let args: Vec<PyObject> = vec![
                PyString::new(py, &request_id).into(),
                py_request.into(),
                sampling_params,
                response_queue_1,
            ];
            let put_arg = PyTuple::new(py, args)?;

            let locals = TaskLocals::new(self.event_loop.bind(py).clone());
            pyo3_async_runtimes::into_future_with_locals(
                &locals.clone_ref(py),
                self.request_queue
                    .bind(py)
                    .call_method1("put", (put_arg,))?,
            )
        })?;
        queue_fut.await?;

        // Read response
        let from_vllm = Python::with_gil(|py| {
            let locals = TaskLocals::new(self.event_loop.bind(py).clone());
            pyo3_async_runtimes::tokio::into_stream_with_locals_v1(
                locals,
                self.py_main_mod
                    .bind(py)
                    .call_method1("run_response", (response_queue_2.bind(py),))?,
            )
        })?;
        let mut from_vllm = Box::pin(from_vllm);

        let cancel_token = self.cancel_token.clone();
        let output = stream! {
            let mut num_output_tokens_so_far = 0;
            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::trace!(request_id, "VllmEngine.generate stopped by cancel token");
                        break;
                    }
                    item = from_vllm.next() => {
                        match item {
                            None => {
                                yield Annotated::from_data(LLMEngineOutput::stop());
                                break;
                            },
                            Some(item) => {
                                match vllm_to_dynamo(item).await {
                                    Ok(Some(mut response)) => {
                                        // The response includes all the tokens.
                                        // We only want the delta.
                                        if response.token_ids.is_empty() {
                                            yield Annotated::from_data(response);
                                            break;
                                        } else {
                                            let next_total_toks = response.token_ids.len();
                                            drop(response.token_ids.drain(0..num_output_tokens_so_far));
                                            num_output_tokens_so_far = next_total_toks;
                                            yield Annotated::from_data(response);
                                        }
                                    }
                                    Ok(None) => {
                                        yield Annotated::from_data(LLMEngineOutput::stop());
                                        break;
                                    },
                                    Err(err) => {
                                        tracing::error!(request_id, %err, "vllm_to_dynamo error");
                                        break;
                                    }
                                }
                            }
                        }
                    }
                } // tokio::select!
            } // loop
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

// TODO this will panic if invalid vllm, can't do anything with wrong vllm.
// But should we return an error instead?
fn sampling_params_type() -> PyObject {
    Python::with_gil(|py| {
        let vllm_module: PyObject = match py.import("vllm") {
            Ok(m) => m.into(),
            Err(err) => {
                panic!("Failed to import python 'vllm' module. Are we running in the correct venv? {err}");
            }
        };

        let version = vllm_module
            .getattr(py, "__version__")
            .expect("vllm missing __version__ field")
            .extract::<String>(py)
            .expect("vllm.__version__ is not a string");
        if !version.starts_with(VLLM_VERSION) {
            panic!("Expected vllm version {VLLM_VERSION}, found {version}");
        }

        let sample_params_type: PyObject = vllm_module
            .getattr(py, "SamplingParams")
            .expect("vllm module missing SamplingParams type.");
        sample_params_type
    })
}

/// Create a Python asyncio.Queue. Return two copies of it.
fn make_python_queues(max_size: usize) -> anyhow::Result<(PyObject, PyObject)> {
    Python::with_gil(|py| -> Result<(PyObject, PyObject), String> {
        let module: PyObject = match py.import("asyncio") {
            Ok(m) => m.into(),
            Err(err) => {
                panic!("Failed to import python 'asyncio' module. Is Python installed? {err}");
            }
        };
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("maxsize", max_size)
            .map_err(|err| format!("Failed setting maxsize in dict to {max_size}: {err}"))?;
        let q = module
            .call_method(py, "Queue", (), Some(&kwargs))
            .map_err(|e| format!("Failed to call asyncio.Queue: {}", e))?;
        Ok((q.clone(), q))
    })
    .map_err(|err| anyhow::anyhow!("{err}"))
}

fn make_python_event() -> anyhow::Result<(PyObject, PyObject)> {
    Python::with_gil(|py| -> Result<(PyObject, PyObject), String> {
        let module: PyObject = match py.import("asyncio") {
            Ok(m) => m.into(),
            Err(err) => {
                panic!("Failed to import python 'asyncio' module. Is Python installed? {err}");
            }
        };
        let ev = module
            .call_method0(py, "Event")
            .map_err(|e| format!("Failed to call asyncio.Event: {}", e))?;
        Ok((ev.clone(), ev))
    })
    .map_err(|err| anyhow::anyhow!("{err}"))
}

/// Start asyncio event loop and block on it forever
#[allow(clippy::too_many_arguments)]
async fn run_vllm_worker(
    cancel_token: CancellationToken,
    tx: Sender<Arc<PyObject>>,
    py_main_mod: Arc<PyObject>,
    request_queue: PyObject, // asyncio.Queue
    ready_event: PyObject,   // asyncio.Event
    model_path: &Path,
    node_conf: MultiNodeConfig,
    tensor_parallel_size: u32,
    extra_engine_args: Option<PathBuf>,
) -> anyhow::Result<()> {
    let model_path_str = model_path.display().to_string();
    let extra_engine_args_str = &extra_engine_args
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    let event_loop: PyObject = Python::with_gil(|py| -> PyResult<PyObject> {
        let aio: PyObject = py.import("asyncio").map(|p| p.into())?;
        aio.call_method0(py, "new_event_loop")
    })?;
    let event_loop = Arc::new(event_loop);
    let _ = tx.send(event_loop.clone());
    let event_loop_forever = event_loop.clone();
    tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            let _ = event_loop_forever.call_method0(py, "run_forever");
        });
    });

    let vllm_fut = Python::with_gil(|py| {
        // These go directly to vllm's AsyncEngineArgs
        let kwargs: Vec<(&str, PyObject)> = vec![
            ("model", PyString::new(py, &model_path_str).into()),
            ("task", PyString::new(py, "generate").into()),
            (
                "skip_tokenizer_init",
                // Safety: true always converts to python object
                true.into_pyobject(py).unwrap().to_owned().into(),
            ),
            (
                "tensor_parallel_size",
                // Safety: A u32 should always convert safely
                tensor_parallel_size.into_pyobject(py).unwrap().into(),
            ),
            (
                "pipeline_parallel_size",
                // Safety: A u32 should always convert safely
                node_conf.num_nodes.into_pyobject(py).unwrap().into(),
            ),
            (
                "enable_prefix_caching",
                // Safety: true always converts to python object
                true.into_pyobject(py).unwrap().to_owned().into(),
            ),
        ];
        let kwargs = kwargs.into_py_dict(py)?;

        let locals = TaskLocals::new(event_loop.bind(py).clone());
        pyo3_async_runtimes::into_future_with_locals(
            &locals.clone_ref(py),
            py_main_mod
                .call_method(
                    py,
                    "main",
                    (
                        request_queue.bind(py),
                        ready_event.bind(py),
                        extra_engine_args_str,
                    ),
                    Some(&kwargs),
                )?
                .into_bound(py),
        )
    })?;

    tokio::select! {
        _ = cancel_token.cancelled() => {
            tracing::trace!("VllmEngine worker stopped by cancel token");
            // Stop vllm
            let vllm_stop_fut = Python::with_gil(|py| {
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::into_future_with_locals(
                    &locals.clone_ref(py),
                    request_queue.call_method1(py, "put", (py.None(),))?.into_bound(py)
                )
            })?;
            tokio::select! {
                _ = vllm_stop_fut => {}
                _ = tokio::time::sleep(VLLM_SHUTDOWN_TIMEOUT) => {
                    tracing::warn!("Timeout waiting for vllm to shut down. Process may still be running");
                }
            };
        }
        _ = vllm_fut => {
            tracing::warn!("VllmEngine worker unexpected worker task completed");
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum ResponseProcessingError {
    #[error("python exception: {0}")]
    PythonException(String),
}

#[derive(Debug, Clone, Deserialize, FromPyObject)]
pub struct CompletionOutput {
    pub index: usize,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub cumulative_logprob: Option<f64>,
    pub logprobs: Option<Vec<f64>>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, FromPyObject)]
pub struct RequestMetrics {
    pub arrival_time: f64,
    pub last_token_time: f64,
    pub first_scheduled_time: f64,
    pub first_token_time: f64,
    pub time_in_queue: f64,
    pub finished_time: Option<f64>,
    pub scheduler_time: f64,
    pub model_forward_time: Option<f64>,
    pub model_execute_time: Option<f64>,
    pub spec_token_acceptance_counts: Vec<u32>,
}

// Matches vllm Python type:
// RequestOutput(request_id=b87cf9dd-f66f-422f-ada9-b3e08c642c03, prompt=None, prompt_token_ids=[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='', token_ids=(785, 6722), cumulative_logprob=None, logprobs=None, finish_reason=None, stop_reason=None)], finished=False, metrics=RequestMetrics(arrival_time=1744820354.1364133, last_token_time=1744820354.206939, first_scheduled_time=1744820354.137031, first_token_time=1744820354.180786, time_in_queue=0.0006177425384521484, finished_time=None, scheduler_time=0.00044727100248564966, model_forward_time=None, model_execute_time=None, spec_token_acceptance_counts=[0]), lora_request=None, num_cached_tokens=0, multi_modal_placeholders={})
#[derive(Debug, Clone, Deserialize, FromPyObject)]
pub struct RequestOutput {
    pub request_id: String, // this is a uuid
    pub prompt: Option<String>,
    pub prompt_token_ids: Vec<u32>,
    pub encoder_prompt: Option<String>,
    pub encoder_prompt_token_ids: Option<Vec<u32>>,
    pub prompt_logprobs: Option<Vec<f32>>,
    pub outputs: Vec<CompletionOutput>,
    pub finished: bool,
    pub metrics: RequestMetrics,
    //pub lora_request: Option<serde_json::Value>,
    pub num_cached_tokens: usize,
    //pub multi_modal_placeholders: HashMap<String, serde_json::Value>,
}

impl From<RequestOutput> for LLMEngineOutput {
    fn from(mut req: RequestOutput) -> LLMEngineOutput {
        if req.outputs.is_empty() {
            // TODO should this be an error?
            return LLMEngineOutput::stop();
        }
        let out = req.outputs.remove(0);
        let finish_reason = out
            .finish_reason
            .map(|fr| match FinishReason::from_str(&fr) {
                Ok(fr) => fr,
                Err(err) => {
                    let s = format!("Unsupported finish reason from vllm: {fr}: {err}");
                    tracing::error!("{s}");
                    FinishReason::Error(s)
                }
            });
        LLMEngineOutput {
            token_ids: out.token_ids,
            tokens: None,
            text: None,
            cum_log_probs: out.cumulative_logprob,
            log_probs: out.logprobs,
            finish_reason,
        }
    }
}

// Convert the vllm type to the dynamo type
async fn vllm_to_dynamo(
    item: Result<Py<PyAny>, PyErr>,
) -> Result<Option<LLMEngineOutput>, ResponseProcessingError> {
    // Handle errors first
    let item = item.map_err(|e| {
        println!();
        Python::with_gil(|py| e.display(py));
        ResponseProcessingError::PythonException(e.to_string())
    })?;

    // None is how Python tells us the request is complete
    if Python::with_gil(|py| item.is_none(py)) {
        return Ok(None);
    }

    Python::with_gil(|py| match item.extract::<RequestOutput>(py) {
        Ok(response) => Ok(Some(response.into())),
        Err(err) => {
            tracing::trace!(%err, "Err extract python into RequestOutput. Usually means end of response.");
            Ok(None)
        }
    })
}

async fn wait_for_vllm(
    event_loop: Arc<PyObject>,
    ready_event_rs: PyObject,
) -> anyhow::Result<PyObject> {
    let maybe_py_fut = Python::with_gil(|py| -> PyResult<PyObject> {
        ready_event_rs
            .bind(py)
            .call_method0("wait")
            .map(|p| p.into())
    });
    let py_fut = match maybe_py_fut {
        Ok(fut) => fut,
        Err(err) => {
            anyhow::bail!("Failed calling python event.wait() waiting for vllm to start: {err}");
        }
    };
    let rs_fut = Python::with_gil(|py| {
        let locals = TaskLocals::new(event_loop.bind(py).clone());
        pyo3_async_runtimes::into_future_with_locals(&locals.clone_ref(py), py_fut.bind(py).clone())
    })?;
    rs_fut.await.map_err(|err| err.into())
}
