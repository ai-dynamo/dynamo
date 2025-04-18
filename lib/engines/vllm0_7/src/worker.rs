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

use std::collections::HashMap;
use std::env;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use std::vec::IntoIter;

use async_zmq::{SinkExt, StreamExt};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::CancellationToken;
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyBytes, PyString},
};
use tokio::io::AsyncBufReadExt;
use tokio::sync::mpsc::{error::SendError, Sender};
use tokio::task::JoinHandle;

use dynamo_llm::kv_router::protocols::ForwardPassMetrics;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_llm::protocols::common::FinishReason;
use dynamo_llm::{engines::MultiNodeConfig, kv_router::publisher::KvMetricsPublisher};

/// Wait this long for the vllm sub-process to stop after we send it a KILL
const VLLM_STOP_TIMEOUT: Duration = Duration::from_millis(1500);

// The minor revision version of vllm that this engine supports. 0.8+ is in a different engine.
const VLLM_VERSION: &str = "0.7";

type RequestID = String;

pub struct VllmWorker {
    /// How we receive work requests
    tx: Sender<WorkRequest>,

    /// Handle of the task that reads from `tx` and forwards those requests over zmq to vllm
    _input_loop: JoinHandle<()>,

    /// Handle of the task that reads vllm's responses from zmq and dispatches them to the correct
    /// active request.
    _output_loop: JoinHandle<()>,

    /// Handle of the vllm background process
    vllm: Option<JoinHandle<()>>,

    // We don't need to hold on to this, it's already shared between input_loop and output_loop
    // But later we'll probably want stats - how many active requests etc, so keep it here
    _active_requests: Arc<tokio::sync::Mutex<HashMap<RequestID, ActiveRequest>>>,

    // Need to keep this alive
    // TODO: With async_zmq we possibly don't need this at all
    #[allow(dead_code)]
    zmq_context: async_zmq::Context,
}

/// How we get asked to do some work. These get unpacked and forwarded to vllm.
pub struct WorkRequest {
    pub request: PreprocessedRequest,
    pub request_id: RequestID,
    pub response_channel: Sender<Annotated<LLMEngineOutput>>,
}

/// A request currently being process by vllm
struct ActiveRequest {
    tx: Sender<Annotated<LLMEngineOutput>>,
    num_output_tokens_so_far: usize,
}

/// Python imports
struct Imports {
    pickle_module: PyObject,
    tokens_prompt_type: PyObject,
    sample_params_type: PyObject,
    rpc_type: PyObject,
    startup_type: PyObject,
}

/// All the zmq sockets we used. This object only used to passing them around to avoid large
/// tuples.
struct Sockets {
    #[allow(dead_code)]
    context: async_zmq::Context, // we have to keep this alive

    // Control socket, how we ask vllm engine to start.
    // Not the best name, but this is what vllm calls it internally.
    data: async_zmq::Dealer<IntoIter<Vec<u8>>, Vec<u8>>,
    // Requests from us to the vllm engine
    input: async_zmq::Push<IntoIter<Vec<u8>>, Vec<u8>>,
    // Responses from the vllm engine back to us
    output: async_zmq::Pull,
    // Heartbeat messages from vllm process
    heartbeat: async_zmq::Pull,
    // NOTE: Metrics socket usage is custom to our patch of vllm, and may not
    // be present when running upstream vllm.
    // Metrics messages from vllm process
    metrics: async_zmq::Pull,
}

/// The message vllm sends us over zmq when it's ready to work.
#[derive(FromPyObject, Debug)]
struct RPCStartupResponse {
    #[allow(dead_code)]
    tracing_enabled: bool,
}

/// What vllm sends us. Usually it contains a single token.
#[allow(dead_code)]
#[derive(FromPyObject, Debug)]
pub struct RequestOutput {
    request_id: String,
    prompt: Option<String>,
    prompt_token_ids: Option<Vec<u32>>,
    prompt_logprobs: Option<Vec<Option<HashMap<u32, Logprob>>>>,
    outputs: Vec<CompletionOutput>,
    finished: bool,
    //metrics: Optional[RequestMetrics] = None,
    //lora_request: Optional[LoRARequest] = None,
    encoder_prompt: Option<String>,
    encoder_prompt_token_ids: Option<Vec<u32>>,
    num_cached_tokens: Option<u32>,
}

#[allow(dead_code)]
#[derive(FromPyObject, Debug)]
pub struct CompletionOutput {
    index: u32,
    text: String,
    token_ids: Vec<u32>,
    cumulative_logprob: Option<f32>,
    logprobs: Option<Vec<HashMap<u32, Logprob>>>,
    finish_reason: Option<String>,
    //stop_reason: Union[int, str, None] = None
    //lora_request: Optional[LoRARequest] = None
}

#[allow(dead_code)]
#[derive(FromPyObject, Debug)]
struct Logprob {
    logprob: f32,
    rank: Option<u32>,
    decoded_token: Option<String>,
}

/// Main entry point
pub async fn start(
    cancel_token: CancellationToken,
    sock_code: &str,
    model_path: &Path,
    _node_conf: MultiNodeConfig,
    tensor_parallel_size: u32,
    extra_engine_args: Option<PathBuf>,
    // When using our vllm fork, this is how we publish it's KV metrics for the KV router
    kv_metrics_publisher: Option<Arc<KvMetricsPublisher>>,
) -> anyhow::Result<VllmWorker> {
    pyo3::prepare_freethreaded_python(); // or enable feature "auto-initialize"
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let _ = Python::with_gil(|py| crate::fix_venv(venv, py));
    }

    let py_imports = Arc::new(python_imports());
    let Sockets {
        context,
        data,
        input,
        output,
        heartbeat,
        metrics,
    } = zmq_sockets(sock_code)?;

    let vllm_process = start_vllm(
        model_path,
        &py_imports,
        data,
        tensor_parallel_size,
        extra_engine_args,
        kv_metrics_publisher.is_some(),
    )
    .await?;
    let vllm_join_handle = watch_vllm(cancel_token.clone(), vllm_process);

    tokio::spawn(heartbeat_loop(cancel_token.clone(), heartbeat));
    tokio::spawn(metrics_loop(
        cancel_token.clone(),
        metrics,
        kv_metrics_publisher.clone(),
    ));

    let active_requests = Arc::new(tokio::sync::Mutex::new(HashMap::new()));
    let (tx, rx) = tokio::sync::mpsc::channel(8);

    let input_loop_handle = {
        let cancel_token = cancel_token.clone();
        let py_imports = py_imports.clone();
        let active_requests = active_requests.clone();
        tokio::spawn(input_loop(
            cancel_token,
            py_imports,
            input,
            active_requests,
            rx,
        ))
    };
    let output_loop_handle = {
        let cancel_token = cancel_token.clone();
        let py_imports = py_imports.clone();
        let active_requests = active_requests.clone();
        tokio::spawn(output_loop(
            cancel_token,
            py_imports,
            output,
            active_requests,
        ))
    };

    Ok(VllmWorker {
        tx,
        zmq_context: context,
        _input_loop: input_loop_handle,
        _output_loop: output_loop_handle,
        vllm: Some(vllm_join_handle),
        _active_requests: active_requests,
    })
}

/// Import all the python packages we'll need. `vllm` particularly takes a few seconds.
fn python_imports() -> Imports {
    Python::with_gil(|py| {
        let pickle_module: PyObject = match py.import("pickle") {
            Ok(m) => m.into(),
            Err(err) => {
                // There is no vllm without python
                panic!("Failed to import python 'pickle' module. Is Python installed? {err}");
            }
        };

        let vllm_module: PyObject = match py.import("vllm") {
            Ok(m) => m.into(),
            Err(err) => {
                panic!("Failed to import python 'vllm' module. Are we running in the correct venv? {err}");
            }
        };

        // While we're here check vllm version
        let version = vllm_module
            .getattr(py, "__version__")
            .expect("vllm missing __version__ field")
            .extract::<String>(py)
            .expect("vllm.__version__ is not a string");
        if !version.starts_with(VLLM_VERSION) {
            panic!("Expected vllm version {VLLM_VERSION}, found {version}");
        }

        let tokens_prompt_type: PyObject = vllm_module.getattr(py, "TokensPrompt").unwrap();
        let sample_params_type: PyObject = vllm_module.getattr(py, "SamplingParams").unwrap();

        let mod_multiprocessing = py.import("vllm.engine.multiprocessing").unwrap();
        let rpc_type: PyObject = mod_multiprocessing
            .getattr("RPCProcessRequest")
            .unwrap()
            .into();
        let startup_type: PyObject = mod_multiprocessing
            .getattr("RPCStartupRequest")
            .unwrap()
            .into();

        Imports {
            pickle_module,
            tokens_prompt_type,
            sample_params_type,
            rpc_type,
            startup_type,
        }
    })
}

/// Create all the zmq sockets we're going to use.
fn zmq_sockets(sock_code: &str) -> anyhow::Result<Sockets> {
    let zmq_context = async_zmq::Context::new();
    let input = async_zmq::push(&format!("ipc:///tmp/{sock_code}_input_socket"))?
        .with_context(&zmq_context)
        .connect()?;

    let output = async_zmq::pull(&format!("ipc:///tmp/{sock_code}_output_socket"))?
        .with_context(&zmq_context)
        .connect()?;

    let data = async_zmq::dealer(&format!("ipc:///tmp/{sock_code}_data_socket"))?
        .with_context(&zmq_context)
        .connect()?;

    let heartbeat = async_zmq::pull(&format!("ipc:///tmp/{sock_code}_health_socket"))?
        .with_context(&zmq_context)
        .connect()?;

    let metrics = async_zmq::pull(&format!("ipc:///tmp/{sock_code}_metrics_socket"))?
        .with_context(&zmq_context)
        .connect()?;

    // TODO: NIXL/Prefill sockets here in the future for disagg?

    Ok(Sockets {
        context: zmq_context,
        data,
        input,
        output,
        heartbeat,
        metrics,
    })
}

/// Start the vllm python sub-process and wait for it to start
async fn start_vllm(
    model_path: &Path,
    python_imports: &Imports,
    mut data_socket: async_zmq::Dealer<IntoIter<Vec<u8>>, Vec<u8>>,
    tensor_parallel_size: u32,
    extra_engine_args: Option<PathBuf>,
    with_kv_routing: bool,
) -> anyhow::Result<tokio::process::Child> {
    let mut vllm_args = vec![
        "--internal-vllm-process".to_string(),
        format!("--model-path={}", model_path.display()),
        format!("--tensor-parallel-size={tensor_parallel_size}"),
    ];
    if let Some(args_path) = extra_engine_args {
        vllm_args.push(format!("--extra-engine-args={}", args_path.display()));
    }
    if with_kv_routing {
        vllm_args.push("--router-mode=kv".to_string());
    }

    let self_path = std::env::current_exe()?;
    let mut proc = tokio::process::Command::new(self_path)
        .env("VLLM_LOGGING_LEVEL", "DEBUG")
        .args(&vllm_args)
        .kill_on_drop(false)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = tokio::io::BufReader::new(proc.stdout.take().unwrap());
    let stderr = tokio::io::BufReader::new(proc.stderr.take().unwrap());

    tokio::spawn(async move {
        let mut lines = stdout.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let mut line_parts = line.splitn(4, ' ');
            let mut log_level = line_parts.next().unwrap_or_default();
            // Skip date (0) and time (1). Print last (2) which is everything else.
            let line = line_parts.nth(2).unwrap_or_default();
            if line.starts_with("custom_op.py:68") || line.trim().is_empty() {
                // Skip a noisy line
                // custom_op.py:68] custom op <the op> enabled
                continue;
            }
            if line.contains("ERROR") {
                log_level = "ERROR";
            }
            match log_level {
                "DEBUG" => tracing::debug!("VLLM: {line}"),
                "INFO" => tracing::debug!("VLLM: {line}"), // VLLM is noisy in debug mode
                "WARNING" => tracing::warn!("VLLM: {line}"),
                "ERROR" => tracing::error!("VLLM: {line}"),
                level => tracing::info!("VLLM: {level} {line}"),
            }
        }
    });
    tokio::spawn(async move {
        let mut lines = stderr.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }
            tracing::warn!("VLLM: {line}");
        }
    });

    let start_req_bytes: Vec<u8> = Python::with_gil(|py| {
        let start_req = python_imports
            .startup_type
            .getattr(py, "IS_SERVER_READY")
            .unwrap();
        let pickle_dumps = python_imports.pickle_module.getattr(py, "dumps").unwrap();
        pickle_dumps
            .call1(py, (start_req,))
            .unwrap()
            .extract(py)
            .unwrap()
    });
    data_socket.send(vec![start_req_bytes].into()).await?;
    let start_resp: Vec<u8> = match data_socket.next().await {
        Some(Ok(r)) => {
            if !r.is_empty() {
                r[0].deref().to_vec()
            } else {
                anyhow::bail!("vllm failed to start. No response on dealer/data socket");
            }
        }
        Some(Err(err)) => {
            anyhow::bail!("vllm failed to start. Error reading from dealer/data socket: {err}");
        }
        None => {
            anyhow::bail!("vllm failed to start. dealer/data socket is closed.");
        }
    };
    let resp: RPCStartupResponse = Python::with_gil(|py| {
        let pickle_loads = python_imports.pickle_module.getattr(py, "loads").unwrap();
        pickle_loads
            .call1(py, (start_resp,))
            .unwrap()
            .extract(py)
            .unwrap()
    });
    tracing::debug!("vllm zmq backend is ready: {resp:?}");

    Ok(proc)
}

// Stop the vllm process when we stop, and prevent it going zombie.
fn watch_vllm(
    cancel_token: CancellationToken,
    mut vllm_process: tokio::process::Child,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        cancel_token.cancelled().await;
        tokio::select! {
            _ = vllm_process.wait() => {
                return;
            },
            _ = tokio::time::sleep(VLLM_STOP_TIMEOUT) => { }
        }
        if let Err(err) = vllm_process.start_kill() {
            tracing::error!("Failing killing vllm subprocess: {err}");
            return;
        }
        tokio::select! {
            _ = vllm_process.wait() => { },
            _ = tokio::time::sleep(VLLM_STOP_TIMEOUT) => {
                tracing::warn!("Timeout waiting for vllm sub-process to stop after kill");
            }
        }
    })
}

// How we know vllm engine is alive. It sends "SUCCESS" as a pickled string every 10s.
// Runs outside of tokio on a regular thread.
// TODO: If we don't get heartbeats we should, euh, do something. vllm is gone. At least
// de-register the model.
async fn heartbeat_loop(cancel_token: CancellationToken, mut socket: async_zmq::Pull) {
    loop {
        let maybe_hb = tokio::select! {
            _ = cancel_token.cancelled() => {
                break;
            }
            maybe_hb = socket.next() => {
                maybe_hb
            }
        };
        let b = match maybe_hb {
            Some(Ok(b)) => b[0].deref().to_vec(),
            Some(Err(err)) => {
                tracing::error!("Error reading from vllm heartbeat socket: {err}");
                break;
            }
            None => {
                tracing::debug!("vllm heartbeat socket closed");
                break;
            }
        };
        let s: String = match serde_pickle::from_slice(&b, Default::default()) {
            Ok(s) => s,
            Err(err) => {
                tracing::error!("Error de-serializing vllm heartbeat response. It was probably Exception not str. {err}");
                break;
            }
        };
        if s != "SUCCESS" {
            tracing::error!("vllm heartbeat error, expected 'SUCCESS' got '{s}'");
            break;
        }
    }
}

// NOTE: Custom to our patch of vllm.
async fn metrics_loop(
    cancel_token: CancellationToken,
    mut socket: async_zmq::Pull,
    publisher: Option<Arc<KvMetricsPublisher>>,
) {
    loop {
        let maybe_metrics = tokio::select! {
            _ = cancel_token.cancelled() => {
                break;
            }
            maybe_metrics = socket.next() => {
                maybe_metrics
            }
        };
        let b = match maybe_metrics {
            Some(Ok(b)) => b[0].deref().to_vec(),
            Some(Err(err)) => {
                tracing::error!("Error reading from vllm metrics socket: {err}");
                break;
            }
            None => {
                tracing::debug!("vllm metrics socket closed");
                break;
            }
        };

        // Try to deserialize directly into ForwardPassMetrics using Python's pickle module
        let metrics_result = Python::with_gil(|py| -> Result<ForwardPassMetrics, String> {
            let pickle = py
                .import("pickle")
                .map_err(|e| format!("Failed to import pickle: {}", e))?;
            let loads = pickle
                .getattr("loads")
                .map_err(|e| format!("Failed to get loads function: {}", e))?;
            let bytes = PyBytes::new(py, &b);

            let result = loads
                .call1((bytes,))
                .map_err(|e| format!("Failed to call pickle.loads: {}", e))?;

            // Try to extract the attributes from the Python object
            let extract_field = |field: &str| -> Result<u64, String> {
                result
                    .getattr(field)
                    .map_err(|e| format!("Field '{}' not found: {}", field, e))?
                    .extract::<u64>()
                    .map_err(|e| format!("Failed to extract '{}' as u64: {}", field, e))
            };

            let extract_float_field = |field: &str| -> Result<f32, String> {
                result
                    .getattr(field)
                    .map_err(|e| format!("Field '{}' not found: {}", field, e))?
                    .extract::<f32>()
                    .map_err(|e| format!("Failed to extract '{}' as f32: {}", field, e))
            };

            // Give default values for any fields not found
            let request_active_slots = extract_field("request_active_slots").unwrap_or(0);
            let request_total_slots = extract_field("request_total_slots").unwrap_or(0);
            let kv_active_blocks = extract_field("kv_active_blocks").unwrap_or(0);
            let kv_total_blocks = extract_field("kv_total_blocks").unwrap_or(0);
            let num_requests_waiting = extract_field("num_requests_waiting").unwrap_or(0);
            let gpu_cache_usage_perc = extract_float_field("gpu_cache_usage_perc").unwrap_or(0.0);
            let gpu_prefix_cache_hit_rate =
                extract_float_field("gpu_prefix_cache_hit_rate").unwrap_or(0.0);

            Ok(ForwardPassMetrics {
                request_active_slots,
                request_total_slots,
                kv_active_blocks,
                kv_total_blocks,
                num_requests_waiting,
                gpu_cache_usage_perc,
                gpu_prefix_cache_hit_rate,
            })
        });

        match metrics_result {
            Ok(metrics) => {
                if let Some(metrics_publisher) = publisher.as_ref() {
                    if let Err(err) = metrics_publisher.publish(metrics.into()) {
                        tracing::error!(%err, "Failed publishing KV metrics");
                    }
                }
            }
            Err(err) => {
                tracing::error!("Error deserializing vllm metrics with Python pickle: {err}");
            }
        }
    }
}

fn from_vllm(output: CompletionOutput, previous_total_toks: usize) -> LLMEngineOutput {
    let finish_reason = match output.finish_reason.as_deref() {
        Some("stop") => Some(FinishReason::Stop),
        Some("abort") => Some(FinishReason::Cancelled),
        Some("length") => Some(FinishReason::Length),
        Some(unknown) => {
            tracing::info!("Unknown vllm stop reason '{unknown}'. Please add to vllm.rs");
            Some(FinishReason::Stop)
        }
        None => None,
    };

    LLMEngineOutput {
        // todo - propagate mdcsum
        token_ids: output.token_ids[previous_total_toks..].into(),
        tokens: None,
        text: None,
        //text: if output.text.is_empty() { None } else { Some(output.text) },
        cum_log_probs: output.cumulative_logprob.map(|v| v as f64),
        log_probs: None, // TODO  output.logprobs
        finish_reason,
    }
}

async fn input_loop(
    cancel_token: CancellationToken,
    py_imports: Arc<Imports>,
    mut input_socket: async_zmq::Push<IntoIter<Vec<u8>>, Vec<u8>>,
    active_requests: Arc<tokio::sync::Mutex<HashMap<RequestID, ActiveRequest>>>,
    mut rx: tokio::sync::mpsc::Receiver<WorkRequest>,
) {
    loop {
        let work_request = tokio::select! {
            _ = cancel_token.cancelled() => {
                tracing::trace!("VllmWorker.input_loop exit");
                break;
            }
            req = rx.recv() => {
                match req {
                    Some(req) => req,
                    None => {
                        tracing::trace!("VllmWorker input_loop socket closed");
                        break;
                    }
                }
            }
        };

        let request_id = work_request.request_id;
        let token_ids = work_request.request.token_ids.clone();
        let temperature: f64 = work_request
            .request
            .sampling_options
            .temperature
            .unwrap_or(0.0)
            .into();

        // Parts that don't change
        let (py_request_id, sampling_params) = Python::with_gil(|py| {
            let py_temp: PyObject = temperature.into_pyobject(py).unwrap().into();
            let mut sp_kwargs = vec![("temperature", py_temp)];
            if let Some(max_tokens) = work_request.request.stop_conditions.max_tokens {
                let py_max_tokens: PyObject = max_tokens.into_pyobject(py).unwrap().into();
                // vllm defaults this to 16
                sp_kwargs.push(("max_tokens", py_max_tokens));
            }
            let sp_kwargs = sp_kwargs.into_py_dict(py).unwrap();
            let sampling_params = py_imports
                .sample_params_type
                .call(py, (), Some(&sp_kwargs))
                .unwrap();
            let py_request_id: PyObject = PyString::new(py, &request_id).into();
            (py_request_id, sampling_params)
        });

        let pickled_req: Vec<u8> = Python::with_gil(|py| {
            let token_prompt_kwargs = [("prompt_token_ids", token_ids.clone())]
                .into_py_dict(py)
                .unwrap();
            let prompt_obj = py_imports
                .tokens_prompt_type
                .call(py, (), Some(&token_prompt_kwargs))
                .unwrap();

            let rpc_kwargs = [
                ("prompt", prompt_obj),
                ("params", sampling_params.clone()),
                ("request_id", py_request_id.clone()),
            ]
            .into_py_dict(py)
            .unwrap();
            let req = py_imports.rpc_type.call(py, (), Some(&rpc_kwargs)).unwrap();

            let pickle_dumps = py_imports.pickle_module.getattr(py, "dumps").unwrap();
            pickle_dumps.call1(py, (req,)).unwrap().extract(py).unwrap()
        });

        let new_active_request = ActiveRequest {
            tx: work_request.response_channel,
            num_output_tokens_so_far: 0,
        };
        active_requests
            .lock()
            .await
            .insert(request_id, new_active_request);

        if let Err(err) = input_socket.send(vec![pickled_req].into()).await {
            tracing::error!("Error sending new request to vllm over zmq: {err}");
        }
    }
}

/// Read from vllm's output zmq socket, find which request it is for and forward over that channel.
async fn output_loop(
    cancel_token: CancellationToken,
    py_imports: Arc<Imports>,
    mut output_socket: async_zmq::Pull,
    active_requests: Arc<tokio::sync::Mutex<HashMap<RequestID, ActiveRequest>>>,
) {
    loop {
        let mut bb = tokio::select! {
            _ = cancel_token.cancelled() => {
                tracing::trace!("VllmWorker.output_loop exit");
                break;
            }
            from_vllm = output_socket.next() => {
                match from_vllm {
                    Some(Ok(b)) => b,
                    Some(Err(err)) => {
                        tracing::error!("Error reading from vllm zmq output: {err}");
                        continue; // hope lives eternal
                    }
                    None => {
                        tracing::debug!("zmq output socket closed");
                        break;
                    }
                }
            }
        };

        let frame = bb.remove(0);
        let mut reqs_out: Vec<RequestOutput> = Python::with_gil(|py| {
            let pickle_loads = py_imports.pickle_module.getattr(py, "loads").unwrap();
            let frame_bytes = PyBytes::new(py, &frame);
            pickle_loads
                .call1(py, (frame_bytes,))
                .unwrap()
                .extract(py)
                .unwrap()
        });
        if reqs_out.is_empty() {
            tracing::debug!("Received message from vllm with no content");
            continue;
        }
        let req_out = reqs_out.remove(0);

        if req_out.finished {
            // The last token is the eos_token, don't forward it
            // TODO: Look at req_out.finish_reason (Option<String>) and set out correctly.
            let out = Annotated::from_data(LLMEngineOutput::stop());
            let maybe_active = active_requests.lock().await.remove(&req_out.request_id);
            match maybe_active {
                Some(active) => {
                    let _ = active.tx.send(out).await;
                }
                None => {
                    tracing::warn!(
                        req_out.request_id,
                        "Missing active request to notify of stop"
                    );
                }
            }
            continue;
        }

        for vllm_output in req_out.outputs.into_iter() {
            let next_total_toks = vllm_output.token_ids.len();

            match active_requests.lock().await.get_mut(&req_out.request_id) {
                Some(active) => {
                    let out = from_vllm(vllm_output, active.num_output_tokens_so_far);
                    active.num_output_tokens_so_far = next_total_toks;
                    let _ = active.tx.send(Annotated::from_data(out)).await;
                }
                None => {
                    tracing::warn!(req_out.request_id, "Missing active request");
                }
            }
        }
    }
}

impl VllmWorker {
    /// Send a request to vllm
    pub async fn enqueue_request(&self, r: WorkRequest) -> Result<(), SendError<WorkRequest>> {
        self.tx.send(r).await
    }

    /// Get the vllm sub-process handle, so we can await it and prevent it going zombie.
    pub fn take_vllm_handle(&mut self) -> JoinHandle<()> {
        self.vllm.take().unwrap()
    }
}
