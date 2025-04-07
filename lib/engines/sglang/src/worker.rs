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

use std::{
    collections::HashMap,
    env, fmt,
    os::fd::{FromRawFd as _, RawFd},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
    vec::IntoIter,
};

use anyhow::Context as _;
use async_zmq::{SinkExt, StreamExt};
use libc::c_int;
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{IntoPyDict, PyBytes, PyString},
};
use regex::Regex;
use tokio::sync::mpsc::Sender;
use tokio::{io::AsyncBufReadExt, sync::mpsc::error::SendError};
use tokio::{io::AsyncReadExt as _, task::JoinHandle};

use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::runtime::CancellationToken;

use dynamo_llm::engines::MultiNodeConfig;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_llm::protocols::common::FinishReason;
use dynamo_llm::protocols::TokenIdType;

use crate::MultiGPUConfig;

/// Wait this long for the sglang sub-process to stop after we send it a KILL
const SGLANG_STOP_TIMEOUT: Duration = Duration::from_millis(1500);

/// Match sglang python log entries, e.g "[2025-01-30 11:23:16] Some text we want"
const SGLANG_LOG_RE: &str =
    r"(?<timestamp>\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] )?(?<message>.*)";

/// Identify sglang log entries with this prefix
const LOG_PREFIX: &str = "SGLANG";

/// Part of what sglang sends us over it's pipe when it's ready
const READY_BYTES: [u8; 5] = [b'r', b'e', b'a', b'd', b'y'];

type RequestID = String;

pub struct SgLangWorker {
    /// How we receive work requests
    tx: Sender<WorkRequest>,

    /// Handle of the task that reads from `tx` and forwards those requests over zmq to vllm
    _input_loop: JoinHandle<()>,

    /// Handle of the task that reads sglang's responses from zmq and dispatches them to the correct
    /// active request.
    _output_loop: JoinHandle<()>,

    /// Handle of the vllm background process
    sglang: Option<JoinHandle<()>>,

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
    num_output_tokens_so_far: Option<i32>,
}

/// Python imports
struct Imports {
    pickle_module: PyObject,
    sampling_params_type: PyObject,
    rpc_type: PyObject,
}

/// All the zmq sockets we used. This object only used to passing them around to avoid large
/// tuples.
struct Sockets {
    #[allow(dead_code)]
    context: async_zmq::Context, // we have to keep this alive

    // Requests from us to the sglang engine
    // scheduler_input_ipc_name,
    input: async_zmq::Push<IntoIter<Vec<u8>>, Vec<u8>>,

    // Responses from the sglang engine back to us
    // tokenizer_ipc_name
    output: async_zmq::Pull,
}

/// What sglang sends us.
#[allow(dead_code)]
#[derive(FromPyObject, Debug)]
pub struct BatchTokenIDOut {
    // The request id
    rids: Vec<String>,

    // The finish reason
    // sglang implements finish reason as subclasses of BaseFinishReason
    //  e.g. `class FINISH_LENGTH(BaseFinishReason):` and lots of others
    finished_reasons: Vec<Option<SgLangFinishReason>>,

    // For incremental decoding
    // The version id to sync decode status with in detokenizer_manager
    vids: Vec<i32>,
    decoded_texts: Vec<String>,
    decode_ids: Vec<Vec<u32>>,
    read_offsets: Vec<i32>,
    // Only used when `--skip-tokenizer-init` is on
    output_ids: Option<Vec<i32>>,
    // Detokenization configs
    skip_special_tokens: Vec<bool>,
    spaces_between_special_tokens: Vec<bool>,
    no_stop_trim: Vec<bool>,

    // Token counts
    prompt_tokens: Vec<i32>,
    completion_tokens: Vec<i32>,
    cached_tokens: Vec<i32>,
    spec_verify_ct: Vec<i32>,

    // Logprobs
    input_token_logprobs_val: Option<Vec<f64>>,
    input_token_logprobs_idx: Option<Vec<i32>>,
    output_token_logprobs_val: Option<Vec<f64>>,
    output_token_logprobs_idx: Option<Vec<i32>>,
    // These in Python are all `List[List]`, so guess
    input_top_logprobs_val: Option<Vec<Vec<f64>>>,
    input_top_logprobs_idx: Option<Vec<Vec<i32>>>,
    output_top_logprobs_val: Option<Vec<Vec<f64>>>,
    output_top_logprobs_idx: Option<Vec<Vec<i32>>>,
}

#[derive(Debug, Copy, Clone)]
enum SgLangFinishReason {
    Matched,
    Length,
    Abort,
}

impl fmt::Display for SgLangFinishReason {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SgLangFinishReason::Matched => write!(f, "Finished due to a successful match"),
            SgLangFinishReason::Length => {
                write!(f, "Finished due to reaching the specified length")
            }
            SgLangFinishReason::Abort => write!(f, "Operation was aborted"),
        }
    }
}

impl<'py> FromPyObject<'py> for SgLangFinishReason {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        // The object we have is a subclass of sglang's BaseFinishReason, one subclass
        // per finish reason. I don't know how to identify the class, but if we force
        // it to a string I _think_ it ends up calling `json_str` in the subclass.
        // Also the string uses single quotes in the JSON, I don't know why.
        let json_str = obj.str()?.to_string().replace("'", "\"");
        let as_map: HashMap<String, serde_json::Value> =
            serde_json::from_str(&json_str).map_err(|err| {
                tracing::error!("SgLangFinishReason JSON convert err: {err}. JSON: {json_str}");
                PyTypeError::new_err(format!("serde_json err: {err}. JSON: {json_str}"))
            })?;
        let Some(type_serde) = as_map.get("type") else {
            return Err(PyTypeError::new_err("Finish reason missing 'type' JSON field. See sglang's schedule_batch.py BaseFinishReason"));
        };
        let Some(type_str) = type_serde.as_str() else {
            return Err(PyTypeError::new_err("Finish reason 'type' JSON field is not a string. See sglang's schedule_batch.py BaseFinishReason"));
        };
        match type_str {
            "stop" => Ok(SgLangFinishReason::Matched),
            "length" => Ok(SgLangFinishReason::Length),
            "abort" => Ok(SgLangFinishReason::Abort),
            x => {
                tracing::warn!("Unknown sglang BaseFinishReason type '{x}'. Using Abort instead.");
                Ok(SgLangFinishReason::Abort)
            }
        }
    }
}

impl From<SgLangFinishReason> for FinishReason {
    fn from(sfr: SgLangFinishReason) -> Self {
        use SgLangFinishReason::*;
        match sfr {
            Matched => FinishReason::Stop,
            Length => FinishReason::Length,
            Abort => FinishReason::Cancelled, // or FinishReason::Error ?
        }
    }
}

/* What we send to sglang
class TokenizedGenerateReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The image inputs
    image_inputs: dict
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # Whether to stream output
    stream: bool

    # LoRA related
    lora_path: Optional[str] = None  # None means just use the base model
    # The input embeds
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None

    # Session info for continual prompting
    session_params: Optional[SessionParams] = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[str] = None

class SamplingParams:
    max_new_tokens: int = 128,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    min_new_tokens: int = 0,
    spaces_between_special_tokens: bool = True,
    n: int = 1,
    json_schema: Optional[str] = None,
    regex: Optional[str] = None,
    ebnf: Optional[str] = None,
    no_stop_trim: bool = False,
    ignore_eos: bool = False,
    skip_special_tokens: bool = True,
    custom_params: Optional[Dict[str, Any]] = None,
*/

/// Main entry point
pub async fn start(
    cancel_token: CancellationToken,
    sock_code: &str,
    model_path: &Path,
    node_conf: MultiNodeConfig,
    tp_size: u32,
    base_gpu_id: u32,
    extra_engine_args: Option<PathBuf>,
) -> anyhow::Result<SgLangWorker> {
    pyo3::prepare_freethreaded_python();
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let _ = Python::with_gil(|py| crate::fix_venv(venv, py));
    }

    let Sockets {
        context,
        input,
        output,
    } = zmq_sockets(sock_code)?;
    let py_imports = Arc::new(python_imports());

    if tp_size < node_conf.num_nodes {
        anyhow::bail!(
            "Need at least as many GPUs as nodes. Pass --tensor-parallel-size >= --num-nodes."
        );
    }
    let tp_size_per_node = tp_size / node_conf.num_nodes;
    let tp_rank_start = tp_size_per_node * node_conf.node_rank;
    let tp_rank_end = tp_size_per_node * (node_conf.node_rank + 1);

    // Start all the sglang workers. They communicate amongst themselves using torch distributed
    // and nccl. They must all start at once.
    let mut sglang_join_handle = None;
    let mut process_group = Vec::with_capacity(tp_size as usize);
    for tp_rank in tp_rank_start..tp_rank_end {
        let gpu_id = base_gpu_id + tp_rank % tp_size_per_node;
        let gpu_conf = MultiGPUConfig {
            tp_size,
            tp_rank,
            gpu_id,
        };
        let (sglang_process, ready_fd) = start_sglang(
            model_path,
            node_conf.clone(),
            gpu_conf,
            extra_engine_args.clone(),
        )
        .await?;
        process_group.push((tp_rank, ready_fd));
        let watcher_join_handle = watch_sglang(cancel_token.clone(), sglang_process);
        // TODO: Do we want to hold on to this?
        // Do we need it for the other sub-processes?
        if sglang_join_handle.is_none() {
            sglang_join_handle = Some(watcher_join_handle);
        }
    }

    for (tp_rank, ready_fd) in process_group.into_iter() {
        wait_for_sglang(tp_rank, ready_fd).await?;
    }

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

    Ok(SgLangWorker {
        tx,
        zmq_context: context,
        _input_loop: input_loop_handle,
        _output_loop: output_loop_handle,
        sglang: sglang_join_handle,
        _active_requests: active_requests,
    })
}

/// Import all the python packages we'll need.
fn python_imports() -> Imports {
    Python::with_gil(|py| {
        let pickle_module: PyObject = match py.import("pickle") {
            Ok(m) => m.into(),
            Err(err) => {
                // There is no sglang without python
                panic!("Failed to import python 'pickle' module. Is Python installed? {err}");
            }
        };

        // This one is a sanity check
        if let Err(err) = py.import("sglang") {
            panic!("Failed to import python 'sglang' module. Are we running in the correct venv? {err}");
        };
        let mod_iostruct: PyObject = match py.import("sglang.srt.managers.io_struct") {
            Ok(m) => m.into(),
            Err(err) => {
                panic!("Failed to import sglang.srt.managers.io_struct. Did sglang change? {err}");
            }
        };
        let rpc_type = mod_iostruct
            .getattr(py, "TokenizedGenerateReqInput")
            .unwrap();

        let mod_sampling: PyObject = match py.import("sglang.srt.sampling.sampling_params") {
            Ok(m) => m.into(),
            Err(err) => {
                panic!(
                    "Failed to import sglang.srt.sampling.sampling_params. Did sglang change? {err}"
                );
            }
        };
        let sampling_params_type: PyObject = mod_sampling.getattr(py, "SamplingParams").unwrap();

        Imports {
            pickle_module,
            sampling_params_type,
            rpc_type,
        }
    })
}

/// Create all the zmq sockets we're going to use.
fn zmq_sockets(sock_code: &str) -> anyhow::Result<Sockets> {
    let zmq_context = async_zmq::Context::new();

    // Scheduler (rank 0) to receive inputs from us
    let input = async_zmq::push(&format!("ipc:///tmp/{sock_code}_input_socket"))?
        .with_context(&zmq_context)
        .bind()?;

    // Use to receive replies from scheduler.
    let output = async_zmq::pull(&format!("ipc:///tmp/{sock_code}_output_socket"))?
        .with_context(&zmq_context)
        .bind()?;

    Ok(Sockets {
        context: zmq_context,
        input,
        output,
    })
}

/// Start the python sub-process and wait for it to be ready
async fn start_sglang(
    model_path: &Path,
    node_conf: MultiNodeConfig,
    gpu_conf: MultiGPUConfig,
    extra_engine_args: Option<PathBuf>,
) -> anyhow::Result<(tokio::process::Child, RawFd)> {
    // This pipe is how sglang tells us it's ready
    let mut pipe_fds: [libc::c_int; 2] = [-1, -1];
    unsafe {
        // Seems to be OK without libc::O_NONBLOCK
        let err = libc::pipe(pipe_fds.as_mut_ptr() as *mut c_int);
        if err != 0 {
            anyhow::bail!("libc::pipe error {err}");
        }
    }
    let sglang_says_hello = pipe_fds[1] as RawFd;
    let tp_rank = gpu_conf.tp_rank;
    let gpu_id = gpu_conf.gpu_id;
    let mut args = vec![
        format!("--internal-sglang-process={sglang_says_hello},{tp_rank},{gpu_id}"),
        format!("--model-path={}", model_path.display()),
        format!("--tensor-parallel-size={}", gpu_conf.tp_size),
        format!("--num-nodes={}", node_conf.num_nodes),
        format!("--node-rank={}", node_conf.node_rank),
    ];
    if let Some(extra_engine_args) = extra_engine_args {
        args.push(format!(
            "--extra-engine-args={}",
            extra_engine_args.display()
        ));
    };
    if node_conf.num_nodes > 1 {
        if node_conf.leader_addr.is_empty() {
            anyhow::bail!("Missing --leader-addr for multi-node");
        }
        args.push(format!("--leader-addr={}", node_conf.leader_addr));
    }
    let self_path = std::env::current_exe()?;
    let mut proc = tokio::process::Command::new(self_path)
        .args(args)
        .kill_on_drop(false)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = tokio::io::BufReader::new(proc.stdout.take().unwrap());
    let stderr = tokio::io::BufReader::new(proc.stderr.take().unwrap());

    // Log sglang's stdout
    // sglang has (almost?) no output on stdout
    tokio::spawn(async move {
        let mut lines = stdout.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!("{LOG_PREFIX}{tp_rank} {line}");
        }
    });

    // Log sglang's stderr
    tokio::spawn(async move {
        // Remove extra date/time entries from stderr, and print with prefix
        let line_re = Regex::new(SGLANG_LOG_RE).unwrap();
        let mut lines = stderr.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(caps) = line_re.captures(&line) {
                match caps.name("timestamp") {
                    Some(_) => {
                        // Skip Python's date/time. Should be normal log
                        tracing::debug!("{LOG_PREFIX}{tp_rank} {}", &caps["message"]);
                    }
                    None => {
                        // No date/time. Usually errors
                        tracing::warn!("{LOG_PREFIX}{tp_rank} {line}");
                    }
                }
            }
        }
    });

    let ready_fd = pipe_fds[0] as RawFd;
    Ok((proc, ready_fd))
}

async fn wait_for_sglang(tp_rank: u32, pipe_fd: RawFd) -> anyhow::Result<()> {
    tracing::info!("Waiting for sglang{tp_rank} to signal that it's ready");

    let mut sglang_ready = unsafe { tokio::fs::File::from_raw_fd(pipe_fd) };
    let mut buf = [0u8; 128]; // Some pickled JSON, about 90 bytes
    let len_read = sglang_ready
        .read(&mut buf)
        .await
        .with_context(|| format!("Failed reading from Rust side of sglang pipe, fd {pipe_fd}",))?;
    let received_bytes = &buf[..len_read];
    /* received_bytes is pickled JSON:
    {
        "status": "ready",
        "max_total_num_tokens": scheduler.max_total_num_tokens,
        "max_req_input_len": scheduler.max_req_input_len,
    }
    We could unpickle it, but this is faster.
     */
    if !received_bytes
        .windows(READY_BYTES.len())
        .any(|candidate| candidate == READY_BYTES)
    {
        anyhow::bail!("Expected sglang pipe to signal ready, but did not contain 'ready' bytes");
    }

    // TODO: warm up the engine

    tracing::info!("sglang{tp_rank} is ready");
    Ok(())
}

// Stop the sglang process when we stop, and prevent it going zombie.
fn watch_sglang(
    cancel_token: CancellationToken,
    mut sglang_process: tokio::process::Child,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        cancel_token.cancelled().await;
        tokio::select! {
            _ = sglang_process.wait() => {
                return;
            },
            _ = tokio::time::sleep(SGLANG_STOP_TIMEOUT) => { }
        }
        if let Err(err) = sglang_process.start_kill() {
            tracing::error!("Failing killing sglang subprocess: {err}");
            return;
        }
        tokio::select! {
            _ = sglang_process.wait() => { },
            _ = tokio::time::sleep(SGLANG_STOP_TIMEOUT) => {
                tracing::warn!("Timeout waiting for sglang sub-process to stop after kill");
            }
        }
    })
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
                tracing::trace!("SgLangWorker.main_loop exit");
                break;
            }
            req = rx.recv() => {
                match req {
                    Some(req) => req,
                    None => {
                        tracing::trace!("SgLangWorker input_loop socket closed");
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
        tracing::trace!("Received work request: {request_id}");

        // Parts that don't change
        let (py_request_id, sampling_params) = Python::with_gil(|py| {
            let py_temp: PyObject = temperature.into_pyobject(py).unwrap().into();
            let mut sp_kwargs = vec![("temperature", py_temp)];
            if let Some(max_tokens) = work_request.request.stop_conditions.max_tokens {
                let py_max_tokens: PyObject = max_tokens.into_pyobject(py).unwrap().into();
                // sglang defaults this to 128
                sp_kwargs.push(("max_new_tokens", py_max_tokens));
            }
            let sp_kwargs = sp_kwargs.into_py_dict(py).unwrap();
            let sampling_params = py_imports
                .sampling_params_type
                .call(py, (), Some(&sp_kwargs))
                .unwrap();
            sampling_params
                .getattr(py, "normalize")
                .unwrap()
                .call1(py, (py.None(),))
                .unwrap();
            let py_request_id: PyObject = PyString::new(py, &request_id).into();
            (py_request_id, sampling_params)
        });

        let pickled_req: Vec<u8> = Python::with_gil(|py| {
            let input_text: PyObject = "".into_pyobject(py).unwrap().into();
            let input_ids: PyObject = token_ids.into_pyobject(py).unwrap().into();
            let image_inputs: PyObject = py.None();
            let return_logprob: PyObject = false.into_pyobject(py).unwrap().to_owned().into();
            let logprob_start_len: PyObject = 0u32.into_pyobject(py).unwrap().into();
            let top_logprobs_num: PyObject = 0u32.into_pyobject(py).unwrap().into();
            let stream: PyObject = true.into_pyobject(py).unwrap().to_owned().into();
            let rpc_pos_args = (
                py_request_id,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                stream,
            );
            //let rpc_kwargs = [].into_py_dict(py).unwrap();
            let req = py_imports
                .rpc_type
                .call(py, rpc_pos_args, None) // Some(&rpc_kwargs))
                .unwrap();

            let pickle_dumps = py_imports.pickle_module.getattr(py, "dumps").unwrap();
            pickle_dumps.call1(py, (req,)).unwrap().extract(py).unwrap()
        });
        let new_active_request = ActiveRequest {
            tx: work_request.response_channel,
            num_output_tokens_so_far: None,
        };
        active_requests
            .lock()
            .await
            .insert(request_id, new_active_request);

        if let Err(err) = input_socket.send(pickled_req.into()).await {
            tracing::error!("Error sending new request to sglang over zmq: {err}");
        }
    }
}

/// Read from sglang's output zmq socket, find which request it is for and forward over that channel.
async fn output_loop(
    cancel_token: CancellationToken,
    py_imports: Arc<Imports>,
    mut output_socket: async_zmq::Pull,
    active_requests: Arc<tokio::sync::Mutex<HashMap<RequestID, ActiveRequest>>>,
) {
    loop {
        let maybe_bb = tokio::select! {
            _ = cancel_token.cancelled() => {
                break;
            }
            maybe_bb = output_socket.next() => {
                maybe_bb
            }
        };
        let mut bb = match maybe_bb {
            Some(Ok(b)) => b,
            Some(Err(err)) => {
                tracing::error!("Error reading from sglang zmq output: {err}");
                continue; // hope live eternal
            }
            None => {
                tracing::debug!("zmq output socket closed");
                break;
            }
        };

        let frame = bb.remove(0);
        let req_out: BatchTokenIDOut = Python::with_gil(|py| {
            let pickle_loads = py_imports.pickle_module.getattr(py, "loads").unwrap();
            let frame_bytes = PyBytes::new(py, &frame);
            let pyobj = pickle_loads.call1(py, (frame_bytes,)).unwrap();
            pyobj.extract(py).unwrap()
        });
        tracing::trace!(?req_out, "from sglang");

        let mut remove_after = vec![];
        for (idx, req_id) in req_out.rids.into_iter().enumerate() {
            let next_total_toks = req_out.decode_ids[idx].len() as i32;

            match active_requests.lock().await.get_mut(&req_id) {
                Some(active) => {
                    let previous_total_toks = active
                        .num_output_tokens_so_far
                        .unwrap_or(req_out.read_offsets[idx])
                        as usize;
                    let sglang_finish_reason = req_out.finished_reasons[idx];
                    let token_ids: Vec<TokenIdType> = if sglang_finish_reason.is_none() {
                        req_out.decode_ids[idx][previous_total_toks..].into()
                    } else {
                        tracing::trace!(
                            req_id,
                            ?sglang_finish_reason,
                            "finished with finish reason"
                        );
                        // Request is over, sglang says so.
                        // The last token is the eos_token, don't forward it
                        remove_after.push(req_id.clone());
                        vec![]
                    };
                    let out = LLMEngineOutput {
                        token_ids,
                        tokens: None,
                        text: None,
                        cum_log_probs: None,
                        log_probs: None,
                        finish_reason: sglang_finish_reason.map(|x| x.into()),
                    };
                    active.num_output_tokens_so_far = Some(next_total_toks);
                    let _ = active.tx.send(Annotated::from_data(out)).await;
                }
                None => {
                    // sglang sends the finish response twice, I don't know why
                    // so only log if it isn't a finished request
                    if req_out.finished_reasons[idx].is_none() {
                        tracing::warn!(req_id, "Missing active request");
                    }
                }
            }
        }
        for req_id in remove_after {
            let _ = active_requests.lock().await.remove(&req_id);
        }
    }
}

impl SgLangWorker {
    /// Send a request to sglang
    pub async fn enqueue_request(&self, r: WorkRequest) -> Result<(), SendError<WorkRequest>> {
        self.tx.send(r).await
    }

    /// Get the sglang sub-process handle, so we can await it and prevent it going zombie.
    pub fn take_sglang_handle(&mut self) -> JoinHandle<()> {
        self.sglang.take().unwrap()
    }
}
