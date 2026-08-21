// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use dynamo_runtime::protocols::annotated::Annotated;
use serde::Deserialize;
use serde_json::{Map, Value, json};

#[cfg(not(test))]
use pyo3::exceptions::PyValueError;
#[cfg(not(test))]
use pyo3::prelude::*;
#[cfg(not(test))]
use pyo3::types::PyModule;
#[cfg(not(test))]
use pythonize::depythonize;

#[cfg(not(test))]
pub fn add_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RequestKey>()?;
    module.add_class::<NativeResponseEgress>()?;
    Ok(())
}

pub(crate) trait ResponseFrameSink: Send + Sync {
    fn send(
        &self,
        frame: Annotated<Value>,
        cancelled: &AtomicBool,
        shutting_down: &AtomicBool,
        send_gate: &Mutex<()>,
    ) -> Result<(), FrameSendError>;
    fn close(&self);
    fn cancel(&self);
    fn close_with_error(&self, message: String) -> bool;
    fn shutdown(&self);
}

#[derive(Debug)]
pub(crate) enum FrameSendError {
    Stopped,
    #[cfg(not(test))]
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[cfg_attr(not(test), pyclass(frozen))]
pub struct RequestKey {
    client_id: u64,
    generation: u64,
}

#[cfg(not(test))]
#[pymethods]
impl RequestKey {
    #[getter]
    #[pyo3(name = "client_id")]
    fn py_client_id(&self) -> u64 {
        self.client_id
    }

    #[getter]
    #[pyo3(name = "generation")]
    fn py_generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Deserialize)]
struct ChoiceDelta {
    index: usize,
    #[serde(default)]
    new_token_ids: Vec<u32>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseEvent {
    client_id: u64,
    generation: u64,
    sequence: u64,
    #[serde(default)]
    outputs: Vec<ChoiceDelta>,
    #[serde(default)]
    is_final: bool,
    #[serde(default)]
    error_msg: Option<String>,
}

#[cfg(test)]
impl ResponseEvent {
    fn tokens_for(
        request: RequestKey,
        sequence: u64,
        new_token_ids: Vec<Vec<u32>>,
        is_final: bool,
    ) -> Self {
        Self {
            client_id: request.client_id,
            generation: request.generation,
            sequence,
            outputs: new_token_ids
                .into_iter()
                .enumerate()
                .map(|(index, new_token_ids)| ChoiceDelta {
                    index,
                    new_token_ids,
                    finish_reason: None,
                    stop_reason: None,
                })
                .collect(),
            is_final,
            error_msg: None,
        }
    }

    fn error_for(request: RequestKey, sequence: u64, message: &str) -> Self {
        Self {
            client_id: request.client_id,
            generation: request.generation,
            sequence,
            outputs: Vec::new(),
            is_final: true,
            error_msg: Some(message.to_string()),
        }
    }
}

#[derive(Debug, Default)]
struct ChoiceState {
    token_ids: Vec<u32>,
    emitted: usize,
    finish_reason: Option<String>,
    stop_reason: Option<String>,
}

#[derive(Debug)]
struct RequestStreamState {
    prompt_tokens: usize,
    choices: Vec<ChoiceState>,
}

impl RequestStreamState {
    fn new(prompt_tokens: usize, num_choices: usize) -> Self {
        Self {
            prompt_tokens,
            choices: (0..num_choices).map(|_| ChoiceState::default()).collect(),
        }
    }

    fn apply(&mut self, response: ResponseEvent) -> Result<Vec<Value>, String> {
        let mut seen = vec![false; self.choices.len()];
        for output in &response.outputs {
            if output.index >= self.choices.len() {
                return Err(format!(
                    "choice index {} is outside registered range 0..{}",
                    output.index,
                    self.choices.len()
                ));
            }
            if std::mem::replace(&mut seen[output.index], true) {
                return Err(format!(
                    "choice index {} appears more than once",
                    output.index
                ));
            }
        }

        for output in &response.outputs {
            let choice = &mut self.choices[output.index];
            choice.token_ids.extend_from_slice(&output.new_token_ids);
            if output.finish_reason.is_some() {
                choice.finish_reason.clone_from(&output.finish_reason);
            }
            if output.stop_reason.is_some() {
                choice.stop_reason.clone_from(&output.stop_reason);
            }
        }

        let completion_tokens = self
            .choices
            .iter()
            .map(|choice| choice.token_ids.len())
            .sum::<usize>();

        Ok(response
            .outputs
            .into_iter()
            .map(|output| {
                let choice = &mut self.choices[output.index];
                let mut frame = Map::new();
                frame.insert(
                    "token_ids".to_string(),
                    json!(choice.token_ids[choice.emitted..]),
                );
                frame.insert("index".to_string(), json!(output.index));
                choice.emitted = choice.token_ids.len();

                let terminal = choice.finish_reason.is_some() || response.is_final;
                if terminal {
                    frame.insert(
                        "finish_reason".to_string(),
                        json!(choice.finish_reason.as_deref().unwrap_or("unknown")),
                    );
                }
                if let Some(stop_reason) = choice.stop_reason.as_deref() {
                    frame.insert("stop_reason".to_string(), json!(stop_reason));
                }
                if terminal {
                    frame.insert(
                        "completion_usage".to_string(),
                        json!({
                            "prompt_tokens": self.prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": self.prompt_tokens + completion_tokens,
                            "prompt_tokens_details": null
                        }),
                    );
                }
                Value::Object(frame)
            })
            .collect())
    }
}

struct RegisteredRequest {
    key: RequestKey,
    next_sequence: AtomicU64,
    response_state: Mutex<RequestStreamState>,
    sink: Arc<dyn ResponseFrameSink>,
    cancelled: AtomicBool,
    send_gate: Mutex<()>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchOutcome {
    pub(crate) completed_requests: Vec<RequestKey>,
    pub(crate) responses_processed: usize,
    pub(crate) responses_dropped: usize,
    pub(crate) frames_sent: usize,
}

#[derive(Debug)]
struct IndexedOutcome {
    ordinal: usize,
    request_key: RequestKey,
    completed: bool,
    processed: bool,
    frames_sent: usize,
}

struct ProcessorShared {
    requests: Mutex<HashMap<u64, Arc<RegisteredRequest>>>,
    next_generation: AtomicU64,
    benchmark_response_work: Duration,
    shutting_down: AtomicBool,
    responses_processed: AtomicUsize,
    responses_dropped: AtomicUsize,
    frames_sent: AtomicUsize,
}

impl Default for ProcessorShared {
    fn default() -> Self {
        Self {
            requests: Mutex::new(HashMap::new()),
            next_generation: AtomicU64::new(1),
            benchmark_response_work: Duration::ZERO,
            shutting_down: AtomicBool::new(false),
            responses_processed: AtomicUsize::new(0),
            responses_dropped: AtomicUsize::new(0),
            frames_sent: AtomicUsize::new(0),
        }
    }
}
impl ProcessorShared {
    fn terminate_protocol_error(
        &self,
        ordinal: usize,
        request: Arc<RegisteredRequest>,
        message: String,
    ) -> IndexedOutcome {
        self.responses_dropped.fetch_add(1, Ordering::Relaxed);
        let frames_sent = usize::from(request.sink.close_with_error(message));
        self.frames_sent.fetch_add(frames_sent, Ordering::Relaxed);
        let completed = !request.cancelled.load(Ordering::Acquire);
        self.remove(request.key, &request);
        IndexedOutcome {
            ordinal,
            request_key: request.key,
            completed,
            processed: false,
            frames_sent,
        }
    }

    fn process_response(&self, ordinal: usize, response: ResponseEvent) -> IndexedOutcome {
        let client_id = response.client_id;
        let request_key = RequestKey {
            client_id,
            generation: response.generation,
        };
        let sequence = response.sequence;
        let request = self
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&client_id)
            .cloned();
        let Some(request) = request else {
            self.responses_dropped.fetch_add(1, Ordering::Relaxed);
            return IndexedOutcome {
                ordinal,
                request_key,
                completed: false,
                processed: false,
                frames_sent: 0,
            };
        };

        if request.key != request_key {
            self.responses_dropped.fetch_add(1, Ordering::Relaxed);
            return IndexedOutcome {
                ordinal,
                request_key,
                completed: false,
                processed: false,
                frames_sent: 0,
            };
        }

        let expected_sequence = request.next_sequence.load(Ordering::Relaxed);
        if sequence < expected_sequence {
            self.responses_dropped.fetch_add(1, Ordering::Relaxed);
            return IndexedOutcome {
                ordinal,
                request_key,
                completed: false,
                processed: false,
                frames_sent: 0,
            };
        }
        if sequence > expected_sequence {
            return self.terminate_protocol_error(
                ordinal,
                request,
                format!(
                    "response sequence gap for client {client_id}: expected sequence {expected_sequence}, received {sequence}"
                ),
            );
        }
        if sequence == u64::MAX && !response.is_final && response.error_msg.is_none() {
            return self.terminate_protocol_error(
                ordinal,
                request,
                format!("response sequence space exhausted for client {client_id}"),
            );
        }

        let work_started = Instant::now();
        self.responses_processed.fetch_add(1, Ordering::Relaxed);
        if request.cancelled.load(Ordering::Acquire) || self.shutting_down.load(Ordering::Acquire) {
            return IndexedOutcome {
                ordinal,
                request_key,
                completed: false,
                processed: true,
                frames_sent: 0,
            };
        }

        let mut terminal = response.is_final || response.error_msg.is_some();
        let mut frames_sent = 0;
        let (frames, close_error) = if let Some(error) = response.error_msg {
            (Vec::new(), Some(error))
        } else {
            let frames = {
                let mut state = request
                    .response_state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if request.cancelled.load(Ordering::Acquire)
                    || self.shutting_down.load(Ordering::Acquire)
                {
                    return IndexedOutcome {
                        ordinal,
                        request_key,
                        completed: false,
                        processed: true,
                        frames_sent: 0,
                    };
                }
                state.apply(response)
            };
            let frames = match frames {
                Ok(frames) => (frames, None),
                Err(error) => {
                    terminal = true;
                    (Vec::new(), Some(error))
                }
            };
            frames
        };

        pad_response_work(
            work_started,
            self.benchmark_response_work,
            &request.cancelled,
            &self.shutting_down,
        );
        if request.cancelled.load(Ordering::Acquire) || self.shutting_down.load(Ordering::Acquire) {
            return IndexedOutcome {
                ordinal,
                request_key,
                completed: false,
                processed: true,
                frames_sent: 0,
            };
        }

        if let Some(error) = close_error {
            if request.sink.close_with_error(error) {
                frames_sent += 1;
                self.frames_sent.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            let mut sink_failed = false;
            for frame in frames {
                match request.sink.send(
                    Annotated::from_data(frame),
                    &request.cancelled,
                    &self.shutting_down,
                    &request.send_gate,
                ) {
                    Ok(()) => {
                        frames_sent += 1;
                        self.frames_sent.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(FrameSendError::Stopped) => {
                        terminal = false;
                        sink_failed = true;
                        break;
                    }
                    #[cfg(not(test))]
                    Err(FrameSendError::Failed(error)) => {
                        tracing::debug!(client_id, %error, "native response sink stopped");
                        request.sink.close();
                        sink_failed = true;
                        terminal = true;
                        break;
                    }
                }
            }
            if terminal && !sink_failed {
                request.sink.close();
            }
        }

        let completed = terminal && !request.cancelled.load(Ordering::Acquire);
        if terminal {
            self.remove(request_key, &request);
        } else {
            request.next_sequence.store(
                sequence.checked_add(1).expect("sequence validated"),
                Ordering::Relaxed,
            );
        }

        IndexedOutcome {
            ordinal,
            request_key,
            completed,
            processed: true,
            frames_sent,
        }
    }

    fn remove(&self, key: RequestKey, request: &Arc<RegisteredRequest>) {
        let mut requests = self
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests
            .get(&key.client_id)
            .is_some_and(|current| current.key == key && Arc::ptr_eq(current, request))
        {
            requests.remove(&key.client_id);
        }
    }

    fn close_all(&self) {
        let requests = self
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain()
            .map(|(_, request)| request)
            .collect::<Vec<_>>();
        for request in requests {
            let _send_guard = request
                .send_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            request.cancelled.store(true, Ordering::Release);
            request.sink.shutdown();
        }
    }
}

struct IndexedResponse {
    ordinal: usize,
    response: ResponseEvent,
}

struct ShardCommand {
    responses: Vec<IndexedResponse>,
    reply: SyncSender<Vec<IndexedOutcome>>,
}

pub(crate) struct ShardedResponseEgress {
    shared: Arc<ProcessorShared>,
    shard_count: usize,
    dispatch: Mutex<()>,
    senders: Mutex<Option<Vec<SyncSender<ShardCommand>>>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl ShardedResponseEgress {
    pub(crate) fn new(shard_count: usize, queue_depth: usize) -> Result<Self, String> {
        Self::new_with_benchmark_work(shard_count, queue_depth, 0.0)
    }

    fn new_with_benchmark_work(
        shard_count: usize,
        queue_depth: usize,
        benchmark_response_work_us: f64,
    ) -> Result<Self, String> {
        if shard_count == 0 {
            return Err("shard_count must be at least 1".to_string());
        }
        if queue_depth == 0 {
            return Err("queue_depth must be at least 1".to_string());
        }
        if !benchmark_response_work_us.is_finite() || benchmark_response_work_us < 0.0 {
            return Err("benchmark_response_work_us must be finite and non-negative".to_string());
        }
        let benchmark_response_work =
            Duration::try_from_secs_f64(benchmark_response_work_us / 1_000_000.0)
                .map_err(|_| "benchmark_response_work_us is too large".to_string())?;

        let shared = Arc::new(ProcessorShared {
            benchmark_response_work,
            ..ProcessorShared::default()
        });
        let mut senders = Vec::with_capacity(shard_count);
        let mut workers = Vec::with_capacity(shard_count);
        for shard in 0..shard_count {
            let (sender, receiver) = mpsc::sync_channel(queue_depth);
            let worker_shared = shared.clone();
            let worker = thread::Builder::new()
                .name(format!("dynamo-egress-shard-{shard}"))
                .spawn(move || run_shard(receiver, worker_shared))
                .map_err(|error| format!("failed to start response shard {shard}: {error}"))?;
            senders.push(sender);
            workers.push(worker);
        }

        Ok(Self {
            shared,
            shard_count,
            dispatch: Mutex::new(()),
            senders: Mutex::new(Some(senders)),
            workers: Mutex::new(workers),
        })
    }

    pub(crate) fn register(
        &self,
        client_id: u64,
        prompt_tokens: usize,
        num_choices: usize,
        sink: Arc<dyn ResponseFrameSink>,
    ) -> Result<RequestKey, String> {
        if num_choices == 0 {
            return Err("num_choices must be at least 1".to_string());
        }

        let mut requests = self
            .shared
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.contains_key(&client_id) {
            return Err(format!("client {client_id} is already registered"));
        }
        let generation = self
            .shared
            .next_generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| "response request generation space is exhausted".to_string())?;
        let key = RequestKey {
            client_id,
            generation,
        };
        requests.insert(
            client_id,
            Arc::new(RegisteredRequest {
                key,
                next_sequence: AtomicU64::new(0),
                response_state: Mutex::new(RequestStreamState::new(prompt_tokens, num_choices)),
                sink,
                cancelled: AtomicBool::new(false),
                send_gate: Mutex::new(()),
            }),
        );
        Ok(key)
    }

    /// Submit events in sequence order for each request. Concurrent producers must serialize
    /// submission per `RequestKey`; a forward sequence gap is a terminal protocol error.
    pub(crate) fn process_batch(
        &self,
        responses: Vec<ResponseEvent>,
    ) -> Result<BatchOutcome, String> {
        let _dispatch = self
            .dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut partitions = (0..self.shard_count)
            .map(|_| Vec::new())
            .collect::<Vec<_>>();
        for (ordinal, response) in responses.into_iter().enumerate() {
            let shard = (response.client_id % self.shard_count as u64) as usize;
            partitions[shard].push(IndexedResponse { ordinal, response });
        }

        let senders = self
            .senders
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .cloned()
            .ok_or_else(|| "response processor is shutting down".to_string())?;
        let mut replies = Vec::new();
        for (shard, responses) in partitions.into_iter().enumerate() {
            if responses.is_empty() {
                continue;
            }
            let (reply, receive) = mpsc::sync_channel(1);
            senders[shard]
                .send(ShardCommand { responses, reply })
                .map_err(|_| format!("response shard {shard} stopped unexpectedly"))?;
            replies.push(receive);
        }

        let mut indexed = Vec::new();
        for reply in replies {
            indexed.extend(
                reply
                    .recv()
                    .map_err(|_| "response shard dropped a batch result".to_string())?,
            );
        }
        indexed.sort_unstable_by_key(|outcome| outcome.ordinal);

        let mut outcome = BatchOutcome::default();
        for response in indexed {
            if response.processed {
                outcome.responses_processed += 1;
            } else {
                outcome.responses_dropped += 1;
            }
            outcome.frames_sent += response.frames_sent;
            if response.completed {
                outcome.completed_requests.push(response.request_key);
            }
        }
        Ok(outcome)
    }

    pub(crate) fn cancel(&self, key: RequestKey) -> bool {
        let request = {
            let mut requests = self
                .shared
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if requests
                .get(&key.client_id)
                .is_some_and(|request| request.key == key)
            {
                requests.remove(&key.client_id)
            } else {
                None
            }
        };
        if let Some(request) = request {
            let _send_guard = request
                .send_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            request.cancelled.store(true, Ordering::Release);
            request.sink.cancel();
            drop(
                request
                    .response_state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            true
        } else {
            false
        }
    }

    pub(crate) fn active_requests(&self) -> usize {
        self.shared
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    #[cfg(not(test))]
    fn responses_processed(&self) -> usize {
        self.shared.responses_processed.load(Ordering::Relaxed)
    }

    #[cfg(not(test))]
    fn responses_dropped(&self) -> usize {
        self.shared.responses_dropped.load(Ordering::Relaxed)
    }

    #[cfg(not(test))]
    fn frames_sent(&self) -> usize {
        self.shared.frames_sent.load(Ordering::Relaxed)
    }
}

fn pad_response_work(
    started: Instant,
    target: Duration,
    cancelled: &AtomicBool,
    shutting_down: &AtomicBool,
) {
    while started.elapsed() < target {
        if cancelled.load(Ordering::Acquire) || shutting_down.load(Ordering::Acquire) {
            return;
        }
        std::hint::spin_loop();
    }
}

impl Drop for ShardedResponseEgress {
    fn drop(&mut self) {
        self.shared.shutting_down.store(true, Ordering::Release);
        self.shared.close_all();
        drop(
            self.senders
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take(),
        );
        let workers = self
            .workers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain(..)
            .collect::<Vec<_>>();
        for worker in workers {
            if worker.join().is_err() {
                tracing::error!("native response shard panicked during shutdown");
            }
        }
    }
}

fn run_shard(receiver: Receiver<ShardCommand>, shared: Arc<ProcessorShared>) {
    while let Ok(command) = receiver.recv() {
        let outcomes = command
            .responses
            .into_iter()
            .map(|response| shared.process_response(response.ordinal, response.response))
            .collect();
        let _ = command.reply.send(outcomes);
    }
}

#[cfg(not(test))]
struct PushFrameSink {
    sink: Arc<crate::push_egress::ResponseSink>,
}

#[cfg(not(test))]
impl ResponseFrameSink for PushFrameSink {
    fn send(
        &self,
        frame: Annotated<Value>,
        cancelled: &AtomicBool,
        shutting_down: &AtomicBool,
        send_gate: &Mutex<()>,
    ) -> Result<(), FrameSendError> {
        self.sink
            .send_annotated(frame, cancelled, shutting_down, send_gate)
            .map_err(|error| match error {
                crate::push_egress::ResponseSendError::Stopped => FrameSendError::Stopped,
                crate::push_egress::ResponseSendError::Failed(message) => {
                    FrameSendError::Failed(message)
                }
            })
    }

    fn close(&self) {
        self.sink.close();
    }

    fn cancel(&self) {
        self.sink.cancel();
    }

    fn close_with_error(&self, message: String) -> bool {
        self.sink.try_close_with_error(message)
    }

    fn shutdown(&self) {
        self.sink.shutdown();
    }
}

#[cfg(not(test))]
#[pyclass]
pub struct NativeResponseEgress {
    processor: ShardedResponseEgress,
}

#[cfg(not(test))]
#[pymethods]
impl NativeResponseEgress {
    #[new]
    #[pyo3(signature = (shards=4, queue_depth=2, benchmark_response_work_us=0.0))]
    fn new(shards: usize, queue_depth: usize, benchmark_response_work_us: f64) -> PyResult<Self> {
        Ok(Self {
            processor: ShardedResponseEgress::new_with_benchmark_work(
                shards,
                queue_depth,
                benchmark_response_work_us,
            )
            .map_err(PyValueError::new_err)?,
        })
    }

    #[pyo3(signature = (client_id, prompt_tokens, num_choices, response_sender))]
    fn register(
        &self,
        client_id: u64,
        prompt_tokens: usize,
        num_choices: usize,
        response_sender: PyRef<'_, crate::push_egress::ResponseSender>,
    ) -> PyResult<RequestKey> {
        self.processor
            .register(
                client_id,
                prompt_tokens,
                num_choices,
                Arc::new(PushFrameSink {
                    sink: response_sender.sink(),
                }),
            )
            .map_err(PyValueError::new_err)
    }

    fn process_batch(
        &self,
        py: Python<'_>,
        responses: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<RequestKey>> {
        let responses = depythonize::<Vec<ResponseEvent>>(responses).map_err(|error| {
            PyValueError::new_err(format!("invalid engine response batch: {error}"))
        })?;
        let outcome = py
            .allow_threads(|| self.processor.process_batch(responses))
            .map_err(PyValueError::new_err)?;
        Ok(outcome.completed_requests)
    }

    fn cancel(&self, py: Python<'_>, request: PyRef<'_, RequestKey>) -> bool {
        let key = *request;
        py.allow_threads(|| self.processor.cancel(key))
    }

    #[getter]
    fn active_requests(&self) -> usize {
        self.processor.active_requests()
    }

    #[getter]
    fn responses_processed(&self) -> usize {
        self.processor.responses_processed()
    }

    #[getter]
    fn responses_dropped(&self) -> usize {
        self.processor.responses_dropped()
    }

    #[getter]
    fn frames_sent(&self) -> usize {
        self.processor.frames_sent()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::Duration;

    use dynamo_runtime::protocols::annotated::Annotated;
    use serde_json::json;

    use super::{
        ChoiceDelta, FrameSendError, RequestKey, RequestStreamState, ResponseEvent,
        ResponseFrameSink, ShardedResponseEgress,
    };

    #[derive(Default)]
    struct RecordingSink {
        frames: Mutex<Vec<Annotated<serde_json::Value>>>,
        closed: AtomicBool,
        errors: Mutex<Vec<String>>,
    }

    impl ResponseFrameSink for RecordingSink {
        fn send(
            &self,
            frame: Annotated<serde_json::Value>,
            cancelled: &AtomicBool,
            shutting_down: &AtomicBool,
            send_gate: &Mutex<()>,
        ) -> Result<(), FrameSendError> {
            let _send_guard = send_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cancelled.load(Ordering::Acquire) || shutting_down.load(Ordering::Acquire) {
                return Err(FrameSendError::Stopped);
            }
            self.frames
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(frame);
            Ok(())
        }

        fn close(&self) {
            self.closed.store(true, Ordering::Relaxed);
        }

        fn cancel(&self) {
            self.close();
        }

        fn close_with_error(&self, message: String) -> bool {
            self.errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(message);
            self.close();
            true
        }

        fn shutdown(&self) {
            self.close();
        }
    }

    struct GateSink {
        entered: mpsc::Sender<u64>,
        client_id: u64,
        released: Arc<(Mutex<bool>, Condvar)>,
    }

    impl GateSink {
        fn release(&self) {
            let (lock, ready) = &*self.released;
            *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
            ready.notify_all();
        }
    }

    impl ResponseFrameSink for GateSink {
        fn send(
            &self,
            _frame: Annotated<serde_json::Value>,
            cancelled: &AtomicBool,
            shutting_down: &AtomicBool,
            send_gate: &Mutex<()>,
        ) -> Result<(), FrameSendError> {
            {
                let _send_guard = send_gate
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if cancelled.load(Ordering::Acquire) || shutting_down.load(Ordering::Acquire) {
                    return Err(FrameSendError::Stopped);
                }
            }
            self.entered
                .send(self.client_id)
                .expect("test receiver remains open");
            let (lock, ready) = &*self.released;
            let mut released = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            while !*released
                && !cancelled.load(Ordering::Acquire)
                && !shutting_down.load(Ordering::Acquire)
            {
                released = ready
                    .wait_timeout(released, Duration::from_millis(10))
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .0;
            }
            if cancelled.load(Ordering::Acquire) || shutting_down.load(Ordering::Acquire) {
                return Err(FrameSendError::Stopped);
            }
            let _send_guard = send_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cancelled.load(Ordering::Acquire) || shutting_down.load(Ordering::Acquire) {
                return Err(FrameSendError::Stopped);
            }
            Ok(())
        }

        fn close(&self) {}

        fn cancel(&self) {
            self.release();
        }

        fn close_with_error(&self, _message: String) -> bool {
            self.close();
            true
        }

        fn shutdown(&self) {
            self.release();
        }
    }

    struct PausingSink {
        entered: mpsc::Sender<()>,
        resume: Mutex<mpsc::Receiver<()>>,
        frames: AtomicUsize,
    }

    impl ResponseFrameSink for PausingSink {
        fn send(
            &self,
            _frame: Annotated<serde_json::Value>,
            cancelled: &AtomicBool,
            shutting_down: &AtomicBool,
            send_gate: &Mutex<()>,
        ) -> Result<(), FrameSendError> {
            let _send_guard = send_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cancelled.load(Ordering::Acquire) || shutting_down.load(Ordering::Acquire) {
                return Err(FrameSendError::Stopped);
            }
            self.entered.send(()).expect("test receiver remains open");
            self.resume
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .recv()
                .expect("test sender remains open");
            self.frames.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn close(&self) {}

        fn cancel(&self) {}

        fn close_with_error(&self, _message: String) -> bool {
            true
        }

        fn shutdown(&self) {}
    }

    #[test]
    fn rejects_zero_shards_and_zero_queue_depth() {
        assert!(ShardedResponseEgress::new(0, 1).is_err());
        assert!(ShardedResponseEgress::new(1, 0).is_err());
    }

    #[test]
    fn rejects_invalid_benchmark_work() {
        assert!(ShardedResponseEgress::new_with_benchmark_work(1, 1, -1.0).is_err());
        assert!(ShardedResponseEgress::new_with_benchmark_work(1, 1, f64::NAN).is_err());
        assert!(ShardedResponseEgress::new_with_benchmark_work(1, 1, f64::MAX).is_err());
    }

    #[test]
    fn builds_delta_frames_with_python_semantic_parity() {
        let mut state = RequestStreamState::new(3, 2);

        let first = state
            .apply(ResponseEvent {
                client_id: 1,
                generation: 1,
                sequence: 0,
                outputs: vec![
                    ChoiceDelta {
                        index: 0,
                        new_token_ids: vec![10],
                        finish_reason: None,
                        stop_reason: None,
                    },
                    ChoiceDelta {
                        index: 1,
                        new_token_ids: vec![20],
                        finish_reason: None,
                        stop_reason: None,
                    },
                ],
                is_final: false,
                error_msg: None,
            })
            .expect("valid first response");
        assert_eq!(
            first,
            vec![
                json!({"token_ids": [10], "index": 0}),
                json!({"token_ids": [20], "index": 1}),
            ]
        );

        let final_frames = state
            .apply(ResponseEvent {
                client_id: 1,
                generation: 1,
                sequence: 1,
                outputs: vec![
                    ChoiceDelta {
                        index: 0,
                        new_token_ids: vec![11, 12],
                        finish_reason: Some("stop".to_string()),
                        stop_reason: None,
                    },
                    ChoiceDelta {
                        index: 1,
                        new_token_ids: vec![21],
                        finish_reason: Some("length".to_string()),
                        stop_reason: Some("eos".to_string()),
                    },
                ],
                is_final: true,
                error_msg: None,
            })
            .expect("valid final response");
        assert_eq!(
            final_frames,
            vec![
                json!({
                    "token_ids": [11, 12],
                    "index": 0,
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 5,
                        "total_tokens": 8,
                        "prompt_tokens_details": null
                    }
                }),
                json!({
                    "token_ids": [21],
                    "index": 1,
                    "finish_reason": "length",
                    "stop_reason": "eos",
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 5,
                        "total_tokens": 8,
                        "prompt_tokens_details": null
                    }
                }),
            ]
        );
    }

    #[test]
    fn reordered_partial_choices_emit_only_supplied_indices() {
        let mut state = RequestStreamState::new(0, 3);

        let frames = state
            .apply(ResponseEvent {
                client_id: 1,
                generation: 1,
                sequence: 0,
                outputs: vec![
                    ChoiceDelta {
                        index: 2,
                        new_token_ids: vec![20],
                        finish_reason: None,
                        stop_reason: None,
                    },
                    ChoiceDelta {
                        index: 0,
                        new_token_ids: vec![10],
                        finish_reason: None,
                        stop_reason: None,
                    },
                ],
                is_final: false,
                error_msg: None,
            })
            .expect("valid partial response");

        assert_eq!(
            frames,
            vec![
                json!({"token_ids": [20], "index": 2}),
                json!({"token_ids": [10], "index": 0}),
            ]
        );
    }

    #[test]
    fn duplicate_choice_indices_are_rejected_before_state_mutation() {
        let mut state = RequestStreamState::new(0, 1);
        let invalid = state.apply(ResponseEvent {
            client_id: 1,
            generation: 1,
            sequence: 0,
            outputs: vec![
                ChoiceDelta {
                    index: 0,
                    new_token_ids: vec![1],
                    finish_reason: None,
                    stop_reason: None,
                },
                ChoiceDelta {
                    index: 0,
                    new_token_ids: vec![2],
                    finish_reason: None,
                    stop_reason: None,
                },
            ],
            is_final: false,
            error_msg: None,
        });
        assert!(invalid.is_err());

        let valid = state
            .apply(ResponseEvent::tokens_for(
                RequestKey {
                    client_id: 1,
                    generation: 1,
                },
                1,
                vec![vec![3]],
                false,
            ))
            .expect("state remains usable");
        assert_eq!(valid, vec![json!({"token_ids": [3], "index": 0})]);
    }

    #[test]
    fn clients_on_different_shards_process_concurrently() {
        let processor = Arc::new(ShardedResponseEgress::new(2, 1).expect("valid processor"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let sink_zero = Arc::new(GateSink {
            entered: entered_tx.clone(),
            client_id: 2,
            released: Arc::new((Mutex::new(false), Condvar::new())),
        });
        let sink_one = Arc::new(GateSink {
            entered: entered_tx,
            client_id: 3,
            released: Arc::new((Mutex::new(false), Condvar::new())),
        });
        let key_zero = processor
            .register(2, 0, 1, sink_zero.clone())
            .expect("register shard zero client");
        let key_one = processor
            .register(3, 0, 1, sink_one.clone())
            .expect("register shard one client");

        let processing = {
            let processor = processor.clone();
            thread::spawn(move || {
                processor.process_batch(vec![
                    ResponseEvent::tokens_for(key_zero, 0, vec![vec![1]], false),
                    ResponseEvent::tokens_for(key_one, 0, vec![vec![2]], false),
                ])
            })
        };

        let first = entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("one worker entered the sink");
        let second_before_release = entered_rx.recv_timeout(Duration::from_millis(250));
        sink_zero.release();
        sink_one.release();
        processing
            .join()
            .expect("batch processor did not panic")
            .expect("batch processor succeeded");

        assert_ne!(second_before_release, Err(mpsc::RecvTimeoutError::Timeout));
        assert_ne!(first, second_before_release.expect("second shard entered"));
    }

    #[test]
    fn same_client_responses_remain_fifo() {
        let processor = ShardedResponseEgress::new(4, 1).expect("valid processor");
        let sink = Arc::new(RecordingSink::default());
        let key = processor
            .register(9, 2, 1, sink.clone())
            .expect("register client 9");

        let outcome = processor
            .process_batch(vec![
                ResponseEvent::tokens_for(key, 0, vec![vec![10]], false),
                ResponseEvent::tokens_for(key, 1, vec![vec![11]], true),
            ])
            .expect("process ordered batch");

        let frames = sink
            .frames
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(frames[0].data, Some(json!({"token_ids": [10], "index": 0})));
        assert_eq!(frames[1].data.as_ref().unwrap()["token_ids"], json!([11]));
        assert_eq!(outcome.completed_requests, vec![key]);
        assert!(sink.closed.load(Ordering::Relaxed));
    }

    #[test]
    fn concurrent_batch_calls_are_serialized() {
        let processor = Arc::new(ShardedResponseEgress::new(2, 1).expect("valid processor"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let blocking_sink = Arc::new(GateSink {
            entered: entered_tx,
            client_id: 2,
            released: Arc::new((Mutex::new(false), Condvar::new())),
        });
        let other_sink = Arc::new(RecordingSink::default());
        let blocking_key = processor
            .register(2, 0, 1, blocking_sink.clone())
            .expect("register blocking client");
        let other_key = processor
            .register(3, 0, 1, other_sink)
            .expect("register other client");

        let first = {
            let processor = processor.clone();
            thread::spawn(move || {
                processor.process_batch(vec![ResponseEvent::tokens_for(
                    blocking_key,
                    0,
                    vec![vec![1]],
                    false,
                )])
            })
        };
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("first batch reached its shard");

        let (done_tx, done_rx) = mpsc::channel();
        let second = {
            let processor = processor.clone();
            thread::spawn(move || {
                let result = processor.process_batch(vec![ResponseEvent::tokens_for(
                    other_key,
                    0,
                    vec![vec![2]],
                    false,
                )]);
                done_tx.send(result).expect("test receiver remains open");
            })
        };
        assert_eq!(
            done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );

        blocking_sink.release();
        first
            .join()
            .expect("first batch did not panic")
            .expect("first batch succeeded");
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("second batch completed after first")
            .expect("second batch succeeded");
        second.join().expect("second batch did not panic");
    }

    #[test]
    fn completion_keys_follow_input_order_across_shards() {
        let processor = ShardedResponseEgress::new(4, 1).expect("valid processor");
        let sink_three = Arc::new(RecordingSink::default());
        let sink_zero = Arc::new(RecordingSink::default());
        let key_three = processor
            .register(3, 0, 1, sink_three)
            .expect("register client 3");
        let key_zero = processor
            .register(4, 0, 1, sink_zero)
            .expect("register client 4");

        let outcome = processor
            .process_batch(vec![
                ResponseEvent::tokens_for(key_three, 0, vec![vec![3]], true),
                ResponseEvent::tokens_for(key_zero, 0, vec![vec![4]], true),
            ])
            .expect("process cross-shard completions");

        assert_eq!(outcome.completed_requests, vec![key_three, key_zero]);
        assert_eq!(processor.active_requests(), 0);
    }

    #[test]
    fn error_closes_request_and_late_response_is_dropped() {
        let processor = ShardedResponseEgress::new(2, 1).expect("valid processor");
        let sink = Arc::new(RecordingSink::default());
        let key = processor
            .register(6, 0, 1, sink.clone())
            .expect("register client 6");

        let error = processor
            .process_batch(vec![ResponseEvent::error_for(key, 0, "engine failed")])
            .expect("process terminal error");
        let late = processor
            .process_batch(vec![ResponseEvent::tokens_for(
                key,
                1,
                vec![vec![99]],
                true,
            )])
            .expect("drop late response");

        assert_eq!(error.completed_requests, vec![key]);
        assert_eq!(late.responses_dropped, 1);
        assert_eq!(
            sink.errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_slice(),
            ["engine failed"]
        );
        assert!(sink.closed.load(Ordering::Relaxed));
    }

    #[test]
    fn cancellation_does_not_wait_for_a_backpressured_shard() {
        let processor = Arc::new(ShardedResponseEgress::new(2, 1).expect("valid processor"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let sink = Arc::new(GateSink {
            entered: entered_tx,
            client_id: 4,
            released: Arc::new((Mutex::new(false), Condvar::new())),
        });
        let key = processor
            .register(4, 0, 1, sink.clone())
            .expect("register client 4");

        let processing = {
            let processor = processor.clone();
            thread::spawn(move || {
                processor.process_batch(vec![ResponseEvent::tokens_for(
                    key,
                    0,
                    vec![vec![1]],
                    false,
                )])
            })
        };
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker entered blocked sink");

        let (cancelled_tx, cancelled_rx) = mpsc::channel();
        let cancelling = {
            let processor = processor.clone();
            thread::spawn(move || {
                cancelled_tx
                    .send(processor.cancel(key))
                    .expect("test receiver remains open");
            })
        };
        let cancelled = cancelled_rx.recv_timeout(Duration::from_millis(250));

        sink.release();
        processing
            .join()
            .expect("batch processor did not panic")
            .expect("batch processor succeeded");
        cancelling.join().expect("cancellation did not panic");

        assert_eq!(cancelled, Ok(true));
        assert_eq!(processor.active_requests(), 0);
    }

    #[test]
    fn cancellation_waits_for_inflight_enqueue_and_prevents_late_frames() {
        let processor = Arc::new(ShardedResponseEgress::new(1, 1).expect("valid processor"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let sink = Arc::new(PausingSink {
            entered: entered_tx,
            resume: Mutex::new(resume_rx),
            frames: AtomicUsize::new(0),
        });
        let key = processor
            .register(8, 0, 1, sink.clone())
            .expect("register client 8");

        let processing = {
            let processor = processor.clone();
            thread::spawn(move || {
                processor.process_batch(vec![ResponseEvent::tokens_for(
                    key,
                    0,
                    vec![vec![1]],
                    false,
                )])
            })
        };
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("send reached its linearization point");

        let (cancelled_tx, cancelled_rx) = mpsc::channel();
        let cancelling = {
            let processor = processor.clone();
            thread::spawn(move || {
                cancelled_tx
                    .send(processor.cancel(key))
                    .expect("test receiver remains open");
            })
        };
        assert_eq!(
            cancelled_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );

        resume_tx.send(()).expect("paused send remains open");
        assert_eq!(cancelled_rx.recv_timeout(Duration::from_secs(1)), Ok(true));
        let frames_at_cancel = sink.frames.load(Ordering::Relaxed);
        processor
            .process_batch(vec![ResponseEvent::tokens_for(
                key,
                1,
                vec![vec![2]],
                false,
            )])
            .expect("late response is dropped");

        processing
            .join()
            .expect("processing did not panic")
            .expect("processing succeeded");
        cancelling.join().expect("cancellation did not panic");
        assert_eq!(frames_at_cancel, 1);
        assert_eq!(sink.frames.load(Ordering::Relaxed), frames_at_cancel);
    }

    #[test]
    fn reused_client_id_gets_a_new_generation_and_rejects_stale_events() {
        let egress = ShardedResponseEgress::new(1, 1).expect("valid egress");
        let first_sink = Arc::new(RecordingSink::default());
        let first_key = egress
            .register(7, 0, 1, first_sink)
            .expect("register first request");
        assert!(egress.cancel(first_key));

        let second_sink = Arc::new(RecordingSink::default());
        let second_key = egress
            .register(7, 0, 1, second_sink.clone())
            .expect("reuse client ID");
        assert_eq!(first_key.client_id, second_key.client_id);
        assert_ne!(first_key.generation, second_key.generation);

        let stale = egress
            .process_batch(vec![ResponseEvent::tokens_for(
                first_key,
                0,
                vec![vec![10]],
                false,
            )])
            .expect("stale event is dropped");
        let current = egress
            .process_batch(vec![ResponseEvent::tokens_for(
                second_key,
                0,
                vec![vec![20]],
                true,
            )])
            .expect("current event is processed");

        assert_eq!(stale.responses_dropped, 1);
        assert_eq!(current.completed_requests, vec![second_key]);
        assert_eq!(second_sink.frames.lock().unwrap().len(), 1);
    }

    #[test]
    fn stale_cancel_does_not_cancel_replacement_registration() {
        let egress = ShardedResponseEgress::new(1, 1).expect("valid egress");
        let first_key = egress
            .register(11, 0, 1, Arc::new(RecordingSink::default()))
            .expect("register first request");
        assert!(egress.cancel(first_key));
        let replacement_key = egress
            .register(11, 0, 1, Arc::new(RecordingSink::default()))
            .expect("register replacement");

        assert!(!egress.cancel(first_key));
        assert_eq!(egress.active_requests(), 1);
        assert!(egress.cancel(replacement_key));
    }

    #[test]
    fn duplicate_sequence_is_dropped_without_mutating_stream_state() {
        let egress = ShardedResponseEgress::new(1, 1).expect("valid egress");
        let sink = Arc::new(RecordingSink::default());
        let key = egress
            .register(13, 0, 1, sink.clone())
            .expect("register request");

        let first = egress
            .process_batch(vec![ResponseEvent::tokens_for(
                key,
                0,
                vec![vec![10]],
                false,
            )])
            .expect("first event is processed");
        let duplicate = egress
            .process_batch(vec![ResponseEvent::tokens_for(
                key,
                0,
                vec![vec![99]],
                false,
            )])
            .expect("duplicate event is dropped");
        let next = egress
            .process_batch(vec![ResponseEvent::tokens_for(
                key,
                1,
                vec![vec![11]],
                true,
            )])
            .expect("next event is processed");

        assert_eq!(first.responses_processed, 1);
        assert_eq!(duplicate.responses_dropped, 1);
        assert_eq!(next.responses_processed, 1);
        let frames = sink.frames.lock().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].data.as_ref().unwrap()["token_ids"], json!([10]));
        assert_eq!(frames[1].data.as_ref().unwrap()["token_ids"], json!([11]));
    }

    #[test]
    fn forward_sequence_gap_terminates_the_registration() {
        let egress = ShardedResponseEgress::new(1, 1).expect("valid egress");
        let sink = Arc::new(RecordingSink::default());
        let key = egress
            .register(17, 0, 1, sink.clone())
            .expect("register request");

        let gap = egress
            .process_batch(vec![ResponseEvent::tokens_for(
                key,
                1,
                vec![vec![10]],
                false,
            )])
            .expect("sequence gap is handled");

        assert_eq!(gap.responses_dropped, 1);
        assert_eq!(gap.completed_requests, vec![key]);
        assert_eq!(egress.active_requests(), 0);
        assert_eq!(sink.errors.lock().unwrap().len(), 1);
        assert!(sink.errors.lock().unwrap()[0].contains("expected sequence 0, received 1"));
    }
}
