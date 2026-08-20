// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dynamo_runtime::dynamo_nvtx_range;
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
    module.add_class::<OwnedTokenEgress>()?;
    Ok(())
}

pub(crate) trait OwnedFrameSink: Send + Sync {
    fn send(&self, frame: Annotated<Value>) -> Result<(), String>;
    fn close(&self);
    fn close_with_error(&self, message: String);
}

#[derive(Debug)]
pub(crate) struct EngineResponse {
    pub(crate) new_token_ids: Vec<Vec<u32>>,
    pub(crate) is_final: bool,
    pub(crate) finish_reasons: Option<Vec<Option<String>>>,
    pub(crate) stop_reasons: Option<Vec<Option<String>>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MockEngineResponse {
    client_id: u64,
    #[serde(default)]
    new_token_ids: Vec<Vec<u32>>,
    #[serde(default)]
    is_final: bool,
    #[serde(default)]
    finish_reasons: Option<Vec<Option<String>>>,
    #[serde(default)]
    stop_reasons: Option<Vec<Option<String>>>,
    #[serde(default)]
    error_msg: Option<String>,
}

#[cfg(test)]
impl MockEngineResponse {
    fn tokens(client_id: u64, new_token_ids: Vec<Vec<u32>>, is_final: bool) -> Self {
        Self {
            client_id,
            new_token_ids,
            is_final,
            finish_reasons: None,
            stop_reasons: None,
            error_msg: None,
        }
    }

    fn error(client_id: u64, error_msg: String) -> Self {
        Self {
            client_id,
            new_token_ids: Vec::new(),
            is_final: true,
            finish_reasons: None,
            stop_reasons: None,
            error_msg: Some(error_msg),
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
pub(crate) struct OwnedResponseState {
    prompt_tokens: usize,
    choices: Vec<ChoiceState>,
}

impl OwnedResponseState {
    pub(crate) fn new(prompt_tokens: usize, num_choices: usize) -> Self {
        Self {
            prompt_tokens,
            choices: (0..num_choices).map(|_| ChoiceState::default()).collect(),
        }
    }

    pub(crate) fn apply(&mut self, response: EngineResponse) -> Vec<Value> {
        for (index, new_tokens) in response.new_token_ids.into_iter().enumerate() {
            let Some(choice) = self.choices.get_mut(index) else {
                break;
            };
            choice.token_ids.extend(new_tokens);
        }

        Self::update_reasons(
            &mut self.choices,
            response.finish_reasons,
            |choice, reason| {
                choice.finish_reason = reason;
            },
        );
        Self::update_reasons(
            &mut self.choices,
            response.stop_reasons,
            |choice, reason| {
                choice.stop_reason = reason;
            },
        );

        let completion_tokens = self
            .choices
            .iter()
            .map(|choice| choice.token_ids.len())
            .sum::<usize>();

        self.choices
            .iter_mut()
            .enumerate()
            .map(|(index, choice)| {
                let mut frame = Map::new();
                frame.insert(
                    "token_ids".to_string(),
                    json!(choice.token_ids[choice.emitted..]),
                );
                frame.insert("index".to_string(), json!(index));
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
            .collect()
    }

    fn update_reasons(
        choices: &mut [ChoiceState],
        reasons: Option<Vec<Option<String>>>,
        update: impl Fn(&mut ChoiceState, Option<String>),
    ) {
        let Some(reasons) = reasons else {
            return;
        };
        for (choice, reason) in choices.iter_mut().zip(reasons) {
            if reason.is_some() {
                update(choice, reason);
            }
        }
    }
}

struct RegisteredRequest {
    response_state: OwnedResponseState,
    sink: Arc<dyn OwnedFrameSink>,
    calibrated_work_us: f64,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchOutcome {
    pub(crate) completed_client_ids: Vec<u64>,
    pub(crate) responses_processed: usize,
    pub(crate) responses_dropped: usize,
    pub(crate) frames_sent: usize,
}

#[derive(Default)]
pub(crate) struct ProcessorCore {
    requests: Mutex<HashMap<u64, Arc<Mutex<RegisteredRequest>>>>,
    responses_processed: AtomicUsize,
    responses_dropped: AtomicUsize,
    frames_sent: AtomicUsize,
}

impl ProcessorCore {
    pub(crate) fn register(
        &self,
        client_id: u64,
        prompt_tokens: usize,
        num_choices: usize,
        sink: Arc<dyn OwnedFrameSink>,
        calibrated_work_us: f64,
    ) -> Result<(), String> {
        if num_choices == 0 {
            return Err("num_choices must be at least 1".to_string());
        }
        if !calibrated_work_us.is_finite() || calibrated_work_us < 0.0 {
            return Err("calibrated_work_us must be finite and non-negative".to_string());
        }

        let mut requests = self
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.contains_key(&client_id) {
            return Err(format!("client {client_id} is already registered"));
        }
        requests.insert(
            client_id,
            Arc::new(Mutex::new(RegisteredRequest {
                response_state: OwnedResponseState::new(prompt_tokens, num_choices),
                sink,
                calibrated_work_us,
            })),
        );
        Ok(())
    }

    pub(crate) fn process_batch(&self, responses: Vec<MockEngineResponse>) -> BatchOutcome {
        let mut outcome = BatchOutcome::default();
        for response in responses {
            let client_id = response.client_id;
            let request = {
                self.requests
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .get(&client_id)
                    .cloned()
            };
            let Some(request) = request else {
                outcome.responses_dropped += 1;
                self.responses_dropped.fetch_add(1, Ordering::Relaxed);
                continue;
            };

            outcome.responses_processed += 1;
            self.responses_processed.fetch_add(1, Ordering::Relaxed);
            let terminal = response.is_final || response.error_msg.is_some();
            let mut request = request
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            if let Some(error) = response.error_msg {
                request.sink.close_with_error(error);
            } else {
                let _nvtx = dynamo_nvtx_range!("rust_egress.handle_build");
                spin_for(request.calibrated_work_us);
                let frames = request.response_state.apply(EngineResponse {
                    new_token_ids: response.new_token_ids,
                    is_final: response.is_final,
                    finish_reasons: response.finish_reasons,
                    stop_reasons: response.stop_reasons,
                });
                for frame in frames {
                    let _send_nvtx = dynamo_nvtx_range!("rust_egress.send");
                    match request.sink.send(Annotated::from_data(frame)) {
                        Ok(()) => {
                            outcome.frames_sent += 1;
                            self.frames_sent.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(error) => {
                            request.sink.close_with_error(error);
                            break;
                        }
                    }
                }
                if response.is_final {
                    request.sink.close();
                }
            }
            drop(request);

            if terminal {
                self.remove(client_id);
                outcome.completed_client_ids.push(client_id);
            }
        }
        outcome
    }

    pub(crate) fn cancel(&self, client_id: u64) -> bool {
        let request = self
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&client_id);
        if let Some(request) = request {
            request
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .sink
                .close();
            true
        } else {
            false
        }
    }

    fn remove(&self, client_id: u64) {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&client_id);
    }

    pub(crate) fn active_requests(&self) -> usize {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

fn spin_for(work_us: f64) {
    if work_us <= 0.0 {
        return;
    }
    let deadline = Instant::now() + Duration::from_secs_f64(work_us / 1_000_000.0);
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

#[cfg(not(test))]
#[pyclass]
pub struct OwnedTokenEgress {
    core: Arc<ProcessorCore>,
}

#[cfg(not(test))]
#[pymethods]
impl OwnedTokenEgress {
    #[new]
    fn new() -> Self {
        Self {
            core: Arc::new(ProcessorCore::default()),
        }
    }

    #[pyo3(signature = (client_id, prompt_tokens, num_choices, response_sender, calibrated_work_us=0.0))]
    fn register(
        &self,
        client_id: u64,
        prompt_tokens: usize,
        num_choices: usize,
        response_sender: PyRef<'_, crate::push_egress::ResponseSender>,
        calibrated_work_us: f64,
    ) -> PyResult<()> {
        self.core
            .register(
                client_id,
                prompt_tokens,
                num_choices,
                response_sender.owned_sink(),
                calibrated_work_us,
            )
            .map_err(PyValueError::new_err)
    }

    fn process_mock_batch(
        &self,
        py: Python<'_>,
        responses: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<u64>> {
        let _convert_nvtx = dynamo_nvtx_range!("pybridge.owned_response_convert");
        let responses = depythonize::<Vec<MockEngineResponse>>(responses).map_err(|error| {
            PyValueError::new_err(format!("invalid mock engine response batch: {error}"))
        })?;
        drop(_convert_nvtx);

        let core = self.core.clone();
        let outcome = py.allow_threads(move || core.process_batch(responses));
        Ok(outcome.completed_client_ids)
    }

    fn cancel(&self, client_id: u64) -> bool {
        self.core.cancel(client_id)
    }

    #[getter]
    fn active_requests(&self) -> usize {
        self.core.active_requests()
    }

    #[getter]
    fn responses_processed(&self) -> usize {
        self.core.responses_processed.load(Ordering::Relaxed)
    }

    #[getter]
    fn responses_dropped(&self) -> usize {
        self.core.responses_dropped.load(Ordering::Relaxed)
    }

    #[getter]
    fn frames_sent(&self) -> usize {
        self.core.frames_sent.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::{
        EngineResponse, MockEngineResponse, OwnedFrameSink, OwnedResponseState, ProcessorCore,
    };
    use dynamo_runtime::protocols::annotated::Annotated;
    use serde_json::json;

    #[derive(Default)]
    struct RecordingSink {
        frames: Mutex<Vec<Annotated<serde_json::Value>>>,
        closed: AtomicBool,
        errors: Mutex<Vec<String>>,
    }

    #[derive(Default)]
    struct FailingSink {
        errors: Mutex<Vec<String>>,
    }

    impl OwnedFrameSink for FailingSink {
        fn send(&self, _frame: Annotated<serde_json::Value>) -> Result<(), String> {
            Err("consumer disconnected".to_string())
        }

        fn close(&self) {}

        fn close_with_error(&self, message: String) {
            self.errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(message);
        }
    }

    impl OwnedFrameSink for RecordingSink {
        fn send(&self, frame: Annotated<serde_json::Value>) -> Result<(), String> {
            self.frames
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(frame);
            Ok(())
        }

        fn close(&self) {
            self.closed.store(true, Ordering::Relaxed);
        }

        fn close_with_error(&self, message: String) {
            self.errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(message);
            self.closed.store(true, Ordering::Relaxed);
        }
    }

    fn assert_send_sync_static<T: Send + Sync + 'static>() {}

    #[test]
    fn owned_response_state_is_send_sync_static() {
        assert_send_sync_static::<OwnedResponseState>();
    }

    #[test]
    fn builds_delta_frames_with_python_semantic_parity() {
        let mut state = OwnedResponseState::new(3, 2);

        let first = state.apply(EngineResponse {
            new_token_ids: vec![vec![10], vec![20]],
            is_final: false,
            finish_reasons: None,
            stop_reasons: None,
        });
        assert_eq!(
            first,
            vec![
                json!({"token_ids": [10], "index": 0}),
                json!({"token_ids": [20], "index": 1}),
            ]
        );

        let final_frames = state.apply(EngineResponse {
            new_token_ids: vec![vec![11, 12], vec![21]],
            is_final: true,
            finish_reasons: Some(vec![Some("stop".to_string()), Some("length".to_string())]),
            stop_reasons: Some(vec![None, Some("eos".to_string())]),
        });
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
    fn final_response_without_finish_reason_uses_unknown() {
        let mut state = OwnedResponseState::new(1, 1);

        let frames = state.apply(EngineResponse {
            new_token_ids: vec![vec![42]],
            is_final: true,
            finish_reasons: None,
            stop_reasons: None,
        });

        assert_eq!(frames[0]["finish_reason"], "unknown");
        assert_eq!(frames[0]["completion_usage"]["completion_tokens"], 1);
    }

    #[test]
    fn extra_engine_choices_are_ignored_like_the_python_path() {
        let mut state = OwnedResponseState::new(0, 1);

        let frames = state.apply(EngineResponse {
            new_token_ids: vec![vec![1], vec![999]],
            is_final: false,
            finish_reasons: None,
            stop_reasons: None,
        });

        assert_eq!(frames, vec![json!({"token_ids": [1], "index": 0})]);
    }

    #[test]
    fn processor_interleaves_clients_without_reordering_each_stream() {
        let processor = ProcessorCore::default();
        let sink_a = Arc::new(RecordingSink::default());
        let sink_b = Arc::new(RecordingSink::default());
        processor
            .register(10, 2, 1, sink_a.clone(), 0.0)
            .expect("register client 10");
        processor
            .register(20, 1, 1, sink_b.clone(), 0.0)
            .expect("register client 20");

        let first = processor.process_batch(vec![
            MockEngineResponse::tokens(10, vec![vec![1]], false),
            MockEngineResponse::tokens(20, vec![vec![7]], false),
        ]);
        assert!(first.completed_client_ids.is_empty());
        assert_eq!(first.responses_processed, 2);

        let final_batch = processor.process_batch(vec![
            MockEngineResponse::tokens(20, vec![vec![8]], true),
            MockEngineResponse::tokens(10, vec![vec![2, 3]], true),
        ]);
        assert_eq!(final_batch.completed_client_ids, vec![20, 10]);

        let frames_a = sink_a
            .frames
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let frames_b = sink_b
            .frames
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            frames_a[0].data,
            Some(json!({"token_ids": [1], "index": 0}))
        );
        assert_eq!(
            frames_a[1].data.as_ref().unwrap()["token_ids"],
            json!([2, 3])
        );
        assert_eq!(
            frames_b[0].data,
            Some(json!({"token_ids": [7], "index": 0}))
        );
        assert_eq!(frames_b[1].data.as_ref().unwrap()["token_ids"], json!([8]));
        assert!(sink_a.closed.load(Ordering::Relaxed));
        assert!(sink_b.closed.load(Ordering::Relaxed));
        assert_eq!(processor.active_requests(), 0);
    }

    #[test]
    fn processor_closes_errors_and_ignores_late_responses() {
        let processor = ProcessorCore::default();
        let sink = Arc::new(RecordingSink::default());
        processor
            .register(30, 0, 1, sink.clone(), 0.0)
            .expect("register client 30");

        let error = processor.process_batch(vec![MockEngineResponse::error(
            30,
            "engine failed".to_string(),
        )]);
        assert_eq!(error.completed_client_ids, vec![30]);
        let late =
            processor.process_batch(vec![MockEngineResponse::tokens(30, vec![vec![99]], true)]);
        assert_eq!(late.responses_dropped, 1);

        assert!(
            sink.errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())[0]
                .contains("engine failed")
        );
        assert!(sink.closed.load(Ordering::Relaxed));
    }

    #[test]
    fn processor_completes_request_when_response_sink_disconnects() {
        let processor = ProcessorCore::default();
        let sink = Arc::new(FailingSink::default());
        processor
            .register(35, 0, 1, sink.clone(), 0.0)
            .expect("register client 35");

        let outcome =
            processor.process_batch(vec![MockEngineResponse::tokens(35, vec![vec![1]], false)]);

        assert_eq!(outcome.completed_client_ids, vec![35]);
        assert_eq!(processor.active_requests(), 0);
        assert_eq!(
            sink.errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_slice(),
            ["consumer disconnected"]
        );
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let processor = ProcessorCore::default();
        let sink = Arc::new(RecordingSink::default());
        processor
            .register(40, 0, 1, sink.clone(), 0.0)
            .expect("first registration");

        let error = processor
            .register(40, 0, 1, sink, 0.0)
            .expect_err("duplicate must fail");
        assert!(error.contains("already registered"));
    }
}
