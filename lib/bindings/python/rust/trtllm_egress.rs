// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use dynamo_runtime::protocols::annotated::Annotated;
use serde::Deserialize;
use serde_json::{Map, Value, json};

pub(crate) trait OwnedFrameSink: Send + Sync {
    fn send(&self, frame: Annotated<Value>) -> Result<(), String>;
    fn close(&self);
    fn close_with_error(&self, message: String);
}

#[derive(Debug, Deserialize)]
pub(crate) struct EngineResponse {
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
impl EngineResponse {
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
}

#[derive(Debug, Default)]
struct ChoiceState {
    token_ids: Vec<u32>,
    emitted: usize,
    finish_reason: Option<String>,
    stop_reason: Option<String>,
}

#[derive(Debug)]
struct OwnedResponseState {
    prompt_tokens: usize,
    choices: Vec<ChoiceState>,
}

impl OwnedResponseState {
    fn new(prompt_tokens: usize, num_choices: usize) -> Self {
        Self {
            prompt_tokens,
            choices: (0..num_choices).map(|_| ChoiceState::default()).collect(),
        }
    }

    fn apply(&mut self, response: EngineResponse) -> Vec<Value> {
        for (index, new_tokens) in response.new_token_ids.into_iter().enumerate() {
            let Some(choice) = self.choices.get_mut(index) else {
                break;
            };
            choice.token_ids.extend(new_tokens);
        }

        Self::update_reasons(
            &mut self.choices,
            response.finish_reasons,
            |choice, reason| choice.finish_reason = reason,
        );
        Self::update_reasons(
            &mut self.choices,
            response.stop_reasons,
            |choice, reason| choice.stop_reason = reason,
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
    response_state: Mutex<OwnedResponseState>,
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

#[derive(Debug)]
struct IndexedOutcome {
    ordinal: usize,
    client_id: u64,
    completed: bool,
    processed: bool,
    frames_sent: usize,
}

#[derive(Default)]
struct ProcessorShared {
    requests: Mutex<HashMap<u64, Arc<RegisteredRequest>>>,
    responses_processed: AtomicUsize,
    responses_dropped: AtomicUsize,
    frames_sent: AtomicUsize,
}

impl ProcessorShared {
    fn process_response(&self, ordinal: usize, response: EngineResponse) -> IndexedOutcome {
        let client_id = response.client_id;
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
                client_id,
                completed: false,
                processed: false,
                frames_sent: 0,
            };
        };

        self.responses_processed.fetch_add(1, Ordering::Relaxed);
        let mut terminal = response.is_final || response.error_msg.is_some();
        let mut frames_sent = 0;

        if let Some(error) = response.error_msg {
            request.sink.close_with_error(error);
        } else {
            spin_for(request.calibrated_work_us);
            let frames = request
                .response_state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .apply(response);
            let mut sink_failed = false;
            for frame in frames {
                match request.sink.send(Annotated::from_data(frame)) {
                    Ok(()) => {
                        frames_sent += 1;
                        self.frames_sent.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(error) => {
                        request.sink.close_with_error(error);
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

        if terminal {
            self.remove(client_id, &request);
        }

        IndexedOutcome {
            ordinal,
            client_id,
            completed: terminal,
            processed: true,
            frames_sent,
        }
    }

    fn remove(&self, client_id: u64, request: &Arc<RegisteredRequest>) {
        let mut requests = self
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests
            .get(&client_id)
            .is_some_and(|current| Arc::ptr_eq(current, request))
        {
            requests.remove(&client_id);
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
            request.sink.close();
        }
    }
}

struct IndexedResponse {
    ordinal: usize,
    response: EngineResponse,
}

struct ShardCommand {
    responses: Vec<IndexedResponse>,
    reply: SyncSender<Vec<IndexedOutcome>>,
}

pub(crate) struct ShardedProcessor {
    shared: Arc<ProcessorShared>,
    shard_count: usize,
    senders: Mutex<Option<Vec<SyncSender<ShardCommand>>>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl ShardedProcessor {
    pub(crate) fn new(shard_count: usize, queue_depth: usize) -> Result<Self, String> {
        if shard_count == 0 {
            return Err("shard_count must be at least 1".to_string());
        }
        if queue_depth == 0 {
            return Err("queue_depth must be at least 1".to_string());
        }

        let shared = Arc::new(ProcessorShared::default());
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
            senders: Mutex::new(Some(senders)),
            workers: Mutex::new(workers),
        })
    }

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
            .shared
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.contains_key(&client_id) {
            return Err(format!("client {client_id} is already registered"));
        }
        requests.insert(
            client_id,
            Arc::new(RegisteredRequest {
                response_state: Mutex::new(OwnedResponseState::new(prompt_tokens, num_choices)),
                sink,
                calibrated_work_us,
            }),
        );
        Ok(())
    }

    pub(crate) fn process_batch(
        &self,
        responses: Vec<EngineResponse>,
    ) -> Result<BatchOutcome, String> {
        let mut partitions = (0..self.shard_count)
            .map(|_| Vec::new())
            .collect::<Vec<_>>();
        for (ordinal, response) in responses.into_iter().enumerate() {
            let shard = response.client_id as usize % self.shard_count;
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
                outcome.completed_client_ids.push(response.client_id);
            }
        }
        Ok(outcome)
    }

    pub(crate) fn cancel(&self, client_id: u64) -> bool {
        let request = self
            .shared
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&client_id);
        if let Some(request) = request {
            request.sink.close();
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
}

impl Drop for ShardedProcessor {
    fn drop(&mut self) {
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
                tracing::error!("owned response shard panicked during shutdown");
            }
        }
        self.shared.close_all();
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

fn spin_for(work_us: f64) {
    if work_us <= 0.0 {
        return;
    }
    let deadline = Instant::now() + Duration::from_secs_f64(work_us / 1_000_000.0);
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::Duration;

    use dynamo_runtime::protocols::annotated::Annotated;
    use serde_json::json;

    use super::{EngineResponse, OwnedFrameSink, ShardedProcessor};

    #[derive(Default)]
    struct RecordingSink {
        frames: Mutex<Vec<Annotated<serde_json::Value>>>,
        closed: AtomicBool,
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

        fn close_with_error(&self, _message: String) {
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

    impl OwnedFrameSink for GateSink {
        fn send(&self, _frame: Annotated<serde_json::Value>) -> Result<(), String> {
            self.entered
                .send(self.client_id)
                .expect("test receiver remains open");
            let (lock, ready) = &*self.released;
            let mut released = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            while !*released {
                released = ready
                    .wait(released)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            Ok(())
        }

        fn close(&self) {}

        fn close_with_error(&self, _message: String) {
            self.close();
        }
    }

    #[test]
    fn rejects_zero_shards_and_zero_queue_depth() {
        assert!(ShardedProcessor::new(0, 1).is_err());
        assert!(ShardedProcessor::new(1, 0).is_err());
    }

    #[test]
    fn clients_on_different_shards_process_concurrently() {
        let processor = Arc::new(ShardedProcessor::new(2, 1).expect("valid processor"));
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
        processor
            .register(2, 0, 1, sink_zero.clone(), 0.0)
            .expect("register shard zero client");
        processor
            .register(3, 0, 1, sink_one.clone(), 0.0)
            .expect("register shard one client");

        let processing = {
            let processor = processor.clone();
            thread::spawn(move || {
                processor.process_batch(vec![
                    EngineResponse::tokens(2, vec![vec![1]], false),
                    EngineResponse::tokens(3, vec![vec![2]], false),
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
        let processor = ShardedProcessor::new(4, 1).expect("valid processor");
        let sink = Arc::new(RecordingSink::default());
        processor
            .register(9, 2, 1, sink.clone(), 0.0)
            .expect("register client 9");

        let outcome = processor
            .process_batch(vec![
                EngineResponse::tokens(9, vec![vec![10]], false),
                EngineResponse::tokens(9, vec![vec![11]], true),
            ])
            .expect("process ordered batch");

        let frames = sink
            .frames
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(frames[0].data, Some(json!({"token_ids": [10], "index": 0})));
        assert_eq!(frames[1].data.as_ref().unwrap()["token_ids"], json!([11]));
        assert_eq!(outcome.completed_client_ids, vec![9]);
        assert!(sink.closed.load(Ordering::Relaxed));
    }

    #[test]
    fn cancellation_does_not_wait_for_a_backpressured_shard() {
        let processor = Arc::new(ShardedProcessor::new(2, 1).expect("valid processor"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let sink = Arc::new(GateSink {
            entered: entered_tx,
            client_id: 4,
            released: Arc::new((Mutex::new(false), Condvar::new())),
        });
        processor
            .register(4, 0, 1, sink.clone(), 0.0)
            .expect("register client 4");

        let processing = {
            let processor = processor.clone();
            thread::spawn(move || {
                processor.process_batch(vec![EngineResponse::tokens(4, vec![vec![1]], false)])
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
                    .send(processor.cancel(4))
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
}
