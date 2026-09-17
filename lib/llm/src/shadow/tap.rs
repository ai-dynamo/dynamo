// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pipeline operator that mirrors requests, and optionally responses, to
//! the tap queues.
//!
//! The request path pays for the select filters, one field copy per distinct
//! filter set, one clock read, and one `try_send` per tap. Encoding and the
//! network send happen on the publisher task. A full queue drops the record
//! and counts it; the request path never waits for a shadow.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use dynamo_runtime::engine::Data;
use dynamo_runtime::pipeline::{
    AsyncEngineContext, AsyncEngineContextProvider, ManyOut, Operator, PipelineOperator,
    ResponseStream, ServerStreamingEngine, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};
use prometheus::IntCounter;
use tokio::sync::mpsc;

use super::config::{Capture, ResponseOptions, TapSpec};
use super::envelope::{
    ENVELOPE_SCHEMA_VERSION, ShadowChoice, ShadowEnvelope, ShadowOrigin, ShadowOutcome,
    ShadowResponse,
};
use super::filter::{Projection, project};
use crate::protocols::TokenIdType;
use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest};

/// What a joined record needs from a response chunk.
pub(crate) trait ShadowChunk {
    fn token_ids(&self) -> &[TokenIdType];
    fn finish_reason(&self) -> Option<&FinishReason>;
    fn index(&self) -> u32;
}

impl ShadowChunk for BackendOutput {
    fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }
    fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }
    fn index(&self) -> u32 {
        self.index.unwrap_or(0)
    }
}

impl ShadowChunk for LLMEngineOutput {
    fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }
    fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }
    fn index(&self) -> u32 {
        self.index.unwrap_or(0)
    }
}

#[derive(Clone)]
pub struct TapCounters {
    pub queued: IntCounter,
    pub dropped: IntCounter,
}

/// The request-path half of one tap. The publisher task owns the receiver.
pub struct TapQueue {
    spec: TapSpec,
    tx: mpsc::Sender<ShadowEnvelope>,
    seq: AtomicU64,
    counters: TapCounters,
}

impl TapQueue {
    pub fn new(
        spec: TapSpec,
        counters: TapCounters,
    ) -> (Arc<Self>, mpsc::Receiver<ShadowEnvelope>) {
        let (tx, rx) = mpsc::channel(spec.capacity);
        let queue = Arc::new(Self {
            spec,
            tx,
            seq: AtomicU64::new(0),
            counters,
        });
        (queue, rx)
    }

    /// `seq` is taken before the send so that a dropped record leaves a gap
    /// the consumer can see.
    fn send(&self, pending: Pending, response: Option<ShadowResponse>) {
        let envelope = ShadowEnvelope {
            schema_version: ENVELOPE_SCHEMA_VERSION,
            tap: self.spec.name.clone(),
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            request_id: pending.request_id,
            origin: pending.origin,
            filters: self.spec.filters.names.clone(),
            arrival_unix_ns: pending.arrival_unix_ns,
            request: pending.request,
            response,
        };
        match self.tx.try_send(envelope) {
            Ok(()) => self.counters.queued.inc(),
            Err(_) => self.counters.dropped.inc(),
        }
    }
}

struct Pending {
    request_id: String,
    origin: ShadowOrigin,
    arrival_unix_ns: u64,
    request: Arc<PreprocessedRequest>,
}

pub struct ShadowTap {
    taps: Vec<Arc<TapQueue>>,
    origin: ShadowOrigin,
}

impl ShadowTap {
    pub fn new(taps: Vec<Arc<TapQueue>>, origin: ShadowOrigin) -> Arc<Self> {
        Arc::new(Self { taps, origin })
    }

    /// Wrap as a `PipelineOperator` over the given response type. The response
    /// type does not appear in the struct, so the caller names it.
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_operator_for<Resp>(
        self: &Arc<Self>,
    ) -> Arc<
        PipelineOperator<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<Resp>>,
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<Resp>>,
        >,
    >
    where
        Resp: Data + ShadowChunk,
    {
        Operator::into_operator(self)
    }

    /// Queue the request-only records and return the recorders for the taps
    /// that also capture the response.
    fn mirror(
        &self,
        request: &PreprocessedRequest,
        context: Arc<dyn AsyncEngineContext>,
    ) -> Vec<Recorder> {
        let request_id = context.id();
        let arrival = SystemTime::now();
        let started = Instant::now();
        let arrival_unix_ns = arrival
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64);

        // Each distinct filter set is copied at most once per request, however
        // many taps use it.
        let mut copies: Vec<(Projection, Arc<PreprocessedRequest>)> = Vec::new();
        let mut recorders = Vec::new();

        for tap in &self.taps {
            if !tap.spec.filters.accepts(request) {
                continue;
            }
            let projection = tap.spec.filters.projection;
            let copy = match copies.iter().find(|(seen, _)| *seen == projection) {
                Some((_, copy)) => Arc::clone(copy),
                None => {
                    let copy = Arc::new(project(request, projection));
                    copies.push((projection, Arc::clone(&copy)));
                    copy
                }
            };
            let pending = Pending {
                request_id: request_id.to_string(),
                origin: self.origin,
                arrival_unix_ns,
                request: copy,
            };
            match tap.spec.capture {
                Capture::Request => tap.send(pending, None),
                Capture::RequestResponse => recorders.push(Recorder::new(
                    Arc::clone(tap),
                    Arc::clone(&context),
                    pending,
                    started,
                    tap.spec.response,
                )),
            }
        }
        recorders
    }
}

#[async_trait]
impl<Resp>
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
    > for ShadowTap
where
    Resp: Data + ShadowChunk,
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    ) -> Result<ManyOut<Annotated<Resp>>> {
        let mut recorders = self.mirror(&request, request.context());
        let response = match next.generate(request).await {
            Ok(response) => response,
            Err(error) => {
                for recorder in &mut recorders {
                    recorder.saw_error = true;
                }
                return Err(error);
            }
        };
        if recorders.is_empty() {
            return Ok(response);
        }

        let context = response.context();
        let recorded = stream::unfold(
            (response, recorders),
            |(mut response, mut recorders)| async move {
                match response.next().await {
                    Some(chunk) => {
                        for recorder in &mut recorders {
                            recorder.observe(&chunk);
                        }
                        Some((chunk, (response, recorders)))
                    }
                    None => {
                        for recorder in &mut recorders {
                            recorder.emit(false);
                        }
                        None
                    }
                }
            },
        );
        Ok(ResponseStream::new(Box::pin(recorded), context))
    }
}

/// Accumulates one response and publishes the joined record exactly once:
/// when the stream ends, or when the stream is dropped first. Without the drop
/// path a cancelled request would vanish from the replay set, and the shadow
/// would see a lighter load than the primary did.
struct Recorder {
    tap: Arc<TapQueue>,
    context: Arc<dyn AsyncEngineContext>,
    pending: Option<Pending>,
    started: Instant,
    options: ResponseOptions,
    saw_error: bool,
    saw_cancel: bool,
    /// Choices the request asked for. The response is complete when this many
    /// choices carried a finish reason other than error or cancel.
    expected_choices: usize,
    finished_choices: usize,
    /// Ordered by index. A request rarely has more than a few choices, so a
    /// sorted `Vec` beats a map.
    choices: Vec<ShadowChoice>,
    first_token: Option<Duration>,
    last_token: Option<Duration>,
    chunk_offsets_ns: Vec<u64>,
}

impl Recorder {
    fn new(
        tap: Arc<TapQueue>,
        context: Arc<dyn AsyncEngineContext>,
        pending: Pending,
        started: Instant,
        options: ResponseOptions,
    ) -> Self {
        let expected_choices = pending
            .request
            .sampling_options
            .n
            .map_or(1, usize::from)
            .max(1);
        Self {
            tap,
            context,
            pending: Some(pending),
            started,
            options,
            saw_error: false,
            saw_cancel: false,
            expected_choices,
            finished_choices: 0,
            choices: Vec::new(),
            first_token: None,
            last_token: None,
            chunk_offsets_ns: Vec::new(),
        }
    }

    fn observe<Resp: ShadowChunk>(&mut self, chunk: &Annotated<Resp>) {
        if chunk.is_error() {
            self.saw_error = true;
        }
        let Some(data) = &chunk.data else {
            return;
        };
        let index = data.index();
        let position = match self
            .choices
            .binary_search_by_key(&index, |choice| choice.index)
        {
            Ok(position) => position,
            Err(position) => {
                self.choices.insert(
                    position,
                    ShadowChoice {
                        index,
                        finish_reason: None,
                        output_tokens: 0,
                        token_ids: Vec::new(),
                    },
                );
                position
            }
        };
        let choice = &mut self.choices[position];

        if let Some(reason) = data.finish_reason() {
            match reason {
                FinishReason::Error(_) => self.saw_error = true,
                FinishReason::Cancelled => self.saw_cancel = true,
                _ if choice.finish_reason.is_none() => self.finished_choices += 1,
                _ => {}
            }
            choice.finish_reason = Some(reason.to_string());
        }
        let tokens = data.token_ids();
        if tokens.is_empty() {
            return;
        }
        choice.output_tokens += tokens.len() as u64;
        if self.options.tokens {
            choice.token_ids.extend_from_slice(tokens);
        }
        let offset = self.started.elapsed();
        self.first_token.get_or_insert(offset);
        self.last_token = Some(offset);
        if self.options.chunk_timing {
            self.chunk_offsets_ns.push(offset.as_nanos() as u64);
        }
    }

    fn emit(&mut self, dropped_early: bool) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        // A client disconnect often reaches the tap as an error chunk. The
        // request context tells the two apart: a replay must treat a cancel as
        // load the client gave up on, not as a failure of the primary.
        let cancelled = self.saw_cancel || self.context.is_stopped() || self.context.is_killed();
        // A choice that finished must not hide another that failed, and a
        // context that is stopped during cleanup must not turn a finished
        // response into a cancel.
        let finished =
            self.finished_choices >= self.expected_choices && !self.saw_error && !self.saw_cancel;
        let outcome = if finished {
            ShadowOutcome::Complete
        } else if cancelled {
            ShadowOutcome::Cancelled
        } else if self.saw_error {
            ShadowOutcome::Error
        } else if dropped_early {
            ShadowOutcome::Cancelled
        } else {
            ShadowOutcome::Complete
        };
        let nanos = |offset: Duration| offset.as_nanos() as u64;
        let response = ShadowResponse {
            outcome,
            output_tokens: self.choices.iter().map(|choice| choice.output_tokens).sum(),
            choices: std::mem::take(&mut self.choices),
            first_token_offset_ns: self.first_token.map(nanos),
            last_token_offset_ns: self.last_token.map(nanos),
            end_offset_ns: nanos(self.started.elapsed()),
            chunk_offsets_ns: std::mem::take(&mut self.chunk_offsets_ns),
        };
        self.tap.send(pending, Some(response));
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.emit(true);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::BTreeMap;

    use dynamo_runtime::pipeline::{AsyncEngine, Context};

    use super::*;
    use crate::shadow::config::ShadowConfig;

    pub(in crate::shadow) fn counters() -> TapCounters {
        let counter = |name: &str| IntCounter::new(name, name).unwrap();
        TapCounters {
            queued: counter("queued"),
            dropped: counter("dropped"),
        }
    }

    fn taps(yaml: &str) -> Vec<(Arc<TapQueue>, mpsc::Receiver<ShadowEnvelope>)> {
        ShadowConfig::from_yaml(yaml)
            .unwrap()
            .taps
            .into_iter()
            .map(|spec| TapQueue::new(spec, counters()))
            .collect()
    }

    fn request(id: &str) -> SingleIn<PreprocessedRequest> {
        let mut request = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.extra_args = Some(serde_json::json!({"k": "v"}));
        Context::with_id_and_metadata(request, id.to_string(), BTreeMap::new())
    }

    fn chunk(tokens: Vec<TokenIdType>, finish: Option<FinishReason>) -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            token_ids: tokens,
            finish_reason: finish,
            ..Default::default()
        })
    }

    /// Records the request it was given and answers with fixed chunks.
    struct Engine {
        chunks: Vec<Annotated<LLMEngineOutput>>,
        seen: std::sync::Mutex<Vec<PreprocessedRequest>>,
        fail: bool,
    }

    impl Engine {
        fn new(chunks: Vec<Annotated<LLMEngineOutput>>) -> Arc<Self> {
            Arc::new(Self {
                chunks,
                seen: Default::default(),
                fail: false,
            })
        }
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        > for Engine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            if self.fail {
                anyhow::bail!("no workers");
            }
            let (request, context) = request.transfer(());
            self.seen.lock().unwrap().push(request);
            Ok(ResponseStream::new(
                Box::pin(stream::iter(self.chunks.clone())),
                context.context(),
            ))
        }
    }

    async fn run(
        tap: &ShadowTap,
        engine: Arc<Engine>,
        id: &str,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        Operator::<_, _, _, ManyOut<Annotated<LLMEngineOutput>>>::generate(tap, request(id), engine)
            .await
    }

    const REQUEST_TAP: &str = "schema_version: 1\ntaps:\n  - {name: a, capture: request, filters: [tokens-only], capacity: 2}\n";
    const JOINED_TAP: &str = "schema_version: 1\ntaps:\n  - {name: j, capture: request_response, emit: joined, response: {chunk_timing: true}}\n";

    #[tokio::test]
    async fn request_passes_through_unchanged_and_is_mirrored_projected() {
        let mut queues = taps(REQUEST_TAP);
        let (queue, mut receiver) = queues.remove(0);
        let tap = ShadowTap::new(vec![queue], ShadowOrigin::Chat);
        let engine = Engine::new(vec![chunk(vec![9], Some(FinishReason::Stop))]);

        let response: Vec<_> = run(&tap, engine.clone(), "req-1")
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(response.len(), 1);
        assert_eq!(response[0].data.as_ref().unwrap().token_ids, vec![9]);

        let downstream = engine.seen.lock().unwrap().pop().unwrap();
        assert!(
            downstream.extra_args.is_some(),
            "the live request keeps every field"
        );

        let envelope = receiver.try_recv().unwrap();
        assert_eq!(envelope.request_id, "req-1");
        assert_eq!(envelope.seq, 0);
        assert_eq!(envelope.origin, ShadowOrigin::Chat);
        assert_eq!(*envelope.request.token_ids, vec![1, 2, 3]);
        assert!(
            envelope.request.extra_args.is_none(),
            "tokens-only drops extra_args"
        );
        assert!(envelope.response.is_none());
        assert!(envelope.arrival_unix_ns > 0);
        assert!(receiver.try_recv().is_err(), "one record per request");
    }

    #[tokio::test]
    async fn stalled_consumer_drops_and_counts_without_blocking() {
        let mut queues = taps(REQUEST_TAP);
        let (queue, mut receiver) = queues.remove(0);
        let tap = ShadowTap::new(vec![queue.clone()], ShadowOrigin::Chat);

        // Nothing drains `receiver`. Capacity is 2.
        let serve = async {
            for index in 0..5 {
                let engine = Engine::new(vec![chunk(vec![9], Some(FinishReason::Stop))]);
                let chunks: Vec<_> = run(&tap, engine, &format!("r{index}"))
                    .await
                    .unwrap()
                    .collect()
                    .await;
                assert_eq!(chunks.len(), 1);
            }
        };
        tokio::time::timeout(Duration::from_secs(5), serve)
            .await
            .expect("a full tap queue must not block serving");

        assert_eq!(queue.counters.queued.get(), 2);
        assert_eq!(queue.counters.dropped.get(), 3);
        assert_eq!(receiver.try_recv().unwrap().seq, 0);
        assert_eq!(receiver.try_recv().unwrap().seq, 1);

        // The next record shows the gap the drops left.
        let engine = Engine::new(vec![]);
        let _ = run(&tap, engine, "r5").await.unwrap();
        assert_eq!(receiver.try_recv().unwrap().seq, 5);
    }

    #[tokio::test]
    async fn joined_record_is_published_once_when_the_stream_ends() {
        let mut queues = taps(JOINED_TAP);
        let (queue, mut receiver) = queues.remove(0);
        let tap = ShadowTap::new(vec![queue], ShadowOrigin::Preprocessed);
        let engine = Engine::new(vec![
            chunk(vec![10, 11], None),
            chunk(vec![12], Some(FinishReason::Stop)),
        ]);

        let mut response = run(&tap, engine, "req-j").await.unwrap();
        assert!(response.next().await.is_some());
        assert!(
            receiver.try_recv().is_err(),
            "nothing is published mid-stream"
        );
        assert!(response.next().await.is_some());
        assert!(response.next().await.is_none());

        let envelope = receiver.try_recv().unwrap();
        assert_eq!(envelope.request_id, "req-j");
        assert_eq!(*envelope.request.token_ids, vec![1, 2, 3]);
        let recorded = envelope.response.unwrap();
        assert_eq!(recorded.outcome, ShadowOutcome::Complete);
        assert_eq!(recorded.choices.len(), 1);
        assert_eq!(recorded.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(recorded.choices[0].token_ids, vec![10, 11, 12]);
        assert_eq!(recorded.output_tokens, 3);
        assert_eq!(recorded.chunk_offsets_ns.len(), 2);
        assert!(recorded.first_token_offset_ns <= recorded.last_token_offset_ns);

        drop(response);
        assert!(
            receiver.try_recv().is_err(),
            "drop after the end publishes nothing more"
        );
    }

    #[tokio::test]
    async fn dropped_stream_publishes_a_cancelled_record() {
        let mut queues = taps(JOINED_TAP);
        let (queue, mut receiver) = queues.remove(0);
        let tap = ShadowTap::new(vec![queue], ShadowOrigin::Chat);
        let engine = Engine::new(vec![chunk(vec![10], None), chunk(vec![11], None)]);

        let mut response = run(&tap, engine, "req-c").await.unwrap();
        assert!(response.next().await.is_some());
        drop(response);

        let recorded = receiver.try_recv().unwrap().response.unwrap();
        assert_eq!(recorded.outcome, ShadowOutcome::Cancelled);
        assert_eq!(recorded.choices[0].token_ids, vec![10]);
    }

    #[tokio::test]
    async fn error_chunk_and_engine_error_publish_error_records() {
        let mut queues = taps(JOINED_TAP);
        let (queue, mut receiver) = queues.remove(0);
        let tap = ShadowTap::new(vec![queue], ShadowOrigin::Chat);

        let engine = Engine::new(vec![chunk(vec![10], None), Annotated::from_error("boom")]);
        let _: Vec<_> = run(&tap, engine, "req-e1").await.unwrap().collect().await;
        let recorded = receiver.try_recv().unwrap().response.unwrap();
        assert_eq!(recorded.outcome, ShadowOutcome::Error);
        assert_eq!(recorded.output_tokens, 1);

        let failing = Arc::new(Engine {
            chunks: vec![],
            seen: Default::default(),
            fail: true,
        });
        assert!(run(&tap, failing, "req-e2").await.is_err());
        let envelope = receiver.try_recv().unwrap();
        assert_eq!(envelope.request_id, "req-e2");
        assert_eq!(envelope.response.unwrap().outcome, ShadowOutcome::Error);
    }

    #[tokio::test]
    async fn taps_with_one_filter_set_share_a_copy_and_selects_skip_quietly() {
        let yaml = "schema_version: 1\ntaps:\n  - {name: a, capture: request, filters: [tokens-only]}\n  - {name: b, capture: request, filters: [tokens-only]}\n  - {name: c, capture: request, models: [other]}\n";
        let mut queues = taps(yaml);
        let (c, mut c_rx) = queues.pop().unwrap();
        let (b, mut b_rx) = queues.pop().unwrap();
        let (a, mut a_rx) = queues.pop().unwrap();
        let tap = ShadowTap::new(vec![a, b, c.clone()], ShadowOrigin::Chat);

        let _ = run(&tap, Engine::new(vec![]), "req-s").await.unwrap();
        let first = a_rx.try_recv().unwrap();
        let second = b_rx.try_recv().unwrap();
        assert!(Arc::ptr_eq(&first.request, &second.request));
        assert_eq!(&*second.tap, "b");

        assert!(
            c_rx.try_recv().is_err(),
            "model filter rejected the request"
        );
        assert_eq!(
            c.counters.dropped.get(),
            0,
            "a filtered request is not a drop"
        );
    }

    #[tokio::test]
    async fn error_chunk_after_a_client_cancel_is_a_cancelled_record() {
        let mut queues = taps(JOINED_TAP);
        let (queue, mut receiver) = queues.remove(0);
        let tap = ShadowTap::new(vec![queue], ShadowOrigin::Chat);
        let engine = Engine::new(vec![
            chunk(vec![10], None),
            Annotated::from_error("stream closed"),
        ]);

        let request = request("req-x");
        let context = request.context();
        let mut response = Operator::<_, _, _, ManyOut<Annotated<LLMEngineOutput>>>::generate(
            &*tap, request, engine,
        )
        .await
        .unwrap();
        assert!(response.next().await.is_some());
        context.stop_generating();
        let _: Vec<_> = response.collect().await;

        let recorded = receiver.try_recv().unwrap().response.unwrap();
        assert_eq!(recorded.outcome, ShadowOutcome::Cancelled);
    }

    fn choice_chunk(
        index: u32,
        tokens: Vec<TokenIdType>,
        finish: Option<FinishReason>,
    ) -> Annotated<LLMEngineOutput> {
        let mut chunk = chunk(tokens, finish);
        chunk.data.as_mut().unwrap().index = Some(index);
        chunk
    }

    #[tokio::test]
    async fn a_finished_choice_does_not_hide_a_failed_choice() {
        let mut queues = taps(JOINED_TAP);
        let (queue, mut receiver) = queues.remove(0);
        let tap = ShadowTap::new(vec![queue], ShadowOrigin::Chat);
        let engine = Engine::new(vec![
            choice_chunk(0, vec![10], None),
            choice_chunk(1, vec![20], None),
            choice_chunk(0, vec![11], Some(FinishReason::Stop)),
            choice_chunk(
                1,
                vec![21],
                Some(FinishReason::Error("worker lost".to_string())),
            ),
        ]);

        let _: Vec<_> = run(&tap, engine, "req-n").await.unwrap().collect().await;
        let recorded = receiver.try_recv().unwrap().response.unwrap();
        assert_eq!(recorded.outcome, ShadowOutcome::Error);
        assert_eq!(recorded.output_tokens, 4);
        assert_eq!(recorded.choices[0].token_ids, vec![10, 11]);
        assert_eq!(recorded.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(recorded.choices[1].token_ids, vec![20, 21]);
    }

    #[tokio::test]
    async fn a_response_is_complete_only_when_every_requested_choice_finished() {
        let mut queues = taps(JOINED_TAP);
        let (queue, mut receiver) = queues.remove(0);
        let tap = ShadowTap::new(vec![queue], ShadowOrigin::Chat);
        let mut two_choices = request("req-n2");
        two_choices.sampling_options.n = Some(2);

        let engine = Engine::new(vec![
            choice_chunk(1, vec![20], Some(FinishReason::Stop)),
            choice_chunk(0, vec![10], None),
        ]);
        let mut response = Operator::<_, _, _, ManyOut<Annotated<LLMEngineOutput>>>::generate(
            &*tap,
            two_choices,
            engine,
        )
        .await
        .unwrap();
        assert!(response.next().await.is_some());
        assert!(response.next().await.is_some());
        drop(response);

        let recorded = receiver.try_recv().unwrap().response.unwrap();
        assert_eq!(recorded.outcome, ShadowOutcome::Cancelled);
        assert_eq!(recorded.choices[0].index, 0, "choices are ordered by index");
        assert_eq!(recorded.choices[1].finish_reason.as_deref(), Some("stop"));
    }

    /// Fails the first attempt the way an unreachable worker does.
    struct FlakyEngine {
        calls: AtomicU64,
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        > for FlakyEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            use dynamo_runtime::error::{DynamoError, ErrorType};
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(anyhow::anyhow!(
                    DynamoError::builder()
                        .error_type(ErrorType::CannotConnect)
                        .message("no responders")
                        .build()
                ));
            }
            let context = request.context();
            let chunks = vec![
                chunk(vec![10], None),
                chunk(vec![11], Some(FinishReason::Stop)),
            ];
            Ok(ResponseStream::new(Box::pin(stream::iter(chunks)), context))
        }
    }

    /// The tap is linked above migration, as in the frontend pipeline. A
    /// request that migration retries is still one client request.
    #[tokio::test]
    async fn migration_retry_yields_one_record_of_each_kind() {
        use dynamo_runtime::pipeline::{SegmentSource, ServiceBackend, Source};

        use crate::http::service::metrics::Metrics;
        use crate::migration::Migration;

        let yaml = "schema_version: 1\ntaps:\n  - {name: r, capture: request}\n  - {name: j, capture: request_response}\n";
        let mut queues = taps(yaml);
        let (joined, mut joined_rx) = queues.pop().unwrap();
        let (requests, mut requests_rx) = queues.pop().unwrap();
        let tap = ShadowTap::new(vec![requests, joined], ShadowOrigin::Preprocessed)
            .into_operator_for::<LLMEngineOutput>();
        let migration = Migration::new(3, None, "m".to_string(), Arc::new(Metrics::new()))
            .into_operator_for::<LLMEngineOutput>();
        let flaky = Arc::new(FlakyEngine {
            calls: AtomicU64::new(0),
        });

        let frontend = SegmentSource::<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        >::new();
        let pipeline = frontend
            .link(tap.forward_edge())
            .unwrap()
            .link(migration.forward_edge())
            .unwrap()
            .link(ServiceBackend::from_engine(flaky.clone()))
            .unwrap()
            .link(migration.backward_edge())
            .unwrap()
            .link(tap.backward_edge())
            .unwrap()
            .link_terminal(frontend)
            .unwrap();

        let chunks: Vec<_> = pipeline
            .generate(request("req-m"))
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(
            flaky.calls.load(Ordering::SeqCst),
            2,
            "migration retried once"
        );
        assert_eq!(chunks.len(), 2);

        assert_eq!(requests_rx.try_recv().unwrap().request_id, "req-m");
        assert!(requests_rx.try_recv().is_err());
        let recorded = joined_rx.try_recv().unwrap().response.unwrap();
        assert_eq!(recorded.outcome, ShadowOutcome::Complete);
        assert_eq!(recorded.choices[0].token_ids, vec![10, 11]);
        assert!(joined_rx.try_recv().is_err());
    }

    #[test]
    fn envelope_round_trips_through_the_event_plane_encoding() {
        let envelope = ShadowEnvelope {
            schema_version: ENVELOPE_SCHEMA_VERSION,
            tap: "t".into(),
            seq: 7,
            request_id: "id".to_string(),
            origin: ShadowOrigin::Completions,
            filters: vec!["tokens-only".to_string()].into(),
            arrival_unix_ns: 42,
            request: Arc::new(project(&request("id"), Projection::TOKENS_ONLY)),
            response: Some(ShadowResponse {
                outcome: ShadowOutcome::Complete,
                choices: vec![ShadowChoice {
                    index: 0,
                    finish_reason: Some("stop".to_string()),
                    output_tokens: 1,
                    token_ids: vec![5],
                }],
                output_tokens: 1,
                first_token_offset_ns: Some(1),
                last_token_offset_ns: Some(1),
                end_offset_ns: 2,
                chunk_offsets_ns: vec![],
            }),
        };
        let bytes = rmp_serde::to_vec_named(&envelope).unwrap();
        let decoded: ShadowEnvelope = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.seq, 7);
        assert_eq!(*decoded.request.token_ids, vec![1, 2, 3]);
        assert_eq!(decoded.response, envelope.response);
        assert_eq!(decoded.origin, ShadowOrigin::Completions);
    }
}
