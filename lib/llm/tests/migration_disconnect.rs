// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Migration's outcome when a client leaves during a retry: the HTTP handler drops the
//! response, and its disconnect monitor kills the request after it. The migration in flight
//! is recorded as cancelled once, whether the kill comes first or the drop does. A panic that
//! unwinds through a retry is not a cancel and records no outcome.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_llm::discovery::ModelManager;
use dynamo_llm::entrypoint::input::PreprocessedRouting;
use dynamo_llm::http::service::metrics::Metrics;
use dynamo_llm::kv_router::{EncoderRouter, PrefillRouter};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use dynamo_runtime::engine::{AsyncEngine, Data, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, Context, Error, ManyOut, RouterMode, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use futures::{StreamExt, stream};
use tokio::sync::mpsc;

type Out = ManyOut<Annotated<LLMEngineOutput>>;
type Engine = Arc<dyn AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error>>;
type Pipeline<Req, T> = Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Annotated<T>>, Error>>;

const SAMPLE_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1"
);

/// One call of [`Worker`].
#[derive(Clone, Debug)]
enum Turn {
    /// Wait in the eager phase for the request's kill, then answer with no item.
    HangUntilKilled,
    /// Stream these items, then end.
    Stream(Vec<Annotated<LLMEngineOutput>>),
}

/// A worker that answers its calls with `turns` in order; the last turn answers every call
/// after it. It reports each call's index on `entered` as the call arrives.
struct Worker {
    turns: Vec<Turn>,
    calls: AtomicUsize,
    entered: mpsc::UnboundedSender<usize>,
}

impl Worker {
    fn new(turns: Vec<Turn>) -> (Arc<Self>, mpsc::UnboundedReceiver<usize>) {
        let (entered, arrivals) = mpsc::unbounded_channel();
        let worker = Arc::new(Self {
            turns,
            calls: AtomicUsize::new(0),
            entered,
        });
        (worker, arrivals)
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for Worker {
    async fn generate(&self, request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
        let context = request.context();
        let index = self.calls.fetch_add(1, Ordering::AcqRel);
        let _ = self.entered.send(index);
        match self.turns[index.min(self.turns.len() - 1)].clone() {
            Turn::HangUntilKilled => {
                context.killed().await;
                Ok(ResponseStream::new(Box::pin(stream::empty()), context))
            }
            Turn::Stream(items) => Ok(ResponseStream::new(Box::pin(stream::iter(items)), context)),
        }
    }
}

fn worker_unavailable() -> Error {
    DynamoError::builder()
        .class(ErrorClass::WorkerUnavailable)
        .message("worker is not ready")
        .build()
        .into()
}

/// An error item a worker's stream carries when the worker goes away mid-stream: Migration
/// retries it.
fn went_away() -> Annotated<LLMEngineOutput> {
    Annotated::from_err(
        DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message("worker went away")
            .build(),
    )
}

fn token(id: u32) -> Annotated<LLMEngineOutput> {
    Annotated::from_data(LLMEngineOutput {
        token_ids: vec![id],
        ..Default::default()
    })
}

fn card() -> ModelDeploymentCard {
    ModelDeploymentCard::load_from_disk(SAMPLE_MODEL, None).unwrap()
}

fn routing(engine: Engine) -> PreprocessedRouting {
    PreprocessedRouting::new(
        engine,
        PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None),
        EncoderRouter::disabled(),
    )
}

/// [`pipeline_c`], recording into `metrics`.
fn pipeline_c_with_metrics(
    engine: Engine,
    migration_limit: u32,
    metrics: Arc<Metrics>,
) -> Pipeline<PreprocessedRequest, LLMEngineOutput> {
    routing(engine)
        .build_preprocessed_pipeline(&card(), migration_limit, None, metrics)
        .unwrap()
}

fn preprocessed() -> SingleIn<PreprocessedRequest> {
    let request = PreprocessedRequest::builder()
        .model("TinyLlama_v1.1".to_string())
        .token_ids(vec![1, 450, 4996])
        .stop_conditions(StopConditions {
            max_tokens: Some(512),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .eos_token_ids(vec![2])
        .annotations(vec![])
        .build()
        .unwrap();
    Context::with_id_and_metadata(request, "req-1".to_string(), BTreeMap::new())
}

async fn rest<T: Data>(stream: ManyOut<T>) -> Vec<T> {
    tokio::time::timeout(Duration::from_secs(5), stream.collect())
        .await
        .expect("the stream ends")
}

/// Every migration outcome `metrics` recorded, as `type/outcome xN`.
fn migration_outcomes(metrics: &Metrics) -> Vec<String> {
    use dynamo_runtime::metrics::prometheus_names::frontend_service::{
        migration_outcome, migration_type,
    };
    let model = card().display_name;
    let mut recorded = Vec::new();
    for kind in [migration_type::NEW_REQUEST, migration_type::ONGOING_REQUEST] {
        for outcome in [
            migration_outcome::SUCCESS,
            migration_outcome::FAILURE,
            migration_outcome::CANCELLED,
        ] {
            let count = metrics.get_migration_duration_sample_count(&model, kind, outcome);
            if count > 0 {
                recorded.push(format!("{kind}/{outcome} x{count}"));
            }
        }
    }
    recorded
}

/// A retry whose dispatch panics unwinds a live request, with no stop or kill: Migration
/// records no outcome for it.
#[tokio::test]
async fn a_retry_dispatch_panic_is_not_a_request_cancellation() {
    struct PanicOnRetry {
        eager_failure: bool,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, Out, Error> for PanicOnRetry {
        async fn generate(&self, request: SingleIn<PreprocessedRequest>) -> Result<Out, Error> {
            if self.calls.fetch_add(1, Ordering::AcqRel) > 0 {
                panic!("injected retry dispatch panic");
            }
            if self.eager_failure {
                Err(worker_unavailable())
            } else {
                Ok(ResponseStream::new(
                    Box::pin(stream::iter(vec![went_away()])),
                    request.context(),
                ))
            }
        }
    }

    let mut outcomes = Vec::new();
    for eager_failure in [true, false] {
        let engine = Arc::new(PanicOnRetry {
            eager_failure,
            calls: AtomicUsize::new(0),
        });
        let metrics = Arc::new(Metrics::new());
        let pipeline = pipeline_c_with_metrics(engine.clone(), 3, metrics.clone());
        let request = preprocessed();
        let context = request.context();
        let task = tokio::spawn(async move {
            let stream = pipeline.generate(request).await.unwrap();
            rest(stream).await;
        });
        let panic = task.await.expect_err("the retry must panic");
        assert!(panic.is_panic());
        let payload = panic.into_panic();
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"injected retry dispatch panic")
        );
        assert_eq!(engine.calls.load(Ordering::Acquire), 2);
        assert!(!context.is_stopped());
        assert!(!context.is_killed());
        outcomes.push(migration_outcomes(&metrics));
    }
    assert_eq!(outcomes, vec![Vec::<String>::new(); 2]);
}

async fn migration_outcome_after_http_stream_drop(kill_first: bool) -> Vec<String> {
    use dynamo_llm::http::service::disconnect::{
        create_connection_monitor, monitor_for_disconnects,
    };
    use dynamo_llm::http::service::metrics::{CancellationLabels, Endpoint};

    let (worker, mut arrivals) = Worker::new(vec![
        Turn::Stream(vec![token(450), went_away()]),
        Turn::HangUntilKilled,
    ]);
    let metrics = Arc::new(Metrics::new());
    let pipeline = pipeline_c_with_metrics(worker.clone(), 3, metrics.clone());
    let request = preprocessed();
    let context = request.context();
    let response = pipeline.generate(request).await.unwrap();
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(metrics.clone()),
        CancellationLabels {
            model: card().display_name.clone(),
            endpoint: Endpoint::Generate.to_string(),
            request_type: "stream".to_string(),
        },
    )
    .await;
    let inflight = metrics.clone().create_inflight_guard(
        &card().display_name,
        Endpoint::Generate,
        true,
        context.id(),
    );
    let monitored = monitor_for_disconnects(
        response.map(|_| Ok(Default::default())),
        context.clone(),
        inflight,
        stream_handle,
    );
    connection_handle.disarm();
    drop(connection_handle);
    let reader = tokio::spawn(async move { monitored.collect::<Vec<_>>().await });
    for index in 0..2 {
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), arrivals.recv())
                .await
                .expect("the retry reaches its dispatch"),
            Some(index),
        );
    }
    assert!(!context.is_stopped());
    assert!(!context.is_killed());
    assert!(migration_outcomes(&metrics).is_empty());
    if kill_first {
        context.kill();
    }
    // Dropping the HTTP body drops its source and signals the disconnect monitor.
    reader.abort();
    let cancelled = tokio::time::timeout(Duration::from_secs(5), reader)
        .await
        .expect("the dropped HTTP body releases its source")
        .expect_err("the body reader was aborted");
    assert!(cancelled.is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), context.killed())
        .await
        .expect("the HTTP disconnect monitor kills the request");
    assert_eq!(worker.calls(), 2);
    assert_eq!(metrics.get_client_disconnect_count(), 1);
    migration_outcomes(&metrics)
}

/// A kill delivered before the HTTP body drops records one cancelled migration.
#[tokio::test]
async fn a_kill_before_http_stream_drop_records_one_cancelled_migration() {
    assert_eq!(
        migration_outcome_after_http_stream_drop(true).await,
        ["ongoing_request/cancelled x1"]
    );
}

/// An HTTP body dropped during a retry records one cancellation after its monitor kills the request.
#[tokio::test]
async fn an_http_stream_drop_during_a_retry_records_one_cancelled_migration() {
    let outcome = migration_outcome_after_http_stream_drop(false).await;
    assert_eq!(outcome, vec!["ongoing_request/cancelled x1"]);
}
