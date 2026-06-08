// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use async_stream::stream;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        RouterMode, SingleIn, async_trait, network::Ingress,
    },
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use tokio::time::{Instant, timeout};

const COMPONENT: &str = "worker";
const TOKENS_PER_RESPONSE: usize = 5;
const FIRST_TOKEN_DELAY: Duration = Duration::from_millis(20);
const INTER_TOKEN_DELAY: Duration = Duration::from_millis(8);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
const PHASE_BASELINE: u8 = 0;
const PHASE_TRANSITION: u8 = 1;
const PHASE_POST: u8 = 2;

static CUTOVER_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Clone)]
struct StreamingHandler {
    name: &'static str,
    active: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
}

impl StreamingHandler {
    fn new(name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            active: Arc::new(AtomicUsize::new(0)),
            total: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn active(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    fn total(&self) -> usize {
        self.total.load(Ordering::SeqCst)
    }
}

struct ActiveGuard(Arc<AtomicUsize>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for StreamingHandler {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let (_request, ctx) = input.into_parts();
        self.total.fetch_add(1, Ordering::SeqCst);
        self.active.fetch_add(1, Ordering::SeqCst);

        let active = self.active.clone();
        let name = self.name;
        let output = stream! {
            let _active = ActiveGuard(active);
            tokio::time::sleep(FIRST_TOKEN_DELAY).await;
            for idx in 0..TOKENS_PER_RESPONSE {
                if idx > 0 {
                    tokio::time::sleep(INTER_TOKEN_DELAY).await;
                }
                yield Annotated::from_data(format!("{name}:{idx}"));
            }
        };

        Ok(ResponseStream::new(Box::pin(output), ctx.context()))
    }
}

#[derive(Clone, Debug)]
struct Sample {
    phase: u8,
    source: String,
    ttft: Duration,
    avg_itl: Duration,
}

fn spawn_engine(
    runtime: DistributedRuntime,
    namespace: &'static str,
    endpoint: &'static str,
    handler: Arc<StreamingHandler>,
) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let ingress = Ingress::for_engine(handler)?;
        let component = runtime.namespace(namespace)?.component(COMPONENT)?;
        component
            .endpoint(endpoint)
            .endpoint_builder()
            .handler(ingress)
            .start()
            .await
    })
}

async fn issue_request(
    router: Arc<PushRouter<String, Annotated<String>>>,
    phase: Arc<AtomicU8>,
) -> Result<Sample> {
    let start = Instant::now();
    let mut response = timeout(
        REQUEST_TIMEOUT,
        router.round_robin("measure cutover".to_string().into()),
    )
    .await
    .context("timed out waiting for routed request")??;

    let mut first_token_at = None;
    let mut previous_token_at = None;
    let mut intervals = Vec::new();
    let mut source = None;
    let mut sample_phase = None;

    while let Some(chunk) = timeout(REQUEST_TIMEOUT, response.next())
        .await
        .context("timed out waiting for response token")?
    {
        let now = Instant::now();
        let chunk = chunk.ok().map_err(|err| anyhow!(err))?;
        let Some(token) = chunk.data else {
            continue;
        };

        if first_token_at.is_none() {
            first_token_at = Some(now);
            sample_phase = Some(phase.load(Ordering::SeqCst));
            source = Some(
                token
                    .split_once(':')
                    .map(|(prefix, _)| prefix.to_string())
                    .unwrap_or(token),
            );
        } else if let Some(previous) = previous_token_at {
            intervals.push(now.duration_since(previous));
        }
        previous_token_at = Some(now);
    }

    let Some(first_token_at) = first_token_at else {
        bail!("response stream completed without tokens");
    };
    let Some(source) = source else {
        bail!("response stream did not identify a source worker");
    };
    let Some(sample_phase) = sample_phase else {
        bail!("response stream did not identify a sample phase");
    };

    let avg_itl = if intervals.is_empty() {
        Duration::ZERO
    } else {
        let total_nanos: u128 = intervals.iter().map(Duration::as_nanos).sum();
        Duration::from_nanos((total_nanos / intervals.len() as u128) as u64)
    };

    Ok(Sample {
        phase: sample_phase,
        source,
        ttft: first_token_at.duration_since(start),
        avg_itl,
    })
}

async fn wait_for<F>(description: &str, mut condition: F) -> Result<()>
where
    F: FnMut() -> bool,
{
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if condition() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    bail!("timed out waiting for {description}")
}

fn p95(values: impl Iterator<Item = Duration>) -> Result<Duration> {
    let mut values: Vec<_> = values.collect();
    if values.is_empty() {
        bail!("cannot compute p95 over an empty sample set");
    }
    values.sort_unstable();
    let idx = ((values.len() * 95).div_ceil(100)).saturating_sub(1);
    Ok(values[idx])
}

#[derive(Debug)]
struct CutoverMetrics {
    baseline_samples: usize,
    transition_samples: usize,
    post_samples: usize,
    baseline_ttft_p95: Duration,
    post_ttft_p95: Duration,
    baseline_itl_p95: Duration,
    post_itl_p95: Duration,
}

async fn run_bulwark_style_shadow_cutover(
    namespace: &'static str,
    endpoint: &'static str,
    client_count: usize,
    warmup_duration: Duration,
    post_duration: Duration,
    latency_budget: Duration,
    min_samples: usize,
) -> Result<CutoverMetrics> {
    dynamo_runtime::logging::init();

    let runtime = Runtime::from_current()?;
    let distributed = DistributedRuntime::new(
        runtime.clone(),
        dynamo_runtime::distributed::DistributedConfig::process_local(),
    )
    .await?;

    let primary = StreamingHandler::new("primary");
    let shadow = StreamingHandler::new("shadow");
    let primary_task = spawn_engine(distributed.clone(), namespace, endpoint, primary.clone());

    let client = distributed
        .namespace(namespace)?
        .component(COMPONENT)?
        .endpoint(endpoint)
        .client()
        .await?;
    client.wait_for_instances().await?;
    assert_eq!(
        client.instance_ids().len(),
        1,
        "router should see one worker"
    );

    let router = Arc::new(
        PushRouter::<String, Annotated<String>>::from_client(
            client.clone(),
            RouterMode::RoundRobin,
        )
        .await?,
    );

    let phase = Arc::new(AtomicU8::new(PHASE_BASELINE));
    let stop = Arc::new(AtomicBool::new(false));
    let (sample_tx, mut sample_rx) =
        tokio::sync::mpsc::unbounded_channel::<Result<Sample, String>>();

    let mut clients = Vec::new();
    for _ in 0..client_count {
        let router = router.clone();
        let phase = phase.clone();
        let stop = stop.clone();
        let sample_tx = sample_tx.clone();
        clients.push(tokio::spawn(async move {
            while !stop.load(Ordering::SeqCst) {
                let result = issue_request(router.clone(), phase.clone())
                    .await
                    .map_err(|err| err.to_string());
                let _ = sample_tx.send(result);
            }
        }));
    }
    drop(sample_tx);

    tokio::time::sleep(warmup_duration).await;

    phase.store(PHASE_TRANSITION, Ordering::SeqCst);
    let shadow_task = spawn_engine(distributed.clone(), namespace, endpoint, shadow.clone());
    wait_for("shadow to receive traffic", || shadow.total() > 0).await?;
    assert_eq!(
        client.instance_ids().len(),
        1,
        "router should still see one worker"
    );

    wait_for("primary in-flight requests to drain", || {
        primary.active() == 0
    })
    .await?;
    phase.store(PHASE_POST, Ordering::SeqCst);
    tokio::time::sleep(post_duration).await;

    stop.store(true, Ordering::SeqCst);
    for client_task in clients {
        client_task.await?;
    }

    let mut samples = Vec::new();
    let mut errors = Vec::new();
    while let Some(result) = sample_rx.recv().await {
        match result {
            Ok(sample) => samples.push(sample),
            Err(err) => errors.push(err),
        }
    }

    distributed.shutdown();
    primary_task.await??;
    shadow_task.await??;
    drop(runtime);

    if !errors.is_empty() {
        bail!("client observed request errors during cutover: {errors:?}");
    }

    let baseline: Vec<_> = samples
        .iter()
        .filter(|sample| sample.phase == PHASE_BASELINE)
        .cloned()
        .collect();
    let post: Vec<_> = samples
        .iter()
        .filter(|sample| sample.phase == PHASE_POST)
        .cloned()
        .collect();
    let transition: Vec<_> = samples
        .iter()
        .filter(|sample| sample.phase == PHASE_TRANSITION)
        .cloned()
        .collect();

    assert!(
        baseline.len() >= min_samples,
        "not enough baseline samples: {}",
        baseline.len()
    );
    assert!(
        post.len() >= min_samples,
        "not enough post-cutover samples: {}",
        post.len()
    );
    assert!(
        !transition.is_empty(),
        "transition phase did not exercise live traffic"
    );

    assert!(
        baseline.iter().all(|sample| sample.source == "primary"),
        "baseline traffic should be served by primary: {baseline:?}"
    );
    assert!(
        post.iter().all(|sample| sample.source == "shadow"),
        "post-drain traffic should be served by shadow only: {post:?}"
    );

    let baseline_ttft_p95 = p95(baseline.iter().map(|sample| sample.ttft))?;
    let post_ttft_p95 = p95(post.iter().map(|sample| sample.ttft))?;
    let baseline_itl_p95 = p95(baseline.iter().map(|sample| sample.avg_itl))?;
    let post_itl_p95 = p95(post.iter().map(|sample| sample.avg_itl))?;

    assert!(
        post_ttft_p95 <= baseline_ttft_p95 + latency_budget,
        "post-cutover TTFT p95 regressed: baseline={baseline_ttft_p95:?}, post={post_ttft_p95:?}"
    );
    assert!(
        post_itl_p95 <= baseline_itl_p95 + latency_budget,
        "post-cutover ITL p95 regressed: baseline={baseline_itl_p95:?}, post={post_itl_p95:?}"
    );

    Ok(CutoverMetrics {
        baseline_samples: baseline.len(),
        transition_samples: transition.len(),
        post_samples: post.len(),
        baseline_ttft_p95,
        post_ttft_p95,
        baseline_itl_p95,
        post_itl_p95,
    })
}

fn print_metrics(label: &str, metrics: &CutoverMetrics) {
    println!(
        "{label}: baseline={}, transition={}, post={}, ttft_p95 {:?}->{:?}, itl_p95 {:?}->{:?}",
        metrics.baseline_samples,
        metrics.transition_samples,
        metrics.post_samples,
        metrics.baseline_ttft_p95,
        metrics.post_ttft_p95,
        metrics.baseline_itl_p95,
        metrics.post_itl_p95
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn bulwark_style_shadow_cutover_under_high_client_load() -> Result<()> {
    let _guard = CUTOVER_TEST_LOCK.lock().await;
    let metrics = run_bulwark_style_shadow_cutover(
        "bulwark_cutover_stress",
        "generate_stress",
        64,
        Duration::from_millis(1500),
        Duration::from_millis(1500),
        Duration::from_millis(75),
        64,
    )
    .await?;
    print_metrics("bulwark cutover stress", &metrics);
    Ok(())
}
