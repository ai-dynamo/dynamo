// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use aisimulate_core::engine::{WorkerType as EngineWorkerType, prefill_handoff_delay_ms};

use crate::common::handoff::HandoffTransferTiming;
use crate::common::protocols::{KvTransferTimingMode, MockEngineArgs, WorkerType};

pub fn prefill_handoff_transfer_timing(
    num_input_tokens: usize,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
    mode: KvTransferTimingMode,
) -> HandoffTransferTiming {
    HandoffTransferTiming {
        mode,
        full_prompt_tokens: num_input_tokens,
        kv_bytes_per_token,
        bandwidth_gb_s: kv_transfer_bandwidth,
    }
}

/// Compute the modeled handoff delay after a prefill worker emits its terminal token.
///
/// NOTE: this intentionally does not model the internal prefill TTFT itself accurately, and the
/// exact prefill/decode boundary is backend dependent. For now we only care about decode-visible
/// TTFT, which is what the client observes, so modeling the delay as prefill-to-decode handoff is
/// good enough.
pub fn compute_prefill_handoff_delay_ms(
    worker_type: WorkerType,
    completed: bool,
    num_input_tokens: usize,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
) -> Option<f64> {
    let worker_type = match worker_type {
        WorkerType::Aggregated => EngineWorkerType::Aggregated,
        WorkerType::Prefill => EngineWorkerType::Prefill,
        WorkerType::Decode => EngineWorkerType::Decode,
    };
    let delay_ms = prefill_handoff_delay_ms(
        worker_type,
        completed,
        num_input_tokens,
        kv_transfer_bandwidth,
        kv_bytes_per_token,
    );
    if let Some(delay_ms) = delay_ms {
        tracing::debug!(
            num_input_tokens,
            bandwidth_gb_s = kv_transfer_bandwidth,
            delay_ms = format!("{delay_ms:.2}"),
            "KV handoff delay for prefill completion"
        );
    }
    delay_ms
}

/// Compute the KV transfer delay duration for a given number of input tokens.
///
/// Returns `None` if KV transfer simulation is disabled (bandwidth is 0 or not configured).
pub fn compute_kv_transfer_delay(
    args: &MockEngineArgs,
    num_input_tokens: usize,
) -> Option<Duration> {
    compute_prefill_handoff_delay_ms(
        args.worker_type,
        true,
        num_input_tokens,
        args.kv_transfer_bandwidth,
        args.kv_bytes_per_token,
    )
    .map(|delay_ms| Duration::from_secs_f64(delay_ms / 1000.0))
}

const SLEEP_BACKEND_ENV: &str = "DYN_MOCKER_SLEEP_BACKEND";

const SLEEP_DRIFT_ENV: &str = "DYN_MOCKER_SLEEP_DRIFT";

/// Which timer primitive serves a [`sleep_until_precise`] deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepBackend {
    /// Platform default: `Timerfd` on Linux, `TimeDriver` everywhere else.
    Auto,
    /// Linux `timerfd`; falls back to `TimeDriver` if unavailable.
    Timerfd,
    /// Tokio's time driver.
    TimeDriver,
}

impl SleepBackend {
    /// Resolves the target's concrete backend.
    pub fn resolve(self) -> SleepBackend {
        match self {
            SleepBackend::TimeDriver => SleepBackend::TimeDriver,
            SleepBackend::Auto | SleepBackend::Timerfd => {
                if cfg!(target_os = "linux") {
                    SleepBackend::Timerfd
                } else {
                    SleepBackend::TimeDriver
                }
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SleepBackend::Auto => "auto",
            SleepBackend::Timerfd => "timerfd",
            SleepBackend::TimeDriver => "time_driver",
        }
    }

    fn parse(value: &str) -> Option<SleepBackend> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(SleepBackend::Auto),
            "timerfd" => Some(SleepBackend::Timerfd),
            "time_driver" | "time-driver" | "timer_driver" | "timer-driver" => {
                Some(SleepBackend::TimeDriver)
            }
            _ => None,
        }
    }
}

static CONFIGURED_SLEEP_BACKEND: LazyLock<SleepBackend> = LazyLock::new(|| {
    let Ok(value) = std::env::var(SLEEP_BACKEND_ENV) else {
        return SleepBackend::Auto;
    };
    match SleepBackend::parse(&value) {
        Some(backend) => backend,
        None => {
            tracing::warn!(
                env = SLEEP_BACKEND_ENV,
                value,
                "unrecognized sleep backend; using auto"
            );
            SleepBackend::Auto
        }
    }
});

static SLEEP_DRIFT_ENABLED: LazyLock<bool> =
    LazyLock::new(|| dynamo_truthy::env_is_truthy(SLEEP_DRIFT_ENV));

/// Returns the backend selected once from the environment.
pub fn configured_sleep_backend() -> SleepBackend {
    *CONFIGURED_SLEEP_BACKEND
}

/// Returns whether precise-sleep drift accounting is enabled.
pub fn sleep_drift_enabled() -> bool {
    *SLEEP_DRIFT_ENABLED
}

/// Timing data for one measured wake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SleepDriftRecord {
    /// Concrete backend that served the wake. Never `SleepBackend::Auto`.
    pub backend: SleepBackend,
    /// Time from the start of the call to the requested deadline.
    pub requested: Duration,
    /// Time actually spent in the call.
    pub actual: Duration,
    /// `actual - requested`, saturating at zero for an early wake.
    pub drift: Duration,
}

// Extends past one second so delayed wakes do not all collapse into the final bucket.
const DRIFT_BUCKET_BOUNDS_SECS: [f64; 12] = [
    0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0,
];

struct DriftHistogram {
    buckets: [AtomicU64; DRIFT_BUCKET_BOUNDS_SECS.len()],
    count: AtomicU64,
    total_nanos: AtomicU64,
    max_nanos: AtomicU64,
}

impl DriftHistogram {
    const fn new() -> Self {
        Self {
            buckets: [const { AtomicU64::new(0) }; DRIFT_BUCKET_BOUNDS_SECS.len()],
            count: AtomicU64::new(0),
            total_nanos: AtomicU64::new(0),
            max_nanos: AtomicU64::new(0),
        }
    }

    fn observe(&self, drift: Duration) -> u64 {
        let secs = drift.as_secs_f64();
        for (bucket, bound) in self.buckets.iter().zip(DRIFT_BUCKET_BOUNDS_SECS) {
            if secs <= bound {
                bucket.fetch_add(1, Ordering::Relaxed);
            }
        }
        let count = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        let nanos = drift.as_nanos().min(u64::MAX as u128) as u64;
        self.total_nanos.fetch_add(nanos, Ordering::Relaxed);
        self.max_nanos.fetch_max(nanos, Ordering::Relaxed);
        count
    }

    fn snapshot(&self, backend: SleepBackend) -> SleepDriftStats {
        SleepDriftStats {
            backend,
            count: self.count.load(Ordering::Relaxed),
            total: Duration::from_nanos(self.total_nanos.load(Ordering::Relaxed)),
            max: Duration::from_nanos(self.max_nanos.load(Ordering::Relaxed)),
            buckets: DRIFT_BUCKET_BOUNDS_SECS
                .iter()
                .zip(self.buckets.iter())
                .map(|(bound, bucket)| (*bound, bucket.load(Ordering::Relaxed)))
                .collect(),
        }
    }
}

static TIMERFD_DRIFT: DriftHistogram = DriftHistogram::new();
static TIME_DRIVER_DRIFT: DriftHistogram = DriftHistogram::new();

/// Aggregated sleep drift for one backend.
#[derive(Clone, Debug, PartialEq)]
pub struct SleepDriftStats {
    pub backend: SleepBackend,
    /// Number of measured wakes; expired deadlines are excluded.
    pub count: u64,
    pub total: Duration,
    pub max: Duration,
    /// Cumulative counts keyed by upper bound in seconds.
    pub buckets: Vec<(f64, u64)>,
}

fn histogram_for(backend: SleepBackend) -> &'static DriftHistogram {
    match backend.resolve() {
        SleepBackend::Timerfd => &TIMERFD_DRIFT,
        _ => &TIME_DRIVER_DRIFT,
    }
}

/// Returns accumulated drift for one backend.
pub fn sleep_drift_stats(backend: SleepBackend) -> SleepDriftStats {
    let backend = backend.resolve();
    histogram_for(backend).snapshot(backend)
}

/// Records and logs one measured wake.
pub fn record_sleep_drift(record: &SleepDriftRecord) {
    let count = histogram_for(record.backend).observe(record.drift);
    tracing::debug!(
        backend = record.backend.label(),
        requested_ms = record.requested.as_secs_f64() * 1_000.0,
        actual_ms = record.actual.as_secs_f64() * 1_000.0,
        drift_ms = record.drift.as_secs_f64() * 1_000.0,
        "precise sleep drift"
    );
    if count.is_multiple_of(DRIFT_SUMMARY_EVERY) {
        log_sleep_drift_summary(record.backend);
    }
}

const DRIFT_SUMMARY_EVERY: u64 = 1_000;

fn log_sleep_drift_summary(backend: SleepBackend) {
    let stats = sleep_drift_stats(backend);
    let mean_ms = if stats.count == 0 {
        0.0
    } else {
        stats.total.as_secs_f64() * 1_000.0 / stats.count as f64
    };
    tracing::info!(
        backend = stats.backend.label(),
        count = stats.count,
        mean_ms,
        max_ms = stats.max.as_secs_f64() * 1_000.0,
        buckets = ?stats.buckets,
        "precise sleep drift summary"
    );
}

/// Sleep for the specified duration using timerfd on Linux for precision.
pub async fn sleep_precise(duration: Duration) {
    sleep_until_precise(Instant::now() + duration).await;
}

/// Sleep until the specified deadline using timerfd on Linux for precision.
///
/// Unlike `sleep_precise`, this accounts for time already elapsed since the
/// deadline's reference point, making it suitable for simulation loops where
/// computation time should be subtracted from the sleep.
pub async fn sleep_until_precise(deadline: Instant) {
    if sleep_drift_enabled() {
        if let Some(record) =
            sleep_until_precise_measured(deadline, configured_sleep_backend()).await
        {
            record_sleep_drift(&record);
        }
        return;
    }
    sleep_until_backend(deadline, configured_sleep_backend()).await;
}

/// Sleeps until `deadline` and returns timing data, or `None` if it has expired.
pub async fn sleep_until_precise_measured(
    deadline: Instant,
    backend: SleepBackend,
) -> Option<SleepDriftRecord> {
    let started = Instant::now();
    let requested = deadline.saturating_duration_since(started);
    let used = sleep_until_backend(deadline, backend).await?;
    let actual = started.elapsed();
    Some(SleepDriftRecord {
        backend: used,
        requested,
        actual,
        drift: actual.saturating_sub(requested),
    })
}

async fn sleep_until_backend(deadline: Instant, backend: SleepBackend) -> Option<SleepBackend> {
    // Scheduler work may consume the modeled delay, especially at high speedup ratios. Avoid
    // allocating and registering a timerfd when there is no remaining time to sleep. Preserve
    // the scheduler loop's cooperative yield so other tasks on the runtime can make progress.
    if deadline <= Instant::now() {
        tokio::task::yield_now().await;
        return None;
    }

    #[cfg(target_os = "linux")]
    if backend.resolve() == SleepBackend::Timerfd {
        // Creation and read failures both fall back to the time driver.
        let timerfd_served = match tokio_timerfd::Delay::new(deadline) {
            Ok(delay) => delay.await.is_ok(),
            Err(_) => false,
        };
        if timerfd_served {
            return Some(SleepBackend::Timerfd);
        }
        tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
        return Some(SleepBackend::TimeDriver);
    }

    #[cfg(not(target_os = "linux"))]
    let _ = backend;

    tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
    Some(SleepBackend::TimeDriver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::Poll;

    #[tokio::test(flavor = "current_thread")]
    async fn test_expired_precise_sleep_yields_to_runtime() {
        let sleep = sleep_until_precise(Instant::now());
        tokio::pin!(sleep);

        let first_poll = futures::poll!(sleep.as_mut());

        assert!(matches!(first_poll, Poll::Pending));
        sleep.await;
    }

    #[test]
    fn test_auto_backend_keeps_platform_default() {
        assert_eq!(configured_sleep_backend(), SleepBackend::Auto);
        assert!(!sleep_drift_enabled());

        #[cfg(target_os = "linux")]
        let expected = SleepBackend::Timerfd;
        #[cfg(not(target_os = "linux"))]
        let expected = SleepBackend::TimeDriver;

        assert_eq!(SleepBackend::Auto.resolve(), expected);
        assert_eq!(SleepBackend::Timerfd.resolve(), expected);
        assert_eq!(SleepBackend::TimeDriver.resolve(), SleepBackend::TimeDriver);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_default_backend_actually_sleeps_on_timerfd() {
        assert_eq!(configured_sleep_backend(), SleepBackend::Auto);

        let record = sleep_until_precise_measured(
            Instant::now() + Duration::from_millis(2),
            SleepBackend::Auto,
        )
        .await
        .expect("a 2ms deadline is not expired, so a timer is armed");

        assert_eq!(
            record.backend,
            SleepBackend::Timerfd,
            "the default path was served by {} on Linux, not timerfd",
            record.backend.label()
        );
    }

    #[test]
    fn test_unrecognized_backend_falls_back_to_auto() {
        assert_eq!(SleepBackend::parse("timerfd"), Some(SleepBackend::Timerfd));
        assert_eq!(
            SleepBackend::parse(" Time-Driver "),
            Some(SleepBackend::TimeDriver)
        );
        assert_eq!(SleepBackend::parse(""), Some(SleepBackend::Auto));
        assert_eq!(SleepBackend::parse("kqueue"), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_short_sleep_does_not_ride_the_one_second_timer() {
        let (registered_tx, registered_rx) = tokio::sync::oneshot::channel();
        let heartbeat = tokio::spawn(async move {
            let sleep = tokio::time::sleep(Duration::from_secs(1));
            tokio::pin!(sleep);
            assert!(
                matches!(futures::poll!(sleep.as_mut()), Poll::Pending),
                "a 1s timer must not be ready on its first poll"
            );
            let _ = registered_tx.send(());
            sleep.await;
        });
        registered_rx
            .await
            .expect("heartbeat task registered its 1s deadline on the time driver");

        let record = sleep_until_precise_measured(
            Instant::now() + Duration::from_millis(2),
            SleepBackend::TimeDriver,
        )
        .await
        .expect("a 2ms deadline is not expired, so a timer is armed");

        assert_eq!(
            record.backend,
            SleepBackend::TimeDriver,
            "requested time_driver but the wake was served by {}",
            record.backend.label()
        );
        assert!(
            record.drift < Duration::from_millis(200),
            "2ms sleep woke {:?} late (actual {:?}) on {}",
            record.drift,
            record.actual,
            record.backend.label()
        );

        heartbeat.abort();
    }

    #[test]
    fn test_drift_histogram_buckets_the_observation() {
        let histogram = DriftHistogram::new();
        histogram.observe(Duration::from_millis(3));
        histogram.observe(Duration::from_millis(1_000));
        histogram.observe(Duration::from_secs(30));

        let stats = histogram.snapshot(SleepBackend::TimeDriver);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.max, Duration::from_secs(30));
        assert_eq!(stats.total, Duration::from_millis(31_003));

        let cumulative = |bound: f64| {
            stats
                .buckets
                .iter()
                .find(|(b, _)| *b == bound)
                .map(|(_, count)| *count)
                .expect("bucket bound present")
        };
        assert_eq!(cumulative(0.001), 0, "3ms must not land in the 1ms bucket");
        assert_eq!(cumulative(0.005), 1);
        assert_eq!(cumulative(0.5), 1, "1s must not land below 1s");
        assert_eq!(cumulative(1.0), 2);
        assert_eq!(
            cumulative(2.0),
            2,
            "30s exceeds every bound and is counted only in the total"
        );
    }

    #[test]
    fn test_prefill_handoff_delay_only_applies_to_completed_prefill() {
        let delay_ms = compute_prefill_handoff_delay_ms(
            WorkerType::Prefill,
            true,
            128,
            Some(1.0),
            Some(1_000_000),
        )
        .expect("prefill completion should produce a handoff delay");
        assert!((delay_ms - 128.0).abs() < 1e-9);

        assert!(
            compute_prefill_handoff_delay_ms(
                WorkerType::Prefill,
                false,
                128,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
        assert!(
            compute_prefill_handoff_delay_ms(
                WorkerType::Decode,
                true,
                128,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
    }
}
