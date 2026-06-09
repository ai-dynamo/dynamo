// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local admission control.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use prometheus::{Gauge, IntCounterVec, Opts, Registry};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::metrics::prometheus_names::name_prefix;

const MODE_NONE: &str = "none";
const MODE_TOKIO_LAG: &str = "tokio-lag";
const REASON_TOKIO_LAG: &str = "tokio_lag";

const DEFAULT_LAG_THRESHOLD_MS: u64 = 50;
const DEFAULT_LAG_CHECK_INTERVAL_MS: u64 = 10;
const DEFAULT_LAG_COOLDOWN_MS: u64 = 1000;

static PROCESS_START: LazyLock<Instant> = LazyLock::new(Instant::now);

pub static ADMISSION_LAG_SECONDS: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::new(
        format!("{}_admission_lag_seconds", name_prefix::FRONTEND),
        "Latest frontend admission Tokio scheduling delay sample.",
    )
    .expect("frontend admission lag gauge")
});

pub static ADMISSION_EFFECTIVE_LAG_SECONDS: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::new(
        format!("{}_admission_effective_lag_seconds", name_prefix::FRONTEND),
        "Effective frontend admission lag used for rejection decisions.",
    )
    .expect("frontend admission effective lag gauge")
});

pub static ADMISSION_OVERLOADED: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::new(
        format!("{}_admission_overloaded", name_prefix::FRONTEND),
        "Whether frontend admission currently considers the frontend overloaded.",
    )
    .expect("frontend admission overloaded gauge")
});

pub static ADMISSION_REJECTIONS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            format!("{}_admission_rejections_total", name_prefix::FRONTEND),
            "Total number of requests rejected by frontend admission control.",
        ),
        &["reason"],
    )
    .expect("frontend admission rejections counter")
});

pub fn register_frontend_admission_metrics(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(ADMISSION_LAG_SECONDS.clone()))?;
    registry.register(Box::new(ADMISSION_EFFECTIVE_LAG_SECONDS.clone()))?;
    registry.register(Box::new(ADMISSION_OVERLOADED.clone()))?;
    registry.register(Box::new(ADMISSION_REJECTIONS_TOTAL.clone()))?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontendAdmissionMode {
    None,
    TokioLag,
}

impl FrontendAdmissionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => MODE_NONE,
            Self::TokioLag => MODE_TOKIO_LAG,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontendAdmissionConfig {
    mode: FrontendAdmissionMode,
    lag_threshold: Duration,
    lag_check_interval: Duration,
    lag_cooldown: Duration,
}

impl Default for FrontendAdmissionConfig {
    fn default() -> Self {
        Self {
            mode: FrontendAdmissionMode::None,
            lag_threshold: Duration::from_millis(DEFAULT_LAG_THRESHOLD_MS),
            lag_check_interval: Duration::from_millis(DEFAULT_LAG_CHECK_INTERVAL_MS),
            lag_cooldown: Duration::from_millis(DEFAULT_LAG_COOLDOWN_MS),
        }
    }
}

impl FrontendAdmissionConfig {
    pub fn new(
        mode: &str,
        lag_threshold_ms: u64,
        lag_check_interval_ms: u64,
        lag_cooldown_ms: u64,
    ) -> anyhow::Result<Self> {
        let mode = match mode {
            MODE_NONE => FrontendAdmissionMode::None,
            MODE_TOKIO_LAG => FrontendAdmissionMode::TokioLag,
            other => anyhow::bail!(
                "invalid frontend admission control mode {other:?}; expected '{MODE_NONE}' or '{MODE_TOKIO_LAG}'"
            ),
        };

        if mode == FrontendAdmissionMode::TokioLag && lag_threshold_ms == 0 {
            anyhow::bail!("frontend lag threshold must be > 0 when tokio-lag admission is enabled");
        }
        if mode == FrontendAdmissionMode::TokioLag && lag_check_interval_ms == 0 {
            anyhow::bail!(
                "frontend lag check interval must be > 0 when tokio-lag admission is enabled"
            );
        }

        Ok(Self {
            mode,
            lag_threshold: Duration::from_millis(lag_threshold_ms),
            lag_check_interval: Duration::from_millis(lag_check_interval_ms),
            lag_cooldown: Duration::from_millis(lag_cooldown_ms),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.mode != FrontendAdmissionMode::None
    }

    pub fn mode(&self) -> &FrontendAdmissionMode {
        &self.mode
    }

    pub fn lag_threshold(&self) -> Duration {
        self.lag_threshold
    }

    pub fn lag_check_interval(&self) -> Duration {
        self.lag_check_interval
    }

    pub fn lag_cooldown(&self) -> Duration {
        self.lag_cooldown
    }
}

#[derive(Debug, Clone)]
pub struct FrontendAdmissionRejection {
    effective_lag: Duration,
    threshold: Duration,
}

impl FrontendAdmissionRejection {
    fn message(&self) -> String {
        format!(
            "Frontend overloaded: Tokio scheduling lag {:.3}ms exceeded configured threshold {:.3}ms",
            self.effective_lag.as_secs_f64() * 1000.0,
            self.threshold.as_secs_f64() * 1000.0,
        )
    }
}

#[derive(Serialize)]
struct AdmissionErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: u16,
}

pub fn rejection_response(rejection: FrontendAdmissionRejection) -> Response {
    let code = StatusCode::SERVICE_UNAVAILABLE;
    (
        code,
        Json(AdmissionErrorBody {
            message: rejection.message(),
            error_type: code
                .canonical_reason()
                .unwrap_or("Service Unavailable")
                .to_string(),
            code: code.as_u16(),
        }),
    )
        .into_response()
}

pub struct FrontendAdmissionController {
    config: FrontendAdmissionConfig,
    last_observed_delay_ns: AtomicU64,
    last_tick_ns: AtomicU64,
    overloaded_until_ns: AtomicU64,
    canary_started: AtomicBool,
}

impl FrontendAdmissionController {
    pub fn new(config: FrontendAdmissionConfig) -> Self {
        let now = now_ns();
        Self {
            config,
            last_observed_delay_ns: AtomicU64::new(0),
            last_tick_ns: AtomicU64::new(now),
            overloaded_until_ns: AtomicU64::new(0),
            canary_started: AtomicBool::new(false),
        }
    }

    pub fn config(&self) -> &FrontendAdmissionConfig {
        &self.config
    }

    pub fn spawn_canary(self: &Arc<Self>, cancel_token: CancellationToken) {
        if self.config.mode() != &FrontendAdmissionMode::TokioLag {
            return;
        }
        if self.canary_started.swap(true, Ordering::AcqRel) {
            return;
        }

        let controller = self.clone();
        tokio::spawn(async move {
            controller.run_tokio_lag_canary(cancel_token).await;
        });
    }

    pub fn admit(&self) -> Result<(), FrontendAdmissionRejection> {
        if self.config.mode() == &FrontendAdmissionMode::None {
            return Ok(());
        }

        let now = now_ns();
        let effective_lag = self.effective_lag(now);
        ADMISSION_EFFECTIVE_LAG_SECONDS.set(duration_from_ns(effective_lag).as_secs_f64());

        let threshold_ns = duration_as_ns(self.config.lag_threshold());
        let overloaded_until = self.overloaded_until_ns.load(Ordering::Acquire);
        if now < overloaded_until {
            ADMISSION_OVERLOADED.set(1.0);
            return Err(self.reject(effective_lag, threshold_ns));
        }

        if effective_lag > threshold_ns {
            let cooldown_until = now.saturating_add(duration_as_ns(self.config.lag_cooldown()));
            self.overloaded_until_ns
                .store(cooldown_until, Ordering::Release);
            ADMISSION_OVERLOADED.set(1.0);
            return Err(self.reject(effective_lag, threshold_ns));
        }

        ADMISSION_OVERLOADED.set(0.0);
        Ok(())
    }

    async fn run_tokio_lag_canary(&self, cancel_token: CancellationToken) {
        let interval = self.config.lag_check_interval();
        loop {
            let expected = Instant::now() + interval;
            tokio::select! {
                _ = tokio::time::sleep_until(expected.into()) => {}
                _ = cancel_token.cancelled() => {
                    tracing::debug!("frontend admission tokio-lag canary shutting down");
                    return;
                }
            }

            let observed_delay = Instant::now().saturating_duration_since(expected);
            self.record_tick(observed_delay);
        }
    }

    fn record_tick(&self, observed_delay: Duration) {
        let observed_delay_ns = duration_as_ns(observed_delay);
        self.last_observed_delay_ns
            .store(observed_delay_ns, Ordering::Release);
        self.last_tick_ns.store(now_ns(), Ordering::Release);
        ADMISSION_LAG_SECONDS.set(observed_delay.as_secs_f64());
    }

    fn effective_lag(&self, now: u64) -> u64 {
        let observed = self.last_observed_delay_ns.load(Ordering::Acquire);
        let last_tick = self.last_tick_ns.load(Ordering::Acquire);
        let stale = now.saturating_sub(last_tick);
        observed.max(stale)
    }

    fn reject(&self, effective_lag_ns: u64, threshold_ns: u64) -> FrontendAdmissionRejection {
        ADMISSION_REJECTIONS_TOTAL
            .with_label_values(&[REASON_TOKIO_LAG])
            .inc();
        FrontendAdmissionRejection {
            effective_lag: duration_from_ns(effective_lag_ns),
            threshold: duration_from_ns(threshold_ns),
        }
    }

    #[cfg(test)]
    pub(crate) fn record_tick_for_test(&self, observed_delay: Duration) {
        self.record_tick(observed_delay);
    }
}

fn now_ns() -> u64 {
    let elapsed = PROCESS_START.elapsed().as_nanos();
    elapsed.min(u64::MAX as u128) as u64
}

fn duration_as_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn duration_from_ns(ns: u64) -> Duration {
    Duration::from_nanos(ns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(threshold_ms: u64, cooldown_ms: u64) -> FrontendAdmissionConfig {
        FrontendAdmissionConfig::new(MODE_TOKIO_LAG, threshold_ms, 10, cooldown_ms).unwrap()
    }

    #[test]
    fn disabled_config_accepts() {
        let controller = FrontendAdmissionController::new(FrontendAdmissionConfig::default());
        controller.record_tick_for_test(Duration::from_secs(10));
        assert!(controller.admit().is_ok());
    }

    #[test]
    fn low_lag_accepts() {
        let controller = FrontendAdmissionController::new(config(50, 1000));
        controller.record_tick_for_test(Duration::from_millis(1));
        assert!(controller.admit().is_ok());
    }

    #[test]
    fn observed_lag_above_threshold_rejects() {
        let controller = FrontendAdmissionController::new(config(50, 1000));
        controller.record_tick_for_test(Duration::from_millis(75));
        assert!(controller.admit().is_err());
    }

    #[test]
    fn stale_heartbeat_rejects() {
        let controller = FrontendAdmissionController::new(config(1, 1000));
        controller.record_tick_for_test(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(2));
        assert!(controller.admit().is_err());
    }

    #[test]
    fn cooldown_prevents_flapping() {
        let controller = FrontendAdmissionController::new(config(50, 1000));
        controller.record_tick_for_test(Duration::from_millis(75));
        assert!(controller.admit().is_err());

        controller.record_tick_for_test(Duration::from_millis(1));
        assert!(controller.admit().is_err());
    }
}
