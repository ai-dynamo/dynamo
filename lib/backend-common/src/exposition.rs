// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridge an engine's Prometheus exposition into the runtime's metric surfaces.
//!
//! This is the Rust twin of `register_engine_metrics_callback` in
//! `components/src/dynamo/common/utils/prometheus.py`. An out-of-process
//! engine serves its registry as text; this module reads that text when the
//! worker's own `/metrics` is scraped, or when the OTLP exporter ticks, filters
//! it to the engine's families, injects the Dynamo identity labels, and hands
//! the result to both runtime surfaces. Nothing is polled and nothing is kept
//! between scrapes: the scraper sets the cadence, exactly as the Python
//! callback does when it collects the in-process registry.
//!
//! Both runtime surfaces invoke callbacks on the blocking pool
//! (`render_scrape_at` and the OTLP exporter wrap collection in
//! `spawn_blocking`), so the source future is driven with
//! `Handle::block_on` from that thread. Calling the callbacks from an async
//! task would panic, the same hazard class as the GIL-taking Python callback.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use dynamo_runtime::metrics::create_metric;
use dynamo_runtime::metrics::prom_text::{TextFormat, parse_text};
use dynamo_runtime::metrics::prom_typed::{BuiltFamily, TypedFamily, build_families};
use futures::future::BoxFuture;
use parking_lot::Mutex;

use crate::error::{BackendError, DynamoError, ErrorType};
use crate::metrics::{EngineMetrics, const_labels};

/// One exposition body as the engine served it.
#[derive(Debug)]
pub struct Exposition {
    pub text: String,
    pub format: TextFormat,
}

pub type ExpositionFuture = BoxFuture<'static, anyhow::Result<Exposition>>;

/// Produces one exposition body per call. Invoked once per scrape and once
/// per OTLP export tick; it must bound its own time and size.
pub type ExpositionSource = Arc<dyn Fn() -> ExpositionFuture + Send + Sync>;

/// Family-name filter, matched against the bare family name as Python's
/// `metric_prefix_filters` and `exclude_prefixes` are.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExpositionFilter {
    /// Keep only families starting with one of these. Empty keeps all.
    pub include_prefixes: Vec<String>,
    /// Drop families starting with one of these.
    pub exclude_prefixes: Vec<String>,
}

impl ExpositionFilter {
    pub fn include(prefixes: &[&str]) -> Self {
        Self {
            include_prefixes: prefixes.iter().map(|p| p.to_string()).collect(),
            exclude_prefixes: Vec::new(),
        }
    }

    pub fn keeps(&self, family: &str) -> bool {
        let included = self.include_prefixes.is_empty()
            || self.include_prefixes.iter().any(|p| family.starts_with(p));
        included && !self.exclude_prefixes.iter().any(|p| family.starts_with(p))
    }
}

/// Which runtime surfaces receive the engine families.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpositionOutputs {
    /// Append to `/metrics` as exposition text.
    pub prometheus: bool,
    /// Register typed families for the OTLP exporter.
    pub otlp: bool,
}

impl ExpositionOutputs {
    pub fn any(self) -> bool {
        self.prometheus || self.otlp
    }
}

pub struct ExpositionSourceConfig {
    /// Shown in logs, for example the engine name and URL.
    pub name: String,
    pub source: ExpositionSource,
    pub filter: ExpositionFilter,
    pub outputs: ExpositionOutputs,
}

impl EngineMetrics {
    /// Register an exposition source on the surfaces `config.outputs` selects.
    ///
    /// Each selected surface fetches independently on its own cadence. On a
    /// failed fetch the surface simply carries no engine families for that
    /// collection; Dynamo's own families are unaffected. The outcome is
    /// reported by `dynamo_component_engine_metrics_scrape_success` and
    /// `dynamo_component_engine_metrics_scrape_duration_seconds`, which are
    /// created only when at least one surface is selected.
    pub fn register_exposition_source(
        &self,
        config: ExpositionSourceConfig,
    ) -> Result<(), DynamoError> {
        if !config.outputs.any() {
            return Ok(());
        }
        let health = ExpositionHealth::new(self)?;
        let collector = Arc::new(Collector {
            name: config.name,
            source: config.source,
            filter: config.filter,
            labels: self.engine_family_labels().clone(),
            health,
            last_succeeded: Mutex::new(None),
        });
        if config.outputs.prometheus {
            let collector = collector.clone();
            self.add_expfmt_callback(Arc::new(move || match collector.collect() {
                Some(families) => render_text(families),
                None => Ok(String::new()),
            }));
        }
        if config.outputs.otlp {
            let collector = collector.clone();
            self.add_typed_callback(Arc::new(move || {
                Ok(collector.collect().map(build_families).unwrap_or_default())
            }));
        }
        Ok(())
    }
}

struct ExpositionHealth {
    success: prometheus::IntGauge,
    duration_seconds: prometheus::Gauge,
}

impl ExpositionHealth {
    fn new(metrics: &EngineMetrics) -> Result<Self, DynamoError> {
        let labels = const_labels(metrics);
        let hierarchy = metrics.hierarchy().as_ref();
        let error = |name: &str, e: anyhow::Error| {
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::Unknown))
                .message(format!("gauge create {name}: {e}"))
                .build()
        };
        let success = create_metric::<prometheus::IntGauge, _>(
            hierarchy,
            "engine_metrics_scrape_success",
            "Whether the last fetch of the engine's metrics endpoint succeeded (1) or failed (0).",
            &labels,
            None,
            None,
        )
        .map_err(|e| error("engine_metrics_scrape_success", e))?;
        let duration_seconds = create_metric::<prometheus::Gauge, _>(
            hierarchy,
            "engine_metrics_scrape_duration_seconds",
            "Wall time of the last fetch and parse of the engine's metrics endpoint.",
            &labels,
            None,
            None,
        )
        .map_err(|e| error("engine_metrics_scrape_duration_seconds", e))?;
        Ok(Self {
            success,
            duration_seconds,
        })
    }
}

struct Collector {
    name: String,
    source: ExpositionSource,
    filter: ExpositionFilter,
    labels: Arc<HashMap<String, String>>,
    health: ExpositionHealth,
    /// Drives state-change logging: one warning when the engine becomes
    /// unreachable, one line when it is back, nothing in between.
    last_succeeded: Mutex<Option<bool>>,
}

impl Collector {
    /// Fetch, parse, filter, and label one exposition. `None` means the
    /// engine families are absent from this collection.
    fn collect(&self) -> Option<Vec<TypedFamily>> {
        let started = Instant::now();
        let outcome = block_on((self.source)()).and_then(|result| {
            let exposition = result?;
            parse_text(&exposition.text, exposition.format).context("parsing engine metrics")
        });
        self.health
            .duration_seconds
            .set(started.elapsed().as_secs_f64());
        match outcome {
            Ok(families) => {
                self.health.success.set(1);
                let families = self.shape(families);
                let previous = self.last_succeeded.lock().replace(true);
                if previous == Some(false) {
                    tracing::info!(source = %self.name, families = families.len(), "engine metrics available again");
                }
                Some(families)
            }
            Err(error) => {
                self.health.success.set(0);
                let previous = self.last_succeeded.lock().replace(false);
                let error = format!("{error:#}");
                if previous == Some(false) {
                    tracing::debug!(source = %self.name, error, "engine metrics still unavailable");
                } else {
                    tracing::warn!(
                        source = %self.name,
                        error,
                        "engine metrics unavailable; Dynamo metrics are served without engine families"
                    );
                }
                None
            }
        }
    }

    fn shape(&self, families: Vec<TypedFamily>) -> Vec<TypedFamily> {
        families
            .into_iter()
            .filter(|family| self.filter.keeps(&family.name))
            .map(|mut family| {
                for sample in &mut family.samples {
                    // Existing engine labels win, as in the Python bridge.
                    for (key, value) in self.labels.iter() {
                        sample
                            .labels
                            .entry(key.clone())
                            .or_insert_with(|| value.clone());
                    }
                }
                family
            })
            .collect()
    }
}

/// Drive the source from the blocking pool the callbacks run on.
fn block_on<T>(future: BoxFuture<'static, T>) -> anyhow::Result<T> {
    let handle = tokio::runtime::Handle::try_current()
        .context("engine metrics source requires a tokio runtime")?;
    Ok(handle.block_on(future))
}

/// Render families as classic exposition text for `/metrics`.
///
/// Sample timestamps are dropped: the scrape instant is the observation time
/// of a proxied sample. Counter families regain their `_total` suffix, which
/// is how classic text names them and what the integrated path emits. Untyped
/// families render as gauges, as the typed builder maps them; the text encoder
/// does not implement the `untyped` word.
fn render_text(mut families: Vec<TypedFamily>) -> anyhow::Result<String> {
    for family in &mut families {
        for sample in &mut family.samples {
            sample.timestamp = None;
        }
        if family.kind == "counter"
            && !family.name.ends_with("_total")
            && family
                .samples
                .iter()
                .any(|sample| sample.name.ends_with("_total"))
        {
            family.name.push_str("_total");
        }
    }
    let families: Vec<_> = build_families(families)
        .into_iter()
        .map(|BuiltFamily { family, .. }| family)
        .collect();
    let mut text = String::new();
    prometheus::TextEncoder::new()
        .encode_utf8(&families, &mut text)
        .context("encoding engine metrics")?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::metrics::TestHierarchy;
    use dynamo_runtime::metrics::MetricsRegistry;
    use prometheus::proto::MetricType;

    const CLASSIC: &str = "\
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine=\"0\",model_name=\"Qwen/Qwen3-0.6B\"} 19132.0 1700000000123
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine=\"0\"} 2
# TYPE vllm:custom untyped
vllm:custom 7
# HELP http_requests_total Frontend requests.
# TYPE http_requests_total counter
http_requests_total{handler=\"/metrics\"} 5
";

    type Scripted = Arc<Mutex<Result<&'static str, &'static str>>>;

    fn scripted_source(script: &Scripted) -> ExpositionSource {
        let script = script.clone();
        Arc::new(move || {
            let script = script.clone();
            Box::pin(async move {
                match *script.lock() {
                    Ok(text) => Ok(Exposition {
                        text: text.to_string(),
                        format: TextFormat::Classic,
                    }),
                    Err(message) => Err(anyhow::anyhow!(message)),
                }
            })
        })
    }

    fn engine_metrics() -> EngineMetrics {
        EngineMetrics::with_engine_config(
            TestHierarchy::new(),
            &EngineConfig {
                model: "/models/qwen".to_string(),
                served_model_name: Some("qwen3".to_string()),
                ..Default::default()
            },
        )
    }

    async fn render(registry: &MetricsRegistry) -> String {
        let registry = registry.clone();
        tokio::task::spawn_blocking(move || registry.prometheus_expfmt_combined())
            .await
            .unwrap()
            .unwrap()
    }

    fn data_line<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
        text.lines()
            .find(|line| line.starts_with(prefix) && !line.starts_with('#'))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn text_surface_filters_labels_and_tracks_health() {
        let script: Scripted = Arc::new(Mutex::new(Ok(CLASSIC)));
        let metrics = engine_metrics();
        metrics
            .register_exposition_source(ExpositionSourceConfig {
                name: "test".to_string(),
                source: scripted_source(&script),
                filter: ExpositionFilter::include(&["vllm:"]),
                outputs: ExpositionOutputs {
                    prometheus: true,
                    otlp: true,
                },
            })
            .unwrap();
        let registry = metrics.hierarchy().get_metrics_registry().clone();

        let text = render(&registry).await;
        assert!(
            text.contains("# TYPE vllm:prompt_tokens_total counter"),
            "{text}"
        );
        // Injected labels fill gaps; the engine's own `model_name` wins.
        assert_eq!(
            data_line(&text, "vllm:prompt_tokens_total").unwrap(),
            "vllm:prompt_tokens_total{engine=\"0\",model=\"qwen3\",model_name=\"Qwen/Qwen3-0.6B\"} 19132"
        );
        assert_eq!(
            data_line(&text, "vllm:num_requests_running").unwrap(),
            "vllm:num_requests_running{engine=\"0\",model=\"qwen3\",model_name=\"qwen3\"} 2"
        );
        // The text encoder has no untyped rendering; untyped families ship as gauges.
        assert!(text.contains("# TYPE vllm:custom gauge"), "{text}");
        assert!(!text.contains("http_requests"), "{text}");
        assert!(
            !text.contains("1700000000"),
            "timestamps must be stripped: {text}"
        );
        assert!(
            data_line(&text, "dynamo_component_engine_metrics_scrape_success")
                .unwrap()
                .ends_with(" 1"),
            "{text}"
        );

        // The typed surface keeps bare counter names and timestamps.
        let collected = {
            let registry = registry.clone();
            tokio::task::spawn_blocking(move || registry.metric_families_combined())
                .await
                .unwrap()
                .unwrap()
        };
        let counter = collected
            .families
            .iter()
            .find(|family| family.name() == "vllm:prompt_tokens")
            .expect("typed counter family");
        assert_eq!(counter.get_field_type(), MetricType::COUNTER);
        assert_eq!(counter.get_metric()[0].timestamp_ms(), 1_700_000_000_123);
        assert!(
            !collected
                .families
                .iter()
                .any(|family| family.name().starts_with("http_"))
        );

        // A failed fetch drops the engine families and flips the gauge.
        *script.lock() = Err("connection refused");
        let text = render(&registry).await;
        assert!(!text.contains("vllm:"), "{text}");
        assert!(
            data_line(&text, "dynamo_component_engine_metrics_scrape_success")
                .unwrap()
                .ends_with(" 0"),
            "{text}"
        );

        // Recovery needs no state reset.
        *script.lock() = Ok(CLASSIC);
        let text = render(&registry).await;
        assert!(text.contains("vllm:num_requests_running"), "{text}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unselected_surfaces_register_nothing() {
        let script: Scripted = Arc::new(Mutex::new(Ok(CLASSIC)));
        let metrics = engine_metrics();
        metrics
            .register_exposition_source(ExpositionSourceConfig {
                name: "test".to_string(),
                source: scripted_source(&script),
                filter: ExpositionFilter::default(),
                outputs: ExpositionOutputs {
                    prometheus: false,
                    otlp: true,
                },
            })
            .unwrap();
        let registry = metrics.hierarchy().get_metrics_registry().clone();
        let text = render(&registry).await;
        assert!(!text.contains("vllm:"), "{text}");
        // Health gauges exist because the typed surface is active.
        assert!(text.contains("dynamo_component_engine_metrics_scrape_success"));

        let metrics = engine_metrics();
        metrics
            .register_exposition_source(ExpositionSourceConfig {
                name: "off".to_string(),
                source: scripted_source(&script),
                filter: ExpositionFilter::default(),
                outputs: ExpositionOutputs::default(),
            })
            .unwrap();
        let text = render(metrics.hierarchy().get_metrics_registry()).await;
        assert!(!text.contains("engine_metrics_scrape"), "{text}");
    }

    #[test]
    fn filter_matches_family_prefixes() {
        let filter = ExpositionFilter::include(&["vllm:", "lmcache:"]);
        assert!(filter.keeps("vllm:prompt_tokens"));
        assert!(filter.keeps("lmcache:hits"));
        assert!(!filter.keeps("http_requests"));
        let filter = ExpositionFilter {
            include_prefixes: Vec::new(),
            exclude_prefixes: vec!["python_".to_string()],
        };
        assert!(filter.keeps("trtllm_request_latency"));
        assert!(!filter.keeps("python_gc_objects"));
    }
}
