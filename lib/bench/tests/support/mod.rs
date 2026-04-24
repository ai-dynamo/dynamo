// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use dynamo_kv_router::indexer::{KvIndexerMetrics, METRIC_WARNING_DUPLICATE_STORE};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::{Context as LayerContext, Layer};
use tracing_subscriber::prelude::*;

struct WarningCounterLayer {
    count: Arc<AtomicUsize>,
    target_prefixes: &'static [&'static str],
}

impl<S> Layer<S> for WarningCounterLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: LayerContext<'_, S>) {
        let metadata = event.metadata();
        let target = metadata.target();
        if matches!(*metadata.level(), Level::WARN | Level::ERROR)
            && self
                .target_prefixes
                .iter()
                .any(|prefix| target.starts_with(prefix))
        {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn warning_counter(target_prefixes: &'static [&'static str]) -> Arc<AtomicUsize> {
    static COUNTER: OnceLock<Arc<AtomicUsize>> = OnceLock::new();

    COUNTER
        .get_or_init(|| {
            let count = Arc::new(AtomicUsize::new(0));
            let subscriber = Registry::default().with(WarningCounterLayer {
                count: Arc::clone(&count),
                target_prefixes,
            });
            tracing::subscriber::set_global_default(subscriber)
                .expect("global warning counter subscriber should initialize once");
            count
        })
        .clone()
}

pub fn reset_warning_count(counter: &Arc<AtomicUsize>) {
    counter.store(0, Ordering::Relaxed);
}

pub fn fixture_path(file_name: &str) -> anyhow::Result<String> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(file_name)
        .to_str()
        .map(str::to_owned)
        .context("fixture path is not valid UTF-8")
}

#[allow(dead_code)]
pub fn duplicate_store_warning_count(metrics: &KvIndexerMetrics) -> u64 {
    metrics
        .kv_cache_event_warnings
        .get_metric_with_label_values(&[METRIC_WARNING_DUPLICATE_STORE])
        .expect("duplicate_store warning metric should exist")
        .get()
}
