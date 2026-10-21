// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Polling `Control.GetLoad` on the serving rank for router KV load, component
//! gauges, and planner forward-pass metrics.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_backend_common::{
    ComponentSnapshot, ForwardPassSnapshot, FpmBinding, FpmPublisher, MetricsBindings,
    SnapshotPublisher,
};
use dynamo_sidecar_common::GrpcEndpoint;
use prost_types::value::Kind;
use tokio::sync::OnceCell;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::client::OpenEngineClient;
use crate::model::DiscoveredEngine;
use crate::pb;

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const MAX_BACKOFF: Duration = Duration::from_secs(1);

#[derive(Default)]
pub(crate) struct Poller {
    supported: OnceCell<bool>,
    state: Mutex<PollerState>,
}

#[derive(Default)]
struct PollerState {
    fpm: Option<FpmPublisher>,
    task: Option<JoinHandle<()>>,
}

impl Poller {
    async fn is_supported(
        &self,
        client: &OpenEngineClient,
        engine: &str,
        endpoint: &GrpcEndpoint,
    ) -> bool {
        *self
            .supported
            .get_or_init(|| async {
                if matches!(client.get_load().await, Err(s) if s.code() == tonic::Code::Unimplemented) {
                    tracing::info!(
                        engine,
                        %endpoint,
                        "the engine does not implement Control.GetLoad; load and forward-pass metrics are disabled"
                    );
                    false
                } else {
                    true
                }
            })
            .await
    }

    pub(crate) async fn snapshot_bindings(
        self: &Arc<Self>,
        client: &Arc<OpenEngineClient>,
        discovered: &DiscoveredEngine,
        endpoint: &GrpcEndpoint,
        cancel: CancellationToken,
    ) -> MetricsBindings {
        if !self
            .is_supported(client, &discovered.engine_name, endpoint)
            .await
        {
            return MetricsBindings::default();
        }
        let Some(card_total) = discovered.total_kv_blocks else {
            return MetricsBindings::default();
        };
        let (poller, client) = (Arc::clone(self), Arc::clone(client));
        let span = tracing::info_span!("load_poller", engine = &*discovered.engine_name, %endpoint);
        MetricsBindings {
            dp_ranks: vec![0],
            on_publisher_ready: Some(Box::new(move |publisher| {
                let mut state = poller.state.lock().expect("load poller state lock");
                if state.task.is_none() {
                    let fpm = state.fpm.take();
                    state.task = Some(tokio::spawn(
                        poll_loop(client, Some((publisher, card_total)), fpm, cancel)
                            .instrument(span),
                    ));
                }
                Ok(())
            })),
        }
    }

    pub(crate) async fn fpm_binding(
        self: &Arc<Self>,
        client: &Arc<OpenEngineClient>,
        discovered: &DiscoveredEngine,
        endpoint: &GrpcEndpoint,
        cancel: CancellationToken,
    ) -> FpmBinding {
        if !self
            .is_supported(client, &discovered.engine_name, endpoint)
            .await
        {
            return FpmBinding::default();
        }
        let (poller, client) = (Arc::clone(self), Arc::clone(client));
        let expects_snapshot = discovered.total_kv_blocks.is_some();
        let span = tracing::info_span!("load_poller", engine = &*discovered.engine_name, %endpoint);
        FpmBinding {
            dp_size: 1,
            on_ready: Some(Box::new(move |ranks| {
                let fpm = ranks.into_iter().next();
                let mut state = poller.state.lock().expect("load poller state lock");
                if expects_snapshot {
                    state.fpm = fpm;
                } else if state.task.is_none() {
                    state.task = Some(tokio::spawn(
                        poll_loop(client, None, fpm, cancel).instrument(span),
                    ));
                }
                Ok(())
            })),
        }
    }

    pub(crate) async fn shutdown(&self) {
        let task = {
            let mut state = self.state.lock().expect("load poller state lock");
            state.fpm = None;
            state.task.take()
        };
        if let Some(task) = task {
            let _ = task.await;
        }
    }
}

async fn poll_loop(
    client: Arc<OpenEngineClient>,
    snapshot: Option<(Arc<SnapshotPublisher>, u64)>,
    fpm: Option<FpmPublisher>,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(POLL_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut backoff = POLL_INTERVAL;
    let mut failing = false;
    let mut last_fpm: Option<(String, u64)> = None;

    while cancel.run_until_cancelled(interval.tick()).await.is_some() {
        let Some(result) = cancel.run_until_cancelled(client.get_load()).await else {
            return;
        };
        match result {
            Ok(load) => {
                if std::mem::replace(&mut failing, false) {
                    backoff = POLL_INTERVAL;
                    tracing::info!("Control.GetLoad is answering again");
                }
                if let Some((publisher, card_total)) = &snapshot
                    && let Some(snap) = snapshot_from_load(&load, *card_total)
                {
                    publisher.publish(0, snap);
                }
                if let Some(fpm) = &fpm
                    && let Some(snap) = parse_forward_pass(&load)
                    && advances(&mut last_fpm, &load.instance_id, snap.counter_id)
                {
                    let _ = fpm.publish(snap);
                }
            }
            Err(status) if status.code() == tonic::Code::Unimplemented => {
                tracing::info!(
                    "the engine answered Control.GetLoad with UNIMPLEMENTED; stopping load polling"
                );
                return;
            }
            Err(status) => {
                if !std::mem::replace(&mut failing, true) {
                    tracing::warn!(
                        code = ?status.code(),
                        status_message = %status.message().escape_debug(),
                        retry_in = ?backoff,
                        "Control.GetLoad failed; keeping the last reported load"
                    );
                }
                let wait = backoff;
                backoff = backoff.saturating_mul(2).min(MAX_BACKOFF);
                if cancel
                    .run_until_cancelled(tokio::time::sleep(wait))
                    .await
                    .is_none()
                {
                    return;
                }
            }
        }
    }
}

pub(crate) fn snapshot_from_load(
    load: &pb::LoadInfo,
    card_total_kv_blocks: u64,
) -> Option<ComponentSnapshot> {
    let used = load.used_kv_blocks?;
    let total = load.total_kv_blocks.unwrap_or(card_total_kv_blocks);
    (total > 0 && used <= total).then(|| ComponentSnapshot {
        kv_used_blocks: used,
        kv_total_blocks: total,
        gpu_cache_usage: (used as f64 / total as f64) as f32,
        kv_cache_hit_rate: None,
        dp_rank: 0,
    })
}

pub(crate) fn advances(last: &mut Option<(String, u64)>, instance: &str, counter: u64) -> bool {
    match last {
        Some((prev_instance, prev_counter)) if prev_instance == instance => (counter
            > *prev_counter)
            .then(|| *prev_counter = counter)
            .is_some(),
        _ => {
            *last = Some((instance.to_owned(), counter));
            counter > 0
        }
    }
}

pub(crate) fn parse_forward_pass(load: &pb::LoadInfo) -> Option<ForwardPassSnapshot> {
    let forward_pass = struct_field(load.attributes.as_ref()?, "forward_pass")?;
    let scheduled = struct_field(forward_pass, "scheduled_requests")?;
    let queued = struct_field(forward_pass, "queued_requests")?;
    Some(ForwardPassSnapshot {
        counter_id: uint_field(forward_pass, "counter_id")?,
        num_prefill_requests: uint_field(scheduled, "num_prefill_requests")?,
        sum_prefill_tokens: uint_field(scheduled, "sum_prefill_tokens")?,
        var_prefill_length: f64_field(scheduled, "var_prefill_length")?,
        sum_prefill_kv_tokens: uint_field(scheduled, "sum_prefill_kv_tokens")?,
        num_decode_requests: uint_field(scheduled, "num_decode_requests")?,
        sum_decode_kv_tokens: uint_field(scheduled, "sum_decode_kv_tokens")?,
        var_decode_kv_tokens: f64_field(scheduled, "var_decode_kv_tokens")?,
        num_queued_prefill: uint_field(queued, "num_prefill_requests")?,
        sum_queued_prefill_tokens: uint_field(queued, "sum_prefill_tokens")?,
        var_queued_prefill_length: f64_field(queued, "var_prefill_length")?,
        num_queued_decode: uint_field(queued, "num_decode_requests")?,
        sum_queued_decode_kv_tokens: uint_field(queued, "sum_decode_kv_tokens")?,
        var_queued_decode_kv_tokens: f64_field(queued, "var_decode_kv_tokens")?,
        wall_time_secs: f64_field(forward_pass, "wall_time")?,
        ..Default::default()
    })
}

fn struct_field<'a>(
    parent: &'a prost_types::Struct,
    name: &str,
) -> Option<&'a prost_types::Struct> {
    match parent.fields.get(name)?.kind.as_ref()? {
        Kind::StructValue(nested) => Some(nested),
        _ => None,
    }
}

fn f64_field(parent: &prost_types::Struct, name: &str) -> Option<f64> {
    match parent.fields.get(name)?.kind.as_ref()? {
        Kind::NumberValue(value) if value.is_finite() => Some(*value),
        _ => None,
    }
}

fn uint_field<T: TryFrom<u64>>(parent: &prost_types::Struct, name: &str) -> Option<T> {
    let value = f64_field(parent, name)?;
    let whole = (value >= 0.0 && value.fract() == 0.0 && value < (u64::MAX as f64))
        .then_some(value as u64)?;
    T::try_from(whole).ok()
}
