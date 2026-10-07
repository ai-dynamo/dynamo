// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use parking_lot::Mutex as ParkingLotMutex;
use std::sync::Mutex as StdMutex;

use dynamo_kv_router::protocols::StorageTier;

use crate::common::handoff::HandoffId;
use crate::common::perf_model::{AisCallback, PerfModel};
use crate::common::protocols::{
    FpmSink, KvCacheEventSink, NativeHostOffloadConfig, RawKvEvent, RawKvEventSink, WorkerType,
};
use crate::live::{LiveEngine, LiveEngineOptions};

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CapturedEffect {
    KvStored,
    KvRemoved,
    Fpm,
    Ack,
}

#[derive(Default)]
struct CapturedEffects {
    kv: StdMutex<Vec<KvCacheEvent>>,
    kv_tiers: StdMutex<Vec<StorageTier>>,
    fpm: StdMutex<Vec<ForwardPassSnapshot>>,
    publication_log: StdMutex<Vec<CapturedEffect>>,
}

impl KvCacheEventSink for CapturedEffects {
    fn publish(&self, event: KvCacheEvent) -> Result<()> {
        self.publication_log.lock().unwrap().push(match event.data {
            KvCacheEventData::Stored(_) => CapturedEffect::KvStored,
            KvCacheEventData::Removed(_) => CapturedEffect::KvRemoved,
            KvCacheEventData::Cleared => CapturedEffect::KvRemoved,
        });
        self.kv.lock().unwrap().push(event);
        Ok(())
    }

    fn publish_with_storage_tier(&self, event: KvCacheEvent, tier: StorageTier) -> Result<()> {
        self.kv_tiers.lock().unwrap().push(tier);
        KvCacheEventSink::publish(self, event)
    }
}

impl FpmSink for CapturedEffects {
    fn publish(&self, snapshot: ForwardPassSnapshot) -> Result<()> {
        self.publication_log
            .lock()
            .unwrap()
            .push(CapturedEffect::Fpm);
        self.fpm.lock().unwrap().push(snapshot);
        Ok(())
    }
}

struct SlowDecode;

impl AisCallback for SlowDecode {
    fn predict_prefill(
        &self,
        _batch_size: usize,
        _effective_isl: usize,
        _prefix: usize,
    ) -> Result<f64> {
        Ok(1.0)
    }

    fn predict_decode(&self, _batch_size: usize, _isl: usize, _osl: usize) -> Result<f64> {
        Ok(100.0)
    }
}

fn args(dp_size: u32) -> MockerConfig {
    let mut args = MockerConfig::from_value(serde_json::json!({})).unwrap();
    args.dp_size = dp_size;
    args.block_size = 4;
    args.num_gpu_blocks = 128;
    args.speedup_ratio = 1_000_000.0;
    args
}

fn request(id: u128, dp_rank: u32) -> DirectRequest {
    DirectRequest {
        tokens: vec![1, 2, 3, 4],
        max_output_tokens: 1,
        output_token_ids: Some(vec![9]),
        uuid: Some(Uuid::from_u128(id)),
        dp_rank,
        ..DirectRequest::default()
    }
}

#[test]
fn cancellation_translation_preserves_explicit_discard() {
    let compatibility = CompatibilityState::new(args(1));
    let request_id = Uuid::from_u128(9);
    for discard_pending_output in [false, true] {
        let translated = translate_command(
            SchedulerCommand::CancelRequest { request_id },
            discard_pending_output,
            &compatibility,
        )
        .unwrap();
        assert!(matches!(
            translated.command,
            Command::CancelRequest {
                request_id: observed,
                discard_pending_output: observed_discard,
            } if observed == request_id && observed_discard == discard_pending_output
        ));
    }
}

#[tokio::test]
async fn noop_cancellation_only_cleans_metadata_when_output_is_discarded() {
    for (suppressed_pending_output, expect_handoff_delay) in [(false, true), (true, false)] {
        let mut engine_args = args(1);
        engine_args.worker_type = WorkerType::Prefill;
        engine_args.kv_transfer_bandwidth = Some(1.0);
        engine_args.kv_transfer_bytes_per_token = Some(1_000_000);
        let compatibility = CompatibilityState::new(engine_args);
        let request_id = Uuid::from_u128(10 + u128::from(suppressed_pending_output));
        compatibility.native_request(request(request_id.as_u128(), 0));

        let (lifecycle_tx, _lifecycle_rx) = mpsc::channel(1);
        let (metrics_tx, _metrics_rx) = watch::channel(MockerMetrics::new(0, 0, 128));
        let dispatch = RankDispatch {
            external_dp_rank: 0,
            output: RankOutputSink::None,
            kv_event_publishers: KvEventPublishers::default(),
            fpm_publisher: FpmPublisher::default(),
            lifecycle_tx,
            metrics_tx,
        };
        let (reply, response) = oneshot::channel();
        let pending = ParkingLotMutex::new(HashMap::from([(
            23,
            PendingCommand {
                reply: Some(reply),
                on_success: Vec::new(),
                on_suppressed_output: vec![Cleanup::Request(request_id)],
                on_error: Vec::new(),
            },
        )]));
        dispatch_command_effects(
            23,
            EngineEffects {
                by_rank: vec![aisimulate_core::engine::generalized::RankEffects {
                    dp_rank: 0,
                    effects: CommandEffects {
                        result: CommandResult::Noop,
                        lifecycle_events: Vec::new(),
                        kv_events: Vec::new(),
                        retired_requests: Vec::new(),
                        metrics: Metrics {
                            dp_rank: 0,
                            total_blocks: 128,
                            ..Metrics::default()
                        },
                        suppressed_pending_output,
                    },
                }],
            },
            true,
            true,
            std::slice::from_ref(&dispatch),
            &compatibility,
            &pending,
            &mut [DeferredCommandPublication::default()],
        )
        .await
        .unwrap();
        assert_eq!(
            response.await.unwrap().unwrap().result,
            if suppressed_pending_output {
                SchedulerCommandResult::Applied
            } else {
                SchedulerCommandResult::Noop
            }
        );

        let output = compatibility.output_signal(aisimulate_core::engine::Output {
            request_id,
            token_id: Some(9),
            completed: true,
            rejected: false,
            cached_tokens: Some(4),
        });
        assert_eq!(output.handoff_delay_ms.is_some(), expect_handoff_delay);
        assert_eq!(output.cached_tokens, Some(4));
    }
}

#[tokio::test]
async fn two_rank_handles_share_one_group_boundary_and_publish_rank_effects() {
    let effects = Arc::new(CapturedEffects::default());
    let (rank0_tx, mut rank0_rx) = mpsc::unbounded_channel();
    let (rank1_tx, mut rank1_rx) = mpsc::unbounded_channel();
    let sinks = [rank0_tx, rank1_tx]
        .into_iter()
        .map(|output_tx| GroupedSchedulerRankSinks {
            output_tx: Some(output_tx),
            kv_event_publishers: KvEventPublishers::new(
                Some(Arc::clone(&effects) as Arc<dyn KvCacheEventSink>),
                None,
            ),
            fpm_publisher: FpmPublisher::new(Some(Arc::clone(&effects) as Arc<dyn FpmSink>)),
        })
        .collect();
    let cancel = CancellationToken::new();
    let GroupedSchedulers {
        schedulers, actor, ..
    } = create_grouped_scheduler(args(2), sinks, Some(cancel.clone())).unwrap();

    schedulers[0].request_sender().send(request(1, 0)).unwrap();
    schedulers[1].request_sender().send(request(2, 1)).unwrap();
    let rank0 = tokio::time::timeout(Duration::from_secs(2), rank0_rx.recv())
        .await
        .unwrap()
        .unwrap();
    let rank1 = tokio::time::timeout(Duration::from_secs(2), rank1_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(rank0.last().unwrap().uuid, Uuid::from_u128(1));
    assert_eq!(rank1.last().unwrap().uuid, Uuid::from_u128(2));
    assert!(!effects.fpm.lock().unwrap().is_empty());

    cancel.cancel();
    actor.await.unwrap().unwrap();
}

/// Encodes every raw batch with the vLLM ZMQ wire encoder, which requires each
/// stored block's token IDs.
#[derive(Default)]
struct ZmqEncodingSink {
    tiers: StdMutex<Vec<StorageTier>>,
}

impl RawKvEventSink for ZmqEncodingSink {
    fn publish(&self, event: RawKvEvent) -> Result<()> {
        self.publish_batch(vec![event])
    }

    fn publish_batch(&self, events: Vec<RawKvEvent>) -> Result<()> {
        crate::services::zmq_events::encode_event_batch(&events, 4, 0)?;
        let mut tiers = self.tiers.lock().unwrap();
        tiers.extend(events.iter().map(|event| event.storage_tier));
        Ok(())
    }
}

#[tokio::test(start_paused = true)]
async fn native_host_offload_restore_wakes_idle_engine_and_publishes_g1_residency() {
    let effects = Arc::new(CapturedEffects::default());
    let zmq = Arc::new(ZmqEncodingSink::default());
    let (output_tx, mut output_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let engine_args = MockerConfig::from_value(serde_json::json!({
        "engine": {
            "num_gpu_blocks": 1,
            "block_size": 4,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 4,
            "kv_cache_bytes_per_token": 250_000,
            "native_host_offload": NativeHostOffloadConfig::new(2).with_bandwidths(0.0, 0.01),
            "timing_model": {
                "type": "fixed",
                "prefill_ms": 0.0,
                "decode_ms": 0.0
            }
        }
    }))
    .unwrap();
    let GroupedSchedulers {
        schedulers, actor, ..
    } = create_grouped_scheduler(
        engine_args,
        vec![GroupedSchedulerRankSinks {
            output_tx: Some(output_tx),
            kv_event_publishers: KvEventPublishers::new(
                Some(Arc::clone(&effects) as Arc<dyn KvCacheEventSink>),
                Some(Arc::clone(&zmq) as Arc<dyn RawKvEventSink>),
            ),
            ..GroupedSchedulerRankSinks::default()
        }],
        Some(cancel.clone()),
    )
    .unwrap();

    for (id, tokens, cached_tokens) in [
        (1, vec![1, 2, 3, 4], 0),
        (2, vec![5, 6, 7, 8], 0),
        (3, vec![1, 2, 3, 4], 4),
    ] {
        let submitted_at = tokio::time::Instant::now();
        schedulers[0]
            .request_sender()
            .send(DirectRequest {
                tokens,
                max_output_tokens: 0,
                uuid: Some(Uuid::from_u128(id)),
                ..DirectRequest::default()
            })
            .unwrap();
        let outputs = tokio::time::timeout(Duration::from_secs(1), output_rx.recv())
            .await
            .expect("host transfer must wake an otherwise idle engine")
            .unwrap();
        let output = outputs.last().unwrap();
        assert_eq!(output.uuid, Uuid::from_u128(id));
        assert!(output.completed);
        assert!(!output.rejected);
        assert_eq!(output.cached_tokens, Some(cached_tokens));
        if id == 3 {
            // One 1 MB block at 0.01 GB/s takes 100 ms to restore from G2.
            assert!(submitted_at.elapsed() >= Duration::from_millis(100));
        }
    }

    cancel.cancel();
    actor.await.unwrap().unwrap();
    let kv = effects.kv.lock().unwrap();
    let tiers = effects.kv_tiers.lock().unwrap();
    assert_eq!(
        kv.len(),
        tiers.len(),
        "every grouped KV event carries its tier"
    );
    let KvCacheEventData::Stored(seed) = &kv[0].data else {
        panic!("seed must publish its initial G1 residency");
    };
    let seed_hash = seed.blocks[0].block_hash;
    let seed_residency = |tier: StorageTier| {
        kv.iter()
            .zip(tiers.iter())
            .filter(|(_, event_tier)| **event_tier == tier)
            .filter_map(|(event, _)| match &event.data {
                KvCacheEventData::Stored(stored)
                    if stored
                        .blocks
                        .iter()
                        .any(|block| block.block_hash == seed_hash) =>
                {
                    Some("stored")
                }
                KvCacheEventData::Removed(removed) if removed.block_hashes.contains(&seed_hash) => {
                    Some("removed")
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    // The G2 restore re-publishes the seed as G1-resident, and its private G2
    // copy is visible as HostPinned, so a KV router can credit it while the
    // seed is host-only.
    assert_eq!(
        seed_residency(StorageTier::Device),
        ["stored", "removed", "stored"]
    );
    assert_eq!(seed_residency(StorageTier::HostPinned), ["stored"]);
    // G2 residency carries no token IDs, so only G1 events reach the vLLM wire.
    let zmq_tiers = zmq.tiers.lock().unwrap();
    assert!(!zmq_tiers.is_empty());
    assert!(zmq_tiers.iter().all(|tier| *tier == StorageTier::Device));
}

#[tokio::test(start_paused = true)]
async fn command_after_a_due_host_restore_drains_the_restore_first() {
    let effects = Arc::new(CapturedEffects::default());
    let (output_tx, mut output_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let engine_args = MockerConfig::from_value(serde_json::json!({
        "engine": {
            "num_gpu_blocks": 1,
            "block_size": 4,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 4,
            "kv_cache_bytes_per_token": 250_000,
            "native_host_offload": NativeHostOffloadConfig::new(2).with_bandwidths(0.0, 0.01),
            "timing_model": {
                "type": "fixed",
                "prefill_ms": 0.0,
                "decode_ms": 0.0
            }
        }
    }))
    .unwrap();
    let GroupedSchedulers {
        schedulers, actor, ..
    } = create_grouped_scheduler(
        engine_args,
        vec![GroupedSchedulerRankSinks {
            output_tx: Some(output_tx),
            kv_event_publishers: KvEventPublishers::new(
                Some(Arc::clone(&effects) as Arc<dyn KvCacheEventSink>),
                None,
            ),
            ..GroupedSchedulerRankSinks::default()
        }],
        Some(cancel.clone()),
    )
    .unwrap();
    let sender = schedulers[0].request_sender();
    let send = |id: u128, tokens: Vec<u32>| {
        sender
            .send(DirectRequest {
                tokens,
                max_output_tokens: 0,
                uuid: Some(Uuid::from_u128(id)),
                ..DirectRequest::default()
            })
            .unwrap();
    };

    // Seed A, then evict it from the single G1 block into G2.
    for (id, tokens) in [(1, vec![1, 2, 3, 4]), (2, vec![5, 6, 7, 8])] {
        send(id, tokens);
        let outputs = output_rx.recv().await.unwrap();
        assert!(outputs.last().unwrap().completed);
    }
    // A's repeat starts a 100 ms G2 restore (1 MB at 0.01 GB/s).
    send(3, vec![1, 2, 3, 4]);
    tokio::time::sleep(Duration::from_millis(10)).await;
    // Queue a submit, then move past the restore deadline. The actor's biased
    // select sees the command before the timer, so the command arrives while
    // the restore is already due.
    send(4, vec![9, 10, 11, 12]);
    tokio::time::advance(Duration::from_millis(200)).await;

    let mut completed = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), async {
        while completed.len() < 2 {
            for output in output_rx.recv().await.unwrap() {
                assert!(!output.rejected);
                if output.completed {
                    completed.push(output.uuid);
                }
            }
        }
    })
    .await
    .expect("a command arriving after the restore deadline must still be applied");
    assert_eq!(completed, [Uuid::from_u128(3), Uuid::from_u128(4)]);

    cancel.cancel();
    actor.await.unwrap().unwrap();
    let kv = effects.kv.lock().unwrap();
    let tiers = effects.kv_tiers.lock().unwrap();
    let KvCacheEventData::Stored(seed) = &kv[0].data else {
        panic!("seed must publish its initial G1 residency");
    };
    let seed_hash = seed.blocks[0].block_hash;
    let device_stores = kv
        .iter()
        .zip(tiers.iter())
        .filter(|(_, tier)| **tier == StorageTier::Device)
        .filter_map(|(event, _)| match &event.data {
            KvCacheEventData::Stored(stored) => Some(stored.blocks[0].block_hash),
            _ => None,
        })
        .collect::<Vec<_>>();
    // Seed, B, the restored seed, then the new request.
    assert_eq!(device_stores.len(), 4);
    assert_eq!(device_stores[0], seed_hash);
    assert_eq!(device_stores[2], seed_hash);
    assert_ne!(device_stores[3], seed_hash);
}

#[tokio::test]
async fn live_engine_rejects_cluster_shared_host_offload() {
    let engine_args = MockerConfig::from_value(serde_json::json!({
        "engine": {
            "kv_cache_bytes_per_token": 1024,
            "native_host_offload": NativeHostOffloadConfig::new(2).cluster_shared("test-kv-layout")
        }
    }))
    .unwrap();
    let Err(error) = create_grouped_scheduler(
        engine_args,
        vec![GroupedSchedulerRankSinks::default()],
        None,
    ) else {
        panic!("live engines must reject a cluster_shared G2 pool");
    };
    assert!(
        error
            .to_string()
            .contains("supported only in offline replay"),
        "{error:#}"
    );
}

#[tokio::test]
async fn same_rank_receive_burst_is_batched_into_one_native_pass() {
    let (output_tx, mut output_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let GroupedSchedulers {
        schedulers, actor, ..
    } = create_grouped_scheduler(
        args(1),
        vec![GroupedSchedulerRankSinks {
            output_tx: Some(output_tx),
            ..GroupedSchedulerRankSinks::default()
        }],
        Some(cancel.clone()),
    )
    .unwrap();

    for id in 10..14 {
        schedulers[0].request_sender().send(request(id, 0)).unwrap();
    }
    let first_pass_outputs = tokio::time::timeout(Duration::from_secs(2), output_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        first_pass_outputs
            .iter()
            .map(|output| output.uuid)
            .collect::<BTreeSet<_>>(),
        (10..14).map(Uuid::from_u128).collect(),
        "all requests queued before the bridge runs must share its first pass"
    );

    cancel.cancel();
    actor.await.unwrap().unwrap();
}

#[tokio::test]
async fn command_ack_and_handoff_lifecycle_round_trip_dynamo_uuid() {
    let handoff_id = HandoffId::from(Uuid::from_u128(99));
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let GroupedSchedulers {
        mut schedulers,
        actor,
        ..
    } = create_grouped_scheduler(
        {
            let mut args = args(1);
            args.worker_type = WorkerType::Prefill;
            args
        },
        vec![GroupedSchedulerRankSinks {
            output_tx: Some(output_tx),
            ..GroupedSchedulerRankSinks::default()
        }],
        Some(cancel.clone()),
    )
    .unwrap();
    let mut lifecycle = schedulers[0].take_lifecycle_receiver().unwrap();
    let command_tx = schedulers[0].command_sender();
    let (reply, response) = oneshot::channel();
    command_tx
        .send(SchedulerCommandEnvelope {
            command: SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: request(7, 0),
            },
            reply,
        })
        .await
        .unwrap();
    let effects = response.await.unwrap().unwrap();
    assert_eq!(
        effects.result,
        SchedulerCommandResult::Submitted(Uuid::from_u128(7))
    );
    let event = tokio::time::timeout(Duration::from_secs(2), lifecycle.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        event,
        SchedulerLifecycleEvent::SourceHeld {
            handoff_id: observed,
            ..
        } if observed == handoff_id
    ));

    cancel.cancel();
    actor.await.unwrap().unwrap();
}

#[tokio::test]
async fn cancellation_lane_bypasses_an_ordinary_command_deferred_mid_pass() {
    let mut slow_args = args(1);
    slow_args.num_gpu_blocks = 2_048;
    slow_args.max_num_batched_tokens = 2_048;
    slow_args.speedup_ratio = 0.001;
    let (admission_tx, mut admission_rx) = mpsc::unbounded_channel();
    let engine = LiveEngine::start_with_options(
        slow_args,
        0,
        LiveEngineOptions {
            admission_tx: Some(admission_tx),
            ..LiveEngineOptions::default()
        },
    )
    .unwrap();
    let first_request_id = Uuid::from_u128(101);
    let first_request = DirectRequest {
        tokens: vec![1; 512],
        uuid: Some(first_request_id),
        ..request(101, 0)
    };
    let first = engine.submit(first_request).await.unwrap();
    admission_rx.recv().await.unwrap();

    let pending_engine = engine.clone();
    let pending = tokio::spawn(async move { pending_engine.submit(request(102, 0)).await });
    tokio::task::yield_now().await;

    let cancellation = tokio::time::timeout(Duration::from_millis(500), first.cancel())
        .await
        .expect("cancellation must bypass the deferred ordinary lane")
        .unwrap();
    assert!(cancellation);
    assert!(!pending.is_finished());

    pending.abort();
    let _ = pending.await;
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn midpass_cancel_ack_precedes_completion_router_effects() {
    let effects = Arc::new(CapturedEffects::default());
    let mut slow_args = args(1);
    slow_args.speedup_ratio = 1.0;
    slow_args.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(SlowDecode)));
    let (output_tx, mut output_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let GroupedSchedulers {
        schedulers, actor, ..
    } = create_grouped_scheduler(
        slow_args,
        vec![GroupedSchedulerRankSinks {
            output_tx: Some(output_tx),
            kv_event_publishers: KvEventPublishers::new(
                Some(Arc::clone(&effects) as Arc<dyn KvCacheEventSink>),
                None,
            ),
            fpm_publisher: FpmPublisher::new(Some(Arc::clone(&effects) as Arc<dyn FpmSink>)),
        }],
        Some(cancel.clone()),
    )
    .unwrap();
    let request_id = Uuid::from_u128(202);
    let mut metrics = schedulers[0].metrics_receiver();
    schedulers[0]
        .request_sender()
        .send(DirectRequest {
            tokens: vec![1; 16],
            max_output_tokens: 8,
            output_token_ids: Some((0..8).collect()),
            uuid: Some(request_id),
            dp_rank: 0,
            ..DirectRequest::default()
        })
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), output_rx.recv())
        .await
        .unwrap()
        .expect("prefill pass should publish an output");
    tokio::time::timeout(Duration::from_secs(2), metrics.changed())
        .await
        .unwrap()
        .unwrap();
    let before_cancel = metrics.borrow().clone();
    assert!(before_cancel.active_decode_blocks > 0);
    effects.publication_log.lock().unwrap().clear();

    // The actor immediately starts the 100ms decode pass after the first
    // output. Leave enough room on both sides of its modeled boundary.
    tokio::time::sleep(Duration::from_millis(10)).await;
    let (reply, response) = oneshot::channel();
    schedulers[0]
        .cancellation_sender()
        .send(SchedulerCancellationEnvelope {
            request_id,
            discard_pending_output: true,
            reply,
        })
        .await
        .unwrap();
    let cancellation = tokio::time::timeout(Duration::from_millis(50), response)
        .await
        .expect("mid-pass cancellation should acknowledge immediately")
        .unwrap()
        .unwrap();
    assert_eq!(cancellation.result, SchedulerCommandResult::Applied);
    effects
        .publication_log
        .lock()
        .unwrap()
        .push(CapturedEffect::Ack);
    assert_eq!(
        effects.publication_log.lock().unwrap().as_slice(),
        &[CapturedEffect::Ack],
        "mid-pass command KV must remain hidden until pass completion"
    );
    tokio::time::timeout(Duration::from_millis(200), async {
        loop {
            metrics.changed().await.unwrap();
            if metrics.borrow().active_decode_blocks == 0 {
                break;
            }
        }
    })
    .await
    .expect("cancellation should publish empty occupancy");
    assert_eq!(
        effects.publication_log.lock().unwrap().as_slice(),
        &[CapturedEffect::Ack],
        "immediate occupancy must not make completion-owned FPM visible"
    );
    tokio::time::timeout(Duration::from_millis(200), async {
        loop {
            if effects
                .publication_log
                .lock()
                .unwrap()
                .contains(&CapturedEffect::Fpm)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("pass completion should publish the post-cancel FPM");
    assert_eq!(
        effects.publication_log.lock().unwrap().as_slice(),
        &[CapturedEffect::Ack, CapturedEffect::Fpm],
        "a prefix-cached cancellation retains its inactive blocks, but pass FPM must remain completion-visible"
    );

    cancel.cancel();
    actor.await.unwrap().unwrap();
}

#[tokio::test]
async fn applied_midpass_cancellation_publishes_empty_occupancy_immediately() {
    let (lifecycle_tx, _lifecycle_rx) = mpsc::channel(1);
    let (metrics_tx, metrics_rx) =
        watch::channel(MockerMetrics::from_parts(0, 7, 128, 1, 0, 0, 0, 0));
    let dispatch = RankDispatch {
        external_dp_rank: 0,
        output: RankOutputSink::None,
        kv_event_publishers: KvEventPublishers::default(),
        fpm_publisher: FpmPublisher::default(),
        lifecycle_tx,
        metrics_tx,
    };
    let compatibility = CompatibilityState::new(args(1));
    let (reply, response) = oneshot::channel();
    let pending = ParkingLotMutex::new(HashMap::from([(
        24,
        PendingCommand {
            reply: Some(reply),
            on_success: Vec::new(),
            on_suppressed_output: Vec::new(),
            on_error: Vec::new(),
        },
    )]));
    let empty = Metrics {
        dp_rank: 0,
        active_blocks: 3,
        total_blocks: 128,
        running_requests: 0,
        waiting_requests: 0,
        ..Metrics::default()
    };
    let mut deferred = vec![DeferredCommandPublication::default()];

    dispatch_command_effects(
        24,
        EngineEffects {
            by_rank: vec![aisimulate_core::engine::generalized::RankEffects {
                dp_rank: 0,
                effects: CommandEffects {
                    result: CommandResult::Applied,
                    lifecycle_events: Vec::new(),
                    kv_events: Vec::new(),
                    retired_requests: vec![Uuid::from_u128(24)],
                    metrics: empty.clone(),
                    suppressed_pending_output: true,
                },
            }],
        },
        true,
        true,
        std::slice::from_ref(&dispatch),
        &compatibility,
        &pending,
        &mut deferred,
    )
    .await
    .unwrap();

    assert_eq!(
        response.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    let observed = metrics_rx.borrow().clone();
    assert_eq!(
        (observed.running_requests, observed.waiting_requests),
        (0, 0)
    );
    assert_eq!(observed.active_decode_blocks, 3);
    assert_eq!(deferred[0].metrics, Some(empty));
}

#[tokio::test]
async fn synthetic_midpass_kv_is_deferred_until_completion_before_fpm() {
    let effects = Arc::new(CapturedEffects::default());
    let (lifecycle_tx, _lifecycle_rx) = mpsc::channel(1);
    let initial_metrics = MockerMetrics::new(0, 7, 128);
    let (metrics_tx, metrics_rx) = watch::channel(initial_metrics);
    let dispatch = RankDispatch {
        external_dp_rank: 0,
        output: RankOutputSink::None,
        kv_event_publishers: KvEventPublishers::new(
            Some(Arc::clone(&effects) as Arc<dyn KvCacheEventSink>),
            None,
        ),
        fpm_publisher: FpmPublisher::new(Some(Arc::clone(&effects) as Arc<dyn FpmSink>)),
        lifecycle_tx,
        metrics_tx,
    };
    let compatibility = CompatibilityState::new(args(1));
    let (reply, response) = oneshot::channel();
    let pending = ParkingLotMutex::new(HashMap::from([(
        17,
        PendingCommand {
            reply: Some(reply),
            on_success: Vec::new(),
            on_suppressed_output: Vec::new(),
            on_error: Vec::new(),
        },
    )]));
    let mut deferred = vec![DeferredCommandPublication::default()];
    let command_effects = EngineEffects {
        by_rank: vec![aisimulate_core::engine::generalized::RankEffects {
            dp_rank: 0,
            effects: CommandEffects {
                result: CommandResult::Applied,
                lifecycle_events: Vec::new(),
                kv_events: vec![KvEvent {
                    tier: aisimulate_core::engine::KvEventTier::Device,
                    event_id: 1,
                    dp_rank: 0,
                    data: KvEventData::Removed {
                        block_hashes: vec![42],
                    },
                }],
                retired_requests: Vec::new(),
                metrics: Metrics {
                    dp_rank: 0,
                    active_blocks: 0,
                    total_blocks: 128,
                    ..Metrics::default()
                },
                suppressed_pending_output: false,
            },
        }],
    };

    dispatch_command_effects(
        17,
        command_effects,
        true,
        false,
        std::slice::from_ref(&dispatch),
        &compatibility,
        &pending,
        &mut deferred,
    )
    .await
    .unwrap();
    assert_eq!(
        response.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    effects
        .publication_log
        .lock()
        .unwrap()
        .push(CapturedEffect::Ack);
    assert_eq!(
        effects.publication_log.lock().unwrap().as_slice(),
        &[CapturedEffect::Ack]
    );
    assert_eq!(metrics_rx.borrow().active_decode_blocks, 7);

    publish_pass_router_effects(
        &dispatch,
        Vec::new(),
        &mut deferred[0].kv,
        ForwardPassMetrics::default(),
    );
    dispatch.publish_metrics(
        deferred[0]
            .metrics
            .take()
            .expect("mid-pass command metrics must be deferred"),
    );
    assert_eq!(
        effects.publication_log.lock().unwrap().as_slice(),
        &[
            CapturedEffect::Ack,
            CapturedEffect::KvRemoved,
            CapturedEffect::Fpm,
        ]
    );
    assert_eq!(metrics_rx.borrow().active_decode_blocks, 0);
}

#[test]
fn completion_metrics_override_the_midpass_command_snapshot() {
    let mut deferred = Some(Metrics {
        dp_rank: 1,
        active_blocks: 99,
        running_requests: 7,
        ..Metrics::default()
    });
    let completed = Metrics {
        dp_rank: 1,
        active_blocks: 3,
        running_requests: 1,
        ..Metrics::default()
    };

    assert_eq!(
        completion_metrics(&mut deferred, completed.clone()),
        completed
    );
    assert!(deferred.is_none());
}

#[tokio::test]
async fn closed_output_receiver_cancels_request_and_releases_native_kv() {
    let mut slow_args = args(1);
    slow_args.speedup_ratio = 1.0;
    slow_args.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(SlowDecode)));
    let effects = Arc::new(CapturedEffects::default());
    let (output_tx, output_rx) = mpsc::unbounded_channel();
    drop(output_rx);
    let cancel = CancellationToken::new();
    let GroupedSchedulers {
        schedulers, actor, ..
    } = create_grouped_scheduler(
        slow_args,
        vec![GroupedSchedulerRankSinks {
            output_tx: Some(output_tx),
            fpm_publisher: FpmPublisher::new(Some(Arc::clone(&effects) as Arc<dyn FpmSink>)),
            ..GroupedSchedulerRankSinks::default()
        }],
        Some(cancel.clone()),
    )
    .unwrap();
    let mut metrics = schedulers[0].metrics_receiver();
    schedulers[0]
        .request_sender()
        .send(DirectRequest {
            tokens: vec![1; 16],
            max_output_tokens: 8,
            output_token_ids: Some((0..8).collect()),
            uuid: Some(Uuid::from_u128(303)),
            dp_rank: 0,
            ..DirectRequest::default()
        })
        .unwrap();

    let drained_result = tokio::time::timeout(Duration::from_millis(500), async {
        loop {
            metrics.changed().await.unwrap();
            let current = metrics.borrow().clone();
            if current.active_decode_blocks == 0
                && current.running_requests == 0
                && current.waiting_requests == 0
            {
                break current;
            }
        }
    })
    .await;
    let drained = drained_result.unwrap_or_else(|_| {
        panic!(
            "closed output must cancel before all eight decode passes finish; latest metrics: {:?}",
            metrics.borrow()
        )
    });
    assert_eq!(drained.gpu_cache_usage_perc, 0.0);
    assert_eq!(
        effects.fpm.lock().unwrap().len(),
        1,
        "output delivery failure must cancel at the same boundary before a decode pass starts"
    );

    cancel.cancel();
    actor.await.unwrap().unwrap();
}
