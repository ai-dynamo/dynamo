// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use anyhow;
use dynamo_llm::block_manager::kv_consolidator::EventSource;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::task::JoinHandle;

/// Capacity of the staging queue between the synchronous connector hooks and
/// the [`Recorder`]'s own event channel. Mirrors that channel's 2,048-event
/// bound, so total retained actions are bounded by ingress + downstream + the
/// single action the forwarding task holds in flight.
const RECORD_INGRESS_CAPACITY: usize = 2048;

/// Drops between overload warnings after the first one. Overload is by
/// definition high-frequency, so an unthrottled warning would become the
/// unbounded thing the queue no longer is.
const DROP_WARN_INTERVAL: u64 = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    GetNumNewMatchedTokens(GetNumNewMatchedTokensInput, GetNumNewMatchedTokensOutput),
    UpdateStateAfterAlloc(UpdateStateAfterAllocInput, UpdateStateAfterAllocOutput),
    BuildConnectorMeta(BuildConnectorMetaInput, BuildConnectorMetaOutput),
    RequestFinished(RequestFinishedInput, RequestFinishedOutput),
    HasSlot(HasSlotInput, HasSlotOutput),
    CreateSlot(CreateSlotInput, CreateSlotOutput),
    ResetCache(ResetCacheInput, ResetCacheOutput),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetNumNewMatchedTokensInput {
    request_id: String,
    request_num_tokens: usize,
    num_computed_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetNumNewMatchedTokensOutput {
    num_new_matched_tokens: usize,
    has_matched: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStateAfterAllocInput {
    request_id: String,
    block_ids: Vec<BlockId>,
    num_external_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStateAfterAllocOutput {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConnectorMetaInput {
    scheduler_output: SchedulerOutput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConnectorMetaOutput {
    metadata: ConnectorMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestFinishedInput {
    request_id: String,
    block_ids: Vec<BlockId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestFinishedOutput {
    is_finished: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasSlotInput {
    request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasSlotOutput {
    result: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSlotInput {
    request: KvbmRequest,
    tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSlotOutput {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetCacheInput {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetCacheOutput {
    result: bool,
}

/// Bounded, explicitly lossy staging queue in front of a [`Recorder`]'s event
/// channel.
///
/// The producers are synchronous vLLM scheduler callbacks reached through
/// pyo3, so they cannot wait for capacity: applying real backpressure here
/// would turn a stall on the recorder's output file into a stall of the
/// serving path. `record` therefore never blocks and never fails — when the
/// queue is full it discards the action and counts it.
///
/// Recording is consequently **lossy under overload**: the JSONL output can
/// contain a gap. The gap is announced rather than silent, through
/// [`Self::dropped_count`], a throttled warning, and the total logged when the
/// ingress is dropped. The same counter covers the other way an action can go
/// missing — the recorder shutting down while producers are still calling —
/// so an action is always either written or counted.
///
/// The *newest* action is discarded, not the oldest. That keeps whatever is
/// retained a FIFO prefix of the action stream instead of a shuffled window,
/// and it means an action that was accepted is never later evicted.
#[derive(Debug)]
struct ActionRecorderIngress<T: Send + 'static> {
    tx: mpsc::Sender<T>,
    dropped: Arc<AtomicU64>,
    capacity: usize,
    // Held rather than detached so the forwarding task's lifetime is tied to
    // this value. It is never aborted: dropping `tx` ends the loop only after
    // it has drained the actions it already accepted. The loop can also end
    // on its own, when the recorder's channel closes under it.
    _forwarder: JoinHandle<()>,
}

impl<T: Send + 'static> ActionRecorderIngress<T> {
    /// The runtime handle and the capacity are parameters rather than a global
    /// lookup and a private constant so that the queue can be constructed —
    /// and its overload behavior exercised — from a plain `cargo test`, with
    /// no GPU, no pyo3 state, and no leader/worker barrier.
    fn new(rt: &Handle, downstream: mpsc::Sender<T>, capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let forwarder = rt.spawn(Self::forward_to_downstream(rx, downstream, dropped.clone()));

        Self {
            tx,
            dropped,
            capacity,
            _forwarder: forwarder,
        }
    }

    /// Enqueue an action, discarding it if the queue is full or the recorder
    /// has already shut down. Callable from a synchronous thread that may or
    /// may not be inside a runtime.
    fn record(&self, item: T) {
        let closed = match self.tx.try_send(item) {
            Ok(()) => return,
            Err(mpsc::error::TrySendError::Full(_)) => false,
            Err(mpsc::error::TrySendError::Closed(_)) => true,
        };

        let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped.is_multiple_of(DROP_WARN_INTERVAL) {
            if closed {
                tracing::warn!(
                    dropped,
                    capacity = self.capacity,
                    "kvbm recorder ingress is closed; dropping action"
                );
            } else {
                tracing::warn!(
                    dropped,
                    capacity = self.capacity,
                    "kvbm recorder ingress is full; dropping newest action"
                );
            }
        }
    }

    fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Move accepted actions to the recorder's own channel until either the
    /// ingress or the recorder goes away.
    ///
    /// A failed `send` means the recorder's receiver has been dropped — after
    /// cancellation, or once a `max_count`/`max_time` limit ends its writer
    /// task. That is terminal, never transient, so the loop stops there
    /// instead of continuing to receive actions it could only throw away: a
    /// forwarder that kept draining would leave `tx` open, so producers would
    /// go on succeeding while every action vanished uncounted, which is the
    /// silent gap the drop counter exists to rule out.
    ///
    /// Closing `rx` on the way out converts that into the accounted case. The
    /// action in flight and everything still buffered are added to `dropped`,
    /// and later `record` calls see `Closed` and count themselves.
    async fn forward_to_downstream(
        mut rx: mpsc::Receiver<T>,
        downstream: mpsc::Sender<T>,
        dropped: Arc<AtomicU64>,
    ) {
        while let Some(msg) = rx.recv().await {
            if downstream.send(msg).await.is_err() {
                rx.close();
                let mut lost: u64 = 1;
                while rx.try_recv().is_ok() {
                    lost += 1;
                }
                let total = dropped.fetch_add(lost, Ordering::Relaxed) + lost;
                tracing::error!(
                    lost,
                    dropped = total,
                    "kvbm recorder channel is closed; discarding buffered actions and \
                     stopping the forwarder"
                );
                return;
            }
        }
    }
}

impl<T: Send + 'static> Drop for ActionRecorderIngress<T> {
    fn drop(&mut self) {
        let dropped = self.dropped_count();
        if dropped > 0 {
            tracing::warn!(
                dropped,
                capacity = self.capacity,
                "kvbm recorder dropped actions under overload; the recorded trace has a gap"
            );
        }
    }
}

#[derive(Debug)]
pub struct KvConnectorLeaderRecorder {
    _recorder: Recorder<Action>, // Keep recorder alive
    ingress: ActionRecorderIngress<Action>,
    connector_leader: Box<dyn Leader>,
}

impl KvConnectorLeaderRecorder {
    pub fn new(
        worker_id: String,
        page_size: usize,
        leader_py: PyKvbmLeader,
        consolidator_vllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
        consolidator_mode: Option<String>,
    ) -> Self {
        tracing::info!(
            "KvConnectorLeaderRecorder initialized with worker_id: {}",
            worker_id
        );

        let leader = leader_py.get_inner().clone();
        let handle: Handle = get_current_tokio_handle();

        let kvbm_metrics = KvbmMetrics::new(
            &KvbmMetricsRegistry::default(),
            kvbm_metrics_endpoint_enabled(),
            parse_kvbm_metrics_port(),
        );
        let kvbm_metrics_clone = kvbm_metrics.clone();

        let token = CancellationToken::new();
        let output_path = "/tmp/records.jsonl";
        tracing::info!("recording events to {}", output_path);

        let recorder = get_current_tokio_handle()
            .block_on(async { Recorder::new(token, &output_path, None, None, None).await })
            .unwrap();

        // todo(kvbm): make this a critical task
        // The queue in front of the writer is bounded, so if writes to
        // `output_path` cannot keep pace the trace loses actions rather than
        // accumulating them: expect gaps in that file, counted and logged by
        // `ActionRecorderIngress`.
        let ingress = ActionRecorderIngress::new(
            &get_current_tokio_handle(),
            recorder.event_sender(),
            RECORD_INGRESS_CAPACITY,
        );

        let slot_manager_cell = Arc::new(OnceLock::new());
        let (leader_ready_tx, leader_ready_rx) = oneshot::channel::<String>();

        {
            let slot_manager_cell = slot_manager_cell.clone();
            // Capture consolidator endpoints for the async block
            let consolidator_vllm_ep = consolidator_vllm_endpoint.clone();
            let consolidator_output_ep = consolidator_output_endpoint.clone();
            let consolidator_mode = super::parse_consolidator_mode(consolidator_mode.clone());

            handle.spawn(async move {
                let ready = leader.wait_worker_sync_ready().await;
                if !ready {
                    tracing::error!(
                        "KvConnectorLeader init aborted: leader worker barrier not ready!",
                    );
                    return;
                }

                let mut block_manager_builder = BlockManagerBuilder::new()
                    .worker_id(0)
                    .leader(leader_py)
                    .page_size(page_size)
                    .disable_device_pool(false)
                    .kvbm_metrics(kvbm_metrics_clone.clone());

                // Add consolidator config if provided
                if let (Some(vllm_ep), Some(output_ep)) =
                    (consolidator_vllm_ep, consolidator_output_ep)
                {
                    block_manager_builder = block_manager_builder.consolidator_config(
                        vllm_ep,
                        Some(output_ep),
                        EventSource::Vllm,
                        consolidator_mode,
                    );
                }

                let block_manager = match block_manager_builder.build().await {
                    Ok(bm) => bm,
                    Err(e) => {
                        tracing::error!("Failed to build BlockManager: {}", e);
                        return;
                    }
                };

                // Create the slot manager now that everything is ready
                let sm = ConnectorSlotManager::new(
                    block_manager.get_block_manager().clone(),
                    leader.clone(),
                    kvbm_metrics_clone.clone(),
                    Some(format!("worker-{}", worker_id)),
                );

                let _ = slot_manager_cell.set(sm);

                if leader_ready_tx.send("finished".to_string()).is_err() {
                    tracing::error!("main routine receiver dropped before result was sent");
                }
            });
        }

        tokio::task::block_in_place(|| {
            handle.block_on(async {
                match leader_ready_rx.await {
                    Ok(_) => tracing::info!("KvConnectorLeader init complete."),
                    Err(_) => tracing::warn!("KvConnectorLeader init channel dropped"),
                }
            });
        });

        let connector_leader = KvConnectorLeader {
            slot_manager: slot_manager_cell,
            block_size: page_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            iteration_counter: 0,
            kvbm_metrics,
        };

        Self {
            _recorder: recorder,
            ingress,
            connector_leader: Box::new(connector_leader),
        }
    }
}

impl Leader for KvConnectorLeaderRecorder {
    #[inline]
    fn slot_manager(&self) -> &ConnectorSlotManager<String> {
        self.connector_leader.slot_manager()
    }
    /// Match the tokens in the request with the available block pools.
    /// Note: the necessary details of the request are captured prior to this call. For vllm,
    /// we make a create slot call prior to this call, so a slot is guaranteed to exist.
    ///
    /// To align with the connector interface, we must ensure that if no blocks are matched, we return (0, false).
    /// In our implementation, if we match any block, we return (num_matched_tokens, true).
    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)> {
        let input_copy = GetNumNewMatchedTokensInput {
            request_id: request_id.clone(),
            request_num_tokens,
            num_computed_tokens,
        };
        let output = self.connector_leader.get_num_new_matched_tokens(
            request_id,
            request_num_tokens,
            num_computed_tokens,
        )?;
        self.ingress.record(Action::GetNumNewMatchedTokens(
            input_copy,
            GetNumNewMatchedTokensOutput {
                num_new_matched_tokens: output.0,
                has_matched: output.1,
            },
        ));
        Ok(output)
    }

    /// We drop the need to pass in the KvCacheBlocks and the num_external_tokens as they are captured
    /// statefully in the [`VllmLeaderKvCacheManagerAndConnector::get_num_new_matched_tokens`] function.
    ///
    /// Note: vLLM will not provide any scheduler output data for requests that are onboarding. it is entirely
    /// on the connector's implementation to handle this case.
    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()> {
        let input_copy = UpdateStateAfterAllocInput {
            request_id: request_id.clone(),
            block_ids: block_ids.clone(),
            num_external_tokens,
        };
        self.connector_leader.update_state_after_alloc(
            request_id,
            block_ids,
            num_external_tokens,
        )?;
        self.ingress.record(Action::UpdateStateAfterAlloc(
            input_copy,
            UpdateStateAfterAllocOutput {},
        ));
        Ok(())
    }

    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>> {
        let input_copy = BuildConnectorMetaInput {
            scheduler_output: scheduler_output.clone(),
        };
        let output = self
            .connector_leader
            .build_connector_metadata(scheduler_output)?;
        self.ingress.record(Action::BuildConnectorMeta(
            input_copy,
            BuildConnectorMetaOutput {
                metadata: serde_json::from_slice(&output)?,
            },
        ));
        Ok(output)
    }

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool> {
        let input_copy = RequestFinishedInput {
            request_id: request_id.clone(),
            block_ids: block_ids.clone(),
        };
        let output = self
            .connector_leader
            .request_finished(request_id, block_ids)?;
        self.ingress.record(Action::RequestFinished(
            input_copy,
            RequestFinishedOutput {
                is_finished: output,
            },
        ));
        Ok(output)
    }

    fn has_slot(&self, request_id: String) -> bool {
        let input_copy = HasSlotInput {
            request_id: request_id.clone(),
        };
        let output = self.connector_leader.has_slot(request_id);
        self.ingress.record(Action::HasSlot(
            input_copy,
            HasSlotOutput { result: output },
        ));
        output
    }

    /// Create a new slot for the given request ID.
    /// This is used to create a new slot for the request.
    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()> {
        let input_copy = CreateSlotInput {
            request: request.clone(),
            tokens: tokens.clone(),
        };
        let _ = self.connector_leader.create_slot(request, tokens);
        self.ingress
            .record(Action::CreateSlot(input_copy, CreateSlotOutput {}));
        Ok(())
    }

    fn reset_cache(&mut self) -> anyhow::Result<bool> {
        let output = self.connector_leader.reset_cache()?;
        self.ingress.record(Action::ResetCache(
            ResetCacheInput {},
            ResetCacheOutput { result: output },
        ));
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the production `Action` the cheapest hook records, tagged with a
    /// monotone id so ordering and loss are both readable off the drained
    /// stream.
    fn has_slot_action(id: usize) -> Action {
        Action::HasSlot(
            HasSlotInput {
                request_id: id.to_string(),
            },
            HasSlotOutput { result: true },
        )
    }

    /// Drain everything the forwarding task will ever deliver.
    ///
    /// The ingress must already have been dropped: that is what closes the
    /// forwarding loop once it has flushed the actions it accepted, which in
    /// turn drops the last downstream sender and terminates this loop. Without
    /// it a stalled-sink test would hang instead of failing.
    async fn drain<T>(rx: &mut mpsc::Receiver<T>) -> Vec<T> {
        let mut drained = Vec::new();
        while let Some(item) = rx.recv().await {
            drained.push(item);
        }
        drained
    }

    /// T1: with a sink that never drains, retention is capped near the
    /// configured capacity instead of growing with the number of pushes, and
    /// the loss is counted rather than silent.
    #[tokio::test]
    async fn stalled_sink_bounds_retention_and_counts_the_loss() {
        const CAPACITY: usize = 4;
        const PUSHED: usize = 100;

        let (downstream_tx, mut downstream_rx) = mpsc::channel::<usize>(1);
        let ingress = ActionRecorderIngress::new(&Handle::current(), downstream_tx, CAPACITY);

        for i in 0..PUSHED {
            ingress.record(i);
        }

        let dropped = ingress.dropped_count();
        assert!(
            dropped > 0,
            "overload against a stalled sink must be counted, got {dropped} drops after {PUSHED} pushes"
        );

        drop(ingress);
        let retained = drain(&mut downstream_rx).await;

        // Ceiling: the ingress queue, plus the one action the forwarding task
        // holds while parked on the downstream `send`, plus the one the
        // downstream channel itself buffers. Deliberately an inequality --
        // whether the forwarder had already parked is a scheduling detail.
        assert!(
            retained.len() <= CAPACITY + 2,
            "retained {} actions, which exceeds the {} the queue can hold",
            retained.len(),
            CAPACITY + 2
        );
        assert_eq!(
            retained.len() as u64 + dropped,
            PUSHED as u64,
            "every action must be either retained or counted as dropped"
        );
    }

    /// T2: the actions that were accepted come out in the order they went in,
    /// and they are a prefix of the pushed stream -- the drop-newest policy,
    /// observed on real `Action` values rather than on the generic parameter.
    #[tokio::test]
    async fn accepted_actions_drain_in_fifo_order_after_the_sink_is_released() {
        const CAPACITY: usize = 4;
        const PUSHED: usize = 32;

        let (downstream_tx, mut downstream_rx) = mpsc::channel::<Action>(1);
        let ingress = ActionRecorderIngress::new(&Handle::current(), downstream_tx, CAPACITY);

        for i in 0..PUSHED {
            ingress.record(has_slot_action(i));
        }
        assert!(
            ingress.dropped_count() > 0,
            "pushing {PUSHED} actions through a capacity-{CAPACITY} queue with a stalled sink must overflow it"
        );

        drop(ingress);
        let ids: Vec<usize> = drain(&mut downstream_rx)
            .await
            .into_iter()
            .map(|action| match action {
                Action::HasSlot(input, _) => input.request_id.parse().unwrap(),
                other => panic!("recorded a different action than the one pushed: {other:?}"),
            })
            .collect();

        assert!(!ids.is_empty(), "the queue must retain what it accepted");
        assert!(
            ids.windows(2).all(|pair| pair[0] < pair[1]),
            "accepted actions must drain in FIFO order, got {ids:?}"
        );
        assert_eq!(
            ids,
            (0..ids.len()).collect::<Vec<_>>(),
            "dropping the newest action must leave a prefix of the pushed stream"
        );
    }

    /// T3: the negative control. A bound that simply threw actions away would
    /// pass T1; this is the test it would fail.
    #[tokio::test]
    async fn a_sink_that_keeps_up_loses_nothing() {
        const CAPACITY: usize = 8;
        const PUSHED: usize = 5;

        let (downstream_tx, mut downstream_rx) = mpsc::channel::<usize>(1);
        let ingress = ActionRecorderIngress::new(&Handle::current(), downstream_tx, CAPACITY);

        let mut received = Vec::new();
        for i in 0..PUSHED {
            ingress.record(i);
            received.push(downstream_rx.recv().await.expect("sink closed early"));
        }

        assert_eq!(
            ingress.dropped_count(),
            0,
            "a sink that keeps up, fed fewer actions than the queue holds, must lose nothing"
        );
        assert_eq!(received, (0..PUSHED).collect::<Vec<_>>());
    }

    /// T4: the other way an action can go missing. When the recorder's own
    /// receiver is gone, nothing downstream can be delivered any more, so
    /// every push must show up in the drop count -- none of it may be
    /// discarded quietly inside the forwarding task.
    #[tokio::test]
    async fn a_departed_recorder_counts_the_actions_it_can_no_longer_take() {
        const CAPACITY: usize = 4;
        const PUSHED: usize = 64;

        let (downstream_tx, downstream_rx) = mpsc::channel::<usize>(1);
        let ingress = ActionRecorderIngress::new(&Handle::current(), downstream_tx, CAPACITY);

        // What `Recorder`'s writer task does when it is cancelled or hits a
        // `max_count`/`max_time` limit: its `event_rx` is dropped.
        drop(downstream_rx);

        for i in 0..PUSHED {
            ingress.record(i);
        }

        // The forwarding task has to be polled before it can notice the
        // closure, so the accounting settles asynchronously. The timeout is a
        // failure bound, not a delay -- it is reached only if some action was
        // neither delivered nor counted.
        let settled = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while ingress.dropped_count() != PUSHED as u64 {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;

        assert!(
            settled.is_ok(),
            "with the recorder gone, all {PUSHED} actions must be counted as dropped, got {}",
            ingress.dropped_count()
        );
    }
}
