// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use anyhow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    GetNumNewMatchedTokens(GetNumNewMatchedTokensInput, GetNumNewMatchedTokensOutput),
    UpdateStateAfterAlloc(UpdateStateAfterAllocInput, UpdateStateAfterAllocOutput),
    BuildConnectorMeta(BuildConnectorMetaInput, BuildConnectorMetaOutput),
    RequestFinished(RequestFinishedInput, RequestFinishedOutput),
    HasSlot(HasSlotInput, HasSlotOutput),
    CreateSlot(CreateSlotInput, CreateSlotOutput),
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

#[derive(Debug)]
pub struct KvConnectorLeaderRecorder {
    _recorder: Recorder<Action>, // Keep recorder alive
    unbounded_tx: mpsc::UnboundedSender<Action>,
    connector_leader: Box<dyn Leader>,
}

impl KvConnectorLeaderRecorder {
    pub fn new(
        worker_id: String,
        drt: PyDistributedRuntime,
        page_size: usize,
        leader_py: PyKvbmLeader,
    ) -> Self {
        tracing::info!(
            "KvConnectorLeaderRecorder initialized with worker_id: {}",
            worker_id
        );

        let leader = leader_py.get_inner().clone();
        let drt = drt.inner().clone();
        let handle: Handle = drt.runtime().primary();

        let kvbm_metrics = KvbmMetrics::new(
            &KvbmMetricsRegistry::default(),
            kvbm_metrics_endpoint_enabled(),
            parse_kvbm_metrics_port(),
        );
        let kvbm_metrics_clone = kvbm_metrics.clone();

        let token = CancellationToken::new();
        let output_path = "/tmp/records.jsonl";
        tracing::info!("recording events to {}", output_path);

        let recorder = drt
            .runtime()
            .primary()
            .block_on(async { Recorder::new(token, &output_path, None, None, None).await })
            .unwrap();

        let (unbounded_tx, unbounded_rx) = mpsc::unbounded_channel();
        let recorder_tx = recorder.event_sender();

        // todo(kvbm): make this a critical task
        drt.runtime()
            .primary()
            .spawn(Self::forward_unbounded_to_sender(unbounded_rx, recorder_tx));

        let slot_manager_cell = Arc::new(OnceLock::new());
        let (leader_ready_tx, leader_ready_rx) = oneshot::channel::<String>();

        {
            let slot_manager_cell = slot_manager_cell.clone();

            handle.spawn(async move {
                let ready = leader.wait_worker_sync_ready().await;
                if !ready {
                    tracing::error!(
                        "KvConnectorLeader init aborted: leader worker barrier not ready!",
                    );
                    return;
                }

                let block_manager = match BlockManagerBuilder::new()
                    .worker_id(0)
                    .leader(leader_py)
                    .page_size(page_size)
                    .disable_device_pool(false)
                    .kvbm_metrics(kvbm_metrics_clone.clone())
                    .build()
                    .await
                {
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
                    drt.clone(),
                    kvbm_metrics_clone.clone(),
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
            unbounded_tx,
            connector_leader: Box::new(connector_leader),
        }
    }

    async fn forward_unbounded_to_sender<T: Send + 'static>(
        mut unbounded_rx: mpsc::UnboundedReceiver<T>,
        bounded_tx: mpsc::Sender<T>,
    ) {
        while let Some(msg) = unbounded_rx.recv().await {
            if bounded_tx.send(msg).await.is_err() {
                tracing::error!("Failed to send message to bounded channel");
            }
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
        let _ = self.unbounded_tx.send(Action::GetNumNewMatchedTokens(
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
        let _ = self.unbounded_tx.send(Action::UpdateStateAfterAlloc(
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
        let _ = self.unbounded_tx.send(Action::BuildConnectorMeta(
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
        let _ = self.unbounded_tx.send(Action::RequestFinished(
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
        let _ = self.unbounded_tx.send(Action::HasSlot(
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
        let _ = self
            .unbounded_tx
            .send(Action::CreateSlot(input_copy, CreateSlotOutput {}));
        Ok(())
    }
}
