// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::{
    block_manager::{
        connector::protocol::{
            LeaderTransferRequest, RequestType, TransferScheduleRequest, TransferType,
        },
        distributed::{BlockTransferPool, BlockTransferRequest, KvbmLeader},
    },
    tokens::TokenBlock,
};
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use dynamo_runtime::traits::{DistributedRuntimeProvider, RuntimeProvider};
use tokio_util::sync::CancellationToken;

use super::*;

#[derive(Debug, thiserror::Error)]
pub enum SlotError {
    #[error("slot not found")]
    NotFound,

    #[error("slot is in an invalid state: {0}")]
    InvalidState(String),

    #[error("slot operation failed: {0}")]
    InvalidOperation(String),

    #[error(transparent)]
    BlockPoolError(#[from] BlockPoolError),
}

pub trait SlotManager<R: RequestKey>: Send + Sync {
    type SlotType: Slot + ?Sized;

    fn has_slot(&self, request_id: &R) -> bool;

    /// Create a new slot for the given request ID, initial tokens and salt hash.
    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError>;

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError>;
    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// The slot was not scheduled in the previous iteration.
    Initialized,

    /// The slot was previously scheduled, but not in the last iteration.
    NotScheduled,

    /// The slot is prepared to load kv blocks from external storage; however, the onboarding operation
    /// has not been triggered yet. The usize is the number of tokens that are ready for onboarding.
    OnboardStaged(usize),

    /// The slot is actively copying blocks to device storage from some external storage(s).
    /// The u64 is the iteration at which the onboarding operation was triggered.
    Onboarding(u64),

    /// The slot is actively prefilling the sequence.
    Prefilling,

    /// The slot is actively participating in a forward pass which will result in one more more tokens
    /// to be applied to the sequence.
    Decoding,

    /// The slot is marked as finished, but not all resources have been released.
    Finishing,

    /// The slot is finished and all resources have been released.
    Finished,
}

pub trait Slot: std::fmt::Debug {
    fn request_id(&self) -> &str;

    fn state(&self) -> SlotState;

    fn sequence(&self) -> &TokenBlockSequence;

    /// The number of tokens that have been computed on the device, i.e. the number of tokens for which we have ownership
    /// of computed kv blocks in the device storage.
    fn computed_tokens(&self) -> usize;

    fn mark_as_scheduled(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_onboarding(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_not_scheduled(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_finished(&mut self, iteration: u64) -> Result<(), SlotError>;

    /// The number of device blocks that have been allocated to the slot.
    fn num_device_blocks_allocated(&self) -> usize;

    /// Find all possible block matches for remaining known tokens in some local storage, i.e. look up and take ownership
    /// of any kv blocks for tokens in the isl that are not already in memory on the device, but on some local storage.
    ///
    /// If external tokens are matched, then the slot will transition to the [`SlotState::Onboarding`] state.
    fn acquire_all_local_matches(&mut self) -> Result<(), SlotError>;

    /// Take all pending operations for the slot.
    fn take_pending_operations(&mut self) -> Vec<WorkerTransferRequest>;
}

pub trait ExternallyManagedDeviceSlot: Slot {
    /// Since we do not control the device pool, nor do we have insight in how the device pool is managed,
    /// we must accept external updates to the computed position.
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError>;

    /// Append the given block ids to the slot.
    ///
    /// The external device block manager has provided a set of mutable blocks to the slot.
    fn append_mutable_device_blocks(&mut self, block_ids: Vec<BlockId>) -> Result<(), SlotError>;
}

pub struct ConnectorSlotManager<R: RequestKey> {
    slots: Mutex<HashMap<R, Arc<Mutex<VllmConnectorSlot>>>>,
    block_manager: VllmBlockManager,
    leader: Arc<KvbmLeader>,
}

impl<R: RequestKey> ConnectorSlotManager<R> {
    pub fn new(block_manager: VllmBlockManager, leader: Arc<KvbmLeader>) -> Self {
        tracing::debug!(
            "creating slot manager with block size: {}",
            block_manager.block_size()
        );
        Self {
            slots: Mutex::new(HashMap::new()),
            block_manager,
            leader,
        }
    }
}

impl<R: RequestKey> SlotManager<R> for ConnectorSlotManager<R> {
    type SlotType = dyn ExternallyManagedDeviceSlot;

    fn has_slot(&self, request_id: &R) -> bool {
        self.slots.lock().unwrap().contains_key(request_id)
    }

    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError> {
        let slot = VllmConnectorSlot::new(
            request_id.to_string(),
            tokens.into(),
            salt_hash,
            self.block_manager.clone(),
            self.leader.clone(),
        );
        self.slots
            .lock()
            .unwrap()
            .insert(request_id.clone(), Arc::new(Mutex::new(slot)));
        Ok(())
    }

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError> {
        let slots = self.slots.lock().unwrap();
        let slot = slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.clone())
    }

    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError> {
        self.slots.lock().unwrap().remove(request_id);
        Ok(())
    }
}

pub struct VllmConnectorSlot {
    request_id: String,

    /// The state of the slot.
    state: SlotState,

    /// Current position in the sequence of tokens that have been computed.
    /// When the slot is initialized, we populate the sequence with the prefill tokens.
    /// However, those tokens are not yet prefilled, so they are not yet represented
    /// in the sequence_position.
    computed_position: usize,

    /// The sequence of token blocks
    sequence: TokenBlockSequence,

    /// The immutable blocks id (device)
    immutable: Vec<usize>,

    /// The mutable blocks id (device)
    mutable: VecDeque<usize>,

    /// Blocks to be onboarded from the host
    /// We must hold these blocks in the slot state until the scheduler trigger the onboarding.
    staging_from_host: Option<Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>>,

    /// Blocks to be onboarded from the disk
    /// We must hold these blocks in the slot state until the scheduler trigger the onboarding.
    staging_from_disk: Option<Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>>,

    /// Blocks to be offloaded to the host
    /// We must hold these blocks in the slot state until the scheduler trigger the offloading.
    offload_to_host: Vec<MutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>,

    /// The host block ids
    offloaded_to_host_blocks: Vec<usize>,

    /// The number of blocks cached from the device
    blocks_cached_from_device: usize,

    /// The number of blocks cached from the host
    blocks_cached_from_host: usize,

    /// The number of blocks cached from the disk
    blocks_cached_from_disk: usize,

    /// Phantom data to ensure the storage type is correct.
    block_manager: VllmBlockManager,

    leader: Arc<KvbmLeader>,

    block_size: usize,

    iteration_first_scheduled: Option<u64>,

    pending_operations: Vec<WorkerTransferRequest>,

    /// use this to issue [`LocalTransferRequest`]s to the transfer engine
    xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,

    rt: tokio::runtime::Runtime,
}

impl VllmConnectorSlot {
    pub fn new(
        request_id: String,
        tokens: Tokens,
        salt_hash: SaltHash,
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
    ) -> Self {
        assert!(!tokens.is_empty(), "tokens must be non-empty");
        let block_size = block_manager.block_size();
        debug_assert!(block_size.is_power_of_two() && block_size <= 1024);
        let sequence = TokenBlockSequence::new(tokens, block_size as u32, Some(salt_hash));

        let (xfer_tx, xfer_rx) = mpsc::unbounded_channel();

        let mut xfer_engine = LocalTransferEngine::new(block_manager.clone(), leader.clone(), xfer_rx);
        let system_cancellation_token = CancellationToken::new();

        // spawn a task to handle the transfer requests
        // use critical task pattern
        let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap();

        let xfer_engine_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token| async move {
                xfer_engine.execute(cancellation_token).await
            },
            system_cancellation_token,
            "LocalTransferEngine",
            &rt.handle(),
        ).unwrap();
        xfer_engine_task.detach();

        tracing::info!("LocalTransferEngine task detached successfully");

        Self {
            request_id,
            sequence,
            block_manager,
            block_size,
            leader,
            xfer_tx,
            rt,

            // default values
            state: SlotState::Initialized,
            iteration_first_scheduled: None,
            computed_position: 0,
            immutable: Vec::new(),
            mutable: VecDeque::new(),
            staging_from_host: None,
            staging_from_disk: None,
            offload_to_host: Vec::new(),
            offloaded_to_host_blocks: Vec::new(),
            pending_operations: Vec::new(),
            blocks_cached_from_device: 0,
            blocks_cached_from_host: 0,
            blocks_cached_from_disk: 0,
        }
    }
}

impl std::fmt::Debug for VllmConnectorSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VllmConnectorSlot")
            .field("state", &self.state)
            .field("computed_position", &self.computed_position)
            .field("num_tokens", &self.sequence.total_tokens())
            .finish()
    }
}

impl Slot for VllmConnectorSlot {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn state(&self) -> SlotState {
        self.state
    }

    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Prefilling;

        let token_blocks = self.sequence.blocks().to_vec();
        let mut block_ids: Vec<_> = self.mutable.iter().map(|b| *b).collect();
        block_ids.truncate(token_blocks.len());

        let local_offload_req = LocalOffloadRequest {
            request_id: self.request_id.clone(),
            block_ids,
            token_blocks,
            operation_id: uuid::Uuid::new_v4(),
        };

        let xfer_req = LocalTransferRequest::Offload(local_offload_req.clone());
        tracing::warn!("local_offload_req: {:?}", local_offload_req);

        if let Err(e) = self.xfer_tx.send(xfer_req) {
            tracing::error!("Failed to send transfer request: {:?}", e);
            return Err(SlotError::InvalidOperation(format!("Transfer engine unavailable: {}", e)));
        }

        Ok(())
    }

    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Decoding;
        Ok(())
    }

    fn mark_as_onboarding(&mut self, iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Onboarding(iteration);
        Ok(())
    }

    fn mark_as_scheduled(&mut self, iteration: u64) -> Result<(), SlotError> {
        if self.iteration_first_scheduled.is_none() {
            self.iteration_first_scheduled = Some(iteration);
        }
        Ok(())
    }

    fn mark_as_not_scheduled(&mut self, _iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::NotScheduled;
        Ok(())
    }

    fn mark_as_finished(&mut self, _iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Finishing;
        Ok(())
    }

    fn sequence(&self) -> &TokenBlockSequence {
        &self.sequence
    }

    fn computed_tokens(&self) -> usize {
        self.computed_position
    }

    fn num_device_blocks_allocated(&self) -> usize {
        self.immutable.len() + self.mutable.len()
    }

    fn take_pending_operations(&mut self) -> Vec<WorkerTransferRequest> {
        std::mem::take(&mut self.pending_operations)
    }

    #[tracing::instrument(level = "debug", skip_all)]
    fn acquire_all_local_matches(&mut self) -> Result<(), SlotError> {
        if !matches!(self.state(), SlotState::Initialized) {
            return Err(SlotError::InvalidOperation(format!(
                "slot must be in the NotScheduled state to acquire local matches; got {:?}",
                self.state()
            )));
        }

        let block_size = self.block_manager.block_size();
        let num_computed_tokens = self.computed_tokens();
        let num_computed_blocks = num_computed_tokens / block_size;
        debug_assert!(num_computed_tokens % block_size == 0);

        let sequence_hashes = self
            .sequence()
            .blocks()
            .iter()
            .skip(num_computed_blocks)
            .map(|b| b.sequence_hash())
            .collect::<Vec<_>>();

        tracing::debug!("matching against {} block hashes", sequence_hashes.len());

        // we should do this opportunistically after this operation is done
        // ideally it was triggered by the match_sequence_hashes_blocking calls directly

        // if let Some(host) = self.block_manager.host() {
        //     host.touch_blocks_blocking(&sequence_hashes)?;
        // }

        // if let Some(disk) = self.block_manager.disk() {
        //     disk.touch_blocks_blocking(&sequence_hashes)?;
        // }

        // we start matching non-device blocks after the device blocks
        let search_offset = num_computed_blocks;

        let mut host_blocks = self
            .block_manager
            .host()
            .map(|host| host.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_host_blocks = host_blocks.len();

        // advance the search offset by the number of matched host blocks
        let search_offset = search_offset + num_matched_host_blocks;

        // start at host offset
        let mut disk_blocks = self
            .block_manager
            .disk()
            .map(|disk| disk.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_disk_blocks = disk_blocks.len();

        let mut num_matched_blocks = num_matched_host_blocks + num_matched_disk_blocks;

        tracing::debug!(
            "matched {} host blocks and {} disk blocks; {} total blocks",
            num_matched_host_blocks,
            num_matched_disk_blocks,
            num_matched_blocks
        );

        if self.sequence.total_tokens() == 95 && self.state() == SlotState::Initialized {
            tracing::warn!("EXPERIMENTAL OVERRIDE: MATCHING FIRST BLOCK");
            num_matched_blocks = 1;
        }

        // early exit if we did not match any blocks
        if num_matched_blocks == 0 {
            return Ok(());
        }

        let mut num_new_matched_tokens = num_matched_blocks * block_size;

        // we are on a block boundary, so we need to throw away the last block
        if num_computed_tokens + num_new_matched_tokens == self.sequence().total_tokens() {
            tracing::debug!("on a block boundary, throwing away the last block");

            // we should have matched at least one block
            assert!(!host_blocks.is_empty() || !disk_blocks.is_empty());

            // pop from disk, or if there are none, then from host
            if disk_blocks.is_empty() {
                host_blocks.pop();
            } else {
                disk_blocks.pop();
            }

            // decrement the number of new matched tokens by the block size
            num_new_matched_tokens -= block_size;
        }

        self.staging_from_host = if !host_blocks.is_empty() {
            Some(host_blocks)
        } else {
            None
        };
        self.staging_from_disk = if !disk_blocks.is_empty() {
            Some(disk_blocks)
        } else {
            None
        };

        self.state = SlotState::OnboardStaged(num_new_matched_tokens);

        Ok(())
    }
}

impl ExternallyManagedDeviceSlot for VllmConnectorSlot {
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError> {
        if self.computed_position + num_tokens > self.sequence().total_tokens() {
            return Err(SlotError::InvalidOperation(format!(
                "cannot advance computed position by {num_tokens} tokens, total tokens is {}",
                self.sequence().total_tokens()
            )));
        }

        self.computed_position += num_tokens;
        Ok(())
    }

    fn append_mutable_device_blocks(&mut self, block_ids: Vec<BlockId>) -> Result<(), SlotError> {
        let count = block_ids.len();
        self.mutable.extend(block_ids);
        tracing::debug!(
            "appended {} mutable device blocks to slot; total device blocks: {}",
            count,
            self.num_device_blocks_allocated()
        );

        if self.sequence.total_tokens() == 95
            && self.staging_from_disk.is_none()
            && self.staging_from_host.is_none()
            && self.state() == SlotState::OnboardStaged(16)
        {
            tracing::warn!("EXPERIMENTAL OVERRIDE: APPENDING MUTABLE BLOCKS");
            assert!(!self.mutable.is_empty());

            tracing::warn!("EXPERIMENTAL OVERRIDE: TRIGGING JUNK H2D XFER");

            let device_block_id = *self.mutable.front().unwrap();
            let host_block_id: usize = 10;

            tracing::warn!("EXPERIMENTAL OVERRIDE: TRIGGING JUNK H2D XFER -1 ");
            let uuid = uuid::Uuid::new_v4();

            let sched_req = WorkerTransferRequest {
                request_id: self.request_id().to_string(),
                uuid,
                request_type: RequestType::Immediate,
                transfer_type: TransferType::Load,
            };

            tracing::warn!("EXPERIMENTAL OVERRIDE: TRIGGING JUNK H2D XFER -2");
            self.pending_operations.push(sched_req);

            let block_xfer_req = BlockTransferRequest {
                from_pool: BlockTransferPool::Host,
                to_pool: BlockTransferPool::Device,
                blocks: vec![(host_block_id, device_block_id)],
                connector_req: Some(LeaderTransferRequest {
                    request_id: self.request_id().to_string(),
                    uuid,
                    requirement: None,
                    request_type: RequestType::Immediate,
                }),
            };

            tracing::warn!("EXPERIMENTAL OVERRIDE: build tmp tokio runtime");

            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap();

            let result: anyhow::Result<()> = rt.block_on(async {
                tracing::warn!("EXPERIMENTAL OVERRIDE: trigger transfer");
                let notify = self.leader.transfer_blocks_request(block_xfer_req).await?;
                // tracing::warn!("EXPERIMENTAL OVERRIDE: await notify");
                // notify
                //     .await
                //     .map_err(|e| anyhow::anyhow!("Notify await failed: {:?}", e))?;
                // tracing::warn!("EXPERIMENTAL OVERRIDE: notify received");
                Ok(())
            });

            tracing::warn!("EXPERIMENTAL OVERRIDE: result: {:?}", result);

            result
                .map_err(|e| anyhow::anyhow!("Transfer blocks request failed: {:?}", e))
                .unwrap();
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
enum LocalTransferRequest {
    Offload(LocalOffloadRequest),
}

#[derive(Debug, Clone)]
struct LocalOffloadRequest {
    request_id: String,
    block_ids: Vec<BlockId>,
    token_blocks: Vec<TokenBlock>,
    operation_id: uuid::Uuid,
}

impl LocalOffloadRequest {
    pub fn new(
        request_id: String,
        block_ids: Vec<BlockId>,
        token_blocks: Vec<TokenBlock>,
        operation_id: uuid::Uuid,
    ) -> Self {
        debug_assert!(block_ids.len() == token_blocks.len());
        Self {
            request_id,
            block_ids,
            token_blocks,
            operation_id,
        }
    }
}

struct LocalTransferEngine {
    block_manager: VllmBlockManager,
    leader: Arc<KvbmLeader>,
    xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
}

impl LocalTransferEngine {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) -> Self {
        Self {
            block_manager,
            leader,
            xfer_rx,
        }
    }

    async fn execute(&mut self, cancellation_token: CancellationToken) -> anyhow::Result<()> {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::debug!("LocalTransferEngine: received cancellation signal");
                    break;
                }
                req = self.xfer_rx.recv() => {
                    match req {
                        Some(req) => {
                            if let Err(e) = self.process_request(req).await {
                                tracing::error!("LocalTransferEngine: error processing request: {:?}", e);
                            }
                        }
                        None => {
                            tracing::debug!("LocalTransferEngine: channel closed");
                            break;
                        }
                    }
                }
            }
        }

        tracing::debug!("LocalTransferEngine: shutting down");
        Ok(())
    }

    async fn process_request(&mut self, req: LocalTransferRequest) -> anyhow::Result<()> {
        match req {
            LocalTransferRequest::Offload(offload_req) => {
                tracing::debug!("Processing offload request for {} blocks", offload_req.block_ids.len());

                // TODO: Implement actual offload logic
                // 1. Acquire mutable host blocks
                let mut host_blocks = self.block_manager.host().unwrap().allocate_blocks(offload_req.block_ids.len()).await?;
                let token_blocks = offload_req.token_blocks;

                let host_block_ids: Vec<usize> = host_blocks.iter().map(|b| b.block_id()).collect();
                let block_pairs: Vec<(usize, usize)> = offload_req.block_ids
                    .into_iter()
                    .zip(host_block_ids.into_iter())
                    .collect();

                // 2. Apply token blocks

                // create an iterator over the mutable blocks zipped with the token blocks
                let mut blocks_to_register = Vec::new();
                let zipped_blocks = host_blocks.into_iter().zip(token_blocks.into_iter());

                // apply the token blocks to the mutable blocks
                for (mut mutable_block, token_block) in zipped_blocks {
                    mutable_block
                        .apply_token_block(token_block.clone())
                        .map_err(|e| anyhow::anyhow!("failed to apply token block: {:?}", e))?;

                    blocks_to_register.push(mutable_block);
                }

                // 3. Issue the offload request using `leader`

                let block_xfer_req = BlockTransferRequest {
                    from_pool: BlockTransferPool::Device,
                    to_pool: BlockTransferPool::Host,
                    blocks: block_pairs,
                    connector_req: Some(LeaderTransferRequest {
                        request_id: offload_req.request_id,
                        uuid: offload_req.operation_id,
                        requirement: None,
                        request_type: RequestType::Scheduled,
                    }),
                };
                let notify_receiver = self.leader.transfer_blocks_request(block_xfer_req).await?;

                // 4. Wait for the offload request to complete
                match notify_receiver.await {
                    Ok(_) => {
                        tracing::debug!("Transfer completed successfully");
                    }
                    Err(_) => {
                        return Err(anyhow::anyhow!("Transfer completion notification failed"));
                    }
                }
                // 5. Register the mutable blocks
                self.block_manager.host().unwrap().register_blocks(blocks_to_register).await?;

                Ok(())
            }
        }
    }
}
