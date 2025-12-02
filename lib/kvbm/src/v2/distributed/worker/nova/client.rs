// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct NovaWorkerClient {
    nova: Arc<Nova>,
    remote: InstanceId,
    g1_handle: Arc<OnceLock<LayoutHandle>>,
    g2_handle: Arc<OnceLock<LayoutHandle>>,
    g3_handle: Arc<OnceLock<LayoutHandle>>,
}

impl WorkerTransfers for NovaWorkerClient {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        // Create a single local event for this operation
        let event = self.nova.events().new_event()?;
        let handle = event.handle();
        let awaiter = self.nova.events().awaiter(handle)?;

        // Convert to serializable options
        // TODO: Extract bounce buffer handle if present in options.bounce_buffer
        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
        };

        // Create the message
        let message = LocalTransferMessage {
            src,
            dst,
            src_block_ids: src_block_ids.to_vec(),
            dst_block_ids: dst_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Spawn a task for the remote instance
        let nova = self.nova.clone();
        let bytes = bytes.clone();
        let remote_instance = self.remote;

        self.nova.tracker().spawn_on(
            async move {
                let result = nova
                    .am_sync("kvbm.worker.local_transfer")?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.nova.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.nova.events().new_event()?;
        let awaiter = self.nova.events().awaiter(event.handle())?;

        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
        };

        let message = RemoteOnboardMessage {
            src,
            dst,
            dst_block_ids: dst_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let nova = self.nova.clone();
        let remote_instance = self.remote;

        self.nova.tracker().spawn_on(
            async move {
                // Use unary instead of am_sync for explicit response handling
                let result = nova
                    .unary("kvbm.worker.remote_onboard")?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.nova.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.nova.events().new_event()?;
        let awaiter = self.nova.events().awaiter(event.handle())?;

        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
        };

        let message = RemoteOffloadMessage {
            src,
            dst,
            src_block_ids: src_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let nova = self.nova.clone();
        let remote_instance = self.remote;

        self.nova.tracker().spawn_on(
            async move {
                // Use unary instead of am_sync for explicit response handling
                let result = nova
                    .unary("kvbm.worker.remote_offload")?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.nova.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn connect_remote(
        &self,
        _instance_id: InstanceId,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        // NovaWorkerClient is a remote proxy - it doesn't store handle mappings locally.
        // The containing ReplicatedWorker handles mapping storage and calls import_metadata
        // on this worker directly for the actual metadata exchange.
        anyhow::bail!(
            "connect_remote not supported on NovaWorkerClient - use ReplicatedWorker instead"
        )
    }

    fn has_remote_metadata(&self, _instance_id: InstanceId) -> bool {
        // NovaWorkerClient doesn't track remote metadata locally.
        // The containing ReplicatedWorker tracks this.
        false
    }

    fn execute_remote_onboard_for_instance(
        &self,
        _instance_id: InstanceId,
        _remote_logical_type: LogicalLayoutHandle,
        _src_block_ids: Vec<BlockId>,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        // NovaWorkerClient doesn't store handle mappings locally.
        // The containing ReplicatedWorker handles mapping lookup and calls
        // execute_remote_onboard with concrete handles.
        anyhow::bail!(
            "execute_remote_onboard_for_instance not supported on NovaWorkerClient - use ReplicatedWorker instead"
        )
    }
}

impl Worker for NovaWorkerClient {
    fn g1_handle(&self) -> Option<LayoutHandle> {
        self.g1_handle.get().copied()
    }

    fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle.get().copied()
    }

    fn g3_handle(&self) -> Option<LayoutHandle> {
        self.g3_handle.get().copied()
    }

    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        // Use unary (not typed_unary) to avoid JSON serialization of bincode data
        let unary_result = self
            .nova
            .unary("kvbm.worker.export_metadata")?
            .instance(self.remote)
            .send();

        // Wrap UnaryResult to convert Bytes to SerializedLayout
        let future = async move {
            let bytes = unary_result.await?;
            Ok(SerializedLayout::from_bytes(bytes))
        };

        Ok(SerializedLayoutResponse::from_boxed(Box::pin(future)))
    }

    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        // Use raw_payload to avoid JSON serialization of bincode data
        let unary_result = self
            .nova
            .unary("kvbm.worker.import_metadata")?
            .raw_payload(metadata.as_bytes().clone())
            .instance(self.remote)
            .send();

        // Response is JSON-serialized Vec<LayoutHandle>
        let future = async move {
            let bytes = unary_result.await?;
            serde_json::from_slice(&bytes).map_err(|e| {
                anyhow::anyhow!("Failed to deserialize import_metadata response: {}", e)
            })
        };

        Ok(ImportMetadataResponse::from_boxed(Box::pin(future)))
    }
}

impl NovaWorkerClient {
    /// Create a new NovaWorkerClient for communicating with a remote worker.
    pub fn new(nova: Arc<Nova>, remote: InstanceId) -> Self {
        Self {
            nova,
            remote,
            g1_handle: Arc::new(OnceLock::new()),
            g2_handle: Arc::new(OnceLock::new()),
            g3_handle: Arc::new(OnceLock::new()),
        }
    }

    /// Configure layout handles from serialized metadata.
    ///
    /// Call this after worker initialization when handles are known from WorkerLayoutResponse.
    /// This allows the NovaWorkerClient to provide layout handles like DirectWorker does.
    ///
    /// # Arguments
    /// * `metadata` - SerializedLayout from WorkerLayoutResponse.metadata
    ///
    /// # Example
    /// ```ignore
    /// let response: WorkerLayoutResponse = worker.initialize(config).await?;
    /// worker_client.configure_layout_handles(&response.metadata)?;
    /// ```
    pub fn configure_layout_handles(&self, metadata: &SerializedLayout) -> Result<()> {
        let unpacked = metadata.unpack()?;
        for desc in &unpacked.layouts {
            match desc.logical_type {
                LogicalLayoutHandle::G1 => {
                    self.g1_handle.set(desc.handle).ok();
                }
                LogicalLayoutHandle::G2 => {
                    self.g2_handle.set(desc.handle).ok();
                }
                LogicalLayoutHandle::G3 => {
                    self.g3_handle.set(desc.handle).ok();
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Get the layout configuration from the remote worker.
    ///
    /// This calls the `kvbm.worker.get_layout_config` handler on the remote worker.
    /// Used by the leader during Phase 3 to gather G1 layout configs from all workers
    /// and validate they match before creating G2/G3 layouts.
    ///
    /// # Returns
    /// A typed unary result that resolves to the layout configuration
    pub fn get_layout_config(&self) -> Result<dynamo_nova::am::TypedUnaryResult<LayoutConfig>> {
        let instance = self.remote;

        let awaiter = self
            .nova
            .typed_unary::<LayoutConfig>("kvbm.worker.get_layout_config")?
            .instance(instance)
            .send();

        Ok(awaiter)
    }

    /// Configure additional layouts (G2, G3) on the remote worker.
    ///
    /// This calls the `kvbm.worker.configure_layouts` handler on the remote worker.
    /// The worker will create host/pinned cache (G2) and optionally disk cache (G3)
    /// based on the provided configuration.
    ///
    /// # Arguments
    /// * `config` - Leader-provided configuration specifying block counts and backends
    ///
    /// # Returns
    /// A typed unary result that resolves to the worker's response with updated metadata
    pub fn configure_layouts(
        &self,
        config: LeaderLayoutConfig,
    ) -> Result<dynamo_nova::am::TypedUnaryResult<WorkerLayoutResponse>> {
        let instance = self.remote;

        let awaiter = self
            .nova
            .typed_unary::<WorkerLayoutResponse>("kvbm.worker.configure_layouts")?
            .payload(config)?
            .instance(instance)
            .send();

        Ok(awaiter)
    }
}
