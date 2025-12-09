// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct NovaWorkerClient {
    nova: Arc<Nova>,
    remote: InstanceId,
    g1_handle: Arc<OnceLock<LayoutHandle>>,
    g2_handle: Arc<OnceLock<LayoutHandle>>,
    g3_handle: Arc<OnceLock<LayoutHandle>>,
    /// Track which remote instances we've connected to for has_remote_metadata()
    connected_instances: Arc<RwLock<HashSet<InstanceId>>>,
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
        let awaiter = self.nova.events().awaiter(event.handle())?;

        // Convert to serializable options (uses From<TransferOptions> impl)
        let options: SerializableTransferOptions = options.into();

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
        let remote_instance = self.remote;

        // Use unary (not am_sync) to wait for transfer completion
        self.nova.tracker().spawn_on(
            async move {
                let result = nova
                    .unary("kvbm.worker.local_transfer")?
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

        let options: SerializableTransferOptions = options.into();

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
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.nova.events().new_event()?;
        let awaiter = self.nova.events().awaiter(event.handle())?;

        let options: SerializableTransferOptions = options.into();

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
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        // Serialize metadata to bytes (SerializedLayout uses bincode internally)
        let serialized_metadata: Vec<Vec<u8>> =
            metadata.iter().map(|m| m.as_bytes().to_vec()).collect();

        let message = ConnectRemoteMessage {
            instance_id,
            metadata: serialized_metadata,
        };
        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Create event for completion tracking
        let event = self.nova.events().new_event()?;
        let awaiter = self.nova.events().awaiter(event.handle())?;

        let nova = self.nova.clone();
        let remote_instance = self.remote;
        let connected = self.connected_instances.clone();
        let target_instance = instance_id;

        self.nova.tracker().spawn_on(
            async move {
                let result = nova
                    .unary("kvbm.worker.connect_remote")?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => {
                        // Track that we've connected to this instance
                        connected.write().insert(target_instance);
                        event.trigger()
                    }
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.nova.runtime(),
        );

        Ok(ConnectRemoteResponse::from_awaiter(awaiter))
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        // Check if we've successfully connected to this instance
        self.connected_instances.read().contains(&instance_id)
    }

    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let message = ExecuteRemoteOnboardForInstanceMessage {
            instance_id,
            remote_logical_type,
            src_block_ids,
            dst,
            dst_block_ids: dst_block_ids.to_vec(),
            options: SerializableTransferOptions::from(options),
        };
        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Create event for completion tracking
        let event = self.nova.events().new_event()?;
        let awaiter = self.nova.events().awaiter(event.handle())?;

        let nova = self.nova.clone();
        let remote_instance = self.remote;

        self.nova.tracker().spawn_on(
            async move {
                let result = nova
                    .unary("kvbm.worker.remote_onboard_for_instance")?
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
            connected_instances: Arc::new(RwLock::new(HashSet::new())),
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
