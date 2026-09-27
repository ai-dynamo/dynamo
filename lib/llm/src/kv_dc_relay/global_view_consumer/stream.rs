// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-producer relay CKF stream decoding and snapshot/delta ingestion.

use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use dynamo_kv_router::global_view::PoolId as RoutingPoolId;
use dynamo_kv_router::global_view::source::{
    PlaneLease, PoolObservation, PoolObservationAssembler, SourcePlane,
};
use dynamo_kv_router::global_view::state::{SignalState, SignalStatus};
use dynamo_kv_router::indexer::cuckoo::{
    CKF_LANE_COUNT, ConsumerInstanceId, GlobalCkfBucketImage, GlobalCkfDelta, GlobalCkfIndexer,
    GlobalCkfIngestOutcome, GlobalCkfLaneIngestor, GlobalCkfManifest, GlobalCkfSnapshot, LaneLease,
    PrefixSearchConfig, ProducerIdentity,
};
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

use super::scorer::{ReadyCkf, RelayCkfOverlapStore};
use crate::global_view::CatalogProjection;
use crate::kv_dc_relay::publication::cbi1::{self, FilterFormat, ImagesFrame, SnapshotAssembly};
use crate::kv_dc_relay::wan::grpc::producer_from_wire;
use crate::kv_dc_relay::wan::grpc::protocol::{
    self as wire, FilterUpdate, FilterUpdateKind, KvPoolDescriptor, KvQueryHashFormat, WorkerRole,
};

struct CkfStreamSession {
    routing_pool_id: RoutingPoolId,
    model: String,
    native_producer: ProducerIdentity,
    block_size: u32,
    is_eagle: bool,
    format: FilterFormat,
    indexer: Arc<GlobalCkfIndexer>,
    ingestor: GlobalCkfLaneIngestor,
    lane_lease: LaneLease,
    snapshot: SnapshotAssembly,
    relay: Option<(u64, u64)>,
    store: Arc<RelayCkfOverlapStore>,
    store_generation: u64,
    assembler: Arc<PoolObservationAssembler>,
    plane_lease: PlaneLease,
}

impl CkfStreamSession {
    fn new(
        routing_pool_id: RoutingPoolId,
        model: String,
        descriptor: &KvPoolDescriptor,
        store: Arc<RelayCkfOverlapStore>,
        assembler: Arc<PoolObservationAssembler>,
    ) -> Result<Self> {
        if assembler.pool_id() != routing_pool_id {
            bail!("CKF session routing pool differs from its observation assembler");
        }
        wire::validate_pool_descriptor(descriptor)?;
        if !descriptor
            .pool_roles
            .contains(&(WorkerRole::Aggregated as i32))
            || !descriptor.registrations.iter().any(|registration| {
                registration.canonical_model_id == model
                    && matches!(
                        registration
                            .target
                            .as_ref()
                            .and_then(|target| target.target.as_ref()),
                        Some(wire::v1::model_target::Target::Base(_))
                    )
            })
        {
            bail!("CKF producer is not an aggregated source for this model");
        }
        let wire_producer = descriptor.producer.as_ref().context("missing producer")?;
        let native_producer = producer_from_wire(wire_producer)?;
        let semantics = descriptor
            .query_semantics
            .as_ref()
            .context("missing query semantics")?;
        let is_eagle = match KvQueryHashFormat::try_from(semantics.hash_format)? {
            KvQueryHashFormat::DynamoStandardV1 => false,
            KvQueryHashFormat::DynamoEagleV1 => true,
            KvQueryHashFormat::Unspecified => bail!("unspecified KV query hash format"),
        };
        let format = FilterFormat::new(
            native_producer.format().seed(),
            native_producer.format().bucket_count(),
        )?;
        let instance = ConsumerInstanceId::new(rand::random());
        let mut lanes = [None; CKF_LANE_COUNT];
        lanes[0] = Some(native_producer.pool_id());
        let manifest = GlobalCkfManifest::new(
            instance,
            native_producer.indexer_domain(),
            native_producer.format(),
            lanes,
        )?;
        let indexer = Arc::new(GlobalCkfIndexer::new(
            manifest,
            PrefixSearchConfig::default(),
        )?);
        let mut ingestor = indexer.claim_lane(0)?;
        let lane_lease = LaneLease::new(instance, 0, 1);
        ingestor.assign(native_producer, lane_lease)?;
        let plane_lease = assembler.open(SourcePlane::KvOverlap)?;
        let store_generation = match store.begin(&routing_pool_id) {
            Ok(generation) => generation,
            Err(error) => {
                assembler.disconnect(plane_lease);
                return Err(error);
            }
        };
        Ok(Self {
            routing_pool_id,
            model,
            native_producer,
            block_size: semantics.kv_block_size,
            is_eagle,
            format,
            indexer,
            ingestor,
            lane_lease,
            snapshot: SnapshotAssembly::new(format),
            relay: None,
            store,
            store_generation,
            assembler,
            plane_lease,
        })
    }

    fn process_update(&mut self, update: &FilterUpdate) -> Result<()> {
        wire::validate_protocol_envelope(update.protocol_version, update.contract_marker)?;
        let frame_producer = update.producer.as_ref().context("missing frame producer")?;
        if producer_from_wire(frame_producer)? != self.native_producer {
            bail!("CKF frame belongs to a different producer generation");
        }
        let relay = update.relay.as_ref().context("missing relay identity")?;
        let relay_key = (relay.drt_instance_id, relay.relay_incarnation);
        if self.relay.is_some_and(|expected| expected != relay_key) {
            bail!("relay incarnation changed within a CKF stream");
        }
        self.relay = Some(relay_key);
        match FilterUpdateKind::try_from(update.kind)? {
            FilterUpdateKind::SnapshotChunk => {
                if update.base_sequence != update.sequence {
                    bail!("snapshot chunk has mismatched base sequence");
                }
                let frame = cbi1::decode(self.format, &update.payload)?;
                let ImagesFrame::SnapshotChunk { header, .. } = &frame else {
                    bail!("snapshot frame contains a non-snapshot CBI1 payload");
                };
                self.check_header(header.dc_id, header.epoch, update.sequence)?;
                if let Some((sequence, images)) = self.snapshot.absorb(&frame)? {
                    let mut buckets = vec![0; self.format.bucket_count];
                    for image in images {
                        buckets[image.bucket as usize] = image.value;
                    }
                    let snapshot = GlobalCkfSnapshot::new(
                        self.native_producer,
                        self.lane_lease,
                        sequence,
                        buckets.into_boxed_slice(),
                    );
                    match self.ingestor.install_snapshot(&snapshot) {
                        GlobalCkfIngestOutcome::SnapshotInstalled { .. } => self.publish()?,
                        outcome => bail!("CKF snapshot rejected: {outcome:?}"),
                    }
                }
            }
            FilterUpdateKind::Delta => {
                let frame = cbi1::decode(self.format, &update.payload)?;
                let ImagesFrame::Delta {
                    header,
                    base_epoch,
                    images,
                } = frame
                else {
                    bail!("delta frame contains a non-delta CBI1 payload");
                };
                self.check_header(header.dc_id, header.epoch, update.sequence)?;
                if base_epoch != update.base_sequence {
                    bail!("delta CBI1 base sequence differs from frame envelope");
                }
                let delta = GlobalCkfDelta::new(
                    self.native_producer,
                    self.lane_lease,
                    update.base_sequence,
                    update.sequence,
                    images
                        .into_iter()
                        .map(|image| GlobalCkfBucketImage::new(image.bucket as usize, image.value))
                        .collect(),
                );
                match self.ingestor.apply_delta(&delta) {
                    GlobalCkfIngestOutcome::DeltaApplied { .. } => self.touch()?,
                    outcome => bail!("CKF delta rejected: {outcome:?}"),
                }
            }
            FilterUpdateKind::Heartbeat => {
                if !update.payload.is_empty()
                    || update.base_sequence != update.sequence
                    || self.ingestor.installed_sequence() != Some(update.sequence)
                {
                    bail!("CKF heartbeat has unexpected payload or sequence");
                }
                self.touch()?;
            }
            FilterUpdateKind::Unspecified => bail!("unspecified CKF frame kind"),
        }
        Ok(())
    }

    fn check_header(&self, dc_id: u64, epoch: u64, sequence: u64) -> Result<()> {
        if dc_id != self.native_producer.dc_id().get() || epoch != sequence {
            bail!("CBI1 header differs from CKF frame envelope");
        }
        Ok(())
    }

    fn publish(&self) -> Result<()> {
        if !self.store.publish(
            &self.routing_pool_id,
            self.store_generation,
            ReadyCkf {
                model: self.model.clone(),
                producer_pool_id: self.native_producer.pool_id(),
                block_size: self.block_size,
                is_eagle: self.is_eagle,
                indexer: Arc::clone(&self.indexer),
                received_at: Instant::now(),
            },
        ) {
            bail!("CKF stream was superseded by a newer producer session");
        }
        self.update_status()
    }

    fn touch(&self) -> Result<()> {
        if !self
            .store
            .touch(&self.routing_pool_id, self.store_generation)
        {
            bail!("CKF stream has no current, ready producer session");
        }
        self.update_status()
    }

    fn update_status(&self) -> Result<()> {
        let received_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        let applied = self.assembler.apply(
            self.plane_lease,
            PoolObservation::KvOverlap {
                status: SignalStatus {
                    state: SignalState::Complete,
                    received_at_unix_ms: Some(received_at_unix_ms),
                    expected_sources: Some(1),
                    observed_sources: Some(1),
                    ..Default::default()
                },
            },
        )?;
        if !applied {
            self.store
                .withdraw(&self.routing_pool_id, self.store_generation);
            bail!("CKF observation stream was superseded");
        }
        Ok(())
    }
}

impl Drop for CkfStreamSession {
    fn drop(&mut self) {
        self.store
            .withdraw(&self.routing_pool_id, self.store_generation);
        self.assembler.disconnect(self.plane_lease);
        self.ingestor.retire();
    }
}

/// Subscribe to a single catalog-selected aggregated producer. The caller
/// cancels this task when the catalog withdraws or replaces that producer and
/// starts a fresh task on reconnect. A closed or invalid stream returns an
/// error; dropping the session removes its overlap score immediately.
pub async fn run_exact_aggregated_producer(
    client: wire::KvEventRelayClient<Channel>,
    routing_pool_id: RoutingPoolId,
    model: String,
    catalog: CatalogProjection,
    subscriber_id: String,
    store: Arc<RelayCkfOverlapStore>,
    assembler: Arc<PoolObservationAssembler>,
    cancel: CancellationToken,
) -> Result<()> {
    let descriptor = catalog
        .sole_aggregated_overlap_producer(&model)
        .context("catalog has no unique aggregated KV producer for this model")?
        .clone();
    let mut session = CkfStreamSession::new(routing_pool_id, model, &descriptor, store, assembler)?;
    let mut stream = client
        .max_decoding_message_size(8 * 1024 * 1024)
        .subscribe_kv_pool(wire::SubscribeKvPoolRequest {
            subscriber_id,
            expected_producer: descriptor.producer,
            contract_marker: wire::RELAY_CONTRACT_MARKER,
        })
        .await?
        .into_inner();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            update = stream.message() => {
                let update = update?.context("CKF relay stream closed")?;
                session.process_update(&update)?;
            }
        }
    }
}

#[cfg(test)]
mod tests;
