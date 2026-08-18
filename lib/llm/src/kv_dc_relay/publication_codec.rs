// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::Bytes;
use dynamo_kv_router::indexer::cuckoo::{DcCkfDelta, DcCkfFormatIdentity, ProducerIdentity};

use super::protocol::wire::images;
use super::publication_hub::HubSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublicationFrameKind {
    SnapshotChunk,
    Delta,
    Heartbeat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PublicationFrame {
    pub(crate) identity: ProducerIdentity,
    pub(crate) base_sequence: u64,
    pub(crate) sequence: u64,
    pub(crate) kind: PublicationFrameKind,
    pub(crate) payload: Bytes,
}

impl PublicationFrame {
    pub(crate) fn queued_bytes(&self) -> usize {
        // Account for the encoded payload plus a conservative protobuf/gRPC envelope budget.
        self.payload.len().saturating_add(256)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum Cbi1AdapterError {
    #[error(transparent)]
    Format(#[from] images::FormatError),
    #[error(transparent)]
    Wire(#[from] images::ImagesWireError),
    #[error(
        "unsupported CKF format identity: version={format_version}, fingerprint_bits={fingerprint_bits}, slots_per_bucket={slots_per_bucket}"
    )]
    UnsupportedFormatIdentity {
        format_version: u16,
        fingerprint_bits: u8,
        slots_per_bucket: u8,
    },
    #[error("snapshot has {actual} buckets, format declares {expected}")]
    SnapshotBucketCount { expected: usize, actual: usize },
    #[error("delta bucket index {0} exceeds the CBI1 u32 address space")]
    BucketIndexOverflow(usize),
}

pub(crate) struct Cbi1SnapshotFrames {
    snapshot: HubSnapshot,
    format: images::FilterFormat,
    next_chunk: usize,
    chunk_count: usize,
}

impl Iterator for Cbi1SnapshotFrames {
    type Item = PublicationFrame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_chunk == self.chunk_count {
            return None;
        }
        let chunk_index = self.next_chunk;
        self.next_chunk += 1;
        let start = chunk_index * images::SNAPSHOT_CHUNK_BUCKETS;
        let end = (start + images::SNAPSHOT_CHUNK_BUCKETS).min(self.snapshot.buckets().len());
        let identity = self.snapshot.identity();
        let sequence = self.snapshot.sequence();
        Some(PublicationFrame {
            identity,
            base_sequence: sequence,
            sequence,
            kind: PublicationFrameKind::SnapshotChunk,
            payload: images::encode_snapshot_chunk(
                self.format,
                identity.dc_id().get(),
                sequence,
                chunk_index,
                self.chunk_count as u32,
                &self.snapshot.buckets()[start..end],
            )
            .into(),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.chunk_count - self.next_chunk;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Cbi1SnapshotFrames {}

pub(crate) fn encode_snapshot(
    snapshot: HubSnapshot,
) -> Result<Cbi1SnapshotFrames, Cbi1AdapterError> {
    let format = filter_format(snapshot.identity().format())?;
    if snapshot.buckets().len() != format.bucket_count {
        return Err(Cbi1AdapterError::SnapshotBucketCount {
            expected: format.bucket_count,
            actual: snapshot.buckets().len(),
        });
    }
    Ok(Cbi1SnapshotFrames {
        chunk_count: snapshot
            .buckets()
            .len()
            .div_ceil(images::SNAPSHOT_CHUNK_BUCKETS),
        snapshot,
        format,
        next_chunk: 0,
    })
}

pub(crate) fn encode_delta(delta: &DcCkfDelta) -> Result<PublicationFrame, Cbi1AdapterError> {
    let identity = delta.identity();
    let format = filter_format(identity.format())?;
    let mut bucket_images = Vec::with_capacity(delta.images().len());
    for image in delta.images() {
        bucket_images.push(images::BucketImage {
            bucket: u32::try_from(image.bucket())
                .map_err(|_| Cbi1AdapterError::BucketIndexOverflow(image.bucket()))?,
            value: image.value(),
        });
    }
    Ok(PublicationFrame {
        identity,
        base_sequence: delta.base_sequence(),
        sequence: delta.sequence(),
        kind: PublicationFrameKind::Delta,
        payload: images::encode_delta(
            format,
            identity.dc_id().get(),
            delta.base_sequence(),
            delta.sequence(),
            &bucket_images,
        )?
        .into(),
    })
}

pub(crate) fn encode_heartbeat(identity: ProducerIdentity, sequence: u64) -> PublicationFrame {
    PublicationFrame {
        identity,
        base_sequence: sequence,
        sequence,
        kind: PublicationFrameKind::Heartbeat,
        payload: Bytes::new(),
    }
}

fn filter_format(format: DcCkfFormatIdentity) -> Result<images::FilterFormat, Cbi1AdapterError> {
    if format.format_version() != images::FORMAT_VERSION
        || format.fingerprint_bits() != images::FINGERPRINT_BITS
        || format.slots_per_bucket() != images::SLOTS_PER_BUCKET
    {
        return Err(Cbi1AdapterError::UnsupportedFormatIdentity {
            format_version: format.format_version(),
            fingerprint_bits: format.fingerprint_bits(),
            slots_per_bucket: format.slots_per_bucket(),
        });
    }
    Ok(images::FilterFormat::new(
        format.seed(),
        format.bucket_count(),
    )?)
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::LocalBlockHash;
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{
        CKF_LANE_COUNT, CkfConfig, ConsumerInstanceId, DcCkfState, GlobalCkfBucketImage,
        GlobalCkfIndexer, GlobalCkfIngestOutcome, GlobalCkfManifest, GlobalCkfSnapshot, LaneLease,
        PrefixSearchConfig, ProducerIdentity,
    };
    use dynamo_kv_router::protocols::{
        BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
        KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, compute_block_hash_for_seq,
        compute_seq_hash_for_block,
    };

    use super::*;
    use crate::kv_dc_relay::identity::{
        CanonicalModelId, CanonicalModelRegistration, KvQueryHashFormat, KvQuerySemantics,
        ModelAlias, ModelTarget,
    };

    fn pool_id(seed: u8) -> PoolId {
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([seed; 16], IdentitySource::Explicit),
                RoutingScopeId::new([seed.wrapping_add(1); 16], IdentitySource::Explicit),
            ),
            DcId::new(7),
        )
    }

    fn stored_event(
        event_id: u64,
        parent_sequence_hash: Option<u64>,
        local_hashes: &[LocalBlockHash],
        sequence_hashes: &[u64],
    ) -> RouterEvent {
        const EXTERNAL_MASK: u64 = 0xC0DE_CAFE_D15C_0A11;
        RouterEvent::new(
            1,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_sequence_hash
                        .map(|hash| ExternalSequenceBlockHash(hash ^ EXTERNAL_MASK)),
                    start_position: None,
                    blocks: local_hashes
                        .iter()
                        .zip(sequence_hashes)
                        .map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(*sequence_hash ^ EXTERNAL_MASK),
                            tokens_hash: *local_hash,
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    fn assert_full_hit(
        consumer: &GlobalCkfIndexer,
        expected_pool: PoolId,
        local_hashes: &[LocalBlockHash],
    ) {
        let result = consumer.find_prefix_matches(local_hashes).unwrap();
        let lane = result.lanes()[0].expect("configured pool lane must be queryable");
        assert_eq!(lane.pool_id(), expected_pool);
        assert_eq!(lane.prefix_depth(), local_hashes.len() as u32);
    }

    fn resolve_registration<'a>(
        registrations: &'a [CanonicalModelRegistration],
        requested_model: &str,
    ) -> &'a CanonicalModelRegistration {
        registrations
            .iter()
            .find(|registration| {
                registration.model().as_str() == requested_model
                    || registration
                        .aliases()
                        .iter()
                        .any(|alias| alias.as_str() == requested_model)
            })
            .expect("request must resolve through the advertised registration")
    }

    fn assert_cbi1_query_round_trip(hash_format: KvQueryHashFormat, via_lora_alias: bool) {
        let semantics = KvQuerySemantics::new(4, hash_format).unwrap();
        let base_model = CanonicalModelId::new("llama").unwrap();
        let registration = if via_lora_alias {
            let adapter = CanonicalModelId::new("tenant-a").unwrap();
            CanonicalModelRegistration::with_target(
                adapter.clone(),
                ModelTarget::Lora {
                    base_model,
                    adapter,
                },
                vec![ModelAlias::new("tenant-chat").unwrap()],
            )
        } else {
            CanonicalModelRegistration::new(base_model, Vec::new())
        };
        let requested_model = if via_lora_alias {
            "tenant-chat"
        } else {
            "llama"
        };
        let registrations = [registration];
        let resolved = resolve_registration(&registrations, requested_model);
        let canonical_lora = resolved.target().adapter().map(CanonicalModelId::as_str);
        let hash_options = BlockHashOptions {
            lora_name: canonical_lora,
            cache_namespace: Some("tenant-ns"),
            is_eagle: Some(semantics.hash_format().is_eagle()),
            ..Default::default()
        };
        let initial_tokens = (0..13).collect::<Vec<u32>>();
        let expanded_tokens = (0..21).collect::<Vec<u32>>();
        let initial_local =
            compute_block_hash_for_seq(&initial_tokens, semantics.kv_block_size(), hash_options);
        let expanded_local =
            compute_block_hash_for_seq(&expanded_tokens, semantics.kv_block_size(), hash_options);
        assert!(expanded_local.starts_with(&initial_local));
        assert!(expanded_local.len() > initial_local.len());
        let initial_sequence = compute_seq_hash_for_block(&initial_local);
        let expanded_sequence = compute_seq_hash_for_block(&expanded_local);

        let mut config = CkfConfig::new(64);
        config.publish_every_n_events = 1;
        let mut producer = DcCkfState::new(config).unwrap();
        let initial =
            producer.apply_event(stored_event(1, None, &initial_local, &initial_sequence));
        assert!(initial.first_error().is_none());
        assert!(initial.publication().is_some());
        let (_, buckets) = producer.barrier_snapshot().unwrap();

        let pool_id =
            pool_id(hash_format.identity_version() as u8 + if via_lora_alias { 10 } else { 0 });
        let identity = ProducerIdentity::new(pool_id, 11, 1, producer.format());
        let consumer_instance = ConsumerInstanceId::new(13);
        let lease = LaneLease::new(consumer_instance, 0, 1);
        let hub_snapshot = HubSnapshot::from_actor(identity, lease, 1, &buckets);
        let wire_format =
            images::FilterFormat::new(identity.format().seed(), identity.format().bucket_count())
                .unwrap();
        let mut assembly = images::SnapshotAssembly::new(wire_format);
        let mut assembled = None;
        for frame in encode_snapshot(hub_snapshot).unwrap() {
            assert_eq!(frame.identity, identity);
            assert_eq!(frame.kind, PublicationFrameKind::SnapshotChunk);
            let decoded = images::decode(wire_format, &frame.payload).unwrap();
            if let Some(snapshot) = assembly.absorb(&decoded).unwrap() {
                assembled = Some(snapshot);
            }
        }
        let (snapshot_sequence, snapshot_images) =
            assembled.expect("all CBI1 snapshot chunks must assemble");
        let mut decoded_buckets = vec![0; wire_format.bucket_count];
        for image in snapshot_images {
            decoded_buckets[image.bucket as usize] = image.value;
        }

        let mut lanes = [None; CKF_LANE_COUNT];
        lanes[0] = Some(pool_id);
        let manifest = GlobalCkfManifest::new(
            consumer_instance,
            pool_id.indexer_domain(),
            producer.format(),
            lanes,
        )
        .unwrap();
        let consumer = GlobalCkfIndexer::new(manifest, PrefixSearchConfig::default()).unwrap();
        let mut ingestor = consumer.claim_lane(0).unwrap();
        ingestor.assign(identity, lease).unwrap();
        assert_eq!(
            ingestor.install_snapshot(&GlobalCkfSnapshot::new(
                identity,
                lease,
                snapshot_sequence,
                decoded_buckets.into_boxed_slice(),
            )),
            GlobalCkfIngestOutcome::SnapshotInstalled { sequence: 1 }
        );
        assert_full_hit(&consumer, pool_id, &initial_local);

        let delta_outcome = producer.apply_event(stored_event(
            2,
            initial_sequence.last().copied(),
            &expanded_local[initial_local.len()..],
            &expanded_sequence[initial_sequence.len()..],
        ));
        assert!(delta_outcome.first_error().is_none());
        let batch = delta_outcome
            .into_publication()
            .expect("publication threshold emits the delta");
        let producer_delta = DcCkfDelta::new(identity, lease, 1, 2, batch.images().to_vec());
        let delta_frame = encode_delta(&producer_delta).unwrap();
        assert_eq!(delta_frame.kind, PublicationFrameKind::Delta);
        let decoded = images::decode(wire_format, &delta_frame.payload).unwrap();
        let images::ImagesFrame::Delta {
            header,
            base_epoch,
            images,
        } = decoded
        else {
            panic!("encoded Relay delta decoded as a snapshot chunk");
        };
        assert_eq!(header.dc_id, identity.dc_id().get());
        let image_count = images.len();
        let consumer_delta = DcCkfDelta::new(
            identity,
            lease,
            base_epoch,
            header.epoch,
            images
                .into_iter()
                .map(|image| GlobalCkfBucketImage::new(image.bucket as usize, image.value))
                .collect(),
        );
        assert_eq!(
            ingestor.apply_delta(&consumer_delta),
            GlobalCkfIngestOutcome::DeltaApplied {
                sequence: 2,
                images: image_count,
            }
        );
        assert_full_hit(&consumer, pool_id, &expanded_local);
    }

    #[test]
    fn cbi1_snapshot_and_delta_preserve_standard_eagle_and_lora_query_spaces() {
        for hash_format in [
            KvQueryHashFormat::DynamoStandardV1,
            KvQueryHashFormat::DynamoEagleV1,
        ] {
            assert_cbi1_query_round_trip(hash_format, false);
            assert_cbi1_query_round_trip(hash_format, true);
        }
    }
}
