// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{SystemTime, UNIX_EPOCH};

use dynamo_kv_router::identity::{
    CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
};
use dynamo_kv_router::indexer::cuckoo::{DcCkfFormatIdentity, ProducerIdentity};
use dynamo_runtime::protocols::EndpointId;

use super::super::identity::{
    CanonicalModelRegistration, DcPoolDescriptor, DcRelayIdentity, KvQueryHashFormat,
    KvQuerySemantics, ModelTarget, WorkerRole,
};
use super::super::protocol as proto;

pub(crate) fn unix_timestamp<const UNITS_PER_SECOND: u128>() -> u64 {
    let units = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .saturating_mul(UNITS_PER_SECOND)
        / 1_000_000_000;
    units.min(u128::from(u64::MAX)) as u64
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum WireConversionError {
    #[error(transparent)]
    InvalidIdentity(#[from] proto::WireIdentityError),
    #[error("wire pool ID has invalid {field} identity source {value}")]
    InvalidIdentitySource { field: &'static str, value: i32 },
}

pub(crate) fn relay_identity_to_wire(identity: DcRelayIdentity) -> proto::RelayIdentity {
    proto::RelayIdentity {
        drt_instance_id: identity.drt_instance_id(),
        relay_incarnation: identity.relay_incarnation(),
    }
}

pub(crate) fn pool_id_to_wire(pool_id: PoolId) -> proto::KvPoolId {
    let domain = pool_id.indexer_domain();
    proto::KvPoolId {
        identity_version: proto::POOL_IDENTITY_VERSION,
        indexer_domain: Some(proto::IndexerDomainId {
            cache_semantics: Some(digest_to_wire(
                domain.cache_semantics().digest(),
                domain.cache_semantics().source(),
            )),
            routing_scope: Some(digest_to_wire(
                domain.routing_scope().digest(),
                domain.routing_scope().source(),
            )),
        }),
        dc_id: pool_id.dc_id().get(),
    }
}

pub(crate) fn pool_id_from_wire(pool_id: &proto::KvPoolId) -> Result<PoolId, WireConversionError> {
    proto::validate_pool_id(pool_id)?;
    let domain = pool_id
        .indexer_domain
        .as_ref()
        .ok_or(proto::WireIdentityError::MissingField("indexer domain"))?;
    let cache = domain
        .cache_semantics
        .as_ref()
        .ok_or(proto::WireIdentityError::MissingField("cache semantics"))?;
    let routing = domain
        .routing_scope
        .as_ref()
        .ok_or(proto::WireIdentityError::MissingField("routing scope"))?;
    Ok(PoolId::new(
        IndexerDomainId::new(
            CacheSemanticsId::new(
                digest_from_wire("cache semantics", &cache.digest)?,
                source_from_wire("cache semantics", cache.source)?,
            ),
            RoutingScopeId::new(
                digest_from_wire("routing scope", &routing.digest)?,
                source_from_wire("routing scope", routing.source)?,
            ),
        ),
        DcId::new(pool_id.dc_id),
    ))
}

pub(crate) fn producer_to_wire(identity: ProducerIdentity) -> proto::ProducerIdentity {
    proto::ProducerIdentity {
        pool_id: Some(pool_id_to_wire(identity.pool_id())),
        producer_incarnation: identity.producer_incarnation(),
        layout_generation: identity.layout_generation(),
        ckf_format: Some(format_to_wire(identity.format())),
    }
}

pub(crate) fn format_to_wire(format: DcCkfFormatIdentity) -> proto::CkfFormat {
    proto::CkfFormat {
        format_version: u32::from(format.format_version()),
        seed: format.seed(),
        bucket_count: format.bucket_count() as u64,
        fingerprint_bits: u32::from(format.fingerprint_bits()),
        slots_per_bucket: u32::from(format.slots_per_bucket()),
    }
}

pub(crate) fn descriptor_to_wire(descriptor: &DcPoolDescriptor) -> proto::KvPoolDescriptor {
    proto::KvPoolDescriptor {
        producer: Some(producer_to_wire(descriptor.producer())),
        serving_endpoint: Some(endpoint_to_wire(descriptor.serving_endpoint())),
        registrations: descriptor
            .registrations()
            .iter()
            .map(registration_to_wire)
            .collect(),
        query_semantics: Some(query_semantics_to_wire(descriptor.query_semantics())),
        pool_roles: descriptor
            .pool_roles()
            .iter()
            .copied()
            .map(worker_role_to_wire)
            .map(|role| role as i32)
            .collect(),
    }
}

pub(crate) const fn worker_role_to_wire(role: WorkerRole) -> proto::WorkerRole {
    match role {
        WorkerRole::Prefill => proto::WorkerRole::Prefill,
        WorkerRole::Decode => proto::WorkerRole::Decode,
        WorkerRole::Encode => proto::WorkerRole::Encode,
        WorkerRole::Aggregated => proto::WorkerRole::Aggregated,
        WorkerRole::Legacy => proto::WorkerRole::Legacy,
    }
}

fn query_semantics_to_wire(semantics: KvQuerySemantics) -> proto::KvQuerySemantics {
    let hash_format = match semantics.hash_format() {
        KvQueryHashFormat::DynamoStandardV1 => proto::KvQueryHashFormat::DynamoStandardV1,
        KvQueryHashFormat::DynamoEagleV1 => proto::KvQueryHashFormat::DynamoEagleV1,
    };
    proto::KvQuerySemantics {
        kv_block_size: semantics.kv_block_size(),
        hash_format: hash_format as i32,
    }
}

pub(crate) fn model_target_to_wire(target: &ModelTarget) -> proto::ModelTarget {
    let target = match target {
        ModelTarget::Base { base_model } => {
            proto::v1::model_target::Target::Base(proto::BaseModelTarget {
                base_model: base_model.as_str().to_string(),
            })
        }
        ModelTarget::Lora {
            base_model,
            adapter,
        } => proto::v1::model_target::Target::Lora(proto::LoraModelTarget {
            base_model: base_model.as_str().to_string(),
            adapter: adapter.as_str().to_string(),
        }),
    };
    proto::ModelTarget {
        target: Some(target),
    }
}

fn registration_to_wire(registration: &CanonicalModelRegistration) -> proto::ModelRegistration {
    proto::ModelRegistration {
        canonical_model_id: registration.model().as_str().to_string(),
        target: Some(model_target_to_wire(registration.target())),
        aliases: registration
            .aliases()
            .iter()
            .map(|alias| alias.as_str().to_string())
            .collect(),
    }
}

pub(crate) fn endpoint_to_wire(endpoint: &EndpointId) -> proto::DynamoEndpointId {
    proto::DynamoEndpointId {
        namespace: endpoint.namespace.clone(),
        component: endpoint.component.clone(),
        endpoint: endpoint.name.clone(),
    }
}

fn digest_to_wire(digest: [u8; 16], source: IdentitySource) -> proto::DigestIdentity {
    proto::DigestIdentity {
        digest: digest.to_vec().into(),
        source: match source {
            IdentitySource::DefaultDerived => proto::IdentitySource::DefaultDerived as i32,
            IdentitySource::Explicit => proto::IdentitySource::Explicit as i32,
        },
    }
}

fn digest_from_wire(field: &'static str, digest: &[u8]) -> Result<[u8; 16], WireConversionError> {
    digest.try_into().map_err(|_| {
        proto::WireIdentityError::DigestLength {
            field,
            actual: digest.len(),
        }
        .into()
    })
}

fn source_from_wire(
    field: &'static str,
    value: i32,
) -> Result<IdentitySource, WireConversionError> {
    match proto::IdentitySource::try_from(value) {
        Ok(proto::IdentitySource::DefaultDerived) => Ok(IdentitySource::DefaultDerived),
        Ok(proto::IdentitySource::Explicit) => Ok(IdentitySource::Explicit),
        _ => Err(WireConversionError::InvalidIdentitySource { field, value }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_id(cache_source: IdentitySource, routing_source: IdentitySource) -> PoolId {
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([0x11; 16], cache_source),
                RoutingScopeId::new([0x22; 16], routing_source),
            ),
            DcId::new(0xAABB_CCDD_EEFF_0011),
        )
    }

    #[test]
    fn dynamo_pool_id_round_trips_through_wire_without_loss() {
        for cache_source in [IdentitySource::DefaultDerived, IdentitySource::Explicit] {
            for routing_source in [IdentitySource::DefaultDerived, IdentitySource::Explicit] {
                let expected = pool_id(cache_source, routing_source);
                let wire = pool_id_to_wire(expected);
                assert_eq!(pool_id_from_wire(&wire).unwrap(), expected);
            }
        }
    }
}
