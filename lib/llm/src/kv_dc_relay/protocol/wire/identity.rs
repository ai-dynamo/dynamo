// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use super::super::{
    CkfFormat, DigestIdentity, DynamoEndpointId, IdentitySource as ProtoIdentitySource, KvPoolId,
    ModelRegistration, POOL_IDENTITY_VERSION, ProducerIdentity, RELAY_CONTRACT_MARKER,
    RELAY_PROTOCOL_VERSION, v1::model_target,
};
use super::images::MAX_BUCKET_COUNT;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireIdentityError {
    #[error("unsupported Relay protocol version {0}")]
    ProtocolVersion(u32),
    #[error("invalid Relay contract marker {0:#010x}")]
    ContractMarker(u32),
    #[error("unsupported pool identity version {0}")]
    PoolIdentityVersion(u32),
    #[error("{0} is missing")]
    MissingField(&'static str),
    #[error("{field} digest has {actual} bytes, expected 16")]
    DigestLength { field: &'static str, actual: usize },
    #[error("{field} has invalid identity source {value}")]
    IdentitySource { field: &'static str, value: i32 },
    #[error("CKF format has zero {0}")]
    ZeroFormatField(&'static str),
    #[error("CKF bucket count does not fit this platform")]
    BucketCountOverflow,
    #[error("CKF bucket count {actual} exceeds the supported maximum {maximum}")]
    BucketCountTooLarge { actual: u64, maximum: usize },
    #[error("layout generation must be nonzero")]
    ZeroLayoutGeneration,
    #[error("{0} must not be empty or contain surrounding whitespace")]
    InvalidText(&'static str),
    #[error("model registration repeats alias {0:?}")]
    DuplicateAlias(String),
}

pub fn validate_contract_marker(contract_marker: u32) -> Result<(), WireIdentityError> {
    if contract_marker != RELAY_CONTRACT_MARKER {
        return Err(WireIdentityError::ContractMarker(contract_marker));
    }
    Ok(())
}

pub fn validate_protocol_envelope(
    protocol_version: u32,
    contract_marker: u32,
) -> Result<(), WireIdentityError> {
    validate_contract_marker(contract_marker)?;
    if protocol_version != RELAY_PROTOCOL_VERSION {
        return Err(WireIdentityError::ProtocolVersion(protocol_version));
    }
    Ok(())
}

pub fn validate_pool_id(pool_id: &KvPoolId) -> Result<(), WireIdentityError> {
    if pool_id.identity_version != POOL_IDENTITY_VERSION {
        return Err(WireIdentityError::PoolIdentityVersion(
            pool_id.identity_version,
        ));
    }
    let domain = pool_id
        .indexer_domain
        .as_ref()
        .ok_or(WireIdentityError::MissingField("indexer domain"))?;
    validate_digest("cache semantics", domain.cache_semantics.as_ref())?;
    validate_digest("routing scope", domain.routing_scope.as_ref())
}

pub fn validate_ckf_format(format: &CkfFormat) -> Result<(), WireIdentityError> {
    for (field, value) in [
        ("format version", u64::from(format.format_version)),
        ("bucket count", format.bucket_count),
        ("fingerprint width", u64::from(format.fingerprint_bits)),
        ("slots per bucket", u64::from(format.slots_per_bucket)),
    ] {
        if value == 0 {
            return Err(WireIdentityError::ZeroFormatField(field));
        }
    }
    let bucket_count =
        usize::try_from(format.bucket_count).map_err(|_| WireIdentityError::BucketCountOverflow)?;
    if bucket_count > MAX_BUCKET_COUNT {
        return Err(WireIdentityError::BucketCountTooLarge {
            actual: format.bucket_count,
            maximum: MAX_BUCKET_COUNT,
        });
    }
    Ok(())
}

pub fn validate_producer_identity(identity: &ProducerIdentity) -> Result<(), WireIdentityError> {
    validate_pool_id(
        identity
            .pool_id
            .as_ref()
            .ok_or(WireIdentityError::MissingField("producer pool ID"))?,
    )?;
    if identity.layout_generation == 0 {
        return Err(WireIdentityError::ZeroLayoutGeneration);
    }
    validate_ckf_format(
        identity
            .ckf_format
            .as_ref()
            .ok_or(WireIdentityError::MissingField("producer CKF format"))?,
    )
}

pub fn validate_endpoint_id(endpoint: &DynamoEndpointId) -> Result<(), WireIdentityError> {
    validate_text("endpoint namespace", &endpoint.namespace)?;
    validate_text("endpoint component", &endpoint.component)?;
    validate_text("endpoint name", &endpoint.endpoint)
}

pub fn validate_model_registration(
    registration: &ModelRegistration,
) -> Result<(), WireIdentityError> {
    validate_text("canonical model ID", &registration.canonical_model_id)?;
    let target = registration
        .target
        .as_ref()
        .and_then(|target| target.target.as_ref())
        .ok_or(WireIdentityError::MissingField("model target"))?;
    match target {
        model_target::Target::Base(base) => validate_text("base model ID", &base.base_model)?,
        model_target::Target::Lora(lora) => {
            validate_text("LoRA base model ID", &lora.base_model)?;
            validate_text("LoRA adapter ID", &lora.adapter)?;
        }
    }

    let mut aliases = HashSet::with_capacity(registration.aliases.len());
    for alias in &registration.aliases {
        validate_text("model alias", alias)?;
        if !aliases.insert(alias) {
            return Err(WireIdentityError::DuplicateAlias(alias.clone()));
        }
    }
    Ok(())
}

fn validate_digest(
    field: &'static str,
    identity: Option<&DigestIdentity>,
) -> Result<(), WireIdentityError> {
    let identity = identity.ok_or(WireIdentityError::MissingField(field))?;
    if identity.digest.len() != 16 {
        return Err(WireIdentityError::DigestLength {
            field,
            actual: identity.digest.len(),
        });
    }
    let source = ProtoIdentitySource::try_from(identity.source).map_err(|_| {
        WireIdentityError::IdentitySource {
            field,
            value: identity.source,
        }
    })?;
    if source == ProtoIdentitySource::Unspecified {
        return Err(WireIdentityError::IdentitySource {
            field,
            value: identity.source,
        });
    }
    Ok(())
}

fn validate_text(field: &'static str, value: &str) -> Result<(), WireIdentityError> {
    if value.is_empty() || value.trim() != value {
        return Err(WireIdentityError::InvalidText(field));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use prost::Message as _;

    use super::super::super::{
        BaseModelTarget, DigestIdentity, IdentitySource, IndexerDomainId, ModelTarget,
    };
    use super::*;

    fn pool_id() -> KvPoolId {
        KvPoolId {
            identity_version: POOL_IDENTITY_VERSION,
            indexer_domain: Some(IndexerDomainId {
                cache_semantics: Some(DigestIdentity {
                    digest: Bytes::from_static(&[0x11; 16]),
                    source: IdentitySource::DefaultDerived as i32,
                }),
                routing_scope: Some(DigestIdentity {
                    digest: Bytes::from_static(&[0x22; 16]),
                    source: IdentitySource::Explicit as i32,
                }),
            }),
            dc_id: 0xAABB_CCDD_EEFF_0011,
        }
    }

    #[test]
    fn full_pool_identity_round_trips_without_loss() {
        let expected = pool_id();
        let decoded = KvPoolId::decode(expected.encode_to_vec().as_slice())
            .expect("pool identity must decode");
        validate_pool_id(&decoded).expect("pool identity must validate");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn retired_v1_envelope_fails_the_clean_break_marker() {
        assert_eq!(
            validate_protocol_envelope(RELAY_PROTOCOL_VERSION, 0),
            Err(WireIdentityError::ContractMarker(0))
        );
        assert_eq!(
            validate_protocol_envelope(RELAY_PROTOCOL_VERSION + 1, RELAY_CONTRACT_MARKER),
            Err(WireIdentityError::ProtocolVersion(
                RELAY_PROTOCOL_VERSION + 1
            ))
        );
    }

    #[test]
    fn pool_identity_rejects_truncated_digest_and_unspecified_source() {
        let mut pool = pool_id();
        pool.indexer_domain
            .as_mut()
            .expect("domain")
            .routing_scope
            .as_mut()
            .expect("routing scope")
            .digest = Bytes::from_static(&[0x22; 15]);
        assert!(matches!(
            validate_pool_id(&pool),
            Err(WireIdentityError::DigestLength { .. })
        ));

        let mut pool = pool_id();
        pool.indexer_domain
            .as_mut()
            .expect("domain")
            .cache_semantics
            .as_mut()
            .expect("cache semantics")
            .source = IdentitySource::Unspecified as i32;
        assert!(matches!(
            validate_pool_id(&pool),
            Err(WireIdentityError::IdentitySource { .. })
        ));
    }

    #[test]
    fn registration_rejects_missing_target_and_duplicate_alias() {
        let missing = ModelRegistration {
            canonical_model_id: "llama".into(),
            target: None,
            aliases: Vec::new(),
        };
        assert_eq!(
            validate_model_registration(&missing),
            Err(WireIdentityError::MissingField("model target"))
        );

        let duplicate = ModelRegistration {
            canonical_model_id: "llama".into(),
            target: Some(ModelTarget {
                target: Some(model_target::Target::Base(BaseModelTarget {
                    base_model: "llama".into(),
                })),
            }),
            aliases: vec!["chat".into(), "chat".into()],
        };
        assert_eq!(
            validate_model_registration(&duplicate),
            Err(WireIdentityError::DuplicateAlias("chat".into()))
        );
    }
}
