// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use super::super::{
    CkfFormat, DigestIdentity, DynamoEndpointId, IdentitySource as ProtoIdentitySource,
    KvPoolDescriptor, KvPoolId, KvQueryHashFormat, KvQuerySemantics, ModelRegistration,
    POOL_IDENTITY_VERSION, ProducerIdentity, RELAY_CONTRACT_MARKER, RELAY_PROTOCOL_VERSION,
    ServingReadinessState, TopologyEntry, WorkerRole, v1::model_target,
};
use super::images::{FINGERPRINT_BITS, FORMAT_VERSION, MAX_BUCKET_COUNT, SLOTS_PER_BUCKET};

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
    #[error(
        "unsupported CBI1 CKF format: version={format_version}, fingerprint_bits={fingerprint_bits}, slots_per_bucket={slots_per_bucket}"
    )]
    UnsupportedCkfFormat {
        format_version: u32,
        fingerprint_bits: u32,
        slots_per_bucket: u32,
    },
    #[error("CKF bucket count {actual} is not a power of two in 2..={maximum}")]
    UnsupportedBucketCount { actual: u64, maximum: usize },
    #[error("layout generation must be nonzero")]
    ZeroLayoutGeneration,
    #[error("KV query block size must be nonzero")]
    ZeroQueryBlockSize,
    #[error("unsupported KV query hash format {0}")]
    QueryHashFormat(i32),
    #[error("{0} must not be empty or contain surrounding whitespace")]
    InvalidText(&'static str),
    #[error("model registration repeats alias {0:?}")]
    DuplicateAlias(String),
    #[error("worker-role set must not be empty")]
    MissingWorkerRoles,
    #[error("unsupported worker role {0}")]
    WorkerRole(i32),
    #[error("worker-role set repeats {0:?}")]
    DuplicateWorkerRole(WorkerRole),
    #[error("duplicate endpoint role {0:?} is not supported")]
    UnsupportedDuplicateEndpointRole(WorkerRole),
    #[error("serving topology has unsupported readiness state {0}")]
    ReadinessState(i32),
    #[error("serving topology must contain at least one member")]
    MissingTopologyMembers,
    #[error("serving topology repeats endpoint {0:?}")]
    DuplicateTopologyMember(String),
    #[error("topology namespace {topology:?} does not match member namespace {member:?}")]
    TopologyNamespaceMismatch { topology: String, member: String },
    #[error("serving topology repeats adapter {0:?}")]
    DuplicateAdapter(String),
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
    if format.format_version != u32::from(FORMAT_VERSION)
        || format.fingerprint_bits != u32::from(FINGERPRINT_BITS)
        || format.slots_per_bucket != u32::from(SLOTS_PER_BUCKET)
    {
        return Err(WireIdentityError::UnsupportedCkfFormat {
            format_version: format.format_version,
            fingerprint_bits: format.fingerprint_bits,
            slots_per_bucket: format.slots_per_bucket,
        });
    }
    let bucket_count =
        usize::try_from(format.bucket_count).map_err(|_| WireIdentityError::BucketCountOverflow)?;
    if bucket_count > MAX_BUCKET_COUNT {
        return Err(WireIdentityError::BucketCountTooLarge {
            actual: format.bucket_count,
            maximum: MAX_BUCKET_COUNT,
        });
    }
    if !bucket_count.is_power_of_two() || bucket_count < 2 {
        return Err(WireIdentityError::UnsupportedBucketCount {
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

pub fn validate_query_semantics(semantics: &KvQuerySemantics) -> Result<(), WireIdentityError> {
    if semantics.kv_block_size == 0 {
        return Err(WireIdentityError::ZeroQueryBlockSize);
    }
    let hash_format = KvQueryHashFormat::try_from(semantics.hash_format)
        .map_err(|_| WireIdentityError::QueryHashFormat(semantics.hash_format))?;
    if hash_format == KvQueryHashFormat::Unspecified {
        return Err(WireIdentityError::QueryHashFormat(semantics.hash_format));
    }
    Ok(())
}

pub fn validate_pool_descriptor(descriptor: &KvPoolDescriptor) -> Result<(), WireIdentityError> {
    validate_producer_identity(
        descriptor
            .producer
            .as_ref()
            .ok_or(WireIdentityError::MissingField("pool producer"))?,
    )?;
    validate_endpoint_id(
        descriptor
            .serving_endpoint
            .as_ref()
            .ok_or(WireIdentityError::MissingField("serving endpoint"))?,
    )?;
    validate_query_semantics(
        descriptor
            .query_semantics
            .as_ref()
            .ok_or(WireIdentityError::MissingField("KV query semantics"))?,
    )?;
    descriptor
        .registrations
        .iter()
        .try_for_each(validate_model_registration)?;
    validate_worker_roles(&descriptor.pool_roles)
}

pub fn validate_worker_roles(roles: &[i32]) -> Result<(), WireIdentityError> {
    if roles.is_empty() {
        return Err(WireIdentityError::MissingWorkerRoles);
    }
    validate_role_set(roles)
}

pub fn validate_topology_entry(entry: &TopologyEntry) -> Result<(), WireIdentityError> {
    validate_text("topology namespace", &entry.namespace)?;
    validate_text("topology canonical model ID", &entry.canonical_model_id)?;
    validate_readiness_state(entry.state)?;
    validate_role_set(&entry.present_roles)?;
    validate_role_set(&entry.missing_roles)?;
    validate_duplicate_endpoint_roles(&entry.duplicate_role_endpoints)?;
    if entry.members.is_empty() {
        return Err(WireIdentityError::MissingTopologyMembers);
    }
    let mut endpoints = HashSet::with_capacity(entry.members.len());
    for member in &entry.members {
        let endpoint = member
            .endpoint
            .as_ref()
            .ok_or(WireIdentityError::MissingField("topology member endpoint"))?;
        validate_endpoint_id(endpoint)?;
        if endpoint.namespace != entry.namespace {
            return Err(WireIdentityError::TopologyNamespaceMismatch {
                topology: entry.namespace.clone(),
                member: endpoint.namespace.clone(),
            });
        }
        validate_worker_roles(&member.roles)?;
        if let Some(pool_id) = member.pool_id.as_ref() {
            validate_pool_id(pool_id)?;
        }
        let endpoint_key = format!(
            "{}.{}.{}",
            endpoint.namespace, endpoint.component, endpoint.endpoint
        );
        if !endpoints.insert(endpoint_key.clone()) {
            return Err(WireIdentityError::DuplicateTopologyMember(endpoint_key));
        }
    }
    let mut adapters = HashSet::with_capacity(entry.adapters.len());
    for adapter in &entry.adapters {
        validate_text("adapter canonical model ID", &adapter.canonical_model_id)?;
        validate_readiness_state(adapter.state)?;
        validate_role_set(&adapter.missing_roles)?;
        if !adapters.insert(&adapter.canonical_model_id) {
            return Err(WireIdentityError::DuplicateAdapter(
                adapter.canonical_model_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_readiness_state(value: i32) -> Result<(), WireIdentityError> {
    ServingReadinessState::try_from(value)
        .map(|_| ())
        .map_err(|_| WireIdentityError::ReadinessState(value))
}

fn validate_duplicate_endpoint_roles(roles: &[i32]) -> Result<(), WireIdentityError> {
    validate_role_set(roles)?;
    for &value in roles {
        let role = WorkerRole::try_from(value).map_err(|_| WireIdentityError::WorkerRole(value))?;
        if !matches!(role, WorkerRole::Prefill | WorkerRole::Decode) {
            return Err(WireIdentityError::UnsupportedDuplicateEndpointRole(role));
        }
    }
    Ok(())
}

fn validate_role_set(roles: &[i32]) -> Result<(), WireIdentityError> {
    let mut unique = HashSet::with_capacity(roles.len());
    for &value in roles {
        let role = WorkerRole::try_from(value).map_err(|_| WireIdentityError::WorkerRole(value))?;
        if role == WorkerRole::Unspecified {
            return Err(WireIdentityError::WorkerRole(value));
        }
        if !unique.insert(role) {
            return Err(WireIdentityError::DuplicateWorkerRole(role));
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
        AdapterReadiness, BaseModelTarget, DigestIdentity, IdentitySource, IndexerDomainId,
        ModelTarget, TopologyMember,
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
    fn v1_envelope_is_accepted_and_mismatches_are_rejected() {
        assert_eq!(
            validate_protocol_envelope(RELAY_PROTOCOL_VERSION, RELAY_CONTRACT_MARKER),
            Ok(())
        );
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
    fn ckf_format_accepts_only_the_executable_cbi1_shape() {
        let valid = CkfFormat {
            format_version: u32::from(FORMAT_VERSION),
            seed: 11,
            bucket_count: 64,
            fingerprint_bits: u32::from(FINGERPRINT_BITS),
            slots_per_bucket: u32::from(SLOTS_PER_BUCKET),
        };
        validate_ckf_format(&valid).expect("CBI1 format must validate");

        for unsupported in [
            CkfFormat {
                format_version: u32::from(FORMAT_VERSION) + 1,
                ..valid
            },
            CkfFormat {
                fingerprint_bits: u32::from(FINGERPRINT_BITS) + 1,
                ..valid
            },
            CkfFormat {
                slots_per_bucket: u32::from(SLOTS_PER_BUCKET) + 1,
                ..valid
            },
        ] {
            assert!(matches!(
                validate_ckf_format(&unsupported),
                Err(WireIdentityError::UnsupportedCkfFormat { .. })
            ));
        }

        assert_eq!(
            validate_ckf_format(&CkfFormat {
                bucket_count: 3,
                ..valid
            }),
            Err(WireIdentityError::UnsupportedBucketCount {
                actual: 3,
                maximum: MAX_BUCKET_COUNT,
            })
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

    #[test]
    fn query_semantics_fail_closed_for_missing_zero_and_unknown_values() {
        let valid = KvQuerySemantics {
            kv_block_size: 64,
            hash_format: KvQueryHashFormat::DynamoStandardV1 as i32,
        };
        validate_query_semantics(&valid).unwrap();

        assert_eq!(
            validate_query_semantics(&KvQuerySemantics {
                kv_block_size: 0,
                ..valid
            }),
            Err(WireIdentityError::ZeroQueryBlockSize)
        );
        for hash_format in [KvQueryHashFormat::Unspecified as i32, 99] {
            assert_eq!(
                validate_query_semantics(&KvQuerySemantics {
                    hash_format,
                    ..valid
                }),
                Err(WireIdentityError::QueryHashFormat(hash_format))
            );
        }

        let descriptor = KvPoolDescriptor {
            producer: None,
            serving_endpoint: None,
            registrations: Vec::new(),
            query_semantics: None,
            pool_roles: Vec::new(),
        };
        assert_eq!(
            validate_pool_descriptor(&descriptor),
            Err(WireIdentityError::MissingField("pool producer"))
        );

        let descriptor = KvPoolDescriptor {
            producer: Some(ProducerIdentity {
                pool_id: Some(pool_id()),
                producer_incarnation: 7,
                layout_generation: 1,
                ckf_format: Some(CkfFormat {
                    format_version: 1,
                    seed: 11,
                    bucket_count: 64,
                    fingerprint_bits: 16,
                    slots_per_bucket: 4,
                }),
            }),
            serving_endpoint: Some(DynamoEndpointId {
                namespace: "prod".into(),
                component: "backend".into(),
                endpoint: "generate".into(),
            }),
            registrations: Vec::new(),
            query_semantics: None,
            pool_roles: vec![WorkerRole::Legacy as i32],
        };
        assert_eq!(
            validate_pool_descriptor(&descriptor),
            Err(WireIdentityError::MissingField("KV query semantics"))
        );

        for roles in [
            Vec::new(),
            vec![WorkerRole::Unspecified as i32],
            vec![99],
            vec![WorkerRole::Decode as i32, WorkerRole::Decode as i32],
        ] {
            assert!(validate_worker_roles(&roles).is_err());
        }
        validate_worker_roles(&[WorkerRole::Prefill as i32, WorkerRole::Decode as i32]).unwrap();
    }

    #[test]
    fn topology_validation_is_fail_closed_for_members_roles_and_namespaces() {
        let valid = TopologyEntry {
            namespace: "prod".into(),
            canonical_model_id: "llama".into(),
            state: ServingReadinessState::Ready as i32,
            present_roles: vec![WorkerRole::Decode as i32],
            missing_roles: Vec::new(),
            members: vec![TopologyMember {
                endpoint: Some(DynamoEndpointId {
                    namespace: "prod".into(),
                    component: "backend".into(),
                    endpoint: "generate".into(),
                }),
                roles: vec![WorkerRole::Decode as i32],
                pool_id: Some(pool_id()),
            }],
            duplicate_role_endpoints: vec![WorkerRole::Prefill as i32],
            legacy_fallback_active: false,
            adapters: vec![AdapterReadiness {
                canonical_model_id: "tenant-a".into(),
                state: ServingReadinessState::Ready as i32,
                missing_roles: Vec::new(),
            }],
        };
        validate_topology_entry(&valid).unwrap();

        let mut missing_members = valid.clone();
        missing_members.members.clear();
        assert_eq!(
            validate_topology_entry(&missing_members),
            Err(WireIdentityError::MissingTopologyMembers)
        );

        let mut unspecified_role = valid.clone();
        unspecified_role.members[0].roles = vec![WorkerRole::Unspecified as i32];
        assert_eq!(
            validate_topology_entry(&unspecified_role),
            Err(WireIdentityError::WorkerRole(
                WorkerRole::Unspecified as i32
            ))
        );

        let mut wrong_namespace = valid.clone();
        wrong_namespace.members[0]
            .endpoint
            .as_mut()
            .unwrap()
            .namespace = "other".into();
        assert!(matches!(
            validate_topology_entry(&wrong_namespace),
            Err(WireIdentityError::TopologyNamespaceMismatch { .. })
        ));

        for role in [
            WorkerRole::Legacy,
            WorkerRole::Aggregated,
            WorkerRole::Encode,
        ] {
            let mut unsupported_duplicate_role = valid.clone();
            unsupported_duplicate_role.duplicate_role_endpoints = vec![role as i32];
            assert_eq!(
                validate_topology_entry(&unsupported_duplicate_role),
                Err(WireIdentityError::UnsupportedDuplicateEndpointRole(role))
            );
        }

        for unsupported_role in [WorkerRole::Unspecified as i32, 99] {
            let mut unsupported_duplicate_role = valid.clone();
            unsupported_duplicate_role.duplicate_role_endpoints = vec![unsupported_role];
            assert_eq!(
                validate_topology_entry(&unsupported_duplicate_role),
                Err(WireIdentityError::WorkerRole(unsupported_role))
            );
        }

        let mut repeated_duplicate_role = valid.clone();
        repeated_duplicate_role.duplicate_role_endpoints =
            vec![WorkerRole::Decode as i32, WorkerRole::Decode as i32];
        assert_eq!(
            validate_topology_entry(&repeated_duplicate_role),
            Err(WireIdentityError::DuplicateWorkerRole(WorkerRole::Decode))
        );

        let mut duplicate_adapter = valid;
        duplicate_adapter
            .adapters
            .push(duplicate_adapter.adapters[0].clone());
        assert_eq!(
            validate_topology_entry(&duplicate_adapter),
            Err(WireIdentityError::DuplicateAdapter("tenant-a".into()))
        );
    }
}
