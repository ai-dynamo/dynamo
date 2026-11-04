// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-independent identities for pools in a Global View.
//!
//! A routing pool is one DGD in the initial deployment. Its identity is stable
//! across process and relay restarts.

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An opaque routing identity. Callers must not parse its representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PoolId(String);

impl PoolId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PoolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable deployment coordinates that a [`PoolIdDeriver`] turns into a [`PoolId`].
///
/// `cluster_id` is the cluster ID the pool's PoolRelay runs with
/// (`--cluster-id`, `DYN_CLUSTER_ID`). Where MCS also reports the pool, that
/// ID must be the cluster's name in the ClusterSet, so a registration and the
/// pool MCS reports have one key. `namespace` is the Kubernetes namespace
/// containing the DGD, not the Dynamo runtime namespace derived by the
/// operator. In JSON it is `{"cluster_id", "namespace", "dgd_name"}`, and
/// reading it checks each field as [`PoolKey::new`] does.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "PoolKeyFields", into = "PoolKeyFields")]
pub struct PoolKey {
    cluster_id: String,
    namespace: String,
    dgd_name: String,
}

#[derive(Serialize, Deserialize)]
struct PoolKeyFields {
    cluster_id: String,
    namespace: String,
    dgd_name: String,
}

impl TryFrom<PoolKeyFields> for PoolKey {
    type Error = PoolKeyError;

    fn try_from(fields: PoolKeyFields) -> Result<Self, Self::Error> {
        Self::new(fields.cluster_id, fields.namespace, fields.dgd_name)
    }
}

impl From<PoolKey> for PoolKeyFields {
    fn from(key: PoolKey) -> Self {
        Self {
            cluster_id: key.cluster_id,
            namespace: key.namespace,
            dgd_name: key.dgd_name,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PoolKeyError {
    #[error("{field} must be non-empty without surrounding whitespace or control characters")]
    InvalidField { field: &'static str },
}

/// The pool's name for people, `<cluster_id>/<namespace>/<dgd_name>`, as logs
/// and metrics show it. A pool's identity is its [`PoolId`].
impl fmt::Display for PoolKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{}",
            self.cluster_id, self.namespace, self.dgd_name
        )
    }
}

impl PoolKey {
    pub fn new(
        cluster_id: impl Into<String>,
        namespace: impl Into<String>,
        dgd_name: impl Into<String>,
    ) -> Result<Self, PoolKeyError> {
        let cluster_id = cluster_id.into();
        let namespace = namespace.into();
        let dgd_name = dgd_name.into();
        validate(&cluster_id, "cluster_id")?;
        validate(&namespace, "namespace")?;
        validate(&dgd_name, "dgd_name")?;
        Ok(Self {
            cluster_id,
            namespace,
            dgd_name,
        })
    }

    pub fn cluster_id(&self) -> &str {
        &self.cluster_id
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn dgd_name(&self) -> &str {
        &self.dgd_name
    }
}

pub(crate) fn validate(value: &str, field: &'static str) -> Result<(), PoolKeyError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(PoolKeyError::InvalidField { field });
    }
    Ok(())
}

/// Global View owns assignment of routing pool IDs. Implementations can
/// change without leaking derivation details into catalog or routing code.
pub trait PoolIdDeriver: Send + Sync {
    fn derive(&self, key: &PoolKey) -> PoolId;
}

/// Initial deterministic derivation, shared by every Global Router replica.
///
/// Changing this algorithm changes pool IDs and requires a deployment
/// migration. The API continues treating IDs as opaque strings.
#[derive(Clone, Copy, Debug, Default)]
pub struct V1PoolIdDeriver;

impl PoolIdDeriver for V1PoolIdDeriver {
    fn derive(&self, key: &PoolKey) -> PoolId {
        let mut hasher = blake3::Hasher::new_derive_key("dynamo.global_view.pool_id.v1");
        for part in [&key.cluster_id, &key.namespace, &key.dgd_name] {
            let bytes = part.as_bytes();
            hasher.update(&(bytes.len() as u64).to_be_bytes());
            hasher.update(bytes);
        }
        PoolId(format!("pool-v1-{}", hasher.finalize().to_hex()))
    }
}

/// The cluster a pool runs in is [`PoolKey::cluster_id`], not a field here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolLocation {
    pub region: String,
    pub availability_zone: Option<String>,
    pub datacenter: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(cluster: &str, namespace: &str, dgd: &str) -> PoolKey {
        PoolKey::new(cluster, namespace, dgd).unwrap()
    }

    #[test]
    fn identity_survives_deriver_restart() {
        let pool = key("ohio", "dynamo", "mocker-1");
        // Pinned: changing the derivation changes every deployed pool's ID.
        assert_eq!(
            V1PoolIdDeriver.derive(&pool).as_str(),
            "pool-v1-2d83e849329ee9362567e384ac43e4d13db27a2b6a00cb79a8be9024504c56ae"
        );
    }

    #[test]
    fn moving_or_renaming_a_dgd_changes_identity() {
        let deriver = V1PoolIdDeriver;
        let original = deriver.derive(&key("ohio", "dynamo", "mocker-1"));
        for changed in [
            key("west", "dynamo", "mocker-1"),
            key("ohio", "other", "mocker-1"),
            key("ohio", "dynamo", "mocker-2"),
        ] {
            assert_ne!(original, deriver.derive(&changed));
        }
    }

    #[test]
    fn component_boundaries_cannot_alias() {
        let deriver = V1PoolIdDeriver;
        assert_ne!(
            deriver.derive(&key("ab", "c", "d")),
            deriver.derive(&key("a", "bc", "d")),
        );
    }

    #[test]
    fn rejects_invalid_coordinates() {
        assert_eq!(
            PoolKey::new("", "dynamo", "mocker-1"),
            Err(PoolKeyError::InvalidField {
                field: "cluster_id"
            })
        );
        assert_eq!(
            PoolKey::new("ohio", " dynamo", "mocker-1"),
            Err(PoolKeyError::InvalidField { field: "namespace" })
        );
        assert_eq!(
            PoolKey::new("ohio", "dynamo", "mocker\n1"),
            Err(PoolKeyError::InvalidField { field: "dgd_name" })
        );
    }

    #[test]
    fn api_representation_is_an_opaque_string() {
        let id = V1PoolIdDeriver.derive(&key("ohio", "dynamo", "mocker-1"));
        assert_eq!(serde_json::to_string(&id).unwrap(), format!("\"{id}\""));
    }

    #[test]
    fn a_key_displays_as_its_name_for_people() {
        assert_eq!(
            key("ohio", "dynamo", "mocker-1").to_string(),
            "ohio/dynamo/mocker-1"
        );
    }

    #[test]
    fn key_json_is_checked_when_read() {
        let pool = key("ohio", "dynamo", "mocker-1");
        let json = serde_json::json!({"cluster_id": "ohio", "namespace": "dynamo", "dgd_name": "mocker-1"});
        assert_eq!(serde_json::to_value(&pool).unwrap(), json);
        assert_eq!(serde_json::from_value::<PoolKey>(json).unwrap(), pool);
        let invalid =
            serde_json::json!({"cluster_id": " ", "namespace": "dynamo", "dgd_name": "mocker-1"});
        assert!(serde_json::from_value::<PoolKey>(invalid).is_err());
    }
}
