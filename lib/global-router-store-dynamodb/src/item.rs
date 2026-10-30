// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use aws_sdk_dynamodb::types::AttributeValue;
use dynamo_global_router::affinity::{AffinityBinding, AffinityError, AffinityKey};

type Item = HashMap<String, AttributeValue>;

/// Full BLAKE3 digest over length-prefixed UTF-8 fields. The derivation context
/// and key prefixes are part of the v1 storage schema and must stay stable.
fn digest(parts: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new_derive_key("dynamo.global_router.affinity.v1");
    for part in parts {
        hasher.update(&(part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

pub(crate) fn key(key: &AffinityKey) -> Item {
    HashMap::from([
        (
            "itemHashKey".into(),
            AttributeValue::S(format!(
                "AFFINITY#v1#{}",
                digest(&[&key.scope, &key.session])
            )),
        ),
        (
            "sortKey".into(),
            AttributeValue::S(format!(
                "MODEL#{}#POLICY#{}",
                digest(&[&key.model]),
                digest(&[&key.policy])
            )),
        ),
    ])
}

pub(crate) fn encode(affinity_key: &AffinityKey, binding: &AffinityBinding) -> Item {
    let mut item = key(affinity_key);
    item.extend([
        ("schemaVersion".into(), AttributeValue::N("1".into())),
        (
            "poolId".into(),
            AttributeValue::S(binding.pool.as_str().to_owned()),
        ),
        (
            "revision".into(),
            AttributeValue::S(binding.revision.clone()),
        ),
        (
            "updatedAt".into(),
            AttributeValue::N(binding.updated_at.to_string()),
        ),
        (
            "expiresAt".into(),
            AttributeValue::N(binding.expires_at.to_string()),
        ),
    ]);
    item
}

pub(crate) fn decode(item: Item) -> Result<AffinityBinding, AffinityError> {
    let invalid = |field: &str| AffinityError::InvalidRecord(format!("missing or invalid {field}"));
    let number = |field: &str| -> Result<u64, AffinityError> {
        item.get(field)
            .and_then(|value| value.as_n().ok())
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| invalid(field))
    };
    let string = |field: &str| -> Result<&str, AffinityError> {
        item.get(field)
            .and_then(|value| value.as_s().ok())
            .map(String::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid(field))
    };
    if number("schemaVersion")? != 1 {
        return Err(AffinityError::InvalidRecord(
            "unsupported schemaVersion".into(),
        ));
    }
    // PoolId is opaque and already supports serde; do not parse its derivation.
    let pool = serde_json::from_value(serde_json::Value::String(string("poolId")?.into()))
        .map_err(|_| invalid("poolId"))?;
    let updated_at = number("updatedAt")?;
    let expires_at = number("expiresAt")?;
    if expires_at <= updated_at {
        return Err(invalid("expiresAt"));
    }
    Ok(AffinityBinding {
        pool,
        revision: string("revision")?.into(),
        updated_at,
        expires_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> AffinityKey {
        AffinityKey {
            scope: "production".into(),
            policy: "headroom-v1".into(),
            model: "qwen".into(),
            session: "session-1".into(),
        }
    }

    #[test]
    fn key_encoding_is_unambiguous_and_scoped() {
        assert_ne!(digest(&["a", "bc"]), digest(&["ab", "c"]));
        let original = example();
        for field in 0..4 {
            let mut other = original.clone();
            match field {
                0 => other.scope.push('x'),
                1 => other.policy.push('x'),
                2 => other.model.push('x'),
                _ => other.session.push('x'),
            }
            assert_ne!(key(&original), key(&other));
        }
        assert!(
            key(&original)
                .values()
                .all(|value| !value.as_s().unwrap().contains("session-1"))
        );
    }

    #[test]
    fn v1_keys_are_stable_across_releases() {
        let item = key(&example());
        assert_eq!(
            item["itemHashKey"].as_s().unwrap(),
            "AFFINITY#v1#71b176389263fe02ad5b65e74f3ee300eaa22daf847f327a3705f006eb23354e"
        );
        assert_eq!(
            item["sortKey"].as_s().unwrap(),
            "MODEL#c5f9ca555abbf21ae912bae80a199e743bf697e5316b5a50bbb2657b1ceb57c6#POLICY#6b8f9859d6125559cb584e0e160fd2f77af057186bf94660470e509c424cc69f"
        );
    }

    #[test]
    fn refuses_corrupt_or_unknown_records() {
        assert!(decode(HashMap::new()).is_err());
        let mut item = key(&example());
        item.insert("schemaVersion".into(), AttributeValue::N("2".into()));
        assert!(decode(item).is_err());
    }
}
