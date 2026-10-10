// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical discovery instance identity shared by a failover cohort.
//!
//! A GMS primary and its warm shadow publish one worker identity: the shadow
//! registers under the same instance id once it owns the failover lock, so
//! routers see the address of a single instance change instead of one worker
//! leaving and another joining.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::{Context, Result, bail};

use crate::config::environment_names::discovery::{
    DYN_DISCOVERY_LOGICAL_INSTANCE_ID, DYN_DISCOVERY_LOGICAL_INSTANCE_KEY,
};

/// Same width as Kubernetes-derived instance ids (fits a JavaScript-safe integer).
const LOGICAL_INSTANCE_ID_MASK: u64 = 0x001F_FFFF_FFFF_FFFF;

/// The configured logical instance id, if any.
pub fn logical_instance_id_from_env() -> Result<Option<u64>> {
    logical_instance_id(
        std::env::var(DYN_DISCOVERY_LOGICAL_INSTANCE_ID)
            .ok()
            .as_deref(),
        std::env::var(DYN_DISCOVERY_LOGICAL_INSTANCE_KEY)
            .ok()
            .as_deref(),
        |name| std::env::var(name).ok(),
    )
}

fn logical_instance_id(
    explicit: Option<&str>,
    key: Option<&str>,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<Option<u64>> {
    if let Some(raw) = explicit.map(str::trim).filter(|raw| !raw.is_empty()) {
        let id = match raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
            Some(hex) => u64::from_str_radix(hex, 16),
            None => raw.parse::<u64>(),
        }
        .with_context(|| format!("invalid {DYN_DISCOVERY_LOGICAL_INSTANCE_ID}={raw}"))?;
        if id == 0 {
            bail!("{DYN_DISCOVERY_LOGICAL_INSTANCE_ID} must be non-zero");
        }
        return Ok(Some(id));
    }
    let Some(key) = key.map(str::trim).filter(|key| !key.is_empty()) else {
        return Ok(None);
    };
    Ok(Some(hash_logical_instance_key(&expand(key, lookup)?)))
}

/// Hash a logical key into an instance id; never zero.
pub fn hash_logical_instance_key(key: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    (hasher.finish() & LOGICAL_INSTANCE_ID_MASK).max(1)
}

/// Expand `$(VAR)` placeholders; an unset variable is an error, so cohort
/// members never silently collapse onto a different id.
fn expand(key: &str, lookup: impl Fn(&str) -> Option<String>) -> Result<String> {
    let mut out = String::with_capacity(key.len());
    let mut rest = key;
    while let Some(start) = rest.find("$(") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after
            .find(')')
            .with_context(|| format!("unterminated $( in {DYN_DISCOVERY_LOGICAL_INSTANCE_KEY}"))?;
        let name = &after[..end];
        let value = lookup(name).with_context(|| {
            format!("{DYN_DISCOVERY_LOGICAL_INSTANCE_KEY} references unset ${name}")
        })?;
        out.push_str(&value);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(name: &str) -> Option<String> {
        match name {
            "POD_NAME" => Some("leader-0".to_string()),
            _ => None,
        }
    }

    #[test]
    fn explicit_id_wins_and_accepts_hex() {
        assert_eq!(
            logical_instance_id(Some("42"), Some("k"), env).unwrap(),
            Some(42)
        );
        assert_eq!(
            logical_instance_id(Some("0x2a"), None, env).unwrap(),
            Some(42)
        );
        assert!(logical_instance_id(Some("0"), None, env).is_err());
        assert!(logical_instance_id(Some("nope"), None, env).is_err());
    }

    #[test]
    fn cohort_members_with_one_key_share_one_id() {
        let primary = logical_instance_id(None, Some("ns/$(POD_NAME)/engine"), env).unwrap();
        let shadow = logical_instance_id(None, Some("ns/$(POD_NAME)/engine"), env).unwrap();
        let other = logical_instance_id(None, Some("ns/other/engine"), env).unwrap();
        assert!(primary.is_some());
        assert_eq!(primary, shadow);
        assert_ne!(primary, other);
        assert!(primary.unwrap() <= LOGICAL_INSTANCE_ID_MASK);
    }

    #[test]
    fn unset_placeholder_and_empty_key_are_handled() {
        assert!(logical_instance_id(None, Some("ns/$(MISSING)"), env).is_err());
        assert_eq!(logical_instance_id(None, Some("  "), env).unwrap(), None);
        assert_eq!(logical_instance_id(None, None, env).unwrap(), None);
    }
}
