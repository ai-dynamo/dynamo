// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A pool's item: its registration as JSON with that JSON's hash, and the
//! lease that decides whether the pool is live.
//!
//! A pool is live while `renewed_at + lease_s + skew >= now`, on the reader's
//! clock against the writer's `renewed_at`. DynamoDB's TTL on `expires_at`
//! only removes items that nobody has renewed for a week.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, anyhow, bail};
use aws_sdk_dynamodb::types::AttributeValue;
use dynamo_global_common::PoolRegistration;

use super::keys;
use crate::backends::clock::millis;

/// How long after its last registration an item stays in the table, as
/// [`keys::EXPIRES_AT`].
pub const GC_AFTER: Duration = Duration::from_secs(7 * 86_400);

/// Whether a pool registered at `renewed_at_ms` with a lease of `lease_s`
/// seconds is live at `now_ms`, allowing `skew` between the writer's clock
/// and the reader's.
pub fn is_live(renewed_at_ms: u64, lease_s: u64, skew: Duration, now_ms: u64) -> bool {
    renewed_at_ms
        .saturating_add(lease_s.saturating_mul(1_000))
        .saturating_add(millis(skew))
        >= now_ms
}

/// blake3 hex of a registration's JSON.
pub fn hash(registration_json: &str) -> String {
    blake3::hash(registration_json.as_bytes())
        .to_hex()
        .to_string()
}

/// A registration ready to write. Its lists must already be sorted, so that
/// the same pool always has the same JSON and hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Encoded {
    pub sort_key: String,
    pub registration: String,
    pub hash: String,
}

pub(super) fn encode(registration: &PoolRegistration) -> Result<Encoded, serde_json::Error> {
    let json = serde_json::to_string(registration)?;
    Ok(Encoded {
        sort_key: keys::sort_key(&registration.key),
        hash: hash(&json),
        registration: json,
    })
}

/// A registration's `UpdateItem`: it sets every attribute but the key, and
/// [`keys::CREATED_AT`] only when absent. The placeholders avoid DynamoDB's
/// reserved words, `hash` among them.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Update {
    pub expression: &'static str,
    pub names: HashMap<String, String>,
    pub values: HashMap<String, AttributeValue>,
}

const UPDATE: &str = "SET #registration = :registration, #hash = :hash, #renewed = :renewed, \
                      #lease = :lease, #expires = :expires, \
                      #created = if_not_exists(#created, :renewed)";

pub(super) fn update(encoded: &Encoded, renewed_at_ms: u64, lease_s: u64) -> Update {
    let names = [
        ("#registration", keys::REGISTRATION),
        ("#hash", keys::HASH),
        ("#renewed", keys::RENEWED_AT),
        ("#lease", keys::LEASE_S),
        ("#expires", keys::EXPIRES_AT),
        ("#created", keys::CREATED_AT),
    ];
    let values = [
        (
            ":registration",
            AttributeValue::S(encoded.registration.clone()),
        ),
        (":hash", AttributeValue::S(encoded.hash.clone())),
        (":renewed", number(renewed_at_ms)),
        (":lease", number(lease_s)),
        (":expires", number(expires_at(renewed_at_ms))),
    ];
    Update {
        expression: UPDATE,
        names: names
            .into_iter()
            .map(|(placeholder, name)| (placeholder.to_owned(), name.to_owned()))
            .collect(),
        values: values
            .into_iter()
            .map(|(placeholder, value)| (placeholder.to_owned(), value))
            .collect(),
    }
}

/// [`keys::EXPIRES_AT`] for a registration at `renewed_at_ms`.
pub(super) fn expires_at(renewed_at_ms: u64) -> u64 {
    (renewed_at_ms / 1_000).saturating_add(GC_AFTER.as_secs())
}

pub(super) fn number(value: u64) -> AttributeValue {
    AttributeValue::N(value.to_string())
}

/// A pool's item as read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Stored {
    pub registration: PoolRegistration,
    pub hash: String,
    pub renewed_at_ms: u64,
    pub lease_s: u64,
}

impl Stored {
    pub fn is_live(&self, skew: Duration, now_ms: u64) -> bool {
        is_live(self.renewed_at_ms, self.lease_s, skew, now_ms)
    }
}

/// Reads a pool's item. Attributes it does not know are ignored.
pub(super) fn decode(item: &HashMap<String, AttributeValue>) -> anyhow::Result<Stored> {
    let sort_key = string(item, keys::SK)?;
    let key = keys::parse_sort_key(sort_key)?;
    let registration: PoolRegistration = serde_json::from_str(string(item, keys::REGISTRATION)?)
        .with_context(|| {
            format!(
                "{sort_key}: attribute {} is not a pool registration",
                keys::REGISTRATION
            )
        })?;
    if registration.key != key {
        bail!(
            "{sort_key}: the registration is for another pool, {:?}",
            registration.key
        );
    }
    Ok(Stored {
        registration,
        hash: string(item, keys::HASH)?.to_owned(),
        renewed_at_ms: unsigned(item, keys::RENEWED_AT)?,
        lease_s: unsigned(item, keys::LEASE_S)?,
    })
}

fn attribute<'a>(
    item: &'a HashMap<String, AttributeValue>,
    name: &str,
) -> anyhow::Result<&'a AttributeValue> {
    item.get(name)
        .ok_or_else(|| anyhow!("attribute {name} is missing"))
}

fn string<'a>(item: &'a HashMap<String, AttributeValue>, name: &str) -> anyhow::Result<&'a str> {
    match attribute(item, name)? {
        AttributeValue::S(value) => Ok(value),
        _ => bail!("attribute {name} is not a string"),
    }
}

fn unsigned(item: &HashMap<String, AttributeValue>, name: &str) -> anyhow::Result<u64> {
    match attribute(item, name)? {
        AttributeValue::N(value) => value
            .parse()
            .with_context(|| format!("attribute {name} is {value}, not a whole number")),
        _ => bail!("attribute {name} is not a number"),
    }
}

#[cfg(test)]
mod tests {
    use dynamo_global_common::{Capacity, PoolKey, PoolLocation};
    use url::Url;

    use super::*;
    use crate::backends::clock::{Clock, ManualClock};

    fn registration() -> PoolRegistration {
        PoolRegistration {
            key: PoolKey::new("aws#west", "dyn\\amo", "mocker-west-1").unwrap(),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: Some("us-west-2a".to_owned()),
                datacenter: None,
            },
            model: "Qwen/Qwen3-0.6B".to_owned(),
            capabilities: vec!["decode".to_owned(), "prefill".to_owned()],
            frontends: vec![Url::parse("http://10.43.1.2:8000/").unwrap()],
            capacity: vec![Capacity {
                sku: "H100".to_owned(),
                workers: 8,
            }],
        }
    }

    /// The item a registration at `renewed_at_ms` leaves where there was
    /// none: each `#name = :name` of the update, and `created_at` from
    /// `:renewed`.
    fn item(
        encoded: &Encoded,
        renewed_at_ms: u64,
        lease_s: u64,
    ) -> HashMap<String, AttributeValue> {
        let update = update(encoded, renewed_at_ms, lease_s);
        let mut item: HashMap<String, AttributeValue> = update
            .names
            .iter()
            .filter_map(|(placeholder, name)| {
                let value = update.values.get(&placeholder.replacen('#', ":", 1))?;
                Some((name.clone(), value.clone()))
            })
            .collect();
        item.insert(
            keys::CREATED_AT.to_owned(),
            update.values[":renewed"].clone(),
        );
        item.insert(
            keys::PK.to_owned(),
            AttributeValue::S(keys::POOLS.to_owned()),
        );
        item.insert(
            keys::SK.to_owned(),
            AttributeValue::S(encoded.sort_key.clone()),
        );
        item
    }

    #[test]
    fn the_update_uses_every_placeholder_it_defines() {
        // DynamoDB rejects an update with a placeholder it does not use.
        let update = update(&encode(&registration()).unwrap(), 1_000, 15);
        let placeholders: Vec<&String> = update.names.keys().chain(update.values.keys()).collect();
        assert_eq!(placeholders.len(), 11);
        for placeholder in placeholders {
            let used = update
                .expression
                .split(|c: char| !(c.is_alphanumeric() || c == '#' || c == ':'))
                .any(|token| token == placeholder);
            assert!(used, "{placeholder} in {}", update.expression);
        }
        assert_eq!(update.names["#created"], keys::CREATED_AT);
        assert!(
            update
                .expression
                .contains("#created = if_not_exists(#created, :renewed)")
        );
    }

    #[test]
    fn a_registration_round_trips() {
        let encoded = encode(&registration()).unwrap();
        assert_eq!(encoded.sort_key, "aws\\#west#dyn\\\\amo#mocker-west-1");
        assert_eq!(encoded.hash, hash(&encoded.registration));
        let stored = decode(&item(&encoded, 1_767_225_600_123, 15)).unwrap();
        assert_eq!(
            stored,
            Stored {
                registration: registration(),
                hash: encoded.hash,
                renewed_at_ms: 1_767_225_600_123,
                lease_s: 15,
            }
        );
    }

    #[test]
    fn the_hash_follows_the_content() {
        let first = encode(&registration()).unwrap();
        assert_eq!(encode(&registration()).unwrap(), first);
        let mut changed = registration();
        changed.model = "Qwen/Qwen3-8B".to_owned();
        assert_ne!(encode(&changed).unwrap().hash, first.hash);
    }

    #[test]
    fn the_item_expires_a_week_after_its_last_registration() {
        let encoded = encode(&registration()).unwrap();
        let item = item(&encoded, 1_000_999, 15);
        assert_eq!(item[keys::EXPIRES_AT], number(1_000 + 604_800));
    }

    #[test]
    fn a_pool_is_live_until_its_lease_and_the_skew_have_passed() {
        let clock = ManualClock::new(10_000);
        let stored = decode(&item(&encode(&registration()).unwrap(), clock.now_ms(), 15)).unwrap();
        let skew = Duration::from_secs(2);

        clock.advance(Duration::from_secs(15));
        assert!(stored.is_live(Duration::ZERO, clock.now_ms()));
        clock.advance(Duration::from_millis(1));
        assert!(!stored.is_live(Duration::ZERO, clock.now_ms()));
        // The skew keeps it live for two more seconds, to the millisecond.
        assert!(stored.is_live(skew, clock.now_ms()));
        clock.advance(Duration::from_millis(1_999));
        assert!(stored.is_live(skew, clock.now_ms()));
        clock.advance(Duration::from_millis(1));
        assert!(!stored.is_live(skew, clock.now_ms()));

        // A reader whose clock is behind the writer's sees it live.
        clock.set(9_000);
        assert!(stored.is_live(Duration::ZERO, clock.now_ms()));
        // No overflow at the ends.
        assert!(is_live(u64::MAX, u64::MAX, Duration::MAX, u64::MAX));
    }

    #[test]
    fn an_item_with_unknown_attributes_decodes() {
        let encoded = encode(&registration()).unwrap();
        let mut item = item(&encoded, 1_000, 15);
        item.insert("added_later".to_owned(), AttributeValue::Bool(true));
        item.remove(keys::CREATED_AT);
        item.remove(keys::EXPIRES_AT);
        assert_eq!(decode(&item).unwrap().registration, registration());
    }

    #[test]
    fn a_damaged_item_does_not_decode() {
        let encoded = encode(&registration()).unwrap();
        let good = item(&encoded, 1_000, 15);
        let mut other = registration();
        other.key = PoolKey::new("aws-east", "dynamo", "mocker").unwrap();
        let damaged: [(&str, Option<AttributeValue>); 9] = [
            (keys::SK, None),
            (keys::SK, Some(AttributeValue::S("a#b".to_owned()))),
            (keys::REGISTRATION, None),
            (keys::REGISTRATION, Some(AttributeValue::S("{".to_owned()))),
            (
                keys::REGISTRATION,
                Some(AttributeValue::S(serde_json::to_string(&other).unwrap())),
            ),
            (keys::HASH, Some(number(1))),
            (keys::RENEWED_AT, None),
            (keys::RENEWED_AT, Some(AttributeValue::N("-1".to_owned()))),
            (keys::LEASE_S, Some(AttributeValue::S("15".to_owned()))),
        ];
        for (name, value) in damaged {
            let mut item = good.clone();
            match value.clone() {
                Some(value) => item.insert(name.to_owned(), value),
                None => item.remove(name),
            };
            assert!(decode(&item).is_err(), "{name} = {value:?}");
        }
    }
}
