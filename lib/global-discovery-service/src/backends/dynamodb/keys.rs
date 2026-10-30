// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The table's keys and attribute names. One item per pool:
//!
//! | `pk` | `sk` |
//! |---|---|
//! | `pool` | `<cluster_id>#<namespace>#<dgd_name>` |
//!
//! Each part of the sort key escapes `#` as `\#` and `\` as `\\`, so a part
//! never holds an unescaped `#`, and every pool has one sort key. Parts that
//! use only the characters Kubernetes allows in names sort by cluster, then
//! namespace, then name, and the pools of one cluster share the prefix
//! `<cluster_id>#`.

use dynamo_global_common::{PoolKey, PoolKeyError};
use thiserror::Error;

/// The partition key attribute. Every pool is in partition [`POOLS`].
pub const PK: &str = "pk";
/// The sort key attribute: [`sort_key`].
pub const SK: &str = "sk";
/// The `PoolRegistration` as JSON, its lists sorted.
pub const REGISTRATION: &str = "registration";
/// blake3 hex of [`REGISTRATION`], to tell a change from a renewal.
pub const HASH: &str = "hash";
/// Unix milliseconds, on the writer's clock, of the last registration.
pub const RENEWED_AT: &str = "renewed_at";
/// The lease, in seconds, that the writer granted.
pub const LEASE_S: &str = "lease_s";
/// Unix milliseconds of the first registration since the item was absent.
pub const CREATED_AT: &str = "created_at";
/// The DynamoDB TTL attribute, in Unix seconds. Garbage collection only:
/// leases are decided from [`RENEWED_AT`] and [`LEASE_S`].
pub const EXPIRES_AT: &str = "expires_at";

/// The partition every pool is in, so that one `Query` lists them.
pub const POOLS: &str = "pool";

const SEPARATOR: char = '#';
const ESCAPE: char = '\\';

/// A sort key that [`sort_key`] did not write.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SortKeyError {
    #[error("sort key {0:?} is not <cluster_id>#<namespace>#<dgd_name> with `#` and `\\` escaped")]
    Shape(String),
    #[error("sort key {sort_key:?}: {error}")]
    Key {
        sort_key: String,
        error: PoolKeyError,
    },
}

/// The sort key of `key`'s item.
pub fn sort_key(key: &PoolKey) -> String {
    let parts = [key.cluster_id(), key.namespace(), key.dgd_name()];
    let mut sort_key = String::with_capacity(parts.iter().map(|part| part.len() + 1).sum());
    for (at, part) in parts.into_iter().enumerate() {
        if at > 0 {
            sort_key.push(SEPARATOR);
        }
        for c in part.chars() {
            if c == SEPARATOR || c == ESCAPE {
                sort_key.push(ESCAPE);
            }
            sort_key.push(c);
        }
    }
    sort_key
}

/// The pool whose item has `sort_key`. Accepts only what [`sort_key`] writes.
pub fn parse_sort_key(sort_key: &str) -> Result<PoolKey, SortKeyError> {
    let shape = || SortKeyError::Shape(sort_key.to_owned());
    let mut parts = vec![String::new()];
    let mut chars = sort_key.chars();
    while let Some(c) = chars.next() {
        match c {
            SEPARATOR => parts.push(String::new()),
            ESCAPE => match chars.next() {
                Some(escaped @ (SEPARATOR | ESCAPE)) => push(&mut parts, escaped),
                _ => return Err(shape()),
            },
            c => push(&mut parts, c),
        }
    }
    let [cluster_id, namespace, dgd_name] = <[String; 3]>::try_from(parts).map_err(|_| shape())?;
    PoolKey::new(cluster_id, namespace, dgd_name).map_err(|error| SortKeyError::Key {
        sort_key: sort_key.to_owned(),
        error,
    })
}

fn push(parts: &mut [String], c: char) {
    if let Some(part) = parts.last_mut() {
        part.push(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(cluster: &str, namespace: &str, dgd: &str) -> PoolKey {
        PoolKey::new(cluster, namespace, dgd).unwrap()
    }

    #[test]
    fn a_key_without_separators_is_written_as_it_is() {
        let pool = key("aws-west", "dynamo", "mocker-west-1");
        assert_eq!(sort_key(&pool), "aws-west#dynamo#mocker-west-1");
        assert_eq!(parse_sort_key("aws-west#dynamo#mocker-west-1"), Ok(pool));
    }

    #[test]
    fn separators_and_escapes_round_trip_in_every_part() {
        let awkward = [
            "#", "\\", "a#b", "a\\b", "#a", "a#", "\\#", "#\\", "##", "\\\\", "a\\#b",
        ];
        for part in awkward {
            for pool in [
                key(part, "dynamo", "mocker"),
                key("west", part, "mocker"),
                key("west", "dynamo", part),
                key(part, part, part),
            ] {
                let written = sort_key(&pool);
                assert_eq!(written.matches(SEPARATOR).count() - escaped(&written), 2);
                assert_eq!(parse_sort_key(&written).as_ref(), Ok(&pool), "{written}");
            }
        }
        assert_eq!(sort_key(&key("a#", "b", "c")), "a\\##b#c");
        assert_eq!(sort_key(&key("a\\", "b", "c")), "a\\\\#b#c");
    }

    /// The `#`s in `sort_key` that are escaped.
    fn escaped(sort_key: &str) -> usize {
        let mut count = 0;
        let mut chars = sort_key.chars();
        while let Some(c) = chars.next() {
            if c == ESCAPE && chars.next() == Some(SEPARATOR) {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn parts_that_share_text_do_not_collide() {
        // Escaping keeps the parts apart where plain joining would not.
        let keys = [
            key("a#b", "c", "d"),
            key("a", "b#c", "d"),
            key("a#", "b", "c"),
            key("a", "#b", "c"),
            key("a\\", "b", "c"),
            key("a", "\\b", "c"),
        ];
        let written: Vec<String> = keys.iter().map(sort_key).collect();
        for (at, sort_key) in written.iter().enumerate() {
            assert!(!written[at + 1..].contains(sort_key), "{sort_key}");
        }
        // A cluster's pools share a prefix no other cluster's pools have.
        assert!(sort_key(&key("c1", "ns", "a")).starts_with("c1#"));
        for other in [
            key("c10", "ns", "a"),
            key("c1#x", "ns", "a"),
            key("c1\\", "ns", "a"),
        ] {
            assert!(!sort_key(&other).starts_with("c1#"), "{other:?}");
        }
    }

    #[test]
    fn kubernetes_names_sort_by_cluster_then_namespace_then_name() {
        let ordered = [
            key("aws-east", "dynamo", "mocker"),
            key("aws-west", "a", "z"),
            key("aws-west", "a-b", "a"),
            key("aws-west", "a.b", "a"),
            key("aws-west", "b", "a"),
            key("aws-west", "b", "a-1"),
            key("aws-west", "b", "a1"),
            key("aws-west-2", "a", "a"),
        ];
        let written: Vec<String> = ordered.iter().map(sort_key).collect();
        let mut sorted = written.clone();
        sorted.sort();
        assert_eq!(written, sorted);
    }

    #[test]
    fn only_what_sort_key_writes_parses() {
        for bad in [
            "", "a#b", "a#b#c#d", "a\\#b#c", "a#b#c\\", "a\\x#b#c", "#b#c", "a# #c",
        ] {
            assert!(parse_sort_key(bad).is_err(), "{bad:?}");
        }
    }
}
