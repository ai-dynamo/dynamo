// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The JSON of the Global Discovery Service's HTTP API. A reader ignores
//! fields it does not know, so the service can add a field before its clients
//! use it.

use std::net::IpAddr;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;
use url::Url;

use crate::pool::{self, PoolKey, PoolLocation};
use crate::replicas::RouterReplica;

/// `POST /pools` body. A PoolRelay sends it on a timer, and each call is also
/// the pool's heartbeat.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolRegistration {
    pub key: PoolKey,
    pub location: PoolLocation,
    /// The model or checkpoint the pool serves. Reading it checks it as
    /// [`PoolKey::new`] checks the key's fields.
    #[serde(deserialize_with = "checked_model")]
    pub model: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Local frontends the Global Router sends requests to.
    #[serde(default)]
    pub frontends: Vec<Url>,
    #[serde(default)]
    pub capacity: Vec<Capacity>,
}

fn checked_model<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let model = String::deserialize(deserializer)?;
    pool::validate(&model, "model").map_err(serde::de::Error::custom)?;
    Ok(model)
}

/// The workers of one hardware SKU.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Capacity {
    pub sku: String,
    pub workers: u32,
}

/// `POST /pools` reply: register again after the heartbeat interval. If the
/// lease passes with no successful registration, the pool drops out of
/// `GET /pools`.
///
/// Both are whole seconds, and the interval is above zero and below the
/// lease. A reply that breaks this does not parse, so it cannot make a
/// PoolRelay register in a tight loop, and every `Lease` can be written as
/// JSON and read back. There is no upper bound: a service bounds the lease it
/// grants, and a client follows any lease it is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "LeaseFields")]
#[must_use = "register again after `heartbeat_interval`"]
pub struct Lease {
    heartbeat_interval_s: u64,
    lease_duration_s: u64,
}

impl Lease {
    pub fn new(heartbeat_interval_s: u64, lease_duration_s: u64) -> Result<Self, LeaseError> {
        if heartbeat_interval_s == 0 || heartbeat_interval_s >= lease_duration_s {
            return Err(LeaseError {
                heartbeat_interval_s,
                lease_duration_s,
            });
        }
        Ok(Self {
            heartbeat_interval_s,
            lease_duration_s,
        })
    }

    pub fn heartbeat_interval(&self) -> Duration {
        Duration::from_secs(self.heartbeat_interval_s)
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs(self.lease_duration_s)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error(
    "heartbeat interval {heartbeat_interval_s}s must be above zero and below the \
     {lease_duration_s}s lease"
)]
pub struct LeaseError {
    pub heartbeat_interval_s: u64,
    pub lease_duration_s: u64,
}

#[derive(Deserialize)]
struct LeaseFields {
    heartbeat_interval_s: u64,
    lease_duration_s: u64,
}

impl TryFrom<LeaseFields> for Lease {
    type Error = LeaseError;

    fn try_from(fields: LeaseFields) -> Result<Self, Self::Error> {
        Self::new(fields.heartbeat_interval_s, fields.lease_duration_s)
    }
}

/// `GET /pools` reply.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolsResponse {
    pub pools: Vec<PoolListing>,
    /// True when the service lists every pool, so a new router can tell a
    /// full list from a partial one.
    pub complete: bool,
}

/// One pool in `GET /pools`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolListing {
    pub key: PoolKey,
    pub location: PoolLocation,
    /// `None` when the service does not know the pool's model.
    pub model: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Local frontends the Global Router sends requests to.
    #[serde(default)]
    pub frontends: Vec<Url>,
    #[serde(default)]
    pub capacity: Vec<Capacity>,
    /// The only PoolRelay allowed to stream this pool's state, by each
    /// identity it can connect with. On a trusted network these are its pod's
    /// addresses, as [`address_identity`] writes them: two on a dual-stack
    /// cluster. Empty when the service does not know the PoolRelay.
    #[serde(default)]
    pub relay_identities: Vec<String>,
}

/// Names a PoolRelay by one of its addresses on a trusted network, as
/// `ip:<address>`. An IPv4-mapped IPv6 address is written as IPv4, as
/// EndpointSlices list it. A router matches a connecting PoolRelay against
/// [`PoolListing::relay_identities`] with this.
pub fn address_identity(address: IpAddr) -> String {
    format!("ip:{}", address.to_canonical())
}

/// `GET /routers` reply: every Global Router replica.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutersResponse {
    pub routers: Vec<RouterReplica>,
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn round_trip<T: Serialize + for<'de> Deserialize<'de>>(json: Value) -> T {
        let parsed: T = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), json);
        parsed
    }

    #[test]
    fn json_matches_the_service_and_its_clients() {
        let key =
            json!({"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"});
        let location =
            json!({"region": "us-west-2", "availability_zone": null, "datacenter": null});
        let mut registration = json!({
            "key": key,
            "location": location,
            "model": "Qwen/Qwen3-0.6B",
            "capabilities": ["aggregated"],
            "frontends": ["http://10.43.1.2:8000/"],
            "capacity": [{"sku": "H100", "workers": 8}]
        });
        round_trip::<PoolRegistration>(registration.clone());
        registration["model"] = json!(" ");
        assert!(serde_json::from_value::<PoolRegistration>(registration).is_err());
        round_trip::<PoolsResponse>(json!({
            "pools": [{
                "key": key,
                "location": location,
                "model": null,
                "capabilities": [],
                "frontends": [],
                "capacity": [],
                "relay_identities": ["ip:10.43.66.175", "ip:fd00::5"]
            }],
            "complete": true
        }));
        round_trip::<RoutersResponse>(json!({
            "routers": [{"replica": "aws-ohio/global-router-0", "url": "http://10.42.0.5:50051/"}]
        }));
        let mapped: IpAddr = "::ffff:10.43.66.175".parse().unwrap();
        assert_eq!(address_identity(mapped), "ip:10.43.66.175");
        assert_eq!(address_identity("fd00::2".parse().unwrap()), "ip:fd00::2");
    }

    #[test]
    fn a_lease_parses_unless_it_would_make_a_relay_spin() {
        let lease = round_trip::<Lease>(json!({"heartbeat_interval_s": 5, "lease_duration_s": 15}));
        assert_eq!(lease.heartbeat_interval(), Duration::from_secs(5));
        assert_eq!(lease.duration(), Duration::from_secs(15));
        // The longest lease a client could be given.
        let longest =
            round_trip::<Lease>(json!({"heartbeat_interval_s": 1, "lease_duration_s": u64::MAX}));
        assert_eq!(longest.duration(), Duration::from_secs(u64::MAX));
        for (interval, lease) in [(0, 15), (15, 15), (20, 15)] {
            let body = json!({"heartbeat_interval_s": interval, "lease_duration_s": lease});
            assert!(
                serde_json::from_value::<Lease>(body).is_err(),
                "{interval}s / {lease}s"
            );
        }
    }

    #[test]
    fn a_reply_without_its_list_does_not_parse() {
        // `{}` must not read as "no routers", or a PoolRelay would drop every
        // router connection.
        assert!(serde_json::from_value::<RoutersResponse>(json!({})).is_err());
        assert!(serde_json::from_value::<PoolsResponse>(json!({"complete": true})).is_err());
    }

    #[test]
    fn a_reader_skips_unknown_fields_and_fills_missing_lists() {
        // Either side can add a field, at any depth, before the other reads it.
        let key = json!({"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1", "new": 1});
        let location =
            json!({"region": "us-west-2", "availability_zone": null, "datacenter": null, "new": 1});
        serde_json::from_value::<PoolRegistration>(json!({
            "key": key, "location": location, "model": "Qwen/Qwen3-0.6B", "new": 1
        }))
        .unwrap();
        serde_json::from_value::<PoolsResponse>(json!({
            "pools": [{
                "key": key, "location": location, "model": null,
                "capacity": [{"sku": "H100", "workers": 8, "new": 1}], "new": 1
            }],
            "complete": true,
            "new": 1
        }))
        .unwrap();
        let _: Lease = serde_json::from_value(
            json!({"heartbeat_interval_s": 5, "lease_duration_s": 15, "new": 1}),
        )
        .unwrap();
        serde_json::from_value::<RoutersResponse>(json!({
            "routers": [{"replica": "aws-ohio/global-router-0", "url": "http://10.42.0.5:50051/", "new": 1}],
            "new": 1
        }))
        .unwrap();
    }
}
