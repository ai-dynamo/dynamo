// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The JSON of the Global Discovery Service's HTTP API. A reader ignores
//! fields it does not know, so the service can add a field before its clients
//! use it.

use std::net::IpAddr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

use crate::pool::{PoolKey, PoolLocation};
use crate::replicas::ReplicaEndpoint;

/// `POST /pools` body. A PoolRelay sends it on a timer, and each call is also
/// the pool's heartbeat.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolRegistration {
    pub key: PoolKey,
    pub location: PoolLocation,
    /// The model or checkpoint the pool serves.
    pub model: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Local frontends the Global Router sends requests to.
    #[serde(default)]
    pub frontends: Vec<Url>,
    #[serde(default)]
    pub capacity: Vec<Capacity>,
}

/// The workers of one hardware SKU.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Capacity {
    pub sku: String,
    pub count: u32,
}

/// `POST /pools` reply: register again after the heartbeat interval. If the
/// lease passes with no successful registration, the pool drops out of
/// `GET /pools`.
///
/// Both are whole seconds, the interval is above zero and below the lease,
/// and the lease is at most [`Lease::MAX_DURATION_S`]. A reply that breaks
/// this does not parse, so it cannot make a PoolRelay register in a tight
/// loop, and every `Lease` can be written as JSON and read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "LeaseFields")]
pub struct Lease {
    heartbeat_interval_s: u64,
    lease_duration_s: u64,
}

impl Lease {
    /// One day.
    pub const MAX_DURATION_S: u64 = 86_400;

    pub fn new(heartbeat_interval_s: u64, lease_duration_s: u64) -> Result<Self, LeaseError> {
        if heartbeat_interval_s == 0
            || heartbeat_interval_s >= lease_duration_s
            || lease_duration_s > Self::MAX_DURATION_S
        {
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

    pub fn lease_duration(&self) -> Duration {
        Duration::from_secs(self.lease_duration_s)
    }
}

/// A lease that [`Lease::new`] refuses.
#[derive(Debug, Error, PartialEq, Eq)]
#[error(
    "heartbeat interval {heartbeat_interval_s}s must be above zero and below the \
     {lease_duration_s}s lease, which may be at most a day"
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
    /// The only PoolRelay allowed to stream this pool's state, when the
    /// service knows it. On a trusted network it is [`address_identity`].
    #[serde(default)]
    pub relay_identity: Option<String>,
}

/// Names a PoolRelay by its address on a trusted network, as
/// `ip:<address>`. An IPv4-mapped IPv6 address is written as IPv4, as
/// EndpointSlices list it. A router matches a connecting PoolRelay against
/// [`PoolListing::relay_identity`] with this.
pub fn address_identity(address: IpAddr) -> String {
    format!("ip:{}", address.to_canonical())
}

/// `GET /routers` reply: every Global Router replica.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutersResponse {
    pub routers: Vec<ReplicaEndpoint>,
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::replicas::ReplicaId;

    fn round_trip<T: Serialize + for<'de> Deserialize<'de>>(json: Value) -> T {
        let parsed: T = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), json);
        parsed
    }

    #[test]
    fn json_matches_the_service_and_its_clients() {
        let key =
            json!({"site_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"});
        let location = json!({
            "region": "us-west-2",
            "availability_zone": null,
            "cluster": "aws-west",
            "datacenter": null
        });
        round_trip::<PoolRegistration>(json!({
            "key": key,
            "location": location,
            "model": "Qwen/Qwen3-0.6B",
            "capabilities": ["aggregated"],
            "frontends": ["http://10.43.1.2:8000/"],
            "capacity": [{"sku": "H100", "count": 8}]
        }));
        let pools = round_trip::<PoolsResponse>(json!({
            "pools": [{
                "key": key,
                "location": location,
                "model": null,
                "capabilities": [],
                "frontends": [],
                "capacity": [],
                "relay_identity": "ip:10.43.66.175"
            }],
            "complete": true
        }));
        assert!(pools.complete);
        let routers = round_trip::<RoutersResponse>(json!({
            "routers": [{"replica": "aws-ohio/global-router-0", "url": "http://10.42.0.5:50051/"}]
        }));
        assert_eq!(
            routers.routers[0].replica,
            ReplicaId("aws-ohio/global-router-0".to_owned())
        );
        let mapped: IpAddr = "::ffff:10.43.66.175".parse().unwrap();
        assert_eq!(address_identity(mapped), "ip:10.43.66.175");
        assert_eq!(address_identity("fd00::2".parse().unwrap()), "ip:fd00::2");
    }

    #[test]
    fn a_lease_that_would_spin_or_overflow_does_not_parse() {
        let lease = round_trip::<Lease>(json!({"heartbeat_interval_s": 5, "lease_duration_s": 15}));
        assert_eq!(lease, Lease::new(5, 15).unwrap());
        assert_eq!(lease.heartbeat_interval(), Duration::from_secs(5));
        let too_long = Lease::MAX_DURATION_S + 1;
        for (interval, lease) in [(0, 15), (15, 15), (20, 15), (5, too_long)] {
            let body = json!({"heartbeat_interval_s": interval, "lease_duration_s": lease});
            assert!(
                serde_json::from_value::<Lease>(body).is_err(),
                "{interval}s / {lease}s"
            );
        }
    }
}
