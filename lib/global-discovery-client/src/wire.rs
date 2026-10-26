// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON exchanged with the Global Discovery Service. The client ignores
//! response fields it does not know, so the service can add fields first.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::replicas::ReplicaEndpoint;

#[derive(Debug, Deserialize)]
pub(crate) struct RoutersResponse {
    pub routers: Vec<ReplicaEndpoint>,
}

/// `POST /pools` reply: the service's timing for registrations. Register again
/// after `heartbeat_interval`; without one, the registration lapses after
/// `lease_duration`. A missing body, a zero interval, or an interval not below
/// the lease does not parse, so a bad reply cannot make a PoolRelay register in
/// a tight loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(try_from = "LeaseBody")]
pub struct Lease {
    pub heartbeat_interval: Duration,
    pub lease_duration: Duration,
}

#[derive(Deserialize)]
struct LeaseBody {
    heartbeat_interval_s: u64,
    lease_duration_s: u64,
}

impl TryFrom<LeaseBody> for Lease {
    type Error = String;

    fn try_from(body: LeaseBody) -> Result<Self, Self::Error> {
        if body.heartbeat_interval_s == 0 || body.heartbeat_interval_s >= body.lease_duration_s {
            return Err(format!(
                "heartbeat interval {}s must be above zero and below the {}s lease",
                body.heartbeat_interval_s, body.lease_duration_s
            ));
        }
        Ok(Self {
            heartbeat_interval: Duration::from_secs(body.heartbeat_interval_s),
            lease_duration: Duration::from_secs(body.lease_duration_s),
        })
    }
}

/// `POST /pools` request body. A PoolRelay sends it on every registration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolRegistration {
    pub key: PoolKey,
    pub location: PoolLocation,
    /// The model or checkpoint the pool serves.
    pub model: String,
    pub capabilities: Vec<String>,
    /// Local frontends the Global Router dispatches requests to.
    pub frontends: Vec<Url>,
    pub capacity: Vec<Capacity>,
}

/// `namespace` is the Kubernetes namespace containing the DGD, not the
/// Dynamo runtime namespace derived by the operator.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoolKey {
    pub site_id: String,
    pub namespace: String,
    pub dgd_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolLocation {
    pub region: String,
    pub availability_zone: Option<String>,
    pub cluster: Option<String>,
    pub datacenter: Option<String>,
}

/// The workers of one hardware SKU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capacity {
    pub sku: String,
    pub count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lease_that_would_spin_does_not_parse() {
        let lease: Lease =
            serde_json::from_str(r#"{ "heartbeat_interval_s": 5, "lease_duration_s": 15 }"#)
                .unwrap();
        assert_eq!(
            lease,
            Lease {
                heartbeat_interval: Duration::from_secs(5),
                lease_duration: Duration::from_secs(15),
            }
        );
        for body in [
            "",
            r#"{ "heartbeat_interval_s": 0, "lease_duration_s": 15 }"#,
            r#"{ "heartbeat_interval_s": 15, "lease_duration_s": 15 }"#,
            r#"{ "heartbeat_interval_s": 20, "lease_duration_s": 15 }"#,
        ] {
            assert!(serde_json::from_str::<Lease>(body).is_err(), "{body:?}");
        }
    }
}
