// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_global_common::{PoolId, PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};
use url::Url;

/// One pool as the Global Router sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pool {
    pub id: PoolId,
    /// The model the pool serves, as clients name it in requests.
    pub model: String,
    pub location: PoolLocation,
    /// Frontends that accept forwarded requests. Each is the root of an
    /// OpenAI-compatible HTTP API, for example `http://10.0.0.5:8000/`.
    pub frontends: Vec<Url>,
    /// The pool's Relay WAN gRPC endpoint, which streams the pool's load,
    /// for example `http://10.0.0.6:50051`. `None` when the pool's load is
    /// not followed.
    pub relay: Option<Url>,
}

impl Pool {
    /// The pool `key` names, serving `model` in `region` from `frontends`,
    /// with no Relay.
    pub fn new(key: &PoolKey, model: &str, region: &str, frontends: Vec<Url>) -> Self {
        Self {
            id: V1PoolIdDeriver.derive(key),
            model: model.to_owned(),
            location: PoolLocation {
                region: region.to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            frontends,
            relay: None,
        }
    }
}
