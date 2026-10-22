// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests of the topology protocol, grouped by the surface they cover.

use super::proto::{
    ChainMember, Endpoint, Execution, GetSnapshotResponse, Identity, Partition, Protocol, Snapshot,
    partition::Placement,
};

mod agent;
mod protocol;
mod relay;
mod resolution;

const CONTROL: Protocol = Protocol::HydraControl;
const STATE: Protocol = Protocol::HydraState;

fn identity() -> Identity {
    Identity {
        deployment: "0d4c7a9e-5b1f-4c38-9a62-3f1e2d7c8b90".to_string(),
        allocation: "llama-0-afd-0".to_string(),
        build: "main/llama3_2-1b-lpx-lp20/7a8a706a".to_string(),
    }
}

fn endpoint(host: &str, port: u32, protocol: Protocol) -> Endpoint {
    Endpoint {
        host: host.to_string(),
        port,
        protocol: protocol as i32,
    }
}

fn execution(id: u32, node_count: u32, control: Endpoint, state: Option<Endpoint>) -> Partition {
    Partition {
        partition_id: id,
        placement: Some(Placement::Execution(Execution {
            node_count,
            control: Some(control),
            state_ingress: state,
        })),
    }
}

fn member(id: u32, owner: u32) -> Partition {
    Partition {
        partition_id: id,
        placement: Some(Placement::ChainMember(ChainMember {
            owner_partition_id: owner,
        })),
    }
}

/// The snapshot in `proto/testdata/get_snapshot_response.txtpb`, written out
/// by hand.
fn snapshot() -> Snapshot {
    let agent = |pod: u32| format!("llama-0-afd-0-agt-{pod}.llama-0.dynamo.svc.cluster.local");
    Snapshot {
        identity: Some(identity()),
        partitions: vec![
            execution(
                1,
                1,
                endpoint(&agent(0), 19877, CONTROL),
                Some(endpoint(&agent(0), 19880, STATE)),
            ),
            execution(4, 2, endpoint(&agent(1), 19877, CONTROL), None),
            member(5, 4),
            execution(300, 1, endpoint("10.0.3.17", 19877, CONTROL), None),
        ],
    }
}

fn response() -> GetSnapshotResponse {
    GetSnapshotResponse {
        snapshot: Some(snapshot()),
    }
}
