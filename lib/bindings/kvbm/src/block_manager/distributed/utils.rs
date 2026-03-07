// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::env;

use dynamo_runtime::config::environment_names::kvbm::leader as env_kvbm_leader;

const DEFAULT_LEADER_ZMQ_HOST: &str = "127.0.0.1";
const DEFAULT_LEADER_ZMQ_PUB_PORT: u16 = 56001;
const DEFAULT_LEADER_ZMQ_ACK_PORT: u16 = 56002;

fn read_env_trimmed(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn parse_port_u16(s: &str) -> Option<u16> {
    match s.parse::<u32>() {
        Ok(v) if (1..=65535).contains(&v) => Some(v as u16),
        _ => None,
    }
}

fn validated_port_from_env(key: &str, default_port: u16) -> u16 {
    if let Some(val) = read_env_trimmed(key) {
        if let Some(p) = parse_port_u16(&val) {
            if p < 1024 {
                tracing::warn!("{key} is a privileged port ({p}); binding may require extra caps");
            }
            return p;
        } else {
            tracing::warn!("{key} invalid value '{val}', falling back to default {default_port}");
        }
    }
    default_port
}

fn get_leader_zmq_host() -> String {
    read_env_trimmed(env_kvbm_leader::DYN_KVBM_LEADER_ZMQ_HOST)
        .unwrap_or_else(|| DEFAULT_LEADER_ZMQ_HOST.to_string())
}

pub fn get_leader_zmq_pub_url() -> String {
    get_leader_zmq_pub_url_for_rank(0)
}

pub fn get_leader_zmq_ack_url() -> String {
    get_leader_zmq_ack_url_for_rank(0)
}

/// Return the ZMQ PUB URL for a specific DP rank.
/// Each rank offsets the base port by `rank * 2` to avoid collisions
/// when multiple DP ranks run in the same process (attention DP).
/// rank 0 → base, rank 1 → base+2, rank 2 → base+4, etc.
/// (offset by 2 because each rank uses two consecutive ports: pub and ack)
pub fn get_leader_zmq_pub_url_for_rank(rank: u16) -> String {
    let base_port: u16 = validated_port_from_env(
        env_kvbm_leader::DYN_KVBM_LEADER_ZMQ_PUB_PORT,
        DEFAULT_LEADER_ZMQ_PUB_PORT,
    );
    let port = base_port + rank * 2;
    format!("tcp://{}:{}", get_leader_zmq_host(), port)
}

/// Return the ZMQ ACK URL for a specific DP rank.
/// See [`get_leader_zmq_pub_url_for_rank`] for the port offset scheme.
pub fn get_leader_zmq_ack_url_for_rank(rank: u16) -> String {
    let base_port: u16 = validated_port_from_env(
        env_kvbm_leader::DYN_KVBM_LEADER_ZMQ_ACK_PORT,
        DEFAULT_LEADER_ZMQ_ACK_PORT,
    );
    let port = base_port + rank * 2;
    format!("tcp://{}:{}", get_leader_zmq_host(), port)
}
