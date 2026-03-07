// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::env;

use dynamo_runtime::config::environment_names::kvbm::leader as env_kvbm_leader;

const DEFAULT_LEADER_ZMQ_HOST: &str = "127.0.0.1";
const DEFAULT_LEADER_ZMQ_PUB_PORT: u16 = 56001;

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
///
/// Port scheme: each rank gets two ports from a single base, interleaved:
///   rank 0 → PUB=base+0, ACK=base+1
///   rank 1 → PUB=base+2, ACK=base+3
///   rank N → PUB=base+N*2, ACK=base+N*2+1
///
/// The PUB base port is used as the single base; the ACK env var is ignored
/// when dp_rank > 0 to prevent overlap between independently-configured bases.
pub fn get_leader_zmq_pub_url_for_rank(rank: u16) -> String {
    let base_port: u16 = validated_port_from_env(
        env_kvbm_leader::DYN_KVBM_LEADER_ZMQ_PUB_PORT,
        DEFAULT_LEADER_ZMQ_PUB_PORT,
    );
    let port = base_port + rank * 2;
    let url = format!("tcp://{}:{}", get_leader_zmq_host(), port);
    tracing::info!(
        "ZMQ PUB URL for dp_rank={}: {} (base_port={})",
        rank, url, base_port
    );
    url
}

/// Return the ZMQ ACK URL for a specific DP rank.
/// See [`get_leader_zmq_pub_url_for_rank`] for the port scheme.
pub fn get_leader_zmq_ack_url_for_rank(rank: u16) -> String {
    let base_port: u16 = validated_port_from_env(
        env_kvbm_leader::DYN_KVBM_LEADER_ZMQ_PUB_PORT,
        DEFAULT_LEADER_ZMQ_PUB_PORT,
    );
    // ACK port = PUB base + rank*2 + 1 (interleaved with PUB)
    let port = base_port + rank * 2 + 1;
    let url = format!("tcp://{}:{}", get_leader_zmq_host(), port);
    tracing::info!(
        "ZMQ ACK URL for dp_rank={}: {} (base_port={})",
        rank, url, base_port
    );
    url
}
