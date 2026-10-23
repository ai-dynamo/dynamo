// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The checks every consumer applies to a snapshot.

use std::collections::HashMap;
use std::net::IpAddr;

use super::check_identity;
use super::proto::{Endpoint, Execution, Identity, Protocol, Snapshot, partition::Placement};

/// The longest DNS name, in bytes, without a trailing dot (RFC 1035).
const MAX_DNS_NAME_BYTES: usize = 253;
/// The longest DNS label, in bytes (RFC 1035).
const MAX_DNS_LABEL_BYTES: usize = 63;

/// Checks that `snapshot` is complete and valid and returns its identity:
/// the identity is valid, partition IDs are unique, each partition has one
/// placement, each execution has a node and a control endpoint, every
/// endpoint is usable for its role, and each chain member's owner is another
/// partition that executes. Partition order is neither checked nor used.
pub(crate) fn check_snapshot(snapshot: &Snapshot) -> Result<&Identity, String> {
    let identity = snapshot
        .identity
        .as_ref()
        .ok_or("the snapshot has no identity")?;
    check_identity(identity)?;
    if snapshot.partitions.is_empty() {
        return Err("the snapshot has no partitions".to_string());
    }
    let mut placements = HashMap::with_capacity(snapshot.partitions.len());
    for partition in &snapshot.partitions {
        let id = partition.partition_id;
        let placement = partition
            .placement
            .as_ref()
            .ok_or_else(|| format!("partition {id} has no placement"))?;
        if placements.insert(id, placement).is_some() {
            return Err(format!("partition {id} appears more than once"));
        }
        if let Placement::Execution(execution) = placement {
            check_execution(execution).map_err(|error| format!("partition {id}: {error}"))?;
        }
    }
    for partition in &snapshot.partitions {
        let id = partition.partition_id;
        let Some(Placement::ChainMember(member)) = &partition.placement else {
            continue;
        };
        let owner = member.owner_partition_id;
        match placements.get(&owner) {
            _ if owner == id => {
                return Err(format!("partition {id}'s chain owner is the member itself"));
            }
            Some(Placement::Execution(_)) => {}
            Some(Placement::ChainMember(_)) => {
                return Err(format!(
                    "partition {id}'s chain owner {owner} is itself a chain member"
                ));
            }
            None => {
                return Err(format!(
                    "partition {id}'s chain owner {owner} is not in the snapshot"
                ));
            }
        }
    }
    Ok(identity)
}

fn check_execution(execution: &Execution) -> Result<(), String> {
    if execution.node_count == 0 {
        return Err("node_count must be at least 1".to_string());
    }
    let control = execution
        .control
        .as_ref()
        .ok_or("the execution has no control endpoint")?;
    check_endpoint(
        "control",
        control,
        &[Protocol::HydraControl, Protocol::LpudRemoted],
    )?;
    if let Some(state_ingress) = &execution.state_ingress {
        check_endpoint("state_ingress", state_ingress, &[Protocol::HydraState])?;
    }
    Ok(())
}

fn check_endpoint(role: &str, endpoint: &Endpoint, protocols: &[Protocol]) -> Result<(), String> {
    if !is_host(&endpoint.host) {
        return Err(format!(
            "{role} host `{}` is neither a DNS name nor an IP address",
            endpoint.host.escape_debug()
        ));
    }
    if !(1..=u32::from(u16::MAX)).contains(&endpoint.port) {
        return Err(format!("{role} port {} is outside 1-65535", endpoint.port));
    }
    let protocol = Protocol::try_from(endpoint.protocol).ok();
    if !protocol.is_some_and(|protocol| protocols.contains(&protocol)) {
        let named = protocol.map_or_else(
            || format!("{} (unknown)", endpoint.protocol),
            |protocol| protocol.as_str_name().to_string(),
        );
        let allowed = protocols
            .iter()
            .map(|protocol| protocol.as_str_name())
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(format!("{role} protocol {named} is not {allowed}"));
    }
    Ok(())
}

/// An IPv4 or IPv6 literal in standard form, without brackets or zone, or a
/// DNS name of RFC 1123 labels: letters, digits and hyphens, not starting or
/// ending with a hyphen.
fn is_host(host: &str) -> bool {
    host.parse::<IpAddr>().is_ok() || is_dns_name(host)
}

/// The last label must not be numeric (RFC 1123 section 2.1): neither all
/// digits nor `0x` and hex digits, so a name is never a form such as `1.2.3`,
/// `010.0.0.1` or `0x7f000001` that libc's `inet_aton` resolves as IPv4
/// shorthand.
fn is_dns_name(name: &str) -> bool {
    name.len() <= MAX_DNS_NAME_BYTES
        && name.split('.').all(|label| {
            (1..=MAX_DNS_LABEL_BYTES).contains(&label.len())
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        })
        && name
            .rsplit('.')
            .next()
            .is_some_and(|last| !is_numeric_label(last))
}

fn is_numeric_label(label: &str) -> bool {
    let hex = label
        .strip_prefix("0x")
        .or_else(|| label.strip_prefix("0X"));
    match hex {
        Some(digits) => !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_hexdigit()),
        None => label.bytes().all(|byte| byte.is_ascii_digit()),
    }
}
