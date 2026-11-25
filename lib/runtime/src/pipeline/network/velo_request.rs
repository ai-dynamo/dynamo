// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

pub(super) mod peers;

use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use velo::{InstanceId, PeerInfo, WorkerAddress};

use crate::protocols::EndpointId;

pub(super) const REQUEST_HANDLER: &str = "dynamo_request";
// A system name lets Velo send the bootstrap without first doing a hello.
pub(super) const PEER_HANDLER: &str = "_dynamo_request_peer";
pub(super) const ACK_SLOTS: usize = 16_384;
const STAGED_WINDOW: usize = 4;

const ADDRESS_PREFIX: &str = "velo://";
const MAX_WORKER_ADDRESS: usize = 8 * 1024;
const MAX_ENDPOINT_PATH: usize = 1024;
const MAX_PEER_IDENTITY: usize = MAX_WORKER_ADDRESS + 128;

#[derive(Debug)]
pub(super) struct RequestAddress<'a> {
    pub(super) peer: PeerInfo,
    pub(super) path: &'a str,
}

pub(super) fn encode_address(peer: &PeerInfo, endpoint: &EndpointId, id: u64) -> Result<String> {
    let path = crate::transports::tcp::instance_path(endpoint, id);
    ensure!(
        path.len() <= MAX_ENDPOINT_PATH,
        "Velo request path for endpoint {endpoint} exceeds {MAX_ENDPOINT_PATH} bytes"
    );
    let address = encode_worker_address(peer.worker_address())
        .with_context(|| format!("invalid Velo address for endpoint {endpoint}"))?;
    Ok(format!(
        "{ADDRESS_PREFIX}{}/{address}/{path}",
        peer.instance_id()
    ))
}

pub(super) fn decode_address(address: &str) -> Result<RequestAddress<'_>> {
    let mut fields = address
        .strip_prefix(ADDRESS_PREFIX)
        .context("expected a velo:// request address")?
        .splitn(3, '/');
    let id = fields.next().context("missing Velo instance ID")?;
    let encoded = fields.next().context("missing Velo worker address")?;
    let path = fields
        .next()
        .filter(|path| !path.is_empty())
        .context("missing Velo endpoint path")?;
    ensure!(
        path.len() <= MAX_ENDPOINT_PATH,
        "Velo request path for endpoint {path} exceeds {MAX_ENDPOINT_PATH} bytes"
    );
    ensure!(id.len() == 36, "expected a hyphenated Velo instance ID");
    let id = InstanceId::from(Uuid::parse_str(id).context("invalid Velo instance ID")?);
    let address = decode_worker_address(encoded)
        .with_context(|| format!("invalid Velo address for endpoint {path}"))?;
    Ok(RequestAddress {
        peer: PeerInfo::new(id, address),
        path,
    })
}

fn encode_worker_address(address: &WorkerAddress) -> Result<String> {
    let size = base64::encoded_len(address.as_bytes().len(), false);
    ensure!(
        size.is_some_and(|size| size <= MAX_WORKER_ADDRESS),
        "encoded Velo worker address exceeds {MAX_WORKER_ADDRESS} bytes"
    );
    Ok(URL_SAFE_NO_PAD.encode(address.as_bytes()))
}

fn decode_worker_address(encoded: &str) -> Result<WorkerAddress> {
    ensure!(!encoded.is_empty(), "empty Velo worker address");
    ensure!(
        encoded.len() <= MAX_WORKER_ADDRESS,
        "encoded Velo worker address exceeds {MAX_WORKER_ADDRESS} bytes"
    );
    Ok(WorkerAddress::from_encoded(
        URL_SAFE_NO_PAD.decode(encoded)?,
    ))
}

#[derive(Serialize, Deserialize)]
struct PeerIdentity<'a> {
    instance_id: InstanceId,
    worker_address: &'a str,
}

pub(super) fn encode_identity(peer: &PeerInfo) -> Result<Bytes> {
    let address = encode_worker_address(peer.worker_address())?;
    Ok(rmp_serde::to_vec_named(&PeerIdentity {
        instance_id: peer.instance_id(),
        worker_address: &address,
    })?
    .into())
}

fn decode_identity(payload: &[u8]) -> Result<PeerIdentity<'_>> {
    ensure!(
        payload.len() <= MAX_PEER_IDENTITY,
        "Velo peer identity is too large"
    );
    Ok(rmp_serde::from_slice(payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(component: &str, name: &str) -> EndpointId {
        EndpointId {
            namespace: "space".into(),
            component: component.into(),
            name: name.into(),
        }
    }

    #[test]
    fn address_round_trips_and_distinguishes_endpoint_instances() {
        let peer = PeerInfo::new(InstanceId::new_v4(), WorkerAddress::empty());
        let first = encode_address(&peer, &endpoint("c", "a/b"), 1).unwrap();
        let other_instance = encode_address(&peer, &endpoint("c", "a/b"), 2).unwrap();
        let other_endpoint = encode_address(&peer, &endpoint("c/a", "b"), 1).unwrap();
        let decoded = decode_address(&first).unwrap();
        assert_eq!(decoded.peer, peer);
        assert_eq!(decoded.path, "1/space/c/a%2Fb");
        assert_ne!(decoded.path, decode_address(&other_instance).unwrap().path);
        assert_ne!(decoded.path, decode_address(&other_endpoint).unwrap().path);
    }

    #[test]
    fn address_bounds_name_the_endpoint() {
        let id = InstanceId::new_v4();
        let name = "a".repeat(MAX_ENDPOINT_PATH - "1/space/c/".len());
        let peer = PeerInfo::new(
            id,
            WorkerAddress::from_encoded(vec![0; MAX_WORKER_ADDRESS * 3 / 4]),
        );
        let limit = encode_address(&peer, &endpoint("c", &name), 1).unwrap();
        assert_eq!(
            decode_address(&limit).unwrap().path.len(),
            MAX_ENDPOINT_PATH
        );

        let overlong = endpoint("c", &(name + "a"));
        let error = encode_address(&peer, &overlong, 1).unwrap_err();
        assert!(error.to_string().contains(&overlong.to_string()));
        let error = decode_address(&(limit + "a")).unwrap_err();
        assert!(error.to_string().contains("1/space/c/"));

        let endpoint = endpoint("c", "generate");
        let too_large = PeerInfo::new(
            id,
            WorkerAddress::from_encoded(vec![0; MAX_WORKER_ADDRESS * 3 / 4 + 1]),
        );
        let error = encode_address(&too_large, &endpoint, 1).unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains(&endpoint.to_string()));
        assert!(error.contains("8192 bytes"));
        let oversized = format!(
            "velo://{id}/{}/1/space/c/generate",
            "A".repeat(MAX_WORKER_ADDRESS + 1)
        );
        assert!(
            format!("{:#}", decode_address(&oversized).unwrap_err()).contains("1/space/c/generate")
        );
    }
}
