// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Valkey module command and response wire codecs.

use super::*;
use std::mem::size_of_val;

pub(super) fn decode_u64_bulk(response: RespValue, command: &str) -> Result<u64> {
    let RespValue::Bulk(payload) = response else {
        bail!("{command} returned a non-bulk reply");
    };
    let bytes: [u8; size_of::<u64>()] = payload.try_into().map_err(|payload: Vec<u8>| {
        anyhow::anyhow!(
            "{command} returned {} bytes; expected {}",
            payload.len(),
            size_of::<u64>()
        )
    })?;
    Ok(u64::from_be_bytes(bytes))
}

pub(super) fn decode_gc_reply(response: RespValue) -> Result<[u64; 8]> {
    let RespValue::Array(values) = response else {
        bail!("DYNKV.GC returned a non-array reply");
    };
    if values.len() != 8 {
        bail!("DYNKV.GC returned {} fields; expected 8", values.len());
    }
    let mut decoded = [0_u64; 8];
    for (index, value) in values.into_iter().enumerate() {
        let RespValue::Integer(value) = value else {
            bail!("DYNKV.GC field {index} was not an integer");
        };
        decoded[index] =
            u64::try_from(value).with_context(|| format!("DYNKV.GC field {index} was negative"))?;
    }
    Ok(decoded)
}

/// Encode one human-readable index-key segment without allowing delimiters in
/// a namespace, component, or configured scope to create the same Valkey key
/// as a different tuple. Percent itself is escaped, making this byte encoding
/// stable and injective for UTF-8 input.
pub(super) fn encode_index_key_segment(value: &str, name: &str) -> Result<String> {
    if value.trim().is_empty() {
        bail!("{name} must not be empty");
    }

    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
    }
    Ok(encoded)
}

pub(super) fn resolve_required_replica_acks(
    configured: Option<u32>,
    endpoint_count: usize,
) -> Result<usize> {
    if endpoint_count == 0 {
        bail!("router_valkey_urls must contain at least one endpoint");
    }
    let required = configured.unwrap_or_else(|| u32::from(endpoint_count > 1));
    if required > MAX_REQUIRED_REPLICA_ACKS {
        bail!("router_valkey_required_replica_acks must be at most {MAX_REQUIRED_REPLICA_ACKS}");
    }
    usize::try_from(required).context("router_valkey_required_replica_acks does not fit usize")
}

pub(super) fn ensure_ok(
    response: std::result::Result<RespValue, RespError>,
) -> std::result::Result<(), RespError> {
    match response? {
        RespValue::Simple(value) if value == "OK" || value == "NOOP" => Ok(()),
        RespValue::Simple(value) => Err(RespError::Protocol(format!(
            "unexpected simple reply {value:?}"
        ))),
        RespValue::Integer(value) => Err(RespError::Protocol(format!(
            "unexpected integer reply {value}"
        ))),
        RespValue::Bulk(_) => Err(RespError::Protocol("unexpected bulk reply".to_string())),
        RespValue::Array(_) => Err(RespError::Protocol("unexpected array reply".to_string())),
        RespValue::Null => Err(RespError::Protocol("unexpected null reply".to_string())),
    }
}

/// Validate a completely consumed APPLY pipeline and report whether every
/// command was an idempotent no-op. The caller may safely reuse the socket on
/// validation failure because all RESP replies have already been read.
pub(super) fn validate_apply_responses(
    responses: &[RespValue],
) -> std::result::Result<bool, RespError> {
    if responses.is_empty() {
        return Err(RespError::Protocol(
            "Valkey APPLY pipeline returned no responses".to_string(),
        ));
    }
    let mut all_noop = true;
    for response in responses {
        match response {
            RespValue::Simple(value) if value == "OK" => all_noop = false,
            RespValue::Simple(value) if value == "NOOP" => {}
            RespValue::Simple(value) => {
                return Err(RespError::Protocol(format!(
                    "unexpected pipelined simple reply {value:?}"
                )));
            }
            other => {
                return Err(RespError::Protocol(format!(
                    "unexpected pipelined Valkey reply: {}",
                    response_kind(other)
                )));
            }
        }
    }
    Ok(all_noop)
}

pub(super) fn response_kind(response: &RespValue) -> &'static str {
    match response {
        RespValue::Simple(_) => "simple string",
        RespValue::Bulk(_) => "bulk string",
        RespValue::Integer(_) => "integer",
        RespValue::Array(_) => "array",
        RespValue::Null => "null",
    }
}

pub(super) fn encode_admission_identity(
    payload: &mut Vec<u8>,
    domain: &[u8],
    nonce: ReservationNonce,
) {
    payload.push(ADMISSION_WIRE_VERSION);
    payload.extend_from_slice(&(domain.len() as u32).to_be_bytes());
    payload.extend_from_slice(domain);
    payload.extend_from_slice(&nonce.client_nonce.to_be_bytes());
    payload.extend_from_slice(&nonce.request_nonce.to_be_bytes());
}

pub(super) fn validate_worker_ranks(dp_ranks: &[DpRank]) -> Result<()> {
    if dp_ranks.is_empty() || dp_ranks.len() > MAX_WORKER_RANKS {
        bail!(
            "Valkey worker registration requires 1..={MAX_WORKER_RANKS} DP ranks; got {}",
            dp_ranks.len()
        );
    }
    if dp_ranks.windows(2).any(|ranks| ranks[0] >= ranks[1]) {
        bail!("Valkey worker registration DP ranks must be sorted and unique");
    }
    Ok(())
}

pub(super) fn encode_worker_lease_registration(
    owner_nonce: u64,
    lease_ms: u64,
    expected_generation: u64,
    dp_ranks: &[DpRank],
) -> Result<Vec<u8>> {
    if owner_nonce == 0 || lease_ms == 0 || lease_ms > MAX_ADMISSION_LEASE_MS {
        bail!("invalid Valkey worker owner nonce or lease duration");
    }
    validate_worker_ranks(dp_ranks)?;
    let mut payload = Vec::with_capacity(1 + 8 + 8 + 8 + 4 + size_of_val(dp_ranks));
    payload.push(WORKER_LEASED_REGISTRATION_WIRE_VERSION);
    payload.extend_from_slice(&owner_nonce.to_be_bytes());
    payload.extend_from_slice(&lease_ms.to_be_bytes());
    payload.extend_from_slice(&expected_generation.to_be_bytes());
    payload.extend_from_slice(&(dp_ranks.len() as u32).to_be_bytes());
    for dp_rank in dp_ranks {
        payload.extend_from_slice(&dp_rank.to_be_bytes());
    }
    Ok(payload)
}

pub(super) fn encode_worker_lease_control(
    worker_id: WorkerId,
    owner_nonce: u64,
    lease_ms: Option<u64>,
) -> Result<Vec<u8>> {
    if owner_nonce == 0
        || lease_ms.is_some_and(|lease_ms| lease_ms == 0 || lease_ms > MAX_ADMISSION_LEASE_MS)
    {
        bail!("invalid Valkey worker owner nonce or lease duration");
    }
    let mut payload = Vec::with_capacity(1 + 8 + 8 + usize::from(lease_ms.is_some()) * 8);
    payload.push(WORKER_LEASE_CONTROL_VERSION);
    payload.extend_from_slice(&worker_id.to_be_bytes());
    payload.extend_from_slice(&owner_nonce.to_be_bytes());
    if let Some(lease_ms) = lease_ms {
        payload.extend_from_slice(&lease_ms.to_be_bytes());
    }
    Ok(payload)
}

pub(super) fn encode_select_reserve(request: &ReservationRequest) -> Result<Vec<u8>> {
    let prefix_bytes = request
        .block_hashes
        .len()
        .checked_mul(size_of::<u64>())
        .context("Valkey admission prefix payload overflow")?;
    let candidate_bytes = request
        .candidates
        .len()
        .checked_mul(size_of::<u64>() + size_of::<u32>() * 2)
        .context("Valkey admission candidate payload overflow")?;
    let mut payload = Vec::with_capacity(
        1 + 4 + request.domain.len() + 8 + 8 + 8 + 4 + prefix_bytes + 4 + candidate_bytes,
    );
    encode_admission_identity(&mut payload, &request.domain, request.nonce);
    payload.extend_from_slice(&request.lease_ms.to_be_bytes());
    payload.extend_from_slice(&(request.block_hashes.len() as u32).to_be_bytes());
    for block_hash in &request.block_hashes {
        payload.extend_from_slice(&block_hash.0.to_be_bytes());
    }
    payload.extend_from_slice(&(request.candidates.len() as u32).to_be_bytes());
    for candidate in &request.candidates {
        payload.extend_from_slice(&candidate.worker.worker_id.to_be_bytes());
        payload.extend_from_slice(&candidate.worker.dp_rank.to_be_bytes());
        payload.extend_from_slice(&candidate.capacity.to_be_bytes());
    }
    Ok(payload)
}

pub(super) fn encode_release(
    request: &ReservationLifecycleRequest,
    expected_expires_at_ms: u64,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(1 + 4 + request.domain.len() + 8 + 8 + 8);
    encode_admission_identity(&mut payload, &request.domain, request.nonce);
    payload.extend_from_slice(&expected_expires_at_ms.to_be_bytes());
    payload
}

pub(super) fn encode_renew(
    request: &ReservationLifecycleRequest,
    expected_expires_at_ms: u64,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(1 + 4 + request.domain.len() + 8 + 8 + 8 + 8);
    encode_admission_identity(&mut payload, &request.domain, request.nonce);
    payload.extend_from_slice(&expected_expires_at_ms.to_be_bytes());
    payload.extend_from_slice(&request.lease_ms.to_be_bytes());
    payload
}

pub(super) fn decode_admission_status(payload: &[u8]) -> Result<bool> {
    if payload.len() != 2 || payload[0] != ADMISSION_WIRE_VERSION {
        bail!("invalid Valkey admission status reply");
    }
    match payload[1] {
        ADMISSION_NO_CAPACITY => Ok(false),
        ADMISSION_RESERVED => Ok(true),
        status => bail!("unknown Valkey admission status {status}"),
    }
}

pub(super) fn decode_reservation_reply(
    payload: &[u8],
    expected_nonce: ReservationNonce,
) -> Result<Option<ReservationGrant>> {
    if payload.len() < 2 || payload[0] != ADMISSION_WIRE_VERSION {
        bail!("invalid Valkey reservation reply");
    }
    match payload[1] {
        ADMISSION_NO_CAPACITY if payload.len() == 2 => Ok(None),
        ADMISSION_RESERVED => {
            const RESERVED_REPLY_BYTES: usize = 1 + 1 + 8 + 8 + 8 + 4 + 8 + 4 + 4;
            if payload.len() != RESERVED_REPLY_BYTES {
                bail!(
                    "invalid Valkey reserved reply length {}; expected {RESERVED_REPLY_BYTES}",
                    payload.len()
                );
            }
            let mut offset = 2;
            let client_nonce = read_u64(payload, &mut offset)?;
            let request_nonce = read_u64(payload, &mut offset)?;
            if (ReservationNonce {
                client_nonce,
                request_nonce,
            }) != expected_nonce
            {
                bail!("Valkey reservation reply nonce does not match request");
            }
            let worker = WorkerWithDpRank::new(
                read_u64(payload, &mut offset)?,
                read_u32(payload, &mut offset)?,
            );
            let expires_at_ms = read_u64(payload, &mut offset)?;
            let matched_blocks = read_u32(payload, &mut offset)?;
            let active_reservations_at_grant = read_u32(payload, &mut offset)?;
            debug_assert_eq!(offset, RESERVED_REPLY_BYTES);
            Ok(Some(ReservationGrant {
                worker,
                expires_at_ms,
                matched_blocks,
                active_reservations_at_grant,
            }))
        }
        status => bail!("unknown Valkey reservation status {status}"),
    }
}

pub(super) fn read_u32(payload: &[u8], offset: &mut usize) -> Result<u32> {
    let bytes = payload
        .get(*offset..*offset + size_of::<u32>())
        .context("truncated Valkey admission u32")?;
    *offset += size_of::<u32>();
    Ok(u32::from_be_bytes(
        bytes.try_into().expect("fixed-size slice"),
    ))
}

pub(super) fn read_u64(payload: &[u8], offset: &mut usize) -> Result<u64> {
    let bytes = payload
        .get(*offset..*offset + size_of::<u64>())
        .context("truncated Valkey admission u64")?;
    *offset += size_of::<u64>();
    Ok(u64::from_be_bytes(
        bytes.try_into().expect("fixed-size slice"),
    ))
}

pub(super) fn encode_event(event: &RouterEvent) -> Vec<u8> {
    let mut payload = Vec::with_capacity(64);
    payload.push(WIRE_VERSION);
    let kind = match &event.event.data {
        KvCacheEventData::Stored(_) => EVENT_STORE,
        KvCacheEventData::Removed(_) => EVENT_REMOVE,
        KvCacheEventData::Cleared => EVENT_CLEAR,
    };
    payload.push(kind);
    payload.extend_from_slice(&event.worker_id.to_be_bytes());
    payload.extend_from_slice(&event.event.dp_rank.to_be_bytes());
    payload.extend_from_slice(&event.event.event_id.to_be_bytes());
    match &event.event.data {
        KvCacheEventData::Stored(store) => {
            payload.extend_from_slice(
                &store
                    .parent_hash
                    .map_or(ROOT_PARENT, |hash| hash.0)
                    .to_be_bytes(),
            );
            payload.extend_from_slice(&(store.blocks.len() as u32).to_be_bytes());
            for block in &store.blocks {
                payload.extend_from_slice(&block.block_hash.0.to_be_bytes());
                payload.extend_from_slice(&block.tokens_hash.0.to_be_bytes());
            }
        }
        KvCacheEventData::Removed(remove) => {
            payload.extend_from_slice(&(remove.block_hashes.len() as u32).to_be_bytes());
            for block_hash in &remove.block_hashes {
                payload.extend_from_slice(&block_hash.0.to_be_bytes());
            }
        }
        KvCacheEventData::Cleared => {}
    }
    payload
}

pub(super) fn encode_rank_snapshot(
    worker_id: WorkerId,
    dp_rank: DpRank,
    events: &[RouterEvent],
) -> Result<Vec<u8>> {
    let event_count = u32::try_from(events.len())
        .context("Valkey rank snapshot contains more than u32::MAX events")?;
    let mut payload = Vec::new();
    payload.push(WIRE_VERSION);
    payload.extend_from_slice(&event_count.to_be_bytes());
    for event in events {
        if event.worker_id != worker_id
            || event.event.dp_rank != dp_rank
            || !event.storage_tier.is_gpu()
            || !matches!(&event.event.data, KvCacheEventData::Stored(_))
        {
            bail!(
                "Valkey rank snapshot must contain only device-tier STORE events for worker {worker_id} rank {dp_rank}"
            );
        }
        let encoded = encode_event(event);
        let event_length = u32::try_from(encoded.len())
            .context("Valkey rank snapshot event exceeds u32::MAX bytes")?;
        payload.extend_from_slice(&event_length.to_be_bytes());
        payload.extend_from_slice(&encoded);
    }
    Ok(payload)
}

pub(super) fn encode_match(block_hashes: &[LocalBlockHash]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(5 + block_hashes.len() * 8);
    payload.push(WIRE_VERSION);
    payload.extend_from_slice(&(block_hashes.len() as u32).to_be_bytes());
    for block_hash in block_hashes {
        payload.extend_from_slice(&block_hash.0.to_be_bytes());
    }
    payload
}

pub(super) fn decode_match(payload: &[u8]) -> Result<MatchDetails> {
    let Some((&version, remaining)) = payload.split_first() else {
        bail!("empty match response");
    };
    if version != WIRE_VERSION || remaining.len() < 4 {
        bail!("unsupported match response version {version}");
    }
    let count = u32::from_be_bytes(remaining[..4].try_into().expect("length checked")) as usize;
    let entries = &remaining[4..];
    if entries.len() != count.saturating_mul(MATCH_ENTRY_BYTES) {
        bail!("invalid match response length for {count} workers");
    }

    let mut scores = FxHashMap::default();
    let mut last_matched_hashes = FxHashMap::default();
    for entry in entries.chunks_exact(MATCH_ENTRY_BYTES) {
        let worker = WorkerWithDpRank::new(
            u64::from_be_bytes(entry[..8].try_into().expect("fixed-size entry")),
            u32::from_be_bytes(entry[8..12].try_into().expect("fixed-size entry")),
        );
        let matched_blocks =
            u32::from_be_bytes(entry[12..16].try_into().expect("fixed-size entry"));
        let last_hash = ExternalSequenceBlockHash(u64::from_be_bytes(
            entry[16..24].try_into().expect("fixed-size entry"),
        ));
        if matched_blocks == 0 {
            continue;
        }
        scores.insert(worker, matched_blocks);
        last_matched_hashes.insert(worker, last_hash);
    }

    Ok(MatchDetails {
        overlap_scores: OverlapScores {
            scores,
            frequencies: Vec::new(),
        },
        last_matched_hashes,
    })
}
