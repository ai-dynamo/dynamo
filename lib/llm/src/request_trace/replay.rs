// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-oriented request hash capture for live traces.

use bytemuck::cast_slice;
use std::sync::{Arc, Mutex, OnceLock};

use dynamo_kv_router::protocols::{
    BlockHashOptions, LocalBlockHash, XXH3_SEED, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};
use dynamo_tokens::compute_hash_v2;

use crate::protocols::TokenIdType;

use super::RequestReplayMetrics;

const HASH_ALGORITHM: &str = "blake3-keyed-chain64-v1";
static HASH_KEY: OnceLock<Option<Arc<ReplayHashKey>>> = OnceLock::new();

struct ReplayHashKey {
    key: [u8; 32],
    id: String,
}

impl ReplayHashKey {
    fn from_hex(encoded: &str) -> anyhow::Result<Self> {
        let encoded = encoded.trim();
        anyhow::ensure!(
            encoded.len() == 64 && encoded.is_ascii(),
            "trace hash key must contain 64 hexadecimal characters"
        );
        let mut key = [0; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).map_err(|_| {
                anyhow::anyhow!("trace hash key must contain 64 hexadecimal characters")
            })?;
        }
        let id = blake3::keyed_hash(&key, b"dynamo-request-trace-key-id-v1")
            .to_hex()
            .to_string();
        Ok(Self { key, id })
    }

    fn block(&self, parent: Option<u64>, tokens: &[TokenIdType], block_size: usize) -> u64 {
        let mut hasher = blake3::Hasher::new_keyed(&self.key);
        hasher.update(HASH_ALGORITHM.as_bytes());
        hasher.update(&(block_size as u64).to_le_bytes());
        hasher.update(&[u8::from(parent.is_some())]);
        hasher.update(&parent.unwrap_or_default().to_le_bytes());
        hasher.update(&(tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hasher.update(&token.to_le_bytes());
        }
        u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap())
    }

    fn input_hashes(&self, tokens: &[TokenIdType], block_size: usize) -> Vec<u64> {
        let mut parent = None;
        tokens
            .chunks(block_size)
            .map(|block| {
                let hash = self.block(parent, block, block_size);
                parent = Some(hash);
                hash
            })
            .collect()
    }
}

pub(super) fn init_hash_key() -> anyhow::Result<()> {
    use dynamo_runtime::config::environment_names::llm::request_trace::DYN_REQUEST_TRACE_HASH_KEY_FILE;
    if HASH_KEY.get().is_some() {
        return Ok(());
    }
    let key = match std::env::var_os(DYN_REQUEST_TRACE_HASH_KEY_FILE) {
        Some(path) => {
            let encoded = std::fs::read_to_string(path)
                .map_err(|_| anyhow::anyhow!("unable to read request trace hash key file"))?;
            Some(Arc::new(ReplayHashKey::from_hex(&encoded)?))
        }
        None => None,
    };
    let _ = HASH_KEY.set(key);
    Ok(())
}

pub(crate) fn replay_metrics(
    token_ids: &[TokenIdType],
    trace_block_size: usize,
) -> Option<RequestReplayMetrics> {
    replay_metrics_with_key(
        token_ids,
        trace_block_size,
        HASH_KEY.get().and_then(Option::as_ref),
    )
}

fn replay_metrics_with_key(
    token_ids: &[TokenIdType],
    trace_block_size: usize,
    key: Option<&Arc<ReplayHashKey>>,
) -> Option<RequestReplayMetrics> {
    if trace_block_size == 0 {
        return None;
    }

    Some(RequestReplayMetrics {
        trace_block_size,
        input_length: token_ids.len(),
        input_sequence_hashes: key.map_or_else(
            || input_sequence_hashes(token_ids, trace_block_size),
            |key| key.input_hashes(token_ids, trace_block_size),
        ),
        output_sequence_hashes: Vec::new(),
        hash_algorithm: key.map(|_| HASH_ALGORITHM.to_string()),
        hash_key_id: key.map(|key| key.id.clone()),
    })
}

pub(crate) type SharedOutputSequenceHashCapture = Arc<Mutex<OutputSequenceHashCapture>>;

/// Captures a generated continuation as sequence hashes while retaining no more
/// than one unfinished KV block of raw token IDs.
pub(crate) struct OutputSequenceHashCapture {
    key: Arc<ReplayHashKey>,
    trace_block_size: usize,
    pending_tokens: Vec<TokenIdType>,
    parent_sequence_hash: Option<u64>,
    output_sequence_hashes: Vec<u64>,
    output_tokens_seen: bool,
}

impl OutputSequenceHashCapture {
    fn new(
        input_tokens: &[TokenIdType],
        replay: &RequestReplayMetrics,
        key: Arc<ReplayHashKey>,
    ) -> Self {
        let input_remainder = input_tokens.len() % replay.trace_block_size;
        let full_input_blocks = input_tokens.len() / replay.trace_block_size;
        let parent_sequence_hash = if input_remainder == 0 {
            replay.input_sequence_hashes.last().copied()
        } else {
            full_input_blocks
                .checked_sub(1)
                .and_then(|index| replay.input_sequence_hashes.get(index).copied())
        };

        Self {
            key,
            trace_block_size: replay.trace_block_size,
            pending_tokens: input_tokens[input_tokens.len() - input_remainder..].to_vec(),
            parent_sequence_hash,
            output_sequence_hashes: Vec::new(),
            output_tokens_seen: false,
        }
    }

    pub(crate) fn record(&mut self, token_ids: &[TokenIdType]) {
        self.output_tokens_seen |= !token_ids.is_empty();
        for &token_id in token_ids {
            self.pending_tokens.push(token_id);
            if self.pending_tokens.len() == self.trace_block_size {
                let block = std::mem::take(&mut self.pending_tokens);
                self.push_block(&block);
            }
        }
    }

    pub(crate) fn sequence_hashes(&self) -> Vec<u64> {
        if !self.output_tokens_seen {
            return Vec::new();
        }

        let mut hashes = self.output_sequence_hashes.clone();
        if !self.pending_tokens.is_empty() {
            let sequence_hash = self.key.block(
                self.parent_sequence_hash,
                &self.pending_tokens,
                self.trace_block_size,
            );
            hashes.push(sequence_hash);
        }
        hashes
    }

    fn push_block(&mut self, tokens: &[TokenIdType]) {
        let sequence_hash =
            self.key
                .block(self.parent_sequence_hash, tokens, self.trace_block_size);
        self.parent_sequence_hash = Some(sequence_hash);
        self.output_sequence_hashes.push(sequence_hash);
    }
}

pub(crate) fn output_sequence_hash_capture(
    input_tokens: &[TokenIdType],
    replay: &RequestReplayMetrics,
) -> Option<SharedOutputSequenceHashCapture> {
    let key = HASH_KEY.get()?.as_ref()?;
    if replay.hash_key_id.as_deref() != Some(key.id.as_str()) {
        return None;
    }
    Some(Arc::new(Mutex::new(OutputSequenceHashCapture::new(
        input_tokens,
        replay,
        key.clone(),
    ))))
}

pub(crate) fn input_sequence_hashes(
    token_ids: &[TokenIdType],
    trace_block_size: usize,
) -> Vec<u64> {
    assert!(
        trace_block_size > 0,
        "request trace replay block size must be positive"
    );

    // Keep this identical to the router/mocker sequence-aware hashing path so
    // replay preserves shared-prefix identity.
    let block_size = trace_block_size as u32;
    let mut block_hashes =
        compute_block_hash_for_seq(token_ids, block_size, BlockHashOptions::default());

    let full_token_count = block_hashes.len() * trace_block_size;
    if full_token_count < token_ids.len() {
        block_hashes.push(partial_local_block_hash(&token_ids[full_token_count..]));
    }

    compute_seq_hash_for_block(&block_hashes)
}

fn partial_local_block_hash(tokens: &[TokenIdType]) -> LocalBlockHash {
    LocalBlockHash(compute_hash_v2(cast_slice(tokens), XXH3_SEED))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn test_key() -> Arc<ReplayHashKey> {
        Arc::new(ReplayHashKey::from_hex(&"01".repeat(32)).unwrap())
    }

    #[test]
    fn keyed_metadata_serializes_without_secrets_and_legacy_stays_compatible() {
        let key = test_key();
        let keyed = replay_metrics_with_key(&[1, 2, 3], 2, Some(&key)).unwrap();
        let json = serde_json::to_value(&keyed).unwrap();
        assert_eq!(json["hash_algorithm"], HASH_ALGORITHM);
        assert_eq!(json["hash_key_id"], key.id);
        assert!(!json.to_string().contains(&"01".repeat(32)));
        assert_eq!(keyed.input_sequence_hashes, key.input_hashes(&[1, 2, 3], 2));

        let legacy = replay_metrics_with_key(&[1, 2, 3], 2, None).unwrap();
        assert_eq!(
            legacy.input_sequence_hashes,
            input_sequence_hashes(&[1, 2, 3], 2)
        );
        assert!(legacy.output_sequence_hashes.is_empty());
        let json = serde_json::to_value(&legacy).unwrap();
        assert!(json.get("hash_key_id").is_none());
        assert!(json.get("hash_algorithm").is_none());
        assert!(json.get("output_sequence_hashes").is_none());
        assert_eq!(
            serde_json::from_value::<RequestReplayMetrics>(json).unwrap(),
            legacy
        );
    }

    fn output_sequence_hash_capture(
        input: &[u32],
        replay: &RequestReplayMetrics,
    ) -> SharedOutputSequenceHashCapture {
        let key = test_key();
        let mut replay = replay.clone();
        replay.input_sequence_hashes = key.input_hashes(input, replay.trace_block_size);
        Arc::new(Mutex::new(OutputSequenceHashCapture::new(
            input, &replay, key,
        )))
    }

    #[test]
    fn shared_prefix_has_same_leading_sequence_hashes() {
        let prefix = vec![1_u32, 2, 3, 4];
        let extended = vec![1_u32, 2, 3, 4, 5, 6];

        let prefix_hashes = input_sequence_hashes(&prefix, 2);
        let extended_hashes = input_sequence_hashes(&extended, 2);

        assert_eq!(prefix_hashes.len(), 2);
        assert_eq!(extended_hashes.len(), 3);
        assert_eq!(extended_hashes[..2], prefix_hashes[..]);
    }

    #[test]
    fn same_tokens_at_different_positions_have_different_sequence_hashes() {
        let hashes = input_sequence_hashes(&[1_u32, 2, 1, 2], 2);

        assert_eq!(hashes.len(), 2);
        assert_ne!(hashes[0], hashes[1]);
    }

    #[test]
    fn empty_input_has_empty_sequence_hashes() {
        assert!(input_sequence_hashes(&[], 64).is_empty());
    }

    #[test]
    fn long_input_hashes_cover_every_token() {
        let tokens = (0..131_072_u32).collect::<Vec<_>>();
        let started = Instant::now();
        let hashes = input_sequence_hashes(&tokens, 64);
        eprintln!(
            "hashed {} input tokens into {} sequence hashes in {:?}",
            tokens.len(),
            hashes.len(),
            started.elapsed()
        );

        assert_eq!(hashes.len(), tokens.len() / 64);
    }

    #[test]
    fn output_hashes_extend_the_input_chain_across_a_partial_block() {
        let input = vec![1_u32, 2, 3];
        let replay = super::replay_metrics(&input, 2).unwrap();
        let capture = output_sequence_hash_capture(&input, &replay);
        capture.lock().unwrap().record(&[4, 5, 6]);

        let mut expected = input;
        expected.extend([4, 5, 6]);
        let expected = test_key().input_hashes(&expected, 2);
        assert_eq!(capture.lock().unwrap().sequence_hashes(), expected[1..]);
    }

    #[test]
    fn output_hashes_keep_no_raw_tokens_after_a_full_block() {
        let input = vec![1_u32, 2];
        let replay = super::replay_metrics(&input, 2).unwrap();
        let capture = output_sequence_hash_capture(&input, &replay);
        capture.lock().unwrap().record(&[3, 4]);

        let capture = capture.lock().unwrap();
        assert!(capture.pending_tokens.is_empty());
        assert_eq!(capture.sequence_hashes().len(), 1);
    }

    #[test]
    fn output_hashes_do_not_depend_on_backend_chunk_boundaries() {
        let input = vec![1_u32, 2, 3];
        let replay = super::replay_metrics(&input, 2).unwrap();
        let one_chunk = output_sequence_hash_capture(&input, &replay);
        one_chunk.lock().unwrap().record(&[4, 5, 6, 7]);

        let many_chunks = output_sequence_hash_capture(&input, &replay);
        many_chunks.lock().unwrap().record(&[4]);
        many_chunks.lock().unwrap().record(&[5, 6]);
        many_chunks.lock().unwrap().record(&[7]);

        assert_eq!(
            one_chunk.lock().unwrap().sequence_hashes(),
            many_chunks.lock().unwrap().sequence_hashes()
        );
    }

    #[test]
    fn rotation_changes_input_and_output_identities() {
        let first = test_key();
        let second = ReplayHashKey::from_hex(&"02".repeat(32)).unwrap();
        assert_ne!(first.id, second.id);
        assert_ne!(
            first.input_hashes(&[1, 2, 3], 2),
            second.input_hashes(&[1, 2, 3], 2)
        );
        assert_ne!(
            first.block(Some(17), &[4], 2),
            second.block(Some(17), &[4], 2)
        );
        assert_ne!(first.block(Some(17), &[4], 2), first.block(None, &[4], 2));
    }

    #[test]
    fn invalid_keys_are_rejected_without_echoing_contents() {
        for invalid in [
            "secret-not-a-key".to_string(),
            "z".repeat(64),
            "é".repeat(32),
        ] {
            let error = ReplayHashKey::from_hex(&invalid).err().unwrap().to_string();
            assert!(!error.contains(&invalid));
        }
    }

    #[test]
    fn keyed_continuations_match_later_inputs_at_every_boundary() {
        let tokens: Vec<u32> = (1..20).collect();
        let key = test_key();
        for input_len in 0..tokens.len() {
            let replay = super::replay_metrics(&tokens[..input_len], 4).unwrap();
            let capture = output_sequence_hash_capture(&tokens[..input_len], &replay);
            capture.lock().unwrap().record(&tokens[input_len..]);
            assert_eq!(
                capture.lock().unwrap().sequence_hashes(),
                key.input_hashes(&tokens, 4)[input_len / 4..]
            );
        }
    }
}
