// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use anyhow::{Result, bail};
use async_trait::async_trait;
use dynamo_tokenizers::{
    Encoding,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Mutex,
};

use super::{
    PrefixHash, RespValue, SharedTokenizerCache, TokenizerCacheEntry, TokenizerCacheHit,
    TokenizerCacheL2Backend, ValkeyTokenizerCache, decode_tokens, encode_tokens, parse_endpoint,
    parse_longest_prefix_response,
};
use crate::valkey_transport::ValkeySentinelConfig;

#[test]
fn valkey_token_value_round_trips_and_rejects_malformed_payloads() {
    let tokens: Arc<[u32]> = vec![0, 1, 255, 65_535, u32::MAX].into();
    let encoded = encode_tokens(&tokens).unwrap();

    assert_eq!(decode_tokens(&encoded).unwrap().as_ref(), tokens.as_ref());
    assert!(decode_tokens(&[]).is_err());
    assert!(decode_tokens(&[99, 0, 0, 0, 0]).is_err());

    let mut truncated = encoded;
    truncated.pop();
    assert!(decode_tokens(&truncated).is_err());

    let mut corrupted = encode_tokens(&tokens).unwrap();
    corrupted[5] ^= 0xff;
    assert!(decode_tokens(&corrupted).is_err());
}

#[test]
fn longest_prefix_response_contains_only_the_selected_value() {
    let tokens: Arc<[u32]> = vec![11, 22, 33].into();
    let response = RespValue::Array(vec![
        RespValue::Integer(1024),
        RespValue::Bulk(encode_tokens(&tokens).unwrap()),
    ]);

    let hit = parse_longest_prefix_response(response, 7, 1024)
        .unwrap()
        .unwrap();

    assert_eq!(hit.candidate_index, 1030);
    assert_eq!(hit.tokens.as_ref(), tokens.as_ref());
    assert!(
        parse_longest_prefix_response(RespValue::Null, 7, 1024)
            .unwrap()
            .is_none()
    );
    assert!(
        parse_longest_prefix_response(
            RespValue::Array(vec![RespValue::Integer(1025), RespValue::Bulk(Vec::new())]),
            7,
            1024,
        )
        .is_err()
    );
}

#[test]
fn tokenizer_valkey_endpoint_validation_rejects_credentials_and_paths() {
    assert_eq!(
        parse_endpoint("valkey://127.0.0.1:6379/").unwrap(),
        "127.0.0.1:6379"
    );
    assert!(parse_endpoint("valkey://user@127.0.0.1:6379").is_err());
    assert!(parse_endpoint("valkey://127.0.0.1:6379/1").is_err());
    assert!(parse_endpoint("localhost").is_err());
}

async fn read_resp_request(stream: &mut TcpStream) -> Vec<Vec<u8>> {
    async fn line(stream: &mut TcpStream) -> Vec<u8> {
        let mut value = Vec::new();
        loop {
            let mut byte = [0_u8; 1];
            stream.read_exact(&mut byte).await.unwrap();
            value.push(byte[0]);
            if value.ends_with(b"\r\n") {
                value.truncate(value.len() - 2);
                return value;
            }
        }
    }

    let header = line(stream).await;
    let count = std::str::from_utf8(&header[1..])
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mut arguments = Vec::with_capacity(count);
    for _ in 0..count {
        let length_line = line(stream).await;
        let length = std::str::from_utf8(&length_line[1..])
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let mut argument = vec![0_u8; length];
        stream.read_exact(&mut argument).await.unwrap();
        let mut terminator = [0_u8; 2];
        stream.read_exact(&mut terminator).await.unwrap();
        assert_eq!(&terminator, b"\r\n");
        arguments.push(argument);
    }
    arguments
}

#[tokio::test]
async fn longest_prefix_lookup_returns_one_value_for_1024_populated_candidates() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let expected_tokens: Arc<[u32]> = (0..4096).collect::<Vec<_>>().into();
    let encoded = encode_tokens(&expected_tokens).unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let arguments = read_resp_request(&mut stream).await;
        assert_eq!(arguments[0], b"EVAL");
        assert!(arguments[1].windows(3).any(|window| window == b"GET"));
        assert_eq!(arguments[2], b"1024");
        assert_eq!(arguments.len(), 1027);
        let response = format!("*2\r\n:1024\r\n${}\r\n", encoded.len());
        stream.write_all(response.as_bytes()).await.unwrap();
        stream.write_all(&encoded).await.unwrap();
        stream.write_all(b"\r\n").await.unwrap();
    });
    let backend = ValkeyTokenizerCache::new(
        &endpoint,
        "dynamo:tokenizer:test:v1",
        60,
        std::time::Duration::from_secs(1),
        1,
    )
    .unwrap();
    let candidates: Vec<PrefixHash> = (0_u16..1024)
        .map(|index| {
            let mut hash = [0_u8; 32];
            hash[..2].copy_from_slice(&index.to_be_bytes());
            hash
        })
        .collect();

    let hit = backend
        .get_longest("aggregate-budget", &candidates)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(hit.candidate_index, 1023);
    assert_eq!(hit.tokens.as_ref(), expected_tokens.as_ref());
    server.await.unwrap();
}

#[tokio::test]
async fn cancelled_partial_reply_is_never_reused_by_the_pool() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let (partial_tx, partial_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        assert_eq!(read_resp_request(&mut first).await, vec![b"PING".to_vec()]);
        first.write_all(b"+STALE").await.unwrap();
        partial_tx.send(()).unwrap();

        let (mut fresh, _) = listener.accept().await.unwrap();
        assert_eq!(read_resp_request(&mut fresh).await, vec![b"PING".to_vec()]);
        fresh.write_all(b"+FRESH\r\n").await.unwrap();
    });
    let backend = Arc::new(
        ValkeyTokenizerCache::new(
            &endpoint,
            "dynamo:tokenizer:test:v1",
            60,
            std::time::Duration::from_millis(100),
            1,
        )
        .unwrap(),
    );

    let abandoned = tokio::spawn({
        let backend = Arc::clone(&backend);
        async move { backend.command(&[b"PING"]).await }
    });
    partial_rx.await.unwrap();
    abandoned.abort();
    assert!(abandoned.await.unwrap_err().is_cancelled());

    let response = backend.command(&[b"PING"]).await.unwrap();

    assert!(matches!(response, RespValue::Simple(value) if value == "FRESH"));
    server.await.unwrap();
}

#[tokio::test]
async fn tokenizer_policy_rejects_excessive_response_nesting() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let _ = read_resp_request(&mut stream).await;
        let response = format!("{}+OK\r\n", "*1\r\n".repeat(9));
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    let backend = ValkeyTokenizerCache::new(
        &endpoint,
        "dynamo:tokenizer:test:v1",
        60,
        std::time::Duration::from_secs(1),
        1,
    )
    .unwrap();

    let error = backend.command(&[b"PING"]).await.unwrap_err();

    assert!(error.to_string().contains("nesting depth"));
    server.await.unwrap();
}

fn sentinel_reply(primary: &str) -> Vec<u8> {
    let (host, port) = primary.rsplit_once(':').unwrap();
    format!(
        "*2\r\n${}\r\n{}\r\n${}\r\n{}\r\n",
        host.len(),
        host,
        port.len(),
        port
    )
    .into_bytes()
}

async fn spawn_tokenizer_sentinel(
    primary_responses: Vec<String>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        for primary in primary_responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            assert_eq!(
                read_resp_request(&mut stream).await,
                vec![
                    b"SENTINEL".to_vec(),
                    b"GET-MASTER-ADDR-BY-NAME".to_vec(),
                    b"dynamo-tokenizer".to_vec(),
                ]
            );
            stream.write_all(&sentinel_reply(&primary)).await.unwrap();
        }
    });
    (endpoint, server)
}

async fn accept_role(listener: &TcpListener) {
    let (mut stream, _) = listener.accept().await.unwrap();
    assert_eq!(read_resp_request(&mut stream).await, vec![b"ROLE".to_vec()]);
    stream.write_all(b"*1\r\n$6\r\nmaster\r\n").await.unwrap();
}

#[tokio::test]
async fn tokenizer_valkey_re_resolves_sentinel_primary_and_retries() {
    let old_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let old_endpoint = old_listener.local_addr().unwrap().to_string();
    let old_server = tokio::spawn(async move {
        accept_role(&old_listener).await;
        let (mut stream, _) = old_listener.accept().await.unwrap();
        assert_eq!(read_resp_request(&mut stream).await, vec![b"PING".to_vec()]);
        drop(stream);
    });
    let new_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let new_endpoint = new_listener.local_addr().unwrap().to_string();
    let new_server = tokio::spawn(async move {
        accept_role(&new_listener).await;
        let (mut stream, _) = new_listener.accept().await.unwrap();
        assert_eq!(read_resp_request(&mut stream).await, vec![b"PING".to_vec()]);
        stream.write_all(b"+PONG\r\n").await.unwrap();
    });

    let mut sentinel_endpoints = Vec::new();
    let mut sentinel_servers = Vec::new();
    for _ in 0..3 {
        let (endpoint, server) =
            spawn_tokenizer_sentinel(vec![old_endpoint.clone(), new_endpoint.clone()]).await;
        sentinel_endpoints.push(endpoint);
        sentinel_servers.push(server);
    }
    let sentinel =
        ValkeySentinelConfig::new(&sentinel_endpoints.join(","), "dynamo-tokenizer", Some(2))
            .unwrap();
    let backend = ValkeyTokenizerCache::new_with_sentinel(
        sentinel,
        "dynamo:tokenizer:test:v1",
        60,
        std::time::Duration::from_secs(2),
        1,
    )
    .unwrap();

    let response = backend.command(&[b"PING"]).await.unwrap();

    assert!(matches!(response, RespValue::Simple(value) if value == "PONG"));
    old_server.await.unwrap();
    new_server.await.unwrap();
    for server in sentinel_servers {
        server.await.unwrap();
    }
}

#[tokio::test]
async fn tokenizer_sentinel_quorum_ignores_one_stalled_witness_at_default_deadline() {
    let primary_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let primary_endpoint = primary_listener.local_addr().unwrap().to_string();
    let primary_server = tokio::spawn(async move {
        accept_role(&primary_listener).await;
        let (mut stream, _) = primary_listener.accept().await.unwrap();
        assert_eq!(read_resp_request(&mut stream).await, vec![b"PING".to_vec()]);
        stream.write_all(b"+PONG\r\n").await.unwrap();
    });

    let stalled_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let stalled_endpoint = stalled_listener.local_addr().unwrap().to_string();
    let stalled_server = tokio::spawn(async move {
        let (mut stream, _) = stalled_listener.accept().await.unwrap();
        let _ = read_resp_request(&mut stream).await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    });
    let mut sentinel_endpoints = vec![stalled_endpoint];
    let mut sentinel_servers = Vec::new();
    for _ in 0..2 {
        let (endpoint, server) = spawn_tokenizer_sentinel(vec![primary_endpoint.clone()]).await;
        sentinel_endpoints.push(endpoint);
        sentinel_servers.push(server);
    }
    let sentinel =
        ValkeySentinelConfig::new(&sentinel_endpoints.join(","), "dynamo-tokenizer", Some(2))
            .unwrap();
    let backend = ValkeyTokenizerCache::new_with_sentinel(
        sentinel,
        "dynamo:tokenizer:test:v1",
        60,
        std::time::Duration::from_millis(20),
        1,
    )
    .unwrap();

    let response = backend.command(&[b"PING"]).await.unwrap();

    assert!(matches!(response, RespValue::Simple(value) if value == "PONG"));
    primary_server.await.unwrap();
    for server in sentinel_servers {
        server.await.unwrap();
    }
    stalled_server.abort();
}

#[tokio::test]
async fn tokenizer_sentinel_failure_is_shared_by_concurrent_lookups() {
    let query_count = Arc::new(AtomicUsize::new(0));
    let mut sentinel_endpoints = Vec::new();
    let mut sentinel_servers = Vec::new();
    for _ in 0..3 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        sentinel_endpoints.push(listener.local_addr().unwrap().to_string());
        let query_count = Arc::clone(&query_count);
        sentinel_servers.push(tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_resp_request(&mut stream).await;
            query_count.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }));
    }
    let sentinel =
        ValkeySentinelConfig::new(&sentinel_endpoints.join(","), "dynamo-tokenizer", Some(2))
            .unwrap();
    let backend = Arc::new(
        ValkeyTokenizerCache::new_with_sentinel(
            sentinel,
            "dynamo:tokenizer:test:v1",
            60,
            std::time::Duration::from_millis(20),
            1,
        )
        .unwrap(),
    );

    let started = std::time::Instant::now();
    let mut lookups = Vec::new();
    for _ in 0..8 {
        let backend = Arc::clone(&backend);
        lookups.push(tokio::spawn(
            async move { backend.command(&[b"PING"]).await },
        ));
    }
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        for lookup in lookups {
            assert!(lookup.await.unwrap().is_err());
        }
    })
    .await
    .unwrap();

    assert!(started.elapsed() < std::time::Duration::from_secs(1));
    assert_eq!(query_count.load(Ordering::SeqCst), 3);
    let cached_started = std::time::Instant::now();
    assert!(backend.command(&[b"PING"]).await.is_err());
    assert!(cached_started.elapsed() < std::time::Duration::from_millis(100));
    assert_eq!(query_count.load(Ordering::SeqCst), 3);
    for server in sentinel_servers {
        server.abort();
    }
}

#[derive(Default)]
struct ByteTokenizer {
    encoded_bytes: AtomicUsize,
}

impl ByteTokenizer {
    fn encoded_bytes(&self) -> usize {
        self.encoded_bytes.load(Ordering::Relaxed)
    }
}

impl Encoder for ByteTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        self.encoded_bytes.fetch_add(input.len(), Ordering::Relaxed);
        Ok(Encoding::Sp(input.bytes().map(u32::from).collect()))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for ByteTokenizer {
    fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<DecodeResult> {
        let bytes = token_ids
            .iter()
            .map(|token| u8::try_from(*token))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(DecodeResult::Complete(String::from_utf8(bytes)?))
    }
}

impl Tokenizer for ByteTokenizer {}

struct CountingTokenizer {
    inner: Arc<dyn Tokenizer>,
    encoded_bytes: AtomicUsize,
}

impl CountingTokenizer {
    fn new(inner: Arc<dyn Tokenizer>) -> Self {
        Self {
            inner,
            encoded_bytes: AtomicUsize::new(0),
        }
    }

    fn encoded_bytes(&self) -> usize {
        self.encoded_bytes.load(Ordering::Relaxed)
    }
}

impl Encoder for CountingTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        self.encoded_bytes.fetch_add(input.len(), Ordering::Relaxed);
        self.inner.encode(input)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.encoded_bytes.fetch_add(
            inputs.iter().map(|input| input.len()).sum(),
            Ordering::Relaxed,
        );
        self.inner.encode_batch(inputs)
    }
}

impl Decoder for CountingTokenizer {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CountingTokenizer {}

#[derive(Default)]
struct MemoryL2 {
    entries: Mutex<HashMap<PrefixHash, Arc<[u32]>>>,
    lookups: AtomicUsize,
    fail_reads: AtomicBool,
}

#[async_trait]
impl TokenizerCacheL2Backend for MemoryL2 {
    async fn get_longest(
        &self,
        _namespace: &str,
        candidates: &[PrefixHash],
    ) -> Result<Option<TokenizerCacheHit>> {
        self.lookups.fetch_add(1, Ordering::Relaxed);
        if self.fail_reads.load(Ordering::Relaxed) {
            bail!("injected L2 failure");
        }
        let entries = self.entries.lock().await;
        Ok(candidates
            .iter()
            .enumerate()
            .rev()
            .find_map(|(candidate_index, hash)| {
                entries.get(hash).map(|tokens| TokenizerCacheHit {
                    candidate_index,
                    tokens: Arc::clone(tokens),
                })
            }))
    }

    async fn put(&self, _namespace: &str, entries: Vec<TokenizerCacheEntry>) -> Result<()> {
        let mut stored = self.entries.lock().await;
        stored.extend(entries.into_iter().map(|entry| (entry.hash, entry.tokens)));
        Ok(())
    }
}

fn cache(tokenizer: Arc<ByteTokenizer>, l2: Arc<MemoryL2>) -> SharedTokenizerCache {
    let tokenizer: Arc<dyn Tokenizer> = tokenizer;
    let l2: Arc<dyn TokenizerCacheL2Backend> = l2;
    SharedTokenizerCache::new(
        tokenizer,
        vec!["<s>".to_string(), "</s>".to_string()],
        "test-tokenizer",
        Some(l2),
    )
    .unwrap()
}

#[tokio::test]
async fn large_growing_prefix_hits_l1_then_hydrates_another_frontend_from_l2() {
    let shared_l2 = Arc::new(MemoryL2::default());
    let first_tokenizer = Arc::new(ByteTokenizer::default());
    let first_frontend = cache(Arc::clone(&first_tokenizer), Arc::clone(&shared_l2));
    let shared_prefix = format!("<s>{}</s>", "large-prefix-".repeat(32 * 1024));
    let first_prompt = format!("{shared_prefix}first request");
    let second_prompt = format!("{shared_prefix}second request");

    let first = first_frontend.encode(&first_prompt).await.unwrap();
    assert_eq!(
        first.token_ids(),
        first_prompt.bytes().map(u32::from).collect::<Vec<_>>()
    );
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if !shared_l2.entries.lock().await.is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("write-behind should publish the shared prefix");
    let lookups_after_first = shared_l2.lookups.load(Ordering::Relaxed);

    let second = first_frontend.encode(&second_prompt).await.unwrap();
    assert_eq!(
        second.token_ids(),
        second_prompt.bytes().map(u32::from).collect::<Vec<_>>()
    );
    assert_eq!(
        shared_l2.lookups.load(Ordering::Relaxed),
        lookups_after_first,
        "a growing prefix already present in local L1 must not query Valkey"
    );

    let second_tokenizer = Arc::new(ByteTokenizer::default());
    let second_frontend = cache(Arc::clone(&second_tokenizer), Arc::clone(&shared_l2));
    let third = second_frontend.encode(&second_prompt).await.unwrap();
    assert_eq!(
        third.token_ids(),
        second_prompt.bytes().map(u32::from).collect::<Vec<_>>()
    );
    assert!(
        second_tokenizer.encoded_bytes() < second_prompt.len() / 100,
        "a cold frontend should tokenize only the suffix after its shared L2 hit"
    );
}

#[tokio::test]
async fn deeper_l2_prefix_wins_over_an_older_l1_prefix() {
    let shared_l2 = Arc::new(MemoryL2::default());
    let local_tokenizer = Arc::new(ByteTokenizer::default());
    let local = cache(Arc::clone(&local_tokenizer), Arc::clone(&shared_l2));
    let old_prompt = "<s>old context</s>local tail";
    let growing_prompt = "<s>old context</s><s>new shared turn</s>latest tail";
    local.encode(old_prompt).await.unwrap();

    let remote_tokenizer = Arc::new(ByteTokenizer::default());
    let remote = cache(remote_tokenizer, Arc::clone(&shared_l2));
    remote.encode(growing_prompt).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if shared_l2.entries.lock().await.len() >= 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("both old and growing prefixes should reach L2");

    let lookups_before = shared_l2.lookups.load(Ordering::Relaxed);
    let encoded_bytes_before = local_tokenizer.encoded_bytes();
    let encoded = local.encode(growing_prompt).await.unwrap();

    assert_eq!(
        encoded.token_ids(),
        growing_prompt.bytes().map(u32::from).collect::<Vec<_>>()
    );
    assert!(shared_l2.lookups.load(Ordering::Relaxed) > lookups_before);
    assert!(
        local_tokenizer.encoded_bytes() - encoded_bytes_before < "latest tail".len() * 2,
        "the deeper shared prefix should avoid re-tokenizing the intervening turn"
    );
}

#[tokio::test]
async fn shared_l2_bounds_tokenization_work_across_many_growing_turns() {
    let shared_l2 = Arc::new(MemoryL2::default());
    let mut history = format!("<s>system\n{}</s>", "shared-system-context ".repeat(4096));
    let mut uncached_input_bytes = 0_usize;
    let mut cached_input_bytes = 0_usize;

    for turn in 0..32 {
        history.push_str(&format!(
            "<s>user\nturn {turn}: {}</s><s>assistant\n",
            "growing user context ".repeat(256)
        ));
        let prompt = format!("{history}answer for turn {turn}");
        uncached_input_bytes += prompt.len();

        let tokenizer = Arc::new(ByteTokenizer::default());
        let frontend = cache(Arc::clone(&tokenizer), Arc::clone(&shared_l2));
        let entries_before = shared_l2.entries.lock().await.len();
        let encoded = frontend.encode(&prompt).await.unwrap();
        cached_input_bytes += tokenizer.encoded_bytes();
        assert_eq!(
            encoded.token_ids(),
            prompt.bytes().map(u32::from).collect::<Vec<_>>()
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if shared_l2.entries.lock().await.len() > entries_before {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("each growing turn should publish its deepest boundary");
        history.push_str(&format!("answer for turn {turn}</s>"));
    }

    assert!(
        cached_input_bytes * 5 < uncached_input_bytes,
        "shared-prefix tokenization should consume under 20% of full re-tokenization work; cached={cached_input_bytes}, uncached={uncached_input_bytes}"
    );
}

#[tokio::test]
async fn l2_failure_falls_back_to_local_tokenization() {
    let shared_l2 = Arc::new(MemoryL2::default());
    shared_l2.fail_reads.store(true, Ordering::Relaxed);
    let tokenizer = Arc::new(ByteTokenizer::default());
    let frontend = cache(Arc::clone(&tokenizer), shared_l2);
    let prompt = format!("<s>{}</s>tail", "failure-safe-prefix".repeat(1024));

    let encoded = frontend.encode(&prompt).await.unwrap();

    assert_eq!(
        encoded.token_ids(),
        prompt.bytes().map(u32::from).collect::<Vec<_>>()
    );
    assert_eq!(tokenizer.encoded_bytes(), prompt.len());
}

#[tokio::test]
async fn unavailable_or_stalled_valkey_falls_back_within_the_lookup_deadline() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let stalled_server = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    });
    let backend = Arc::new(
        ValkeyTokenizerCache::new(
            &endpoint,
            "dynamo:tokenizer:test:v1",
            60,
            std::time::Duration::from_millis(10),
            1,
        )
        .unwrap(),
    );
    let l2: Arc<dyn TokenizerCacheL2Backend> = backend;
    let tokenizer = Arc::new(ByteTokenizer::default());
    let inner: Arc<dyn Tokenizer> = tokenizer.clone();
    let frontend = SharedTokenizerCache::new(
        inner,
        vec!["<s>".to_string(), "</s>".to_string()],
        "stalled-valkey-test",
        Some(l2),
    )
    .unwrap();
    let prompt = format!("<s>{}</s>tail", "deadline-prefix".repeat(1024));
    let started = std::time::Instant::now();

    let encoded = frontend.encode(&prompt).await.unwrap();

    stalled_server.abort();
    assert_eq!(
        encoded.token_ids(),
        prompt.bytes().map(u32::from).collect::<Vec<_>>()
    );
    assert_eq!(tokenizer.encoded_bytes(), prompt.len());
    assert!(started.elapsed() < std::time::Duration::from_millis(250));
}

#[tokio::test]
async fn real_hf_tokenizer_matches_uncached_for_a_large_shared_prefix() {
    let tokenizer_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
    );
    let load_tokenizer = || -> Arc<dyn Tokenizer> {
        Arc::new(
            dynamo_tokenizers::HuggingFaceTokenizer::from_file(tokenizer_path)
                .expect("load TinyLlama tokenizer"),
        )
    };
    let shared_l2 = Arc::new(MemoryL2::default());
    let first_counting = Arc::new(CountingTokenizer::new(load_tokenizer()));
    let first_inner: Arc<dyn Tokenizer> = first_counting;
    let first_l2: Arc<dyn TokenizerCacheL2Backend> = shared_l2.clone();
    let first_frontend = SharedTokenizerCache::new(
        first_inner,
        vec!["<s>".to_string(), "</s>".to_string()],
        "real-tokenizer-test",
        Some(first_l2),
    )
    .unwrap();
    let shared_prefix = format!(
        "<s>system\n{}</s><s>user\n",
        "A long shared document used by every request. ".repeat(4096)
    );
    let first_prompt = format!("{shared_prefix}first question</s>");
    let second_prompt = format!("{shared_prefix}second question</s>");
    first_frontend.encode(&first_prompt).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if !shared_l2.entries.lock().await.is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let expected = load_tokenizer().encode(&second_prompt).unwrap();
    let second_counting = Arc::new(CountingTokenizer::new(load_tokenizer()));
    let second_inner: Arc<dyn Tokenizer> = second_counting.clone();
    let second_l2: Arc<dyn TokenizerCacheL2Backend> = shared_l2;
    let second_frontend = SharedTokenizerCache::new(
        second_inner,
        vec!["<s>".to_string(), "</s>".to_string()],
        "real-tokenizer-test",
        Some(second_l2),
    )
    .unwrap();

    let encoded = second_frontend.encode(&second_prompt).await.unwrap();

    assert_eq!(encoded.token_ids(), expected.token_ids());
    assert!(
        second_counting.encoded_bytes() < second_prompt.len() / 100,
        "the cold frontend should only tokenize the new user suffix"
    );
}

#[tokio::test]
#[ignore = "requires DYN_TOKENIZER_CACHE_TEST_URL pointing to a disposable Valkey server"]
async fn live_valkey_shares_a_large_growing_prefix_between_frontends() {
    let url = std::env::var("DYN_TOKENIZER_CACHE_TEST_URL")
        .expect("DYN_TOKENIZER_CACHE_TEST_URL is required");
    let namespace = format!("test-{}", std::process::id());
    let backend = Arc::new(
        ValkeyTokenizerCache::new(
            &url,
            "dynamo:tokenizer:test:v1",
            60,
            std::time::Duration::from_millis(250),
            4,
        )
        .unwrap(),
    );
    let first_tokenizer = Arc::new(ByteTokenizer::default());
    let first_l2: Arc<dyn TokenizerCacheL2Backend> = backend.clone();
    let first_frontend = SharedTokenizerCache::new(
        first_tokenizer,
        vec!["<s>".to_string(), "</s>".to_string()],
        namespace.clone(),
        Some(first_l2),
    )
    .unwrap();
    let shared_prefix = format!("<s>{}</s>", "large-prefix-".repeat(16 * 1024));
    let first_prompt = format!("{shared_prefix}first request");
    let second_prompt = format!("{shared_prefix}second request");

    first_frontend.encode(&first_prompt).await.unwrap();
    let candidates = first_frontend.candidates(&second_prompt);
    let hashes: Vec<PrefixHash> = candidates.iter().map(|candidate| candidate.hash).collect();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if backend
                .get_longest(&namespace, &hashes)
                .await
                .unwrap()
                .is_some()
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("write-behind should publish to Valkey");

    let second_tokenizer = Arc::new(ByteTokenizer::default());
    let second_l2: Arc<dyn TokenizerCacheL2Backend> = backend;
    let second_frontend = SharedTokenizerCache::new(
        second_tokenizer.clone(),
        vec!["<s>".to_string(), "</s>".to_string()],
        namespace,
        Some(second_l2),
    )
    .unwrap();
    let encoded = second_frontend.encode(&second_prompt).await.unwrap();

    assert_eq!(
        encoded.token_ids(),
        second_prompt.bytes().map(u32::from).collect::<Vec<_>>()
    );
    assert!(second_tokenizer.encoded_bytes() < second_prompt.len() / 100);
}
