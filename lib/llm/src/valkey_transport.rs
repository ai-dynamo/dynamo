// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared bounded RESP transport and Sentinel discovery for Valkey clients.

use std::{io, sync::Arc, time::Duration};

use anyhow::{Result, bail};
use futures::{StreamExt, stream};
use rustc_hash::FxHashMap;
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufStream},
    net::TcpStream,
    time::timeout,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(4);
const SENTINEL_QUERY_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_VALKEY_SENTINEL_ENDPOINTS: usize = 16;
const SENTINEL_QUERY_CONCURRENCY: usize = 4;
const MAX_RESP_NESTING_DEPTH: usize = 8;
const ROUTER_MAX_RESP_LINE: usize = 1024 * 1024;
const ROUTER_MAX_RESP_BULK: usize = 64 * 1024 * 1024;
const ROUTER_MAX_RESP_ARRAY_ITEMS: usize = 1024 * 1024;
const ROUTER_MAX_RESP_BYTES: usize = 64 * 1024 * 1024;
const SENTINEL_ADDRESS_RESP_POLICY: RespPolicy =
    RespPolicy::bounded(SENTINEL_QUERY_TIMEOUT, 256, 2_048, 2, 4_096);
const SENTINEL_ROLE_RESP_POLICY: RespPolicy =
    RespPolicy::bounded(SENTINEL_QUERY_TIMEOUT, 512, 2_048, 1_024, 64 * 1_024);

#[derive(Clone, Copy)]
pub(crate) struct RespPolicy {
    connect_timeout: Duration,
    command_timeout: Duration,
    max_line_bytes: usize,
    max_bulk_bytes: usize,
    max_array_items: usize,
    max_response_bytes: usize,
    max_nesting_depth: usize,
}

impl RespPolicy {
    pub(crate) const fn bounded(
        command_timeout: Duration,
        max_line_bytes: usize,
        max_bulk_bytes: usize,
        max_array_items: usize,
        max_response_bytes: usize,
    ) -> Self {
        Self {
            connect_timeout: CONNECT_TIMEOUT,
            command_timeout,
            max_line_bytes,
            max_bulk_bytes,
            max_array_items,
            max_response_bytes,
            max_nesting_depth: MAX_RESP_NESTING_DEPTH,
        }
    }
}

impl Default for RespPolicy {
    fn default() -> Self {
        Self::bounded(
            COMMAND_TIMEOUT,
            ROUTER_MAX_RESP_LINE,
            ROUTER_MAX_RESP_BULK,
            ROUTER_MAX_RESP_ARRAY_ITEMS,
            ROUTER_MAX_RESP_BYTES,
        )
    }
}

#[derive(Debug, Error)]
pub(crate) enum RespError {
    #[error("Valkey server error: {0}")]
    Server(String),
    #[error("Valkey I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Valkey protocol error: {0}")]
    Protocol(String),
    #[error("Valkey command timed out")]
    Timeout,
}

#[derive(Debug)]
pub(crate) enum RespValue {
    Simple(String),
    Bulk(Vec<u8>),
    Integer(i64),
    Array(Vec<RespValue>),
    Null,
}

pub(crate) struct RespConnection {
    stream: BufStream<TcpStream>,
    pub(crate) endpoint: Arc<str>,
    pub(crate) topology_generation: u64,
    policy: RespPolicy,
}

impl RespConnection {
    pub(crate) async fn connect(
        endpoint: &str,
        topology_generation: u64,
    ) -> std::result::Result<Self, RespError> {
        Self::connect_with_policy(endpoint, topology_generation, RespPolicy::default()).await
    }

    pub(crate) async fn connect_with_policy(
        endpoint: &str,
        topology_generation: u64,
        policy: RespPolicy,
    ) -> std::result::Result<Self, RespError> {
        let stream = timeout(policy.connect_timeout, TcpStream::connect(endpoint))
            .await
            .map_err(|_| RespError::Timeout)??;
        stream.set_nodelay(true)?;
        Ok(Self {
            stream: BufStream::new(stream),
            endpoint: Arc::from(endpoint),
            topology_generation,
            policy,
        })
    }

    pub(crate) async fn command(
        &mut self,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        let mut request = Vec::with_capacity(request_capacity(arguments));
        append_request(&mut request, arguments);
        timeout(self.policy.command_timeout, async {
            self.stream.write_all(&request).await?;
            self.stream.flush().await?;
            self.read_response().await
        })
        .await
        .map_err(|_| RespError::Timeout)?
    }

    pub(crate) async fn command_pipeline(
        &mut self,
        commands: &[&[&[u8]]],
    ) -> std::result::Result<Vec<RespValue>, RespError> {
        if commands.is_empty() {
            return Err(RespError::Protocol(
                "Valkey command pipeline must not be empty".to_string(),
            ));
        }
        let mut request = Vec::with_capacity(
            commands
                .iter()
                .map(|arguments| request_capacity(arguments))
                .sum(),
        );
        for arguments in commands {
            append_request(&mut request, arguments);
        }
        timeout(self.policy.command_timeout, async {
            self.stream.write_all(&request).await?;
            self.stream.flush().await?;
            let mut responses = Vec::with_capacity(commands.len());
            for _ in commands {
                responses.push(self.read_response().await?);
            }
            Ok(responses)
        })
        .await
        .map_err(|_| RespError::Timeout)?
    }

    async fn read_response(&mut self) -> std::result::Result<RespValue, RespError> {
        let mut remaining = self.policy.max_response_bytes;
        self.read_response_bounded(&mut remaining, 0).await
    }

    async fn read_response_bounded(
        &mut self,
        remaining: &mut usize,
        depth: usize,
    ) -> std::result::Result<RespValue, RespError> {
        let mut marker = [0_u8; 1];
        self.read_exact_bounded(&mut marker, remaining).await?;
        match marker[0] {
            b'+' => Ok(RespValue::Simple(self.read_line(remaining).await?)),
            b'-' => Err(RespError::Server(self.read_line(remaining).await?)),
            b':' => {
                let line = self.read_line(remaining).await?;
                let value = line.parse::<i64>().map_err(|error| {
                    RespError::Protocol(format!("invalid integer reply {line:?}: {error}"))
                })?;
                Ok(RespValue::Integer(value))
            }
            b'$' => self.read_bulk(remaining).await,
            b'*' => {
                if depth >= self.policy.max_nesting_depth {
                    return Err(RespError::Protocol(format!(
                        "RESP array nesting depth exceeds maximum {}",
                        self.policy.max_nesting_depth
                    )));
                }
                self.read_array(remaining, depth + 1).await
            }
            marker => Err(RespError::Protocol(format!(
                "unsupported RESP reply marker 0x{marker:02x}"
            ))),
        }
    }

    async fn read_bulk(
        &mut self,
        remaining: &mut usize,
    ) -> std::result::Result<RespValue, RespError> {
        let Some(length) = self
            .read_length("bulk", self.policy.max_bulk_bytes, remaining)
            .await?
        else {
            return Ok(RespValue::Null);
        };
        let mut payload = vec![0_u8; length];
        self.read_exact_bounded(&mut payload, remaining).await?;
        self.read_terminator("bulk", remaining).await?;
        Ok(RespValue::Bulk(payload))
    }

    async fn read_array(
        &mut self,
        remaining: &mut usize,
        depth: usize,
    ) -> std::result::Result<RespValue, RespError> {
        let Some(length) = self
            .read_length("array", self.policy.max_array_items, remaining)
            .await?
        else {
            return Ok(RespValue::Null);
        };
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            values.push(Box::pin(self.read_response_bounded(remaining, depth)).await?);
        }
        Ok(RespValue::Array(values))
    }

    async fn read_length(
        &mut self,
        kind: &str,
        maximum: usize,
        remaining: &mut usize,
    ) -> std::result::Result<Option<usize>, RespError> {
        let line = self.read_line(remaining).await?;
        let length = line.parse::<isize>().map_err(|error| {
            RespError::Protocol(format!("invalid {kind} length {line:?}: {error}"))
        })?;
        if length == -1 {
            return Ok(None);
        }
        if length < 0 || length as usize > maximum {
            return Err(RespError::Protocol(format!(
                "invalid RESP {kind} length {length}; maximum is {maximum}"
            )));
        }
        Ok(Some(length as usize))
    }

    async fn read_line(&mut self, remaining: &mut usize) -> std::result::Result<String, RespError> {
        let mut line = Vec::new();
        loop {
            if line.len() >= self.policy.max_line_bytes {
                return Err(RespError::Protocol(
                    "RESP line exceeds maximum length".to_string(),
                ));
            }
            let mut byte = [0_u8; 1];
            self.read_exact_bounded(&mut byte, remaining).await?;
            line.push(byte[0]);
            if line.ends_with(b"\r\n") {
                line.truncate(line.len() - 2);
                return String::from_utf8(line)
                    .map_err(|error| RespError::Protocol(format!("non-UTF8 RESP line: {error}")));
            }
        }
    }

    async fn read_terminator(
        &mut self,
        kind: &str,
        remaining: &mut usize,
    ) -> std::result::Result<(), RespError> {
        let mut terminator = [0_u8; 2];
        self.read_exact_bounded(&mut terminator, remaining).await?;
        if terminator != *b"\r\n" {
            return Err(RespError::Protocol(format!(
                "{kind} reply lacks CRLF terminator"
            )));
        }
        Ok(())
    }

    async fn read_exact_bounded(
        &mut self,
        destination: &mut [u8],
        remaining: &mut usize,
    ) -> std::result::Result<(), RespError> {
        *remaining = remaining.checked_sub(destination.len()).ok_or_else(|| {
            RespError::Protocol(format!(
                "RESP reply exceeds aggregate {} byte limit",
                self.policy.max_response_bytes
            ))
        })?;
        self.stream.read_exact(destination).await?;
        Ok(())
    }
}

fn request_capacity(arguments: &[&[u8]]) -> usize {
    32 + arguments
        .iter()
        .map(|argument| argument.len() + 32)
        .sum::<usize>()
}

fn append_request(request: &mut Vec<u8>, arguments: &[&[u8]]) {
    request.extend_from_slice(format!("*{}\r\n", arguments.len()).as_bytes());
    for argument in arguments {
        request.extend_from_slice(format!("${}\r\n", argument.len()).as_bytes());
        request.extend_from_slice(argument);
        request.extend_from_slice(b"\r\n");
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ValkeySentinelConfig {
    pub(crate) endpoints: Arc<[String]>,
    pub(crate) master_name: Arc<[u8]>,
    pub(crate) quorum: usize,
}

impl ValkeySentinelConfig {
    pub(crate) fn new(urls: &str, master_name: &str, quorum: Option<usize>) -> Result<Self> {
        let mut endpoints = Vec::new();
        for endpoint in urls
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            if endpoints.len() == MAX_VALKEY_SENTINEL_ENDPOINTS {
                bail!(
                    "Valkey Sentinel URLs must contain at most {MAX_VALKEY_SENTINEL_ENDPOINTS} endpoints"
                );
            }
            endpoints.push(parse_endpoint(endpoint)?);
        }
        if endpoints.is_empty() {
            bail!("Valkey Sentinel URLs must contain at least one endpoint");
        }
        let mut unique = FxHashMap::default();
        for endpoint in &endpoints {
            if unique.insert(endpoint.as_str(), ()).is_some() {
                bail!("Valkey Sentinel endpoints must be distinct; duplicate {endpoint:?}");
            }
        }
        if master_name.is_empty()
            || master_name.len() > 128
            || master_name.bytes().any(|byte| byte.is_ascii_whitespace())
        {
            bail!("Valkey Sentinel master name must be one non-empty token of at most 128 bytes");
        }
        let quorum = quorum.unwrap_or(endpoints.len() / 2 + 1);
        if quorum == 0 || quorum > endpoints.len() {
            bail!(
                "Valkey Sentinel quorum must be in 1..={}; got {quorum}",
                endpoints.len()
            );
        }
        if quorum <= endpoints.len() / 2 {
            bail!(
                "Valkey Sentinel resolver quorum must be a strict majority of {} endpoints; got {quorum}",
                endpoints.len()
            );
        }
        Ok(Self {
            endpoints: endpoints.into(),
            master_name: Arc::from(master_name.as_bytes()),
            quorum,
        })
    }

    pub(crate) fn validate_degraded_writes(&self) -> Result<()> {
        if self.endpoints.len() < 3 {
            bail!(
                "degraded Valkey writes require at least three distinct Sentinel witnesses; got {}",
                self.endpoints.len()
            );
        }
        Ok(())
    }

    async fn query_endpoint(&self, endpoint: &str) -> std::result::Result<String, RespError> {
        timeout(SENTINEL_QUERY_TIMEOUT, async {
            let mut connection =
                RespConnection::connect_with_policy(endpoint, 0, SENTINEL_ADDRESS_RESP_POLICY)
                    .await?;
            let response = connection
                .command(&[
                    b"SENTINEL",
                    b"GET-MASTER-ADDR-BY-NAME",
                    self.master_name.as_ref(),
                ])
                .await?;
            decode_sentinel_primary(response)
        })
        .await
        .map_err(|_| RespError::Timeout)?
    }

    pub(crate) async fn resolve_primary(&self) -> std::result::Result<String, RespError> {
        let mut queries = stream::iter(self.endpoints.iter().cloned())
            .map(|endpoint| async move { self.query_endpoint(&endpoint).await })
            .buffer_unordered(SENTINEL_QUERY_CONCURRENCY);
        let mut votes = FxHashMap::<String, usize>::default();
        let mut failures = 0_usize;
        while let Some(result) = queries.next().await {
            match result {
                Ok(endpoint) => {
                    let vote_count = votes.entry(endpoint.clone()).or_default();
                    *vote_count += 1;
                    if *vote_count >= self.quorum {
                        return Ok(endpoint);
                    }
                }
                Err(error) => {
                    failures += 1;
                    tracing::debug!(error = %error, "Valkey Sentinel primary query failed");
                }
            }
        }
        votes
            .iter()
            .max_by_key(|(_, votes)| **votes)
            .filter(|(_, votes)| **votes >= self.quorum)
            .map(|(endpoint, _)| endpoint.clone())
            .ok_or_else(|| {
                RespError::Protocol(format!(
                    "Valkey Sentinel quorum not reached: required {}, votes {votes:?}, failures {failures}",
                    self.quorum
                ))
            })
    }

    pub(crate) async fn resolve_validated_primary(&self) -> std::result::Result<String, RespError> {
        let endpoint = self.resolve_primary().await?;
        timeout(SENTINEL_QUERY_TIMEOUT, async {
            let mut connection =
                RespConnection::connect_with_policy(&endpoint, 0, SENTINEL_ROLE_RESP_POLICY)
                    .await?;
            validate_primary_role_response(connection.command(&[b"ROLE"]).await?)?;
            Ok::<String, RespError>(endpoint)
        })
        .await
        .map_err(|_| RespError::Timeout)?
    }

    pub(crate) async fn resolve_validated_primary_endpoint(&self) -> Result<String> {
        self.resolve_validated_primary()
            .await
            .map_err(anyhow::Error::from)
    }
}

fn decode_sentinel_primary(response: RespValue) -> std::result::Result<String, RespError> {
    let RespValue::Array(mut fields) = response else {
        return Err(RespError::Protocol(
            "Valkey Sentinel returned a non-array primary address".to_string(),
        ));
    };
    if fields.len() != 2 {
        return Err(RespError::Protocol(format!(
            "Valkey Sentinel returned {} primary-address fields; expected 2",
            fields.len()
        )));
    }
    let port = fields.pop().expect("two fields checked");
    let host = fields.pop().expect("two fields checked");
    let (RespValue::Bulk(host), RespValue::Bulk(port)) = (host, port) else {
        return Err(RespError::Protocol(
            "Valkey Sentinel primary address must contain bulk strings".to_string(),
        ));
    };
    let host = std::str::from_utf8(&host)
        .map_err(|error| RespError::Protocol(format!("non-UTF8 Sentinel host: {error}")))?;
    if host.is_empty() || host.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(RespError::Protocol(
            "Valkey Sentinel returned an invalid primary host".to_string(),
        ));
    }
    let port = std::str::from_utf8(&port)
        .map_err(|error| RespError::Protocol(format!("non-UTF8 Sentinel port: {error}")))?
        .parse::<u16>()
        .map_err(|error| RespError::Protocol(format!("invalid Sentinel port: {error}")))?;
    if port == 0 {
        return Err(RespError::Protocol(
            "Valkey Sentinel returned primary port zero".to_string(),
        ));
    }
    Ok(
        if host.contains(':') && !(host.starts_with('[') && host.ends_with(']')) {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        },
    )
}

pub(crate) fn validate_primary_role_response(
    response: RespValue,
) -> std::result::Result<(), RespError> {
    let RespValue::Array(fields) = response else {
        return Err(RespError::Protocol(
            "Valkey ROLE returned a non-array reply".to_string(),
        ));
    };
    let Some(role) = fields.first() else {
        return Err(RespError::Protocol(
            "Valkey ROLE returned an empty reply".to_string(),
        ));
    };
    let is_master = match role {
        RespValue::Bulk(role) => role.as_slice() == b"master",
        RespValue::Simple(role) => role == "master",
        _ => false,
    };
    if !is_master {
        return Err(RespError::Server(
            "DYNKV_NOT_PRIMARY Sentinel candidate failed ROLE validation".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn parse_endpoint(url: &str) -> Result<String> {
    let endpoint = url
        .strip_prefix("valkey://")
        .or_else(|| url.strip_prefix("redis://"))
        .unwrap_or(url);
    let endpoint = endpoint.strip_suffix('/').unwrap_or(endpoint);
    if endpoint.contains('@') {
        bail!(
            "invalid Valkey endpoint {url:?}; expected host:port or valkey://host:port without credentials"
        );
    }
    if endpoint.contains('/') {
        bail!("invalid Valkey endpoint {url:?}; endpoint URLs must not contain paths");
    }
    if endpoint.contains(',') {
        bail!("invalid Valkey endpoint {url:?}; expected one host:port endpoint");
    }
    if endpoint.is_empty()
        || endpoint.len() > 2_048
        || endpoint.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        bail!("invalid Valkey endpoint {url:?}; expected host:port");
    }
    let (host, port) = endpoint
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("invalid Valkey endpoint {url:?}; expected host:port"))?;
    let port = port.parse::<u16>().map_err(|_| {
        anyhow::anyhow!("invalid Valkey endpoint {url:?}; port must be in 1..=65535")
    })?;
    if host.is_empty() || port == 0 {
        bail!("invalid Valkey endpoint {url:?}; host must be non-empty and port in 1..=65535");
    }
    Ok(endpoint.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use tokio::{io::AsyncWriteExt, net::TcpListener};

    async fn assert_nested_reply_is_rejected(policy: RespPolicy) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let response = format!("{}+OK\r\n", "*1\r\n".repeat(9));
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let mut connection = RespConnection::connect_with_policy(&endpoint, 0, policy)
            .await
            .unwrap();

        let error = connection.command(&[b"PING"]).await.unwrap_err();

        assert!(error.to_string().contains("nesting depth"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn router_policy_rejects_excessive_response_nesting() {
        assert_nested_reply_is_rejected(RespPolicy::default()).await;
    }

    #[tokio::test]
    async fn sentinel_policies_reject_excessive_response_nesting() {
        assert_nested_reply_is_rejected(SENTINEL_ADDRESS_RESP_POLICY).await;
        assert_nested_reply_is_rejected(SENTINEL_ROLE_RESP_POLICY).await;
    }

    #[test]
    fn sentinel_config_rejects_unbounded_endpoint_fanout() {
        let urls = (0..17)
            .map(|index| format!("127.0.0.1:{}", 20_000 + index))
            .collect::<Vec<_>>()
            .join(",");

        let error = ValkeySentinelConfig::new(&urls, "router", None).unwrap_err();

        assert!(error.to_string().contains("at most 16"));
    }

    #[tokio::test]
    async fn sentinel_address_reply_uses_a_command_specific_budget() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream.write_all(b"*2\r\n$2049\r\n").await.unwrap();
        });
        let config = ValkeySentinelConfig::new(&endpoint, "router", None).unwrap();

        let error = config.query_endpoint(&endpoint).await.unwrap_err();

        assert!(error.to_string().contains("maximum is 2048"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn sentinel_role_reply_uses_a_command_specific_budget() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream.write_all(b"*1\r\n$2049\r\n").await.unwrap();
        });
        let mut connection =
            RespConnection::connect_with_policy(&endpoint, 0, SENTINEL_ROLE_RESP_POLICY)
                .await
                .unwrap();

        let error = connection.command(&[b"ROLE"]).await.unwrap_err();

        assert!(error.to_string().contains("maximum is 2048"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn sentinel_queries_have_bounded_parallelism() {
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let mut endpoints = Vec::new();
        let mut servers = Vec::new();
        for _ in 0..5 {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            endpoints.push(listener.local_addr().unwrap().to_string());
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            servers.push(tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(25)).await;
                stream
                    .write_all(b"*2\r\n$9\r\n127.0.0.1\r\n$4\r\n6379\r\n")
                    .await
                    .unwrap();
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        let config = ValkeySentinelConfig::new(&endpoints.join(","), "router", Some(3)).unwrap();

        assert_eq!(config.resolve_primary().await.unwrap(), "127.0.0.1:6379");
        assert_eq!(maximum.load(Ordering::SeqCst), SENTINEL_QUERY_CONCURRENCY);
        for server in servers {
            // Quorum resolution intentionally drops outstanding witness
            // queries. Some listeners may therefore never be accepted.
            server.abort();
            let _ = server.await;
        }
    }

    #[test]
    fn endpoint_parser_reports_credentials_paths_and_lists_at_the_boundary() {
        assert!(
            parse_endpoint("valkey://user@host:6379")
                .unwrap_err()
                .to_string()
                .contains("credentials")
        );
        assert!(
            parse_endpoint("valkey://host:6379/1")
                .unwrap_err()
                .to_string()
                .contains("paths")
        );
        assert!(
            parse_endpoint("host:6379,other:6379")
                .unwrap_err()
                .to_string()
                .contains("one host:port")
        );
    }

    #[tokio::test]
    async fn aggregate_response_budget_bounds_nested_bulk_values() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream
                .write_all(b"*2\r\n$8\r\n12345678\r\n$8\r\nabcdefgh\r\n")
                .await
                .unwrap();
        });
        let policy = RespPolicy::bounded(Duration::from_secs(1), 64, 16, 2, 24);
        let mut connection = RespConnection::connect_with_policy(&endpoint, 0, policy)
            .await
            .unwrap();

        let error = connection.command(&[b"PING"]).await.unwrap_err();

        assert!(error.to_string().contains("aggregate 24 byte limit"));
        server.await.unwrap();
    }
}
