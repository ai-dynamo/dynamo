// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fetch staged sources under both the Dynamo and native encoder URL policies.

use std::{collections::HashSet, time::Duration};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_backend_common::DynamoError;
use dynamo_llm::preprocessor::media::{MediaFetcher, max_data_url_bytes};
use dynamo_sidecar_common::DEFAULT_MAX_GRPC_MESSAGE_SIZE;
use prost::Message;

use crate::client;
use crate::proto::encoder::{StageRequest, StagedMediaPolicy};

#[derive(Clone)]
pub(crate) struct StagedMedia {
    policy: MediaFetcher,
    http: reqwest::Client,
    max_bytes: usize,
    timeout: Option<Duration>,
}

impl StagedMedia {
    pub(crate) fn new(native: &StagedMediaPolicy) -> Result<Self, DynamoError> {
        Self::with_policy(native, MediaFetcher::from_env())
    }

    fn with_policy(
        native: &StagedMediaPolicy,
        mut policy: MediaFetcher,
    ) -> Result<Self, DynamoError> {
        if !native.allowed_domains.is_empty() {
            let mut allowed: HashSet<String> = native
                .allowed_domains
                .iter()
                .map(|domain| {
                    // Python normalizes IPv6 hosts without url::Url's brackets.
                    if let Ok(ip) = domain.parse::<std::net::Ipv6Addr>() {
                        format!("[{ip}]")
                    } else {
                        domain.clone()
                    }
                })
                .collect();
            if let Some(domains) = &policy.allowed_media_domains {
                allowed.retain(|domain| domains.contains(domain));
            }
            // Some(empty) denies every remote host when the two policies disagree.
            policy.allowed_media_domains = Some(allowed);
        }
        let timeout = native
            .timeout_secs
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|_| client::protocol_error("invalid staged media timeout"))?;
        if timeout.is_some_and(|value| value.is_zero()) {
            return Err(client::protocol_error("invalid staged media timeout"));
        }
        let http = policy
            .http_client_builder()
            .no_proxy()
            .build()
            .map_err(|error| {
                client::protocol_error(format!("cannot build staged media client: {error}"))
            })?;
        Ok(Self {
            policy,
            http,
            max_bytes: usize::try_from(native.max_bytes).unwrap_or(usize::MAX),
            timeout,
        })
    }

    async fn fetch(
        &self,
        source: &str,
        modality: &str,
        max_uri_len: usize,
    ) -> Result<String, DynamoError> {
        let url = url::Url::parse(source)
            .map_err(|_| client::invalid_request("invalid staged media URL"))?;
        self.policy
            .check_if_url_allowed_with_dns(&url)
            .await
            .map_err(|_| {
                client::invalid_request("staged media URL is unavailable or blocked by policy")
            })?;
        let mut response = self
            .http
            .get(url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|_| client::invalid_request("failed to download staged media"))?;
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .map(str::trim);
        let mime = match content_type {
            Some(mime)
                if mime.split_once('/').is_some_and(|(kind, subtype)| {
                    kind.eq_ignore_ascii_case(modality)
                        && !subtype.is_empty()
                        && subtype
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"-.+".contains(&b))
                }) =>
            {
                mime
            }
            // Object stores commonly serve untyped bytes. The native decoder
            // determines the media format from their contents.
            None | Some("application/octet-stream") => "application/octet-stream",
            _ => {
                return Err(client::invalid_request(
                    "staged media response has an incompatible content type",
                ));
            }
        };
        let mut uri = format!("data:{mime};base64,");
        let max_encoded = max_uri_len.saturating_sub(uri.len());
        let mut max_bytes = (max_encoded / 4) * 3;
        if self.max_bytes != 0 {
            max_bytes = max_bytes.min(self.max_bytes);
        }
        let too_large =
            || client::invalid_request("staged media exceeds the encoder or gRPC size limit");
        if response
            .content_length()
            .is_some_and(|length| length > max_bytes as u64)
        {
            return Err(too_large());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| client::invalid_request("failed to read staged media"))?
        {
            if chunk.len() > max_bytes.saturating_sub(bytes.len()) {
                return Err(too_large());
            }
            bytes.extend_from_slice(&chunk);
        }
        if bytes.is_empty() {
            return Err(client::invalid_request("staged media is empty"));
        }
        STANDARD.encode_string(bytes, &mut uri);
        Ok(uri)
    }
}

/// Rewrite only the private Stage RPC. The original request keeps its media
/// identity for KV routing, downstream validation, and retries.
pub(crate) async fn prepare(
    message: &mut StageRequest,
    fetcher: Option<&StagedMedia>,
) -> Result<(), DynamoError> {
    prepare_with_limits(
        message,
        fetcher,
        max_data_url_bytes(),
        DEFAULT_MAX_GRPC_MESSAGE_SIZE,
    )
    .await
}

async fn prepare_with_limits(
    message: &mut StageRequest,
    fetcher: Option<&StagedMedia>,
    inline_limit: usize,
    wire_limit: usize,
) -> Result<(), DynamoError> {
    let started = tokio::time::Instant::now();
    let mut wire_len = message.encoded_len();
    if wire_len > wire_limit {
        return Err(client::invalid_request(
            "staged media exceeds the gRPC size limit",
        ));
    }
    for (modality, sources, default_timeout) in [
        ("image", &mut message.mm_items, Duration::from_secs(3)),
        ("video", &mut message.video_urls, Duration::from_secs(10)),
        ("audio", &mut message.audio_urls, Duration::from_secs(5)),
    ] {
        for source in sources {
            if source.starts_with("data:") {
                if source.len() > inline_limit {
                    return Err(client::invalid_request(
                        "staged data URL exceeds DYN_MM_MAX_DATA_URL_MB",
                    ));
                }
                continue;
            }
            let Some(fetcher) = fetcher else {
                if modality != "image" {
                    return Err(client::invalid_request(
                        "remote staged video or audio requires the encoder's media policy",
                    ));
                }
                // v1 encoders do not publish their native domain and size
                // restrictions. They still fetch their own original image URLs.
                continue;
            };
            let old_len = source.len() + prost::length_delimiter_len(source.len());
            let available = wire_limit - (wire_len - old_len);
            // Reserve the longest length prefix allowed by the gRPC limit.
            let max_uri_len = available.saturating_sub(prost::length_delimiter_len(wire_limit));
            let deadline = fetcher.timeout.unwrap_or(default_timeout).min(
                fetcher
                    .policy
                    .timeout
                    .unwrap_or(Duration::MAX)
                    .saturating_sub(started.elapsed()),
            );
            let uri = tokio::time::timeout(deadline, fetcher.fetch(source, modality, max_uri_len))
                .await
                .map_err(|_| client::connection_timeout("staged media download timed out"))??;
            wire_len = wire_len - old_len + uri.len() + prost::length_delimiter_len(uri.len());
            *source = uri;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    async fn serve(reply: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let reply = reply.to_owned();
        tokio::spawn(async move {
            let (mut conn, _) = listener.accept().await.unwrap();
            let mut headers = BufReader::new(&mut conn).lines();
            while let Some(line) = headers.next_line().await.unwrap() {
                if line.is_empty() {
                    break;
                }
            }
            conn.write_all(reply.as_bytes()).await.unwrap();
        });
        url
    }

    fn test_fetcher(max_bytes: usize) -> StagedMedia {
        let policy = MediaFetcher {
            allow_private_ips: true,
            allow_direct_ip: true,
            allow_direct_port: true,
            allowed_media_domains: Some(["127.0.0.1".to_owned()].into()),
            ..Default::default()
        };
        let http = policy.http_client_builder().no_proxy().build().unwrap();
        StagedMedia {
            policy,
            http,
            max_bytes,
            timeout: None,
        }
    }

    #[test]
    fn disjoint_allowlists_deny_all_remote_hosts() {
        let native = StagedMediaPolicy {
            allowed_domains: vec!["encoder.test".into()],
            ..Default::default()
        };
        let policy = MediaFetcher {
            allowed_media_domains: Some(["frontend.test".to_owned()].into()),
            ..Default::default()
        };
        let fetcher = StagedMedia::with_policy(&native, policy).unwrap();
        for host in ["encoder.test", "frontend.test"] {
            assert!(
                fetcher
                    .policy
                    .check_if_url_allowed(&format!("https://{host}/clip").parse().unwrap())
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn stage_message_accounts_for_inline_sources_and_base64_expansion() {
        let mut message = StageRequest {
            mm_items: vec!["data:image/png;base64,AAAA".into()],
            ..Default::default()
        };
        assert!(
            prepare_with_limits(&mut message, None, 4, 100)
                .await
                .is_err()
        );
        let source = serve("HTTP/1.1 200 OK\r\nContent-Type: video/webm\r\nContent-Length: 30\r\nConnection: close\r\n\r\n012345678901234567890123456789").await;
        message.video_urls.push(source);
        let wire_limit = message.encoded_len();
        assert!(
            prepare_with_limits(&mut message, Some(&test_fetcher(100)), 100, wire_limit)
                .await
                .unwrap_err()
                .to_string()
                .contains("size limit")
        );
    }

    #[tokio::test]
    async fn staged_audio_fetches_once_under_both_encoder_and_frontend_limits() {
        let reply = "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc";
        let mut message = StageRequest {
            audio_urls: vec![serve(reply).await],
            ..Default::default()
        };
        assert!(prepare(&mut message, None).await.is_err());
        prepare(&mut message, Some(&test_fetcher(3))).await.unwrap();
        assert_eq!(message.audio_urls, ["data:audio/wav;base64,YWJj"]);
        message.audio_urls = vec![serve(reply).await];
        assert!(prepare(&mut message, Some(&test_fetcher(2))).await.is_err());
        message.audio_urls = vec![serve(&reply.replace("audio/wav", "text/html")).await];
        assert!(prepare(&mut message, Some(&test_fetcher(3))).await.is_err());
    }

    #[tokio::test]
    async fn staged_video_snapshots_the_response_and_checks_every_redirect_and_chunk() {
        let mut fetcher = test_fetcher(3);
        let response = "HTTP/1.1 200 OK\r\nContent-Type: video/webm\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n3\r\nabc\r\n0\r\n\r\n";
        let source = serve(response).await;
        let mut message = StageRequest {
            video_urls: vec![source.clone()],
            ..Default::default()
        };
        assert!(prepare(&mut message, None).await.is_err());
        prepare(&mut message, Some(&fetcher)).await.unwrap();
        assert_eq!(message.video_urls, ["data:video/webm;base64,YWJj"]);

        fetcher.max_bytes = 2;
        message.video_urls = vec![serve(response).await];
        assert!(
            prepare(&mut message, Some(&fetcher))
                .await
                .unwrap_err()
                .to_string()
                .contains("size limit")
        );
        fetcher.max_bytes = 3;
        let target = serve(response).await.replace("127.0.0.1", "localhost");
        let redirect = format!(
            "HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        message.video_urls = vec![serve(&redirect).await];
        assert!(prepare(&mut message, Some(&fetcher)).await.is_err());
        // A legacy encoder still receives its image URL, preserving its native
        // allowlist. The video code above requires the E policy before fetching.
        let mut legacy = StageRequest {
            mm_items: vec![source.clone()],
            ..Default::default()
        };
        prepare(&mut legacy, None).await.unwrap();
        assert_eq!(legacy.mm_items, [source]);
    }
}
