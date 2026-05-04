// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::Result;
use ipnet::IpNet;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::Policy;

use dynamo_memory::nixl::NixlAgent;
use dynamo_protocols::types::ChatCompletionRequestUserMessageContentPart;

use super::common::EncodedMediaData;
use super::decoders::{Decoder, MediaDecoder};
use super::rdma::{DataType, RdmaMediaDataDescriptor, get_nixl_agent};
use lru::LruCache;
use parking_lot::Mutex;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const DEFAULT_HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_REDIRECTS: usize = 3;

// IP ranges that must never be reachable from a user-controlled URL.
// Source: RFC1918 (private), RFC6598 (CGNAT), RFC5735 (loopback, link-local,
// 0.0.0.0/8), RFC4193 (ULA), RFC4291 (IPv6 loopback / link-local), RFC6890
// (reserved). Link-local 169.254/16 covers the AWS / OpenStack metadata IP.
//
// Keep this list in sync with the Python counterpart
// (components/src/dynamo/common/multimodal/url_validator.py::_BLOCKED_IP_NETWORKS).
static BLOCKED_IP_NETWORKS: LazyLock<Vec<IpNet>> = LazyLock::new(|| {
    [
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "255.255.255.255/32",
        "::/128",
        "::1/128",
        "::ffff:0:0/96",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    ]
    .iter()
    .map(|s| s.parse().expect("invalid CIDR in BLOCKED_IP_NETWORKS"))
    .collect()
});

// Hostnames we reject by literal match without any DNS lookup. Defends
// against /etc/hosts tricks or malicious resolvers that alias metadata /
// internal-service names to attacker IPs. Match is case-insensitive.
//
// Keep this list in sync with the Python counterpart
// (components/src/dynamo/common/multimodal/url_validator.py::_BLOCKED_HOSTS).
static BLOCKED_HOSTS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "kubernetes.default",
        "kubernetes.default.svc",
    ]
    .iter()
    .copied()
    .collect()
});

/// Return `true` if `ip` falls inside any of the blocked ranges.
pub fn is_blocked_ip(ip: &IpAddr) -> bool {
    BLOCKED_IP_NETWORKS.iter().any(|net| net.contains(ip))
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MediaFetcher {
    pub user_agent: String,
    pub allow_direct_ip: bool,
    pub allow_direct_port: bool,
    /// When `false` (default), reject URLs that target blocked locations:
    /// an IP literal in the RFC-range blocklist (`BLOCKED_IP_NETWORKS`),
    /// a hostname in the literal blocklist (`BLOCKED_HOSTS`, e.g.
    /// `localhost` / `metadata.google.internal`), or — in
    /// `check_if_url_allowed_with_dns` — a hostname that DNS-resolves
    /// to a blocked IP. The name reads "IP" but semantically this is a
    /// single "allow internal / on-prem targets" switch that covers
    /// both IP and hostname blocklists together: real on-prem
    /// deployments need both at once (private CIDRs *and* internal
    /// service names), and splitting them would give no useful config
    /// while doubling the footgun surface. **Never** set on anything
    /// public-facing.
    pub allow_private_ips: bool,
    pub allowed_media_domains: Option<HashSet<String>>,
    pub timeout: Option<Duration>,
}

impl Default for MediaFetcher {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_HTTP_USER_AGENT.to_string(),
            allow_direct_ip: false,
            allow_direct_port: false,
            allow_private_ips: false,
            allowed_media_domains: None,
            timeout: Some(DEFAULT_HTTP_TIMEOUT),
        }
    }
}

impl MediaFetcher {
    /// Build a `MediaFetcher` whose defaults respect the shared
    /// `DYN_MM_ALLOW_INTERNAL` environment variable. Mirrors the Python
    /// `UrlValidationPolicy.from_env()` behavior so both fetch paths
    /// (frontend decode in Rust, backend decode in Python) honor the
    /// same on-prem opt-in flag.
    ///
    /// `DYN_MM_ALLOW_INTERNAL=1` flips `allow_direct_ip`,
    /// `allow_direct_port`, and `allow_private_ips` all to `true` at
    /// once.
    pub fn from_env() -> Self {
        let allow_internal = std::env::var("DYN_MM_ALLOW_INTERNAL").ok().as_deref() == Some("1");
        Self {
            allow_direct_ip: allow_internal,
            allow_direct_port: allow_internal,
            allow_private_ips: allow_internal,
            ..Self::default()
        }
    }
}

impl MediaFetcher {
    pub fn check_if_url_allowed(&self, url: &url::Url) -> Result<()> {
        if !matches!(url.scheme(), "http" | "https" | "data") {
            anyhow::bail!("Only HTTP(S) and data URLs are allowed");
        }

        if url.scheme() == "data" {
            return Ok(());
        }

        let host = url
            .host()
            .ok_or_else(|| anyhow::anyhow!("URL has no host component"))?;

        if !self.allow_direct_ip && !matches!(host, url::Host::Domain(_)) {
            anyhow::bail!("Direct IP access is not allowed");
        }
        if !self.allow_direct_port && url.port().is_some() {
            anyhow::bail!("Direct port access is not allowed");
        }

        // Host-level checks: blocked hostnames and IP literals in blocked
        // ranges. DNS-resolved IPs are checked in `check_if_url_allowed_with_dns`.
        if !self.allow_private_ips {
            let ip_literal = match host {
                url::Host::Domain(domain) => {
                    let lowered = domain.trim_end_matches('.').to_ascii_lowercase();
                    if BLOCKED_HOSTS.contains(lowered.as_str()) {
                        anyhow::bail!("Host '{domain}' is blocked (resolves to internal service)");
                    }
                    None
                }
                url::Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
                url::Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
            };
            if let Some(ip) = ip_literal
                && is_blocked_ip(&ip)
            {
                anyhow::bail!("IP literal '{ip}' is in a blocked range");
            }
        }

        if let Some(allowed_domains) = &self.allowed_media_domains
            && let Some(host_str) = url.host_str()
            && !allowed_domains.contains(host_str)
        {
            anyhow::bail!("Host '{host_str}' is not in the allowed_media_domains list");
        }

        Ok(())
    }

    /// Full policy check: runs `check_if_url_allowed` and, for hostname
    /// URLs, resolves DNS and checks each resulting IP against the blocked ranges.
    pub async fn check_if_url_allowed_with_dns(&self, url: &url::Url) -> Result<()> {
        self.check_if_url_allowed(url)?;

        // Only hostnames need DNS resolution; IP-literal hosts were already
        // checked against the blocklist above.
        if self.allow_private_ips || url.scheme() == "data" {
            return Ok(());
        }
        let Some(url::Host::Domain(host)) = url.host() else {
            return Ok(());
        };

        let port = url.port_or_known_default().unwrap_or(0);
        let iter = tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| anyhow::anyhow!("Could not resolve host '{host}': {e}"))?;
        for sock_addr in iter {
            let ip = sock_addr.ip();
            if is_blocked_ip(&ip) {
                anyhow::bail!("Host '{host}' resolves to blocked IP '{ip}'");
            }
        }
        Ok(())
    }
}

/// DNS resolver that filters out blocked IP ranges before reqwest sees them.
///
/// Attached to the shared `reqwest::Client` via `ClientBuilder::dns_resolver`.
/// reqwest calls this for every hostname it needs to resolve — including
/// redirect targets — so DNS rebinding can't slip a blocked IP past us:
/// reqwest literally never learns about the blocked addresses.
struct BlocklistResolver {
    allow_private_ips: bool,
}

impl Resolve for BlocklistResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        let allow_private = self.allow_private_ips;
        Box::pin(async move {
            let iter = tokio::net::lookup_host((host.as_str(), 0_u16)).await?;
            let addrs: Vec<SocketAddr> = if allow_private {
                iter.collect()
            } else {
                iter.filter(|sa| !is_blocked_ip(&sa.ip())).collect()
            };
            if addrs.is_empty() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::AddrNotAvailable,
                    format!("no non-blocked addresses for host '{host}'"),
                ))
                    as Box<dyn std::error::Error + Send + Sync>);
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

/// Byte-budgeted LRU of decoded + NIXL-registered media descriptors.
///
/// Capacity is denominated in bytes (raw decoded payload, derived from
/// `tensor_info.shape * dtype`). Insertion evicts oldest entries until the
/// running total fits the budget; entries larger than the whole budget are
/// inserted and immediately evicted (i.e. effectively not cached).
struct LoaderCache {
    lru: LruCache<u64, RdmaMediaDataDescriptor>,
    bytes_used: u64,
    budget_bytes: u64,
}

impl LoaderCache {
    fn new(budget_bytes: u64) -> Self {
        // `unbounded` capacity — eviction is driven by the byte budget.
        Self {
            lru: LruCache::unbounded(),
            bytes_used: 0,
            budget_bytes,
        }
    }

    fn get(&mut self, key: &u64) -> Option<RdmaMediaDataDescriptor> {
        self.lru.get(key).cloned()
    }

    fn put(&mut self, key: u64, val: RdmaMediaDataDescriptor) {
        let val_bytes = descriptor_bytes(&val);
        // Re-insert path: drop the old entry's bytes from the running total
        // before adding the new one.
        if let Some(old) = self.lru.pop(&key) {
            self.bytes_used = self.bytes_used.saturating_sub(descriptor_bytes(&old));
        }
        self.lru.put(key, val);
        self.bytes_used = self.bytes_used.saturating_add(val_bytes);
        while self.bytes_used > self.budget_bytes && !self.lru.is_empty() {
            if let Some((_, old)) = self.lru.pop_lru() {
                self.bytes_used = self.bytes_used.saturating_sub(descriptor_bytes(&old));
            } else {
                break;
            }
        }
    }

    fn len(&self) -> usize {
        self.lru.len()
    }
}

/// Raw decoded byte size of a descriptor — what the NIXL registration holds.
/// Per-entry bookkeeping (struct fields, NIXL metadata string) is negligible
/// compared to a single decoded image.
fn descriptor_bytes(d: &RdmaMediaDataDescriptor) -> u64 {
    let elem = match d.tensor_info.dtype {
        DataType::UINT8 => 1u64,
    };
    d.tensor_info
        .shape
        .iter()
        .try_fold(1u64, |acc, &x| acc.checked_mul(x as u64))
        .unwrap_or(u64::MAX)
        .saturating_mul(elem)
}

pub struct MediaLoader {
    #[allow(dead_code)]
    media_decoder: MediaDecoder,
    #[allow(dead_code)]
    http_client: reqwest::Client,
    #[allow(dead_code)]
    media_fetcher: MediaFetcher,
    nixl_agent: NixlAgent,
    /// Optional byte-budgeted LRU cache of decoded + NIXL-registered media,
    /// keyed by URL hash. Each cache entry is shared via Arc; the underlying
    /// NIXL registration is kept alive as long as any clone (in cache or
    /// in-flight) holds it. Eviction just drops the cache's reference.
    /// `None` when caching is disabled (budget = 0).
    cache: Option<Arc<Mutex<LoaderCache>>>,
}

impl MediaLoader {
    /// Read the cache budget (in bytes) from `DYN_MULTIMODAL_LOADER_CACHE_GB`.
    /// Value parses as a float number of gibibytes (1 GiB = 1024^3 bytes).
    /// Default `0` (disabled) — opt-in only.
    fn cache_budget_bytes_from_env() -> u64 {
        let gb = std::env::var("DYN_MULTIMODAL_LOADER_CACHE_GB")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0);
        (gb * (1024.0 * 1024.0 * 1024.0)) as u64
    }

    /// Hash a URL/datauri string into a stable u64 cache key.
    fn cache_key(url: &str) -> u64 {
        let mut h = DefaultHasher::new();
        url.hash(&mut h);
        h.finish()
    }

    pub fn new(media_decoder: MediaDecoder, media_fetcher: Option<MediaFetcher>) -> Result<Self> {
        // Fall back to env-aware defaults so `DYN_MM_ALLOW_INTERNAL=1` is
        // honored even when the caller doesn't pass an explicit fetcher.
        let media_fetcher = media_fetcher.unwrap_or_else(MediaFetcher::from_env);

        // Redirect policy: revalidate the policy-visible part of the URL
        // (scheme, IP literals, hostname blocklist, direct-IP / direct-port
        // rules) on every hop. DNS-based attacks on redirect targets are
        // handled by the custom resolver below.
        let fetcher_for_redirects = media_fetcher.clone();
        let redirect_policy = Policy::custom(move |attempt| {
            if attempt.previous().len() >= MAX_REDIRECTS {
                return attempt.error(anyhow::anyhow!("too many redirects (max={MAX_REDIRECTS})"));
            }
            match fetcher_for_redirects.check_if_url_allowed(attempt.url()) {
                Ok(()) => attempt.follow(),
                Err(e) => attempt.error(e),
            }
        });

        let mut http_client_builder = reqwest::Client::builder()
            .user_agent(&media_fetcher.user_agent)
            .redirect(redirect_policy)
            .dns_resolver(Arc::new(BlocklistResolver {
                allow_private_ips: media_fetcher.allow_private_ips,
            }));

        if let Some(timeout) = media_fetcher.timeout {
            http_client_builder = http_client_builder.timeout(timeout);
        }

        let http_client = http_client_builder.build()?;

        let nixl_agent = get_nixl_agent()?;

        let cache = match Self::cache_budget_bytes_from_env() {
            0 => {
                tracing::debug!(
                    "[mm-cache] frontend media cache disabled (DYN_MULTIMODAL_LOADER_CACHE_GB=0)"
                );
                None
            }
            budget => {
                tracing::info!(
                    budget_bytes = budget,
                    "[mm-cache] frontend media cache enabled (DYN_MULTIMODAL_LOADER_CACHE_GB)"
                );
                Some(Arc::new(Mutex::new(LoaderCache::new(budget))))
            }
        };

        Ok(Self {
            media_decoder,
            http_client,
            media_fetcher,
            nixl_agent,
            cache,
        })
    }

    /// Test-only constructor that lets a unit test build a `MediaLoader` with
    /// an explicit byte budget (bypassing the env-var read in `new`).
    /// Pass 0 to disable.
    #[cfg(test)]
    pub fn with_cache_budget_bytes(
        media_decoder: MediaDecoder,
        media_fetcher: Option<MediaFetcher>,
        budget_bytes: u64,
    ) -> Result<Self> {
        let mut loader = Self::new(media_decoder, media_fetcher)?;
        loader.cache = if budget_bytes == 0 {
            None
        } else {
            Some(Arc::new(Mutex::new(LoaderCache::new(budget_bytes))))
        };
        Ok(loader)
    }

    /// Number of entries currently held in the cache (test/observability helper).
    pub fn cache_len(&self) -> usize {
        self.cache.as_ref().map(|c| c.lock().len()).unwrap_or(0)
    }

    pub async fn fetch_and_decode_media_part(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
        media_io_kwargs: Option<&MediaDecoder>,
    ) -> Result<RdmaMediaDataDescriptor> {
        // Image-only fast path: cache lookup keyed by URL/datauri string.
        // Video/audio aren't cached yet (their lifetime/content semantics
        // are different — easy to add later if profiling justifies it).
        // The cache stores the post-decode + NIXL-registered descriptor;
        // a hit short-circuits both the network fetch and the image decode.
        if let (Some(cache), ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part)) =
            (self.cache.as_ref(), oai_content_part)
        {
            // media_io_kwargs is per-request and could change the decode
            // output (resize, normalisation). When it's set we skip the
            // cache to stay correct; in practice it's None on the common
            // path so the hit rate is unaffected.
            if media_io_kwargs.is_none() {
                let key = Self::cache_key(image_part.image_url.url.as_str());
                if let Some(hit) = cache.lock().get(&key) {
                    tracing::debug!(url_hash = key, "[mm-cache] hit");
                    return Ok(hit);
                }
            }
        }

        // fetch the media, decode and NIXL-register
        let decoded = match oai_content_part {
            ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                let mdc_decoder = self
                    .media_decoder
                    .image
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Model does not support image inputs"))?;

                let url = &image_part.image_url.url;
                self.media_fetcher
                    .check_if_url_allowed_with_dns(url)
                    .await?;
                let data = EncodedMediaData::from_url(url, &self.http_client).await?;

                // Use runtime decoder if provided, with MDC limits enforced
                let decoder =
                    mdc_decoder.with_runtime(media_io_kwargs.and_then(|k| k.image.as_ref()));
                decoder.decode_async(data).await?
            }
            #[allow(unused_variables)]
            ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                #[cfg(not(feature = "media-ffmpeg"))]
                anyhow::bail!("Video decoding requires the 'media-ffmpeg' feature to be enabled");

                #[cfg(feature = "media-ffmpeg")]
                {
                    let mdc_decoder =
                        self.media_decoder.video.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("Model does not support video inputs")
                        })?;

                    let url = &video_part.video_url.url;
                    self.media_fetcher
                        .check_if_url_allowed_with_dns(url)
                        .await?;
                    let data = EncodedMediaData::from_url(url, &self.http_client).await?;

                    // Use runtime decoder if provided, with MDC limits enforced
                    let decoder =
                        mdc_decoder.with_runtime(media_io_kwargs.and_then(|k| k.video.as_ref()));
                    decoder.decode_async(data).await?
                }
            }
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => {
                anyhow::bail!("Audio decoding is not supported yet");
            }
            _ => anyhow::bail!("Unsupported media type"),
        };

        let rdma_descriptor = decoded.into_rdma_descriptor(&self.nixl_agent)?;

        // Insert into the cache on the way out. We only cache image inputs
        // (matched in the lookup above) and only when no per-request decoder
        // override is in effect. The descriptor is `Clone` and shares the
        // underlying NIXL registration via `Arc`, so the cached entry stays
        // alive across in-flight requests; eviction just drops the cache's
        // own reference.
        if let (Some(cache), ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part)) =
            (self.cache.as_ref(), oai_content_part)
            && media_io_kwargs.is_none()
        {
            let key = Self::cache_key(image_part.image_url.url.as_str());
            let bytes = descriptor_bytes(&rdma_descriptor);
            cache.lock().put(key, rdma_descriptor.clone());
            tracing::debug!(url_hash = key, bytes, "[mm-cache] insert");
        }

        Ok(rdma_descriptor)
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::super::decoders::ImageDecoder;
    use super::super::rdma::DataType;
    use super::*;
    use dynamo_protocols::types::{ChatCompletionRequestMessageContentPartImage, ImageUrl};

    #[tokio::test]
    async fn test_fetch_and_decode() {
        let test_image_bytes =
            include_bytes!("../../../tests/data/media/llm-optimize-deploy-graphic.png");

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/llm-optimize-deploy-graphic.png")
            .with_status(200)
            .with_header("content-type", "image/png")
            .with_body(&test_image_bytes[..])
            .create_async()
            .await;

        let media_decoder = MediaDecoder {
            image: Some(ImageDecoder::default()),
            #[cfg(feature = "media-ffmpeg")]
            video: None,
        };
        let fetcher = MediaFetcher {
            allow_direct_ip: true,
            allow_direct_port: true,
            // mockito serves on 127.0.0.1 which is in the loopback blocklist.
            allow_private_ips: true,
            ..Default::default()
        };

        let loader: MediaLoader = match MediaLoader::new(media_decoder, Some(fetcher)) {
            Ok(l) => l,
            Err(e) => {
                println!(
                    "test test_fetch_and_decode ... ignored (NIXL/UCX not available: {})",
                    e
                );
                return;
            }
        };

        let image_url = ImageUrl::from(format!("{}/llm-optimize-deploy-graphic.png", server.url()));
        let content_part = ChatCompletionRequestUserMessageContentPart::ImageUrl(
            ChatCompletionRequestMessageContentPartImage { image_url },
        );

        let result = loader
            .fetch_and_decode_media_part(&content_part, None)
            .await;

        let descriptor = match result {
            Ok(descriptor) => descriptor,
            Err(e) if e.to_string().contains("NIXL agent is not available") => {
                println!("test test_fetch_and_decode ... ignored (NIXL agent not available)");
                return;
            }
            Err(e) => panic!("Failed to fetch and decode image: {}", e),
        };
        mock.assert_async().await;
        assert_eq!(descriptor.tensor_info.dtype, DataType::UINT8);

        // Verify image dimensions: 1,999px × 1,125px (width × height)
        // Shape format is [height, width, channels]
        assert_eq!(descriptor.tensor_info.shape.len(), 3);
        assert_eq!(
            descriptor.tensor_info.shape[0], 1125,
            "Height should be 1125"
        );
        assert_eq!(
            descriptor.tensor_info.shape[1], 1999,
            "Width should be 1999"
        );
        assert_eq!(
            descriptor.tensor_info.shape[2], 4,
            "RGBA channels should be 4"
        );

        assert!(
            descriptor.source_storage.is_some(),
            "Source storage should be present"
        );
        assert!(
            descriptor.source_storage.unwrap().is_registered(),
            "Source storage should be registered with NIXL"
        );
    }

    /// With the cache enabled, a second fetch of the same URL must be served
    /// from the cache: the upstream server should see exactly one GET, and
    /// the loader's `cache_len()` should reflect the inserted entry.
    #[tokio::test]
    async fn test_cache_hit_skips_second_fetch() {
        let test_image_bytes =
            include_bytes!("../../../tests/data/media/llm-optimize-deploy-graphic.png");

        let mut server = mockito::Server::new_async().await;
        // Expect EXACTLY one GET; if the cache misses on the second call this
        // assertion will fail because mockito will see a second hit.
        let mock = server
            .mock("GET", "/cache-image.png")
            .with_status(200)
            .with_header("content-type", "image/png")
            .with_body(&test_image_bytes[..])
            .expect(1)
            .create_async()
            .await;

        let media_decoder = MediaDecoder {
            image: Some(ImageDecoder::default()),
            #[cfg(feature = "media-ffmpeg")]
            video: None,
        };
        let fetcher = MediaFetcher {
            allow_direct_ip: true,
            allow_direct_port: true,
            allow_private_ips: true,
            ..Default::default()
        };

        // Budget = 100 MB — comfortably fits one ~9 MB decoded image (1125x1999x4
        // RGBA = 8.57 MB). We're not exercising eviction here, just the hit path.
        let loader = match MediaLoader::with_cache_budget_bytes(
            media_decoder,
            Some(fetcher),
            100 * 1024 * 1024,
        ) {
            Ok(l) => l,
            Err(e) => {
                println!("test_cache_hit_skips_second_fetch ... ignored ({})", e);
                return;
            }
        };

        assert_eq!(loader.cache_len(), 0, "cache should start empty");

        let url_string = format!("{}/cache-image.png", server.url());
        let image_url = ImageUrl::from(url_string);
        let content_part = ChatCompletionRequestUserMessageContentPart::ImageUrl(
            ChatCompletionRequestMessageContentPartImage { image_url },
        );

        // First call — populates cache.
        let first = match loader
            .fetch_and_decode_media_part(&content_part, None)
            .await
        {
            Ok(d) => d,
            Err(e) if e.to_string().contains("NIXL agent is not available") => {
                println!("test_cache_hit_skips_second_fetch ... ignored (NIXL not available)");
                return;
            }
            Err(e) => panic!("first fetch failed: {}", e),
        };
        assert_eq!(loader.cache_len(), 1, "cache should hold one entry");

        // Second call — must be served from cache (no second mock hit).
        let second = loader
            .fetch_and_decode_media_part(&content_part, None)
            .await
            .expect("second fetch should hit cache");

        // Both descriptors should reference the same NIXL-registered storage
        // (cloned `Arc` under the hood — same pointer).
        match (&first.source_storage, &second.source_storage) {
            (Some(a), Some(b)) => assert!(
                Arc::ptr_eq(a, b),
                "cache hit should return the same Arc'd source_storage"
            ),
            _ => panic!("source_storage missing from one of the descriptors"),
        }

        // Mockito asserts exactly-one GET — fails the test if we hit twice.
        mock.assert_async().await;
    }

    /// Cache byte budget is enforced: with a budget that fits exactly two
    /// decoded images, inserting a third evicts the least-recently-used
    /// entry, and re-fetching the evicted entry triggers a real network call
    /// (proving the entry actually left the cache rather than silently
    /// lingering).
    #[tokio::test]
    async fn test_cache_budget_lru_eviction() {
        let test_image_bytes =
            include_bytes!("../../../tests/data/media/llm-optimize-deploy-graphic.png");

        let mut server = mockito::Server::new_async().await;
        // /a.png expects 2 hits: (i) cold insert, (ii) re-fetch after C
        // evicts A. /b.png expects 1: cold insert + a re-fetch served from
        // cache. /c.png expects 1: cold insert (it stays in cache too).
        let mock_a = server
            .mock("GET", "/a.png")
            .with_status(200)
            .with_header("content-type", "image/png")
            .with_body(&test_image_bytes[..])
            .expect(2)
            .create_async()
            .await;
        let mock_b = server
            .mock("GET", "/b.png")
            .with_status(200)
            .with_header("content-type", "image/png")
            .with_body(&test_image_bytes[..])
            .expect(1)
            .create_async()
            .await;
        let mock_c = server
            .mock("GET", "/c.png")
            .with_status(200)
            .with_header("content-type", "image/png")
            .with_body(&test_image_bytes[..])
            .expect(1)
            .create_async()
            .await;

        let media_decoder = MediaDecoder {
            image: Some(ImageDecoder::default()),
            #[cfg(feature = "media-ffmpeg")]
            video: None,
        };
        let fetcher = MediaFetcher {
            allow_direct_ip: true,
            allow_direct_port: true,
            allow_private_ips: true,
            ..Default::default()
        };

        // Single decoded image is ~8.57 MB (1125x1999x4 RGBA). Budget = 18 MB
        // accommodates exactly 2 entries; inserting a 3rd forces an eviction.
        let loader = match MediaLoader::with_cache_budget_bytes(
            media_decoder,
            Some(fetcher),
            18 * 1024 * 1024,
        ) {
            Ok(l) => l,
            Err(e) => {
                println!("test_cache_budget_lru_eviction ... ignored ({})", e);
                return;
            }
        };

        let make_part = |path: &str| {
            let image_url = ImageUrl::from(format!("{}{}", server.url(), path));
            ChatCompletionRequestUserMessageContentPart::ImageUrl(
                ChatCompletionRequestMessageContentPartImage { image_url },
            )
        };

        let part_a = make_part("/a.png");
        let part_b = make_part("/b.png");
        let part_c = make_part("/c.png");

        // Cold-fetch A and B (cache: [B, A], B is MRU).
        for (label, part) in [("a", &part_a), ("b", &part_b)] {
            match loader.fetch_and_decode_media_part(part, None).await {
                Ok(_) => {}
                Err(e) if e.to_string().contains("NIXL agent is not available") => {
                    println!(
                        "test_cache_budget_lru_eviction ... ignored (NIXL not available, fetch={})",
                        label
                    );
                    return;
                }
                Err(e) => panic!("fetch {} failed: {}", label, e),
            }
        }
        assert_eq!(loader.cache_len(), 2);

        // Touch B to make it the LRU front; A becomes the eviction target
        // when the next insert lands. mock_b expects 1 total hit, so this
        // must be served from cache.
        loader
            .fetch_and_decode_media_part(&part_b, None)
            .await
            .expect("b re-fetch should hit cache");

        // Cold-fetch C → cache is full, evict A (LRU), now cache = {C, B}.
        loader
            .fetch_and_decode_media_part(&part_c, None)
            .await
            .expect("c cold fetch should succeed");
        assert_eq!(loader.cache_len(), 2);

        // Re-fetching A should miss the cache (A was evicted), triggering a
        // second network GET — matched by mock_a.expect(2).
        loader
            .fetch_and_decode_media_part(&part_a, None)
            .await
            .expect("a re-fetch after eviction should succeed");
        assert_eq!(loader.cache_len(), 2);

        mock_a.assert_async().await;
        mock_b.assert_async().await;
        mock_c.assert_async().await;
    }
}

#[cfg(test)]
mod tests_non_nixl {
    use super::*;

    #[test]
    fn test_cache_key_is_stable_per_url() {
        // Same URL → same key, every time. Different URLs → different keys.
        let u1 = "http://images.example.com/a.png";
        let u2 = "http://images.example.com/b.png";

        assert_eq!(MediaLoader::cache_key(u1), MediaLoader::cache_key(u1));
        assert_ne!(MediaLoader::cache_key(u1), MediaLoader::cache_key(u2));

        // Datauri strings produce stable keys too — these get long, the hash
        // collapses them to a u64 deterministically.
        let datauri = "data:image/png;base64,iVBORw0KGgoAAAA...".to_string();
        assert_eq!(
            MediaLoader::cache_key(&datauri),
            MediaLoader::cache_key(&datauri)
        );
    }

    #[test]
    fn test_cache_budget_from_env_default_zero() {
        const VAR: &str = "DYN_MULTIMODAL_LOADER_CACHE_GB";
        const GIB: u64 = 1024 * 1024 * 1024;
        // Save and restore so other tests in the same process aren't affected.
        let prev = std::env::var(VAR).ok();
        // SAFETY: env mutation is fine within this single-threaded test;
        // we restore on exit below.
        unsafe {
            std::env::remove_var(VAR);
        }
        assert_eq!(MediaLoader::cache_budget_bytes_from_env(), 0);

        unsafe {
            std::env::set_var(VAR, "1");
        }
        assert_eq!(MediaLoader::cache_budget_bytes_from_env(), GIB);

        // Fractional GB are honoured (lets users set, e.g., 0.5 for 512 MiB).
        unsafe {
            std::env::set_var(VAR, "0.5");
        }
        assert_eq!(MediaLoader::cache_budget_bytes_from_env(), GIB / 2);

        unsafe {
            std::env::set_var(VAR, "not-a-number");
        }
        assert_eq!(
            MediaLoader::cache_budget_bytes_from_env(),
            0,
            "non-numeric value should fall back to 0"
        );

        // Negative values are rejected (treated as 0).
        unsafe {
            std::env::set_var(VAR, "-1");
        }
        assert_eq!(MediaLoader::cache_budget_bytes_from_env(), 0);

        match prev {
            Some(v) => unsafe { std::env::set_var(VAR, v) },
            None => unsafe { std::env::remove_var(VAR) },
        }
    }

    #[test]
    fn test_direct_ip_blocked() {
        let fetcher = MediaFetcher {
            allow_direct_ip: false,
            ..Default::default()
        };

        let url = url::Url::parse("http://192.168.1.1/image.jpg").unwrap();
        let result = fetcher.check_if_url_allowed(&url);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Direct IP access is not allowed")
        );
    }

    #[test]
    fn test_direct_port_blocked() {
        let fetcher = MediaFetcher {
            allow_direct_port: false,
            ..Default::default()
        };

        let url = url::Url::parse("http://example.com:8080/image.jpg").unwrap();
        let result = fetcher.check_if_url_allowed(&url);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Direct port access is not allowed")
        );
    }

    #[test]
    fn test_domain_allowlist() {
        let mut allowed_domains = HashSet::new();
        allowed_domains.insert("trusted.com".to_string());
        allowed_domains.insert("example.com".to_string());

        let fetcher = MediaFetcher {
            allowed_media_domains: Some(allowed_domains),
            ..Default::default()
        };

        // Allowed domain should pass
        let url = url::Url::parse("https://trusted.com/image.jpg").unwrap();
        assert!(fetcher.check_if_url_allowed(&url).is_ok());

        // Disallowed domain should fail
        let url = url::Url::parse("https://untrusted.com/image.jpg").unwrap();
        let result = fetcher.check_if_url_allowed(&url);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("allowed_media_domains")
        );
    }

    #[test]
    fn test_is_blocked_ip_ranges() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.5.5",
            "192.168.1.1",
            "169.254.169.254", // AWS metadata
            "100.64.0.1",      // CGNAT
            "::1",
            "fe80::1",
            "fc00::1",
        ] {
            let addr: IpAddr = ip.parse().unwrap();
            assert!(is_blocked_ip(&addr), "{ip} should be blocked");
        }

        // Public IPs should pass.
        for ip in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            let addr: IpAddr = ip.parse().unwrap();
            assert!(!is_blocked_ip(&addr), "{ip} should not be blocked");
        }
    }

    #[test]
    fn test_blocked_ip_literal_rejected_even_when_direct_ip_allowed() {
        // allow_direct_ip=true lets IP-literal URLs through the early check,
        // but the RFC-range blocklist must still reject cloud-metadata IPs.
        let fetcher = MediaFetcher {
            allow_direct_ip: true,
            ..Default::default()
        };

        let url = url::Url::parse("http://169.254.169.254/latest/meta-data/").unwrap();
        let result = fetcher.check_if_url_allowed(&url);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("is in a blocked range")
        );
    }

    #[test]
    fn test_blocked_hostname_rejected() {
        let fetcher = MediaFetcher::default();
        for host in [
            "localhost",
            "metadata.google.internal",
            "kubernetes.default.svc",
        ] {
            let url = url::Url::parse(&format!("https://{host}/x")).unwrap();
            let result = fetcher.check_if_url_allowed(&url);
            assert!(result.is_err(), "{host} should be blocked");
            assert!(
                result.unwrap_err().to_string().contains("blocked"),
                "{host} error should mention 'blocked'"
            );
        }
    }

    #[test]
    fn test_allow_private_ips_bypasses_blocklist() {
        // allow_private_ips=true is the escape hatch for on-prem / dev envs.
        let fetcher = MediaFetcher {
            allow_direct_ip: true,
            allow_private_ips: true,
            ..Default::default()
        };

        // Both an IP literal in a blocked range and a blocked hostname
        // should pass when the opt-in flag is set.
        assert!(
            fetcher
                .check_if_url_allowed(&url::Url::parse("http://10.0.0.5/x").unwrap())
                .is_ok()
        );
        assert!(
            fetcher
                .check_if_url_allowed(&url::Url::parse("https://localhost/x").unwrap())
                .is_ok()
        );
    }

    #[test]
    fn test_hostname_blocklist_case_insensitive() {
        let fetcher = MediaFetcher::default();
        let url = url::Url::parse("https://Metadata.Google.Internal/x").unwrap();
        let result = fetcher.check_if_url_allowed(&url);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_env_default() {
        // Saving/restoring env vars in tests is racy with parallel tests,
        // so we only assert the "unset" case here (parallel-safe).
        // SAFETY: single-threaded mutation acceptable for this restore.
        unsafe {
            std::env::remove_var("DYN_MM_ALLOW_INTERNAL");
        }
        let f = MediaFetcher::from_env();
        assert!(!f.allow_private_ips);
        assert!(!f.allow_direct_ip);
        assert!(!f.allow_direct_port);
    }

    #[test]
    fn test_hostname_blocklist_strips_trailing_dot() {
        // FQDN form with a trailing dot must still match the blocklist;
        // `metadata.google.internal.` resolves to the same host as
        // `metadata.google.internal` at the DNS layer.
        let fetcher = MediaFetcher::default();
        let url = url::Url::parse("https://metadata.google.internal./x").unwrap();
        let result = fetcher.check_if_url_allowed(&url);
        assert!(result.is_err(), "FQDN with trailing dot should be rejected");
    }

    #[tokio::test]
    async fn test_check_with_dns_data_url_skips_resolution() {
        // data: URLs never touch the network, so the async path must early-return.
        let fetcher = MediaFetcher::default();
        let url = url::Url::parse("data:image/png;base64,iVBORw0KGgoAAAA=").unwrap();
        fetcher.check_if_url_allowed_with_dns(&url).await.unwrap();
    }

    #[tokio::test]
    async fn test_check_with_dns_public_ip_literal_passes() {
        // IP literals were already checked by the sync pass; async path is a no-op.
        let fetcher = MediaFetcher {
            allow_direct_ip: true,
            ..Default::default()
        };
        let url = url::Url::parse("https://8.8.8.8/x").unwrap();
        fetcher.check_if_url_allowed_with_dns(&url).await.unwrap();
    }

    #[tokio::test]
    async fn test_check_with_dns_blocked_hostname_fails_before_resolution() {
        // The sync hostname-blocklist check fires before we attempt any DNS.
        let fetcher = MediaFetcher::default();
        let url = url::Url::parse("https://localhost/x").unwrap();
        let result = fetcher.check_if_url_allowed_with_dns(&url).await;
        assert!(result.is_err());
    }
}
