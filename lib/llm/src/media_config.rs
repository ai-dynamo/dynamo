// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Media decoder and fetcher configuration a model deployment card carries.
//!
//! These are settings only: decoding, URL policy enforcement, and the HTTP client
//! live with the frontend media loader, which takes these values as input.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

pub const DEFAULT_IMAGE_MAX_ALLOC: u64 = 128 * 1024 * 1024; // 128 MB
const ENABLE_LIBJPEG_ENV: &str = "DYN_MM_ENABLE_LIBJPEG";

fn default_enable_libjpeg() -> bool {
    let value = std::env::var(ENABLE_LIBJPEG_ENV).ok();
    libjpeg_enabled(value.as_deref())
}

fn libjpeg_enabled(value: Option<&str>) -> bool {
    value
        .and_then(dynamo_runtime::config::parse_bool_opt)
        .unwrap_or(true)
}

/// Image decoder limits - can only be set via server config, not runtime kwargs.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoderLimits {
    #[serde(default)]
    pub max_image_width: Option<u32>,
    #[serde(default)]
    pub max_image_height: Option<u32>,
    /// Maximum allowed total allocation of the decoder in bytes
    #[serde(default)]
    pub max_alloc: Option<u64>,
}

impl Default for ImageDecoderLimits {
    fn default() -> Self {
        Self {
            max_image_width: None,
            max_image_height: None,
            max_alloc: Some(DEFAULT_IMAGE_MAX_ALLOC),
        }
    }
}

impl ImageDecoderLimits {
    pub fn validate_output(&self, width: u32, height: u32, channels: usize) -> Result<usize> {
        if self.max_image_width.is_some_and(|limit| width > limit)
            || self.max_image_height.is_some_and(|limit| height > limit)
        {
            anyhow::bail!("Image dimensions exceed configured limits: {width}x{height}");
        }

        let nbytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(channels as u64))
            .ok_or_else(|| {
                anyhow::anyhow!("Image allocation size overflow for dimensions: {width}x{height}")
            })?;
        if let Some(limit) = self.max_alloc
            && nbytes > limit
        {
            anyhow::bail!("Image allocation {nbytes} bytes exceeds configured limit {limit} bytes");
        }
        usize::try_from(nbytes)
            .map_err(|_| anyhow::anyhow!("Image allocation does not fit in usize: {nbytes} bytes"))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageDecoder {
    #[serde(default)]
    pub limits: ImageDecoderLimits,
    /// Frontend-local backend selection; never part of model or request configuration.
    #[serde(skip, default = "default_enable_libjpeg")]
    #[schema(ignore)]
    pub enable_libjpeg: bool,
    /// Copied from [`MediaDecoder::apply_exif_orientation`] by the media loader.
    #[serde(skip)]
    #[schema(ignore)]
    pub apply_exif_orientation: bool,
}

impl Default for ImageDecoder {
    fn default() -> Self {
        Self {
            limits: ImageDecoderLimits::default(),
            enable_libjpeg: default_enable_libjpeg(),
            apply_exif_orientation: false,
        }
    }
}

impl ImageDecoder {
    #[doc(hidden)]
    pub fn with_libjpeg_for_benchmark(mut self, enabled: bool) -> Self {
        self.enable_libjpeg = enabled;
        self
    }
}

#[cfg(feature = "media-ffmpeg")]
const DEFAULT_VIDEO_MAX_ALLOC: u64 = 512 * 1024 * 1024; // 512 MB

#[cfg(feature = "media-ffmpeg")]
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoderLimits {
    /// Maximum allowed total allocation of decoded frames in bytes
    #[serde(default)]
    pub max_alloc: Option<u64>,
}

#[cfg(feature = "media-ffmpeg")]
impl Default for VideoDecoderLimits {
    fn default() -> Self {
        Self {
            max_alloc: Some(DEFAULT_VIDEO_MAX_ALLOC),
        }
    }
}

#[cfg(feature = "media-ffmpeg")]
#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoder {
    #[serde(default)]
    pub limits: VideoDecoderLimits,

    /// sample N frames per second
    #[serde(default)]
    pub fps: Option<f64>,
    /// sample at most N frames
    #[serde(default)]
    pub max_frames: Option<u64>,
    /// sample N frames in total (linspace)
    #[serde(default)]
    pub num_frames: Option<u64>,
    /// fail if some frames fail to decode
    #[serde(default)]
    pub strict: bool,
}

/// Media decoder configuration.
/// Used both for MDC server config and runtime `media_io_kwargs`.
/// When used at runtime, limits are enforced from MDC and cannot be overridden.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, ToSchema)]
pub struct MediaDecoder {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageDecoder>,
    #[cfg(feature = "media-ffmpeg")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoDecoder>,
    // TODO: audio decoder
    /// Rotate by the EXIF orientation tag as vllm-rs reads it: no XMP, no PNG eXIf after IDAT.
    /// Not in `image`: older frontends reject unknown `image` fields but ignore them here.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub apply_exif_orientation: bool,
}

const DEFAULT_HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);

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
    /// When `false`, reject every `http(s)://` URL; only `data:` URLs pass.
    #[serde(default = "default_allow_remote_urls")]
    pub allow_remote_urls: bool,
}

fn default_allow_remote_urls() -> bool {
    true
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
            allow_remote_urls: default_allow_remote_urls(),
        }
    }
}

impl MediaFetcher {
    fn with_internal_access(allow_internal: bool) -> Self {
        Self {
            allow_direct_ip: allow_internal,
            allow_direct_port: allow_internal,
            allow_private_ips: allow_internal,
            ..Self::default()
        }
    }

    /// Build a `MediaFetcher` whose defaults respect the shared
    /// `DYN_MM_ALLOW_INTERNAL` environment variable. Mirrors the Python
    /// `UrlValidationPolicy.from_env()` behavior so both fetch paths
    /// (frontend decode in Rust, backend decode in Python) honor the
    /// same on-prem opt-in flag.
    ///
    /// `DYN_MM_ALLOW_INTERNAL=1` flips `allow_direct_ip`,
    /// `allow_direct_port`, and `allow_private_ips` all to `true` at
    /// once.
    ///
    /// `DYN_MM_ALLOW_REMOTE_URLS=0` (or any other false value) sets
    /// `allow_remote_urls` to `false`.
    pub fn from_env() -> Self {
        let allow_internal = std::env::var("DYN_MM_ALLOW_INTERNAL").ok().as_deref() == Some("1");
        let allow_remote = std::env::var("DYN_MM_ALLOW_REMOTE_URLS").ok();
        Self {
            allow_remote_urls: Self::allow_remote_urls_from(allow_remote.as_deref()),
            ..Self::with_internal_access(allow_internal)
        }
    }

    /// Unset or empty keeps remote URLs on. A value that is not a boolean
    /// turns them off, so a typo fails closed.
    fn allow_remote_urls_from(value: Option<&str>) -> bool {
        let Some(value) = value.filter(|v| !v.trim().is_empty()) else {
            return true;
        };
        dynamo_runtime::config::parse_bool(value).unwrap_or_else(|error| {
            tracing::error!(%error, "Invalid DYN_MM_ALLOW_REMOTE_URLS; disabling remote media URLs");
            false
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_orientation_is_off_unless_the_card_sets_it() {
        let old_card: MediaDecoder =
            serde_json::from_value(serde_json::json!({"image": {}})).unwrap();
        assert!(!old_card.apply_exif_orientation);
        let default = serde_json::to_value(&old_card).unwrap();
        assert!(default.get("apply_exif_orientation").is_none());

        let card = serde_json::to_value(MediaDecoder {
            apply_exif_orientation: true,
            ..old_card
        })
        .unwrap();
        assert_eq!(card["apply_exif_orientation"], true);
        assert!(card["image"].get("apply_exif_orientation").is_none());
        assert!(
            serde_json::from_value::<MediaDecoder>(card)
                .unwrap()
                .apply_exif_orientation
        );
    }

    // parse_bool_opt owns the accepted spellings; this locks the default-on
    // behavior and explicit opt-out used by the frontend-local setting.
    #[test]
    fn test_libjpeg_defaults_on_unless_explicitly_disabled() {
        assert!(libjpeg_enabled(None));
        assert!(libjpeg_enabled(Some("")));
        assert!(libjpeg_enabled(Some("invalid")));
        assert!(!libjpeg_enabled(Some("0")));
    }

    #[test]
    fn test_libjpeg_selection_is_not_media_config() {
        let config =
            serde_json::to_value(ImageDecoder::default().with_libjpeg_for_benchmark(false))
                .unwrap();
        assert!(config.get("enable_libjpeg").is_none());

        let decoder: ImageDecoder = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(decoder.enable_libjpeg, default_enable_libjpeg());

        let error =
            serde_json::from_value::<ImageDecoder>(serde_json::json!({"enable_libjpeg": false}))
                .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn test_allow_remote_urls_from_env_value() {
        for off in [Some("0"), Some("disabled")] {
            assert!(!MediaFetcher::allow_remote_urls_from(off));
        }
        for on in [None, Some("1"), Some("")] {
            assert!(MediaFetcher::allow_remote_urls_from(on));
        }
    }

    #[test]
    fn test_internal_access_defaults() {
        let f = MediaFetcher::with_internal_access(false);
        assert!(!f.allow_private_ips);
        assert!(!f.allow_direct_ip);
        assert!(!f.allow_direct_port);

        let f = MediaFetcher::with_internal_access(true);
        assert!(f.allow_private_ips);
        assert!(f.allow_direct_ip);
        assert!(f.allow_direct_port);
    }
}
