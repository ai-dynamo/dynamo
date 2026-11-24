#[cfg(feature = "opencv-video")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::HashSet,
    io::Write,
    path::PathBuf,
    process::{Output, Stdio},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use bytes::{Bytes, BytesMut};
#[cfg(feature = "opencv-video")]
use opencv::{
    core::{Mat, Vector},
    prelude::*,
    videoio,
};
use reqwest::Client;
use tokio::{fs, io::AsyncReadExt, process::Command, task, time};
use tracing::info;
use url::Url;

use crate::audio::decode_audio_mono_f32;

const DEFAULT_VIDEO_PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_IMAGE_MAX_INPUT_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_VIDEO_MAX_INPUT_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_VIDEO_MAX_DECODED_BYTES: usize = 1024 * 1024 * 1024;
const DEFAULT_AUDIO_MAX_INPUT_BYTES: usize = 256 * 1024 * 1024;
const _: () = assert!(DEFAULT_VIDEO_MAX_INPUT_BYTES < DEFAULT_VIDEO_MAX_DECODED_BYTES);
static VIDEO_DECODE_BACKEND: OnceLock<Option<String>> = OnceLock::new();
static LOG_VIDEO_DECODE_TIMING: OnceLock<bool> = OnceLock::new();
static VIDEO_PROCESS_TIMEOUT: OnceLock<Duration> = OnceLock::new();
static IMAGE_MAX_INPUT_BYTES: OnceLock<usize> = OnceLock::new();
static VIDEO_MAX_INPUT_BYTES: OnceLock<usize> = OnceLock::new();
static VIDEO_MAX_DECODED_BYTES: OnceLock<usize> = OnceLock::new();
static AUDIO_MAX_INPUT_BYTES: OnceLock<usize> = OnceLock::new();
#[cfg(feature = "opencv-video")]
static ACTIVE_OPENCV_DECODES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "opencv-video")]
static AVAILABLE_OPENCV_CPUS: OnceLock<usize> = OnceLock::new();
#[cfg(feature = "opencv-video")]
const MAX_OPENCV_DECODER_THREADS: usize = 8;
#[cfg(feature = "opencv-video")]
const OPENCV_DECODE_BURST_COALESCE: Duration = Duration::from_millis(5);
#[cfg(feature = "opencv-video")]
const OPENCV_LOW_CONCURRENCY_LIMIT: usize = 8;
#[cfg(feature = "opencv-video")]
const OPENCV_LOW_CONCURRENCY_CPU_MULTIPLIER: usize = 2;
#[cfg(feature = "opencv-video")]
const OPENCV_HIGH_CONCURRENCY_CPU_BUDGET_NUMERATOR: usize = 6;
#[cfg(feature = "opencv-video")]
const OPENCV_HIGH_CONCURRENCY_CPU_BUDGET_DENOMINATOR: usize = 7;

use super::{
    error::MediaConnectorError,
    types::{
        AudioClip, AudioSource, DecodedRgbFrame, DecodedRgbVideo, ImageDetail, ImageFrame,
        ImageSource, VideoClip, VideoSource,
    },
};

#[derive(Clone)]
pub struct MediaConnectorConfig {
    pub allowed_domains: Option<Vec<String>>,
    pub allowed_local_media_path: Option<PathBuf>,
    pub fetch_timeout: Duration,
}

impl Default for MediaConnectorConfig {
    fn default() -> Self {
        Self {
            allowed_domains: None,
            allowed_local_media_path: None,
            fetch_timeout: Duration::from_secs(10),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageFetchConfig {
    pub detail: ImageDetail,
}

impl Default for ImageFetchConfig {
    fn default() -> Self {
        Self {
            detail: ImageDetail::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VideoFetchConfig {
    pub min_frames: usize,
    pub max_frames: usize,
    pub sample_fps: f32,
}

impl Default for VideoFetchConfig {
    fn default() -> Self {
        Self {
            min_frames: 4,
            max_frames: 768,
            sample_fps: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MediaSource {
    Url(String),
    DataUrl(String),
    InlineBytes(Vec<u8>),
    File(PathBuf),
}

#[derive(Clone)]
pub struct MediaConnector {
    client: Client,
    allowed_domains: Option<HashSet<String>>,
    allowed_local_media_path: Option<PathBuf>,
    fetch_timeout: Duration,
}

impl MediaConnector {
    pub fn new(client: Client, config: MediaConnectorConfig) -> Result<Self, MediaConnectorError> {
        let allowed_domains = config.allowed_domains.map(|domains| {
            domains
                .into_iter()
                .map(|d| d.to_ascii_lowercase())
                .collect::<HashSet<_>>()
        });

        let allowed_local_media_path = if let Some(path) = config.allowed_local_media_path {
            Some(std::fs::canonicalize(path)?)
        } else {
            None
        };

        Ok(Self {
            client,
            allowed_domains,
            allowed_local_media_path,
            fetch_timeout: config.fetch_timeout,
        })
    }

    pub async fn fetch_image(
        &self,
        source: MediaSource,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        match source {
            MediaSource::Url(url) => self.fetch_http_image(url, cfg).await,
            MediaSource::DataUrl(data_url) => self.fetch_data_url(data_url, cfg).await,
            MediaSource::InlineBytes(bytes) => {
                self.decode_image(bytes.into(), cfg.detail, ImageSource::InlineBytes)
                    .await
            }
            MediaSource::File(path) => self.fetch_file(path, cfg).await,
        }
    }

    pub async fn fetch_video(
        &self,
        source: MediaSource,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        match source {
            MediaSource::Url(url) => self.fetch_http_video(url, cfg).await,
            MediaSource::DataUrl(data_url) => self.fetch_video_data_url(data_url, cfg).await,
            MediaSource::InlineBytes(bytes) => {
                self.decode_video(bytes.into(), cfg, VideoSource::InlineBytes)
                    .await
            }
            MediaSource::File(path) => self.fetch_video_file(path, cfg).await,
        }
    }

    pub async fn fetch_audio(
        &self,
        source: MediaSource,
    ) -> Result<Arc<AudioClip>, MediaConnectorError> {
        match source {
            MediaSource::Url(url) => self.fetch_http_audio(url).await,
            MediaSource::DataUrl(data_url) => self.fetch_audio_data_url(data_url).await,
            MediaSource::InlineBytes(bytes) => {
                self.decode_audio(bytes.into(), AudioSource::InlineBytes)
                    .await
            }
            MediaSource::File(path) => self.fetch_audio_file(path).await,
        }
    }

    async fn fetch_http_image(
        &self,
        url: String,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let parsed = Url::parse(&url).map_err(|_| MediaConnectorError::InvalidUrl(url.clone()))?;
        self.ensure_domain_allowed(&parsed)?;

        let mut req = self.client.get(parsed.as_str());
        if self.fetch_timeout > Duration::ZERO {
            req = req.timeout(self.fetch_timeout);
        }

        let resp = req.send().await.map_err(|err| {
            if err.is_timeout() {
                MediaConnectorError::Timeout(self.fetch_timeout)
            } else {
                MediaConnectorError::Http(err)
            }
        })?;

        let resp = resp.error_for_status()?;
        let bytes = collect_http_body_with_limit(resp, image_max_input_bytes(), "image").await?;
        self.decode_image(
            bytes,
            cfg.detail,
            ImageSource::Url {
                url: parsed.to_string(),
            },
        )
        .await
    }

    async fn fetch_data_url(
        &self,
        data_url: String,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let (metadata, data) = data_url
            .split_once(',')
            .ok_or_else(|| MediaConnectorError::DataUrl("missing comma in data url".into()))?;

        if !metadata.ends_with(";base64") {
            return Err(MediaConnectorError::DataUrl(
                "only base64 encoded data URLs are supported".into(),
            ));
        }

        let data = data.trim();
        let decoded = decode_base64_with_limit(data, image_max_input_bytes(), "image")?;
        self.decode_image(decoded.into(), cfg.detail, ImageSource::DataUrl)
            .await
    }

    async fn fetch_video_data_url(
        &self,
        data_url: String,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        let (metadata, data) = data_url
            .split_once(',')
            .ok_or_else(|| MediaConnectorError::DataUrl("missing comma in data url".into()))?;

        if !metadata.ends_with(";base64") {
            return Err(MediaConnectorError::DataUrl(
                "only base64 encoded data URLs are supported".into(),
            ));
        }

        let data = data.trim();
        let decoded = decode_base64_with_limit(data, video_max_input_bytes(), "video")?;
        self.decode_video(decoded.into(), cfg, VideoSource::DataUrl)
            .await
    }

    async fn fetch_audio_data_url(
        &self,
        data_url: String,
    ) -> Result<Arc<AudioClip>, MediaConnectorError> {
        let (metadata, data) = data_url
            .split_once(',')
            .ok_or_else(|| MediaConnectorError::DataUrl("missing comma in data url".into()))?;

        if !metadata.ends_with(";base64") {
            return Err(MediaConnectorError::DataUrl(
                "only base64 encoded data URLs are supported".into(),
            ));
        }

        let data = data.trim();
        let decoded = decode_base64_with_limit(data, audio_max_input_bytes(), "audio")?;
        self.decode_audio(decoded.into(), AudioSource::DataUrl)
            .await
    }

    async fn fetch_file(
        &self,
        path: PathBuf,
        cfg: ImageFetchConfig,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        let allowed_root = self
            .allowed_local_media_path
            .as_ref()
            .ok_or_else(|| MediaConnectorError::DisallowedLocalPath(path.display().to_string()))?;

        let canonical = fs::canonicalize(&path).await?;
        if !canonical.starts_with(allowed_root) {
            return Err(MediaConnectorError::DisallowedLocalPath(
                path.display().to_string(),
            ));
        }

        let bytes = read_file_with_limit(&canonical, image_max_input_bytes(), "image").await?;
        self.decode_image(bytes, cfg.detail, ImageSource::File { path: canonical })
            .await
    }

    async fn fetch_http_video(
        &self,
        url: String,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        let parsed = Url::parse(&url).map_err(|_| MediaConnectorError::InvalidUrl(url.clone()))?;
        self.ensure_domain_allowed(&parsed)?;

        let mut req = self.client.get(parsed.as_str());
        if self.fetch_timeout > Duration::ZERO {
            req = req.timeout(self.fetch_timeout);
        }

        let resp = req.send().await.map_err(|err| {
            if err.is_timeout() {
                MediaConnectorError::Timeout(self.fetch_timeout)
            } else {
                MediaConnectorError::Http(err)
            }
        })?;

        let resp = resp.error_for_status()?;
        let bytes = collect_http_body_with_limit(resp, video_max_input_bytes(), "video").await?;
        self.decode_video(
            bytes,
            cfg,
            VideoSource::Url {
                url: parsed.to_string(),
            },
        )
        .await
    }

    async fn fetch_http_audio(&self, url: String) -> Result<Arc<AudioClip>, MediaConnectorError> {
        let parsed = Url::parse(&url).map_err(|_| MediaConnectorError::InvalidUrl(url.clone()))?;
        self.ensure_domain_allowed(&parsed)?;

        let mut req = self.client.get(parsed.as_str());
        if self.fetch_timeout > Duration::ZERO {
            req = req.timeout(self.fetch_timeout);
        }

        let resp = req.send().await.map_err(|err| {
            if err.is_timeout() {
                MediaConnectorError::Timeout(self.fetch_timeout)
            } else {
                MediaConnectorError::Http(err)
            }
        })?;

        let resp = resp.error_for_status()?;
        let bytes = collect_http_body_with_limit(resp, audio_max_input_bytes(), "audio").await?;
        self.decode_audio(
            bytes,
            AudioSource::Url {
                url: parsed.to_string(),
            },
        )
        .await
    }

    async fn fetch_video_file(
        &self,
        path: PathBuf,
        cfg: VideoFetchConfig,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        let allowed_root = self
            .allowed_local_media_path
            .as_ref()
            .ok_or_else(|| MediaConnectorError::DisallowedLocalPath(path.display().to_string()))?;

        let canonical = fs::canonicalize(&path).await?;
        if !canonical.starts_with(allowed_root) {
            return Err(MediaConnectorError::DisallowedLocalPath(
                path.display().to_string(),
            ));
        }

        let bytes = read_file_with_limit(&canonical, video_max_input_bytes(), "video").await?;
        self.decode_video(bytes, cfg, VideoSource::File { path: canonical })
            .await
    }

    async fn fetch_audio_file(&self, path: PathBuf) -> Result<Arc<AudioClip>, MediaConnectorError> {
        let allowed_root = self
            .allowed_local_media_path
            .as_ref()
            .ok_or_else(|| MediaConnectorError::DisallowedLocalPath(path.display().to_string()))?;

        let canonical = fs::canonicalize(&path).await?;
        if !canonical.starts_with(allowed_root) {
            return Err(MediaConnectorError::DisallowedLocalPath(
                path.display().to_string(),
            ));
        }

        let bytes = read_file_with_limit(&canonical, audio_max_input_bytes(), "audio").await?;
        self.decode_audio(bytes, AudioSource::File { path: canonical })
            .await
    }

    fn ensure_domain_allowed(&self, url: &Url) -> Result<(), MediaConnectorError> {
        if let Some(allowed) = &self.allowed_domains {
            let host = url
                .host_str()
                .map(|h| h.to_ascii_lowercase())
                .ok_or_else(|| MediaConnectorError::InvalidUrl(url.to_string()))?;
            if !allowed.contains(&host) {
                return Err(MediaConnectorError::DisallowedDomain(host));
            }
        }
        Ok(())
    }

    async fn decode_image(
        &self,
        bytes: Bytes,
        detail: ImageDetail,
        source: ImageSource,
    ) -> Result<Arc<ImageFrame>, MediaConnectorError> {
        ensure_input_byte_limit(bytes.len(), image_max_input_bytes(), "image")?;
        let hash = crate::hasher::hash_image(&bytes);

        // Decode JPEGs through libjpeg-turbo (PIL-compatible defaults: accurate
        // IDCT + fancy upsampling) so pixel values match vLLM bit-for-bit; the
        // pure-Rust decoder diverges by a few levels, which the vision encoder
        // amplifies into an embedding shift. Non-JPEG inputs and any turbojpeg
        // failure fall back to the `image` crate.
        let bytes_for_decode = bytes.clone();
        let image = task::spawn_blocking(
            move || -> Result<image::DynamicImage, MediaConnectorError> {
                if let Some(img) = crate::jpeg_turbo::decode_jpeg_rgb(&bytes_for_decode) {
                    return Ok(img);
                }
                let cursor = std::io::Cursor::new(bytes_for_decode);
                let reader = image::ImageReader::new(cursor).with_guessed_format()?;
                Ok(reader.decode()?)
            },
        )
        .await
        .map_err(MediaConnectorError::Blocking)??;

        Ok(Arc::new(ImageFrame::new(
            image, bytes, detail, source, hash,
        )))
    }

    async fn decode_audio(
        &self,
        bytes: Bytes,
        source: AudioSource,
    ) -> Result<Arc<AudioClip>, MediaConnectorError> {
        ensure_input_byte_limit(bytes.len(), audio_max_input_bytes(), "audio")?;
        let hash = crate::hasher::hash_audio(&bytes);
        let decoded = decode_audio_mono_f32(&bytes)
            .await
            .map_err(|e| MediaConnectorError::AudioDecode(e.to_string()))?;
        Ok(Arc::new(AudioClip::new(bytes, decoded, source, hash)))
    }

    async fn decode_video(
        &self,
        bytes: Bytes,
        cfg: VideoFetchConfig,
        source: VideoSource,
    ) -> Result<Arc<VideoClip>, MediaConnectorError> {
        ensure_input_byte_limit(bytes.len(), video_max_input_bytes(), "video")?;
        if cfg.max_frames == 0 {
            return Err(MediaConnectorError::VideoDecode(
                "max_frames must be greater than 0".to_string(),
            ));
        }
        if cfg.min_frames == 0 {
            return Err(MediaConnectorError::VideoDecode(
                "min_frames must be greater than 0".to_string(),
            ));
        }
        if cfg.min_frames > cfg.max_frames {
            return Err(MediaConnectorError::VideoDecode(
                "min_frames must be less than or equal to max_frames".to_string(),
            ));
        }
        if !cfg.sample_fps.is_finite() || cfg.sample_fps <= 0.0 {
            return Err(MediaConnectorError::VideoDecode(
                "sample_fps must be finite and greater than 0".to_string(),
            ));
        }

        let hash = crate::hasher::hash_video(&bytes);
        let decoded = decode_video_frames(bytes.clone(), cfg).await?;

        let clip = match decoded {
            DecodedVideoFrames::Images { frames, sample_fps } => {
                VideoClip::new_with_sample_fps(frames, bytes, source, hash, sample_fps)
            }
            DecodedVideoFrames::Rgb { video, sample_fps } => {
                VideoClip::new_rgb_with_sample_fps(video, bytes, source, hash, sample_fps)
            }
        };
        Ok(Arc::new(clip))
    }
}

async fn read_file_with_limit(
    path: &std::path::Path,
    limit: usize,
    media: &'static str,
) -> Result<Bytes, MediaConnectorError> {
    let file = fs::File::open(path).await?;
    let limit_u64 = u64::try_from(limit).unwrap_or(u64::MAX);
    if file.metadata().await?.len() > limit_u64 {
        return Err(MediaConnectorError::PayloadTooLarge { media, limit });
    }

    // Read at most one byte beyond the limit. The post-read exact check also
    // covers a file growing after the metadata check.
    let mut reader = file.take(limit_u64.saturating_add(1));
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await?;
    ensure_input_byte_limit(bytes.len(), limit, media)?;
    Ok(Bytes::from(bytes))
}

fn ensure_input_byte_limit(
    input_bytes: usize,
    limit: usize,
    media: &'static str,
) -> Result<(), MediaConnectorError> {
    checked_payload_length(0, input_bytes, limit, media).map(|_| ())
}

async fn collect_http_body_with_limit(
    mut response: reqwest::Response,
    limit: usize,
    media: &'static str,
) -> Result<Bytes, MediaConnectorError> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(MediaConnectorError::PayloadTooLarge { media, limit });
    }

    let mut body = BytesMut::new();
    while let Some(chunk) = response.chunk().await? {
        checked_payload_length(body.len(), chunk.len(), limit, media)?;
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

fn checked_payload_length(
    current: usize,
    additional: usize,
    limit: usize,
    media: &'static str,
) -> Result<usize, MediaConnectorError> {
    current
        .checked_add(additional)
        .filter(|length| *length <= limit)
        .ok_or(MediaConnectorError::PayloadTooLarge { media, limit })
}

fn decode_base64_with_limit(
    encoded: &str,
    limit: usize,
    media: &'static str,
) -> Result<Vec<u8>, MediaConnectorError> {
    // A padded base64 encoding of at most `limit` bytes needs no more than
    // ceil(limit / 3) * 4 input bytes. Reject longer strings before the base64
    // decoder allocates; the exact decoded-length check below handles the up
    // to two-byte slack at the boundary.
    let max_encoded_len = (limit as u128).div_ceil(3) * 4;
    if encoded.len() as u128 > max_encoded_len {
        return Err(MediaConnectorError::PayloadTooLarge { media, limit });
    }

    let decoded = BASE64_STANDARD.decode(encoded)?;
    checked_payload_length(0, decoded.len(), limit, media)?;
    Ok(decoded)
}

fn env_byte_limit(cache: &'static OnceLock<usize>, env_var: &str, default: usize) -> usize {
    *cache.get_or_init(|| {
        std::env::var(env_var)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|bytes| *bytes > 0)
            .unwrap_or(default)
    })
}

fn image_max_input_bytes() -> usize {
    env_byte_limit(
        &IMAGE_MAX_INPUT_BYTES,
        "SMG_IMAGE_MAX_INPUT_BYTES",
        DEFAULT_IMAGE_MAX_INPUT_BYTES,
    )
}

fn video_max_input_bytes() -> usize {
    env_byte_limit(
        &VIDEO_MAX_INPUT_BYTES,
        "SMG_VIDEO_MAX_INPUT_BYTES",
        DEFAULT_VIDEO_MAX_INPUT_BYTES,
    )
}

fn audio_max_input_bytes() -> usize {
    env_byte_limit(
        &AUDIO_MAX_INPUT_BYTES,
        "SMG_AUDIO_MAX_INPUT_BYTES",
        DEFAULT_AUDIO_MAX_INPUT_BYTES,
    )
}

enum DecodedVideoFrames {
    Images {
        frames: Vec<image::DynamicImage>,
        sample_fps: f32,
    },
    Rgb {
        video: DecodedRgbVideo,
        sample_fps: f32,
    },
}

async fn decode_video_frames(
    bytes: Bytes,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    #[cfg(feature = "opencv-video")]
    let input_bytes = bytes.len();
    match video_decode_backend_override() {
        Some("ffmpeg") => decode_video_bytes_with_ffmpeg(bytes, cfg).await,
        Some("opencv") => {
            #[cfg(feature = "opencv-video")]
            {
                let opencv_bytes = bytes.clone();
                let result = task::spawn_blocking(move || {
                    decode_video_with_opencv_bytes_logged(opencv_bytes, input_bytes, cfg)
                })
                .await
                .map_err(MediaConnectorError::Blocking)?;
                match result {
                    Ok(frames) => Ok(frames),
                    Err(error) => {
                        if log_video_decode_timing_enabled() {
                            info!(
                                error = %error,
                                "smg_mm_timing video_decode_opencv_buffer_fallback"
                            );
                        }
                        decode_video_bytes_with_tempfile(bytes, cfg)
                            .await
                            .map_err(|fallback_error| {
                                MediaConnectorError::VideoDecode(format!(
                                    "buffered OpenCV decode failed: {error}; tempfile OpenCV fallback failed: {fallback_error}"
                                ))
                            })
                    }
                }
            }
            #[cfg(not(feature = "opencv-video"))]
            {
                Err(MediaConnectorError::VideoDecode(
                    "SMG_VIDEO_DECODE_BACKEND=opencv requires the opencv-video feature".to_string(),
                ))
            }
        }
        Some(backend) => Err(MediaConnectorError::VideoDecode(format!(
            "unsupported SMG_VIDEO_DECODE_BACKEND={backend}; expected auto, opencv, or ffmpeg"
        ))),
        None => {
            #[cfg(feature = "opencv-video")]
            {
                let opencv_bytes = bytes.clone();
                let opencv_result = task::spawn_blocking(move || {
                    decode_video_with_opencv_bytes_logged(opencv_bytes, input_bytes, cfg)
                })
                .await
                .map_err(MediaConnectorError::Blocking)?;
                match opencv_result {
                    Ok(frames) => Ok(frames),
                    Err(opencv_error) => {
                        if log_video_decode_timing_enabled() {
                            info!(
                                error = %opencv_error,
                                "smg_mm_timing video_decode_auto_opencv_fallback"
                            );
                        }
                        decode_video_bytes_with_tempfile(bytes, cfg)
                            .await
                            .map_err(|fallback_error| {
                                MediaConnectorError::VideoDecode(format!(
                                    "buffered OpenCV decode failed: {opencv_error}; tempfile fallback failed: {fallback_error}"
                                ))
                            })
                    }
                }
            }
            #[cfg(not(feature = "opencv-video"))]
            {
                decode_video_bytes_with_ffmpeg(bytes, cfg).await
            }
        }
    }
}

#[cfg(feature = "opencv-video")]
async fn decode_video_bytes_with_tempfile(
    bytes: Bytes,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let input_bytes = bytes.len();
    let input_file = {
        let bytes = bytes.clone();
        task::spawn_blocking(move || write_temp_video_file(&bytes))
            .await
            .map_err(MediaConnectorError::Blocking)??
    };
    decode_video_frames_from_path(input_file.path(), input_bytes, cfg).await
}

async fn decode_video_bytes_with_ffmpeg(
    bytes: Bytes,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let input_bytes = bytes.len();
    let input_file = {
        let bytes = bytes.clone();
        task::spawn_blocking(move || write_temp_video_file(&bytes))
            .await
            .map_err(MediaConnectorError::Blocking)??
    };
    let input_path = input_file.path().to_path_buf();
    decode_video_with_ffmpeg(&input_path, input_bytes, cfg).await
}

#[cfg(feature = "opencv-video")]
async fn decode_video_frames_from_path(
    input_path: &std::path::Path,
    input_bytes: usize,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    match video_decode_backend_override() {
        Some("ffmpeg") => decode_video_with_ffmpeg(input_path, input_bytes, cfg).await,
        Some("opencv") => {
            #[cfg(feature = "opencv-video")]
            {
                let input_path = input_path.to_path_buf();
                task::spawn_blocking(move || {
                    decode_video_with_opencv_logged(&input_path, input_bytes, cfg)
                })
                .await
                .map_err(MediaConnectorError::Blocking)?
            }
            #[cfg(not(feature = "opencv-video"))]
            {
                Err(MediaConnectorError::VideoDecode(
                    "SMG_VIDEO_DECODE_BACKEND=opencv requires the opencv-video feature".to_string(),
                ))
            }
        }
        Some(backend) => Err(MediaConnectorError::VideoDecode(format!(
            "unsupported SMG_VIDEO_DECODE_BACKEND={backend}; expected auto, opencv, or ffmpeg"
        ))),
        None => {
            #[cfg(feature = "opencv-video")]
            {
                // OpenCV samples by frame index while the FFmpeg fallback uses an
                // fps filter, so the fallback can select a different frame set.
                let opencv_input_path = input_path.to_path_buf();
                let opencv_result = task::spawn_blocking(move || {
                    decode_video_with_opencv_logged(&opencv_input_path, input_bytes, cfg)
                })
                .await
                .map_err(MediaConnectorError::Blocking)?;

                match opencv_result {
                    Ok(frames) => Ok(frames),
                    Err(opencv_error) => {
                        if log_video_decode_timing_enabled() {
                            info!(
                                error = %opencv_error,
                                "smg_mm_timing video_decode_auto_opencv_fallback"
                            );
                        }

                        match decode_video_with_ffmpeg(input_path, input_bytes, cfg).await {
                            Ok(frames) => Ok(frames),
                            Err(ffmpeg_error) => Err(MediaConnectorError::VideoDecode(format!(
                                "OpenCV decode failed: {opencv_error}; ffmpeg fallback failed: {ffmpeg_error}"
                            ))),
                        }
                    }
                }
            }

            #[cfg(not(feature = "opencv-video"))]
            {
                decode_video_with_ffmpeg(input_path, input_bytes, cfg).await
            }
        }
    }
}

#[cfg(feature = "opencv-video")]
fn decode_video_with_opencv_logged(
    input_path: &std::path::Path,
    input_bytes: usize,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let started = Instant::now();
    let result = decode_video_with_opencv_file(input_path, cfg);
    match &result {
        Ok(_) => log_video_decode_backend_timing("opencv", started, input_bytes, cfg, None),
        Err(error) => {
            log_video_decode_backend_timing("opencv", started, input_bytes, cfg, Some(error));
        }
    }
    result
}

#[cfg(feature = "opencv-video")]
fn decode_video_with_opencv_bytes_logged(
    bytes: Bytes,
    input_bytes: usize,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let started = Instant::now();
    let result = decode_video_with_opencv_bytes(bytes, cfg);
    match &result {
        Ok(_) => log_video_decode_backend_timing("opencv_buffer", started, input_bytes, cfg, None),
        Err(error) => {
            log_video_decode_backend_timing(
                "opencv_buffer",
                started,
                input_bytes,
                cfg,
                Some(error),
            );
        }
    }
    result
}

fn video_decode_backend_override() -> Option<&'static str> {
    VIDEO_DECODE_BACKEND
        .get_or_init(|| {
            let backend = std::env::var("SMG_VIDEO_DECODE_BACKEND")
                .ok()?
                .trim()
                .to_ascii_lowercase();
            match backend.as_str() {
                "" | "auto" => None,
                _ => Some(backend),
            }
        })
        .as_deref()
}

fn log_video_decode_timing_enabled() -> bool {
    *LOG_VIDEO_DECODE_TIMING.get_or_init(|| {
        std::env::var("SMG_LOG_MM_TIMING")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn log_video_decode_backend_timing(
    backend: &str,
    started: Instant,
    input_bytes: usize,
    cfg: VideoFetchConfig,
    error: Option<&MediaConnectorError>,
) {
    if !log_video_decode_timing_enabled() {
        return;
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match error {
        Some(error) => info!(
            backend,
            ok = false,
            input_bytes,
            min_frames = cfg.min_frames,
            max_frames = cfg.max_frames,
            sample_fps = cfg.sample_fps,
            elapsed_ms,
            error = %error,
            "smg_mm_timing video_decode_backend"
        ),
        None => info!(
            backend,
            ok = true,
            input_bytes,
            min_frames = cfg.min_frames,
            max_frames = cfg.max_frames,
            sample_fps = cfg.sample_fps,
            elapsed_ms,
            "smg_mm_timing video_decode_backend"
        ),
    }
}

#[cfg(feature = "opencv-video")]
fn decode_video_with_opencv_file(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let input = input_path.to_str().ok_or_else(|| {
        MediaConnectorError::VideoDecode(format!(
            "OpenCV video path is not valid UTF-8: {}",
            input_path.display()
        ))
    })?;

    let active_decode = ActiveOpenCvDecode::enter();
    let decoder_threads = opencv_decoder_threads(active_decode.count());
    let capture = open_opencv_video_capture(input, decoder_threads)?;
    decode_video_from_opencv_capture(capture, cfg)
}

#[cfg(feature = "opencv-video")]
fn decode_video_with_opencv_bytes(
    bytes: Bytes,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    let active_decode = ActiveOpenCvDecode::enter();
    let decoder_threads = opencv_decoder_threads(active_decode.count());
    let capture = open_opencv_video_capture_from_buffer(bytes, decoder_threads)?;
    decode_video_from_opencv_capture(capture, cfg)
}

#[cfg(feature = "opencv-video")]
trait OpenCvCaptureOwner {
    fn capture_mut(&mut self) -> &mut videoio::VideoCapture;
}

#[cfg(feature = "opencv-video")]
impl OpenCvCaptureOwner for videoio::VideoCapture {
    fn capture_mut(&mut self) -> &mut videoio::VideoCapture {
        self
    }
}

#[cfg(feature = "opencv-video")]
impl OpenCvCaptureOwner for crate::opencv_buffer::BufferedCapture {
    fn capture_mut(&mut self) -> &mut videoio::VideoCapture {
        self.capture_mut()
    }
}

#[cfg(feature = "opencv-video")]
fn decode_video_from_opencv_capture<C>(
    mut capture: C,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError>
where
    C: OpenCvCaptureOwner,
{
    let capture = capture.capture_mut();
    let total_frames = capture
        .get(videoio::CAP_PROP_FRAME_COUNT)
        .map_err(opencv_decode_error)?
        .round()
        .max(0.0) as usize;
    if total_frames == 0 {
        return Err(MediaConnectorError::VideoDecode(
            "OpenCV reported zero video frames".to_string(),
        ));
    }

    let fps = capture
        .get(videoio::CAP_PROP_FPS)
        .map_err(opencv_decode_error)?;
    let frame_indices = opencv_frame_indices(total_frames, fps, cfg);
    if frame_indices.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "OpenCV video sampling produced no frame indices".to_string(),
        ));
    }

    let sampled_frame_counts = counted_frame_indices(&frame_indices);
    let unique_frame_count = sampled_frame_counts.len();
    let mut rgb_output = None;
    let mut frames = Vec::new();
    frames.try_reserve(frame_indices.len()).map_err(|e| {
        MediaConnectorError::VideoDecode(format!(
            "failed to reserve {} decoded video frame records: {e}",
            frame_indices.len()
        ))
    })?;
    let mut bgr_frame = Mat::default();

    let timeout = video_process_timeout();
    let started = Instant::now();
    // Advance to each sampled frame by SEQUENTIALLY grabbing the intervening frames
    // (cheap decode-without-retrieve) and `read`ing only the sampled ones, instead of
    // calling `set(CAP_PROP_POS_FRAMES)` per frame. OpenCV's POS_FRAMES set flushes/
    // re-seeks the decoder on every call (~10 ms/frame even for adjacent frames);
    // sequential grab is ~1-2 ms/frame. This matches vLLM's OpenCV video backend and is
    // verified bit-exact vs the old per-frame seek on both dense and sparse (non-keyframe)
    // sampling, so accuracy is unchanged. `sampled_frame_counts` is monotonic.
    // Index of the most recently decoded frame (-1 = nothing read yet).
    let mut decoded_pos: i64 = -1;
    for (idx, repeat_count) in sampled_frame_counts {
        if started.elapsed() >= timeout {
            return Err(MediaConnectorError::VideoDecode(format!(
                "OpenCV timed out after {:.3} seconds",
                timeout.as_secs_f64()
            )));
        }

        // Skip-decode the frames between the current position and `idx` so the
        // following `read` lands on `idx` without a decoder flush/seek.
        while decoded_pos + 1 < idx as i64 {
            if started.elapsed() >= timeout {
                return Err(MediaConnectorError::VideoDecode(format!(
                    "OpenCV timed out after {:.3} seconds",
                    timeout.as_secs_f64()
                )));
            }
            if !capture.grab().map_err(opencv_decode_error)? {
                return Err(MediaConnectorError::VideoDecode(format!(
                    "OpenCV could not grab intervening frame to reach sampled frame {idx}"
                )));
            }
            decoded_pos += 1;
        }

        let read_successful = capture.read(&mut bgr_frame).map_err(opencv_decode_error)?;
        decoded_pos = idx as i64;
        if !read_successful || bgr_frame.empty() {
            continue;
        }

        let decoded_width = u32::try_from(bgr_frame.cols()).map_err(|_| {
            MediaConnectorError::VideoDecode(format!(
                "OpenCV produced invalid RGB frame width: {}",
                bgr_frame.cols()
            ))
        })?;
        let decoded_height = u32::try_from(bgr_frame.rows()).map_err(|_| {
            MediaConnectorError::VideoDecode(format!(
                "OpenCV produced invalid RGB frame height: {}",
                bgr_frame.rows()
            ))
        })?;
        if rgb_output.is_none() {
            let frame_size = rawvideo_frame_size(decoded_width, decoded_height)?;
            let decoded_bytes = frame_size.checked_mul(unique_frame_count).ok_or_else(|| {
                MediaConnectorError::VideoDecode(
                    "decoded video byte size overflow while reserving RGB frames".to_string(),
                )
            })?;
            ensure_decoded_byte_limit(decoded_bytes)?;
            rgb_output = Some(
                crate::opencv_buffer::RgbOutputBuffer::with_capacity(decoded_bytes)
                    .map_err(MediaConnectorError::VideoDecode)?,
            );
        }
        let output = rgb_output.as_mut().ok_or_else(|| {
            MediaConnectorError::VideoDecode("missing OpenCV RGB output buffer".to_string())
        })?;
        let frame_size = rawvideo_frame_size(decoded_width, decoded_height)?;
        let new_len = output.len().checked_add(frame_size).ok_or_else(|| {
            MediaConnectorError::VideoDecode(
                "decoded video byte size overflow while appending RGB frame".to_string(),
            )
        })?;
        ensure_decoded_byte_limit(new_len)?;
        let (offset, len) = output
            .push_bgr(&bgr_frame, decoded_width, decoded_height)
            .map_err(MediaConnectorError::VideoDecode)?;
        let frame = DecodedRgbFrame {
            width: decoded_width,
            height: decoded_height,
            offset,
            len,
        };
        for _ in 0..repeat_count {
            frames.push(frame.clone());
        }
    }

    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "OpenCV produced no readable sampled frames".to_string(),
        ));
    }
    if frames.len() != frame_indices.len() {
        return Err(MediaConnectorError::VideoDecode(format!(
            "OpenCV produced {} sampled frames, expected {}",
            frames.len(),
            frame_indices.len()
        )));
    }

    let data = rgb_output
        .ok_or_else(|| {
            MediaConnectorError::VideoDecode("OpenCV produced no RGB output".to_string())
        })?
        .into_bytes();
    let sample_fps = effective_sample_fps(
        (fps.is_finite() && fps > 0.0).then_some(total_frames as f64 / fps),
        cfg,
    );
    Ok(DecodedVideoFrames::Rgb {
        video: DecodedRgbVideo::new(data, frames),
        sample_fps,
    })
}

#[cfg(feature = "opencv-video")]
fn open_opencv_video_capture_from_buffer(
    bytes: Bytes,
    decoder_threads: i32,
) -> Result<crate::opencv_buffer::BufferedCapture, MediaConnectorError> {
    crate::opencv_buffer::open_capture(bytes, decoder_threads).map_err(|error| {
        MediaConnectorError::VideoDecode(format!("OpenCV could not open video buffer: {error}"))
    })
}

#[cfg(feature = "opencv-video")]
fn open_opencv_video_capture(
    input: &str,
    decoder_threads: i32,
) -> Result<videoio::VideoCapture, MediaConnectorError> {
    // CAP_PROP_N_THREADS has ID 70. Referencing the numeric ID keeps builds
    // compatible with pre-4.8 headers; unsupported backends reject it and use
    // the parameter-free fallback below.
    const CAP_PROP_N_THREADS: i32 = 70;
    let params = Vector::from_slice(&[CAP_PROP_N_THREADS, decoder_threads]);
    if let Ok(capture) =
        videoio::VideoCapture::from_file_with_params(input, videoio::CAP_FFMPEG, &params)
    {
        if capture.is_opened().map_err(opencv_decode_error)? {
            return Ok(capture);
        }
    }

    for backend in [videoio::CAP_FFMPEG, videoio::CAP_ANY] {
        let Ok(capture) = videoio::VideoCapture::from_file(input, backend) else {
            continue;
        };
        if capture.is_opened().map_err(opencv_decode_error)? {
            return Ok(capture);
        }
    }

    Err(MediaConnectorError::VideoDecode(format!(
        "OpenCV could not open video: {input}"
    )))
}

#[cfg(feature = "opencv-video")]
struct ActiveOpenCvDecode {
    count: usize,
}

#[cfg(feature = "opencv-video")]
impl ActiveOpenCvDecode {
    fn enter() -> Self {
        ACTIVE_OPENCV_DECODES.fetch_add(1, Ordering::AcqRel);
        // Let a burst of decode tasks become visible before dividing the CPU
        // budget. The fixed window also covers blocking-pool ramp-up, where
        // arrivals may briefly appear stable before the full burst.
        std::thread::sleep(OPENCV_DECODE_BURST_COALESCE);
        Self {
            count: ACTIVE_OPENCV_DECODES.load(Ordering::Acquire),
        }
    }

    fn count(&self) -> usize {
        self.count
    }
}

#[cfg(feature = "opencv-video")]
impl Drop for ActiveOpenCvDecode {
    fn drop(&mut self) {
        ACTIVE_OPENCV_DECODES.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(feature = "opencv-video")]
fn opencv_decoder_threads(active_decodes: usize) -> i32 {
    let available = *AVAILABLE_OPENCV_CPUS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
    });
    adaptive_opencv_decoder_threads(available, active_decodes)
}

#[cfg(feature = "opencv-video")]
fn adaptive_opencv_decoder_threads(available_cpus: usize, active_decodes: usize) -> i32 {
    let available_cpus = available_cpus.max(1);
    let active_decodes = active_decodes.max(1);

    // Once eight or more independent decoders fill the CPU quota, codec-level
    // threading only adds scheduler contention.
    if active_decodes >= OPENCV_LOW_CONCURRENCY_LIMIT && active_decodes >= available_cpus {
        return 1;
    }

    let (decoder_budget, max_threads) = if active_decodes <= OPENCV_LOW_CONCURRENCY_LIMIT {
        let max_threads = if active_decodes <= 2 { 16 } else { 8 };
        (
            available_cpus.saturating_mul(OPENCV_LOW_CONCURRENCY_CPU_MULTIPLIER),
            max_threads,
        )
    } else {
        // Independent decoders supply request-level parallelism at high
        // concurrency. Reserve roughly one seventh of the CPU quota for frame
        // copies, request handling, and other non-decoder work.
        (
            available_cpus
                .saturating_mul(OPENCV_HIGH_CONCURRENCY_CPU_BUDGET_NUMERATOR)
                .div_ceil(OPENCV_HIGH_CONCURRENCY_CPU_BUDGET_DENOMINATOR),
            MAX_OPENCV_DECODER_THREADS,
        )
    };

    (decoder_budget.max(1) / active_decodes).clamp(1, max_threads) as i32
}

#[cfg(feature = "opencv-video")]
fn opencv_frame_indices(total_frames: usize, fps: f64, cfg: VideoFetchConfig) -> Vec<usize> {
    let mut target_frames = if fps.is_finite() && fps > 0.0 {
        let duration = total_frames as f64 / fps;
        (duration * cfg.sample_fps as f64).round() as usize
    } else {
        cfg.max_frames
    };
    target_frames = target_frames.clamp(cfg.min_frames, cfg.max_frames);
    target_frames = target_frames.max(1);
    if target_frames == 1 {
        return vec![0];
    }

    let last = (total_frames - 1) as f64;
    let denom = (target_frames - 1) as f64;
    (0..target_frames)
        .map(|idx| ((idx as f64 * last) / denom).floor() as usize)
        .collect()
}

#[cfg(feature = "opencv-video")]
fn counted_frame_indices(frame_indices: &[usize]) -> Vec<(usize, usize)> {
    let mut counts = Vec::new();
    for &idx in frame_indices {
        if let Some((last_idx, count)) = counts.last_mut() {
            if *last_idx == idx {
                *count += 1;
                continue;
            }
        }
        counts.push((idx, 1));
    }
    counts
}

#[cfg(feature = "opencv-video")]
fn opencv_decode_error(err: opencv::Error) -> MediaConnectorError {
    MediaConnectorError::VideoDecode(format!("OpenCV video decode failed: {err}"))
}

async fn decode_video_with_ffmpeg(
    input_path: &std::path::Path,
    input_bytes: usize,
    cfg: VideoFetchConfig,
) -> Result<DecodedVideoFrames, MediaConnectorError> {
    if let Ok(metadata) = probe_video_metadata(input_path).await {
        let sample_fps = effective_sample_fps(metadata.duration_seconds, cfg);
        let started = Instant::now();
        match decode_video_with_ffmpeg_ppm(input_path, cfg, metadata).await {
            Ok(rgb_video) => {
                log_video_decode_backend_timing("ffmpeg_ppm_file", started, input_bytes, cfg, None);
                return Ok(DecodedVideoFrames::Rgb {
                    video: rgb_video,
                    sample_fps,
                });
            }
            Err(error) => {
                log_video_decode_backend_timing(
                    "ffmpeg_ppm_file",
                    started,
                    input_bytes,
                    cfg,
                    Some(&error),
                );
            }
        }

        let started = Instant::now();
        match decode_video_with_ffmpeg_raw(input_path, cfg, metadata).await {
            Ok(rgb_video) => {
                log_video_decode_backend_timing("ffmpeg_raw_file", started, input_bytes, cfg, None);
                return Ok(DecodedVideoFrames::Rgb {
                    video: rgb_video,
                    sample_fps,
                });
            }
            Err(error) => {
                log_video_decode_backend_timing(
                    "ffmpeg_raw_file",
                    started,
                    input_bytes,
                    cfg,
                    Some(&error),
                );
            }
        }
    }

    let started = Instant::now();
    match decode_video_with_ffmpeg_png(input_path, cfg).await {
        Ok((frames, sample_fps)) => {
            log_video_decode_backend_timing("ffmpeg_png_file", started, input_bytes, cfg, None);
            Ok(DecodedVideoFrames::Images { frames, sample_fps })
        }
        Err(error) => {
            log_video_decode_backend_timing(
                "ffmpeg_png_file",
                started,
                input_bytes,
                cfg,
                Some(&error),
            );
            Err(error)
        }
    }
}

fn write_temp_video_file(bytes: &[u8]) -> Result<tempfile::NamedTempFile, MediaConnectorError> {
    let started = Instant::now();
    let mut input_file = tempfile::Builder::new()
        .prefix("smg-video-")
        .suffix(video_temp_suffix(bytes))
        .tempfile()?;
    input_file.write_all(bytes)?;
    input_file.flush()?;
    if log_video_decode_timing_enabled() {
        info!(
            nbytes = bytes.len(),
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            suffix = video_temp_suffix(bytes),
            "smg_mm_timing video_tempfile_write"
        );
    }
    Ok(input_file)
}

fn video_temp_suffix(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 12 && bytes.get(4..8) == Some(b"ftyp") {
        return ".mp4";
    }
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return ".webm";
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"AVI ") {
        return ".avi";
    }
    if bytes.starts_with(b"OggS") {
        return ".ogv";
    }
    if bytes.starts_with(&[0x00, 0x00, 0x01, 0xba]) {
        return ".mpg";
    }
    ".video"
}

fn video_process_timeout() -> Duration {
    *VIDEO_PROCESS_TIMEOUT.get_or_init(|| {
        std::env::var("SMG_VIDEO_PROCESS_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .map(Duration::from_secs_f64)
            .unwrap_or(DEFAULT_VIDEO_PROCESS_TIMEOUT)
    })
}

fn video_max_decoded_bytes() -> usize {
    env_byte_limit(
        &VIDEO_MAX_DECODED_BYTES,
        "SMG_VIDEO_MAX_DECODED_BYTES",
        DEFAULT_VIDEO_MAX_DECODED_BYTES,
    )
}

fn ensure_decoded_byte_limit(bytes: usize) -> Result<(), MediaConnectorError> {
    let limit = video_max_decoded_bytes();
    if bytes > limit {
        return Err(MediaConnectorError::VideoDecode(format!(
            "decoded video RGB payload would be {bytes} bytes, exceeding SMG_VIDEO_MAX_DECODED_BYTES={limit}"
        )));
    }
    Ok(())
}

fn checked_decoded_rgb_bytes(
    frame_count: usize,
    frame_size: usize,
) -> Result<usize, MediaConnectorError> {
    let bytes = frame_count.checked_mul(frame_size).ok_or_else(|| {
        MediaConnectorError::VideoDecode(format!(
            "decoded video byte size overflow for {frame_count} frames of {frame_size} bytes"
        ))
    })?;
    ensure_decoded_byte_limit(bytes)?;
    Ok(bytes)
}

async fn run_video_command_output(
    mut command: Command,
    program: &'static str,
) -> Result<Output, MediaConnectorError> {
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = command.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            MediaConnectorError::VideoDecode(format!(
                "{program} executable not found; install {program} to decode video_url inputs"
            ))
        } else {
            MediaConnectorError::Io(e)
        }
    })?;

    let timeout = video_process_timeout();
    match time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(error)) => Err(MediaConnectorError::Io(error)),
        Err(_) => Err(MediaConnectorError::VideoDecode(format!(
            "{program} timed out after {:.3} seconds",
            timeout.as_secs_f64()
        ))),
    }
}

async fn decode_video_with_ffmpeg_ppm(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
    metadata: VideoMetadata,
) -> Result<DecodedRgbVideo, MediaConnectorError> {
    let fps_filter = fps_filter_for_metadata(metadata, cfg);
    let max_frames = cfg.max_frames.to_string();
    let frame_size = rawvideo_frame_size(metadata.width, metadata.height)?;
    let target_frames = expected_sampled_frame_count(metadata, cfg);
    let decoded_bytes = checked_decoded_rgb_bytes(target_frames, frame_size)?;
    let output_limit = decoded_bytes
        .checked_add(target_frames.saturating_mul(64))
        .unwrap_or_else(video_max_decoded_bytes)
        .min(video_max_decoded_bytes())
        .to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(input_path)
        .args([
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-fs",
            &output_limit,
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]);
    let output = run_video_command_output(command, "ffmpeg").await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    parse_ppm_rgb_video(Bytes::from(output.stdout))
}

async fn decode_video_with_ffmpeg_raw(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
    metadata: VideoMetadata,
) -> Result<DecodedRgbVideo, MediaConnectorError> {
    let fps_filter = fps_filter_for_metadata(metadata, cfg);
    let max_frames = cfg.max_frames.to_string();
    let frame_size = rawvideo_frame_size(metadata.width, metadata.height)?;
    let target_frames = expected_sampled_frame_count(metadata, cfg);
    let decoded_bytes = checked_decoded_rgb_bytes(target_frames, frame_size)?;
    let output_limit = decoded_bytes.to_string();
    let mut command = Command::new("ffmpeg");
    // Rawvideo has no per-frame header, so we interpret stdout using ffprobe's
    // coded stream dimensions. Disable FFmpeg autorotation here; otherwise a
    // display-matrix rotation can swap output width/height and corrupt framing.
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-noautorotate",
            "-i",
        ])
        .arg(input_path)
        .args([
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-fs",
            &output_limit,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]);
    let output = run_video_command_output(command, "ffmpeg").await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    let frame_count = output.stdout.len() / frame_size;
    checked_decoded_rgb_bytes(frame_count, frame_size)?;
    let mut frames = Vec::new();
    frames.try_reserve(frame_count).map_err(|e| {
        MediaConnectorError::VideoDecode(format!(
            "failed to reserve {frame_count} decoded video frame records: {e}"
        ))
    })?;
    for idx in 0..frame_count {
        frames.push(DecodedRgbFrame {
            width: metadata.width,
            height: metadata.height,
            offset: idx * frame_size,
            len: frame_size,
        });
    }
    let remainder = output.stdout.len() % frame_size;
    if remainder != 0 {
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg rawvideo output has trailing partial frame: {remainder} bytes"
        )));
    }
    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "ffmpeg produced no frames".to_string(),
        ));
    }
    Ok(DecodedRgbVideo::new(Bytes::from(output.stdout), frames))
}

async fn decode_video_with_ffmpeg_png(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
) -> Result<(Vec<image::DynamicImage>, f32), MediaConnectorError> {
    let (fps_filter, sample_fps) = sampling_filter_for_video(input_path, cfg).await;
    let max_frames = cfg.max_frames.to_string();
    let output_limit = video_max_decoded_bytes().to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(input_path)
        .args([
            "-vf",
            &fps_filter,
            "-frames:v",
            &max_frames,
            "-fs",
            &output_limit,
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]);
    let output = run_video_command_output(command, "ffmpeg").await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffmpeg failed: {stderr}"
        )));
    }

    let pngs = split_png_stream(&output.stdout)?;
    let mut frames = Vec::with_capacity(pngs.len());
    let mut decoded_bytes = 0usize;
    for png in pngs {
        let image = image::load_from_memory(png)?;
        let frame_size = rawvideo_frame_size(image.width(), image.height())?;
        decoded_bytes = decoded_bytes.checked_add(frame_size).ok_or_else(|| {
            MediaConnectorError::VideoDecode("PNG decoded byte size overflow".to_string())
        })?;
        ensure_decoded_byte_limit(decoded_bytes)?;
        frames.push(image);
    }
    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "ffmpeg produced no frames".to_string(),
        ));
    }
    Ok((frames, sample_fps))
}

#[derive(Debug, Clone, Copy)]
struct VideoMetadata {
    width: u32,
    height: u32,
    duration_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct ProbedVideoInfo {
    width: Option<u32>,
    height: Option<u32>,
    duration_seconds: Option<f64>,
}

async fn probe_video_metadata(
    input_path: &std::path::Path,
) -> Result<VideoMetadata, MediaConnectorError> {
    let info = probe_video_info(input_path).await?;
    let width = info.width.ok_or_else(|| {
        MediaConnectorError::VideoDecode("ffprobe did not return video width".to_string())
    })?;
    let height = info.height.ok_or_else(|| {
        MediaConnectorError::VideoDecode("ffprobe did not return video height".to_string())
    })?;
    Ok(VideoMetadata {
        width,
        height,
        duration_seconds: info.duration_seconds,
    })
}

async fn probe_video_info(
    input_path: &std::path::Path,
) -> Result<ProbedVideoInfo, MediaConnectorError> {
    let mut command = Command::new("ffprobe");
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration,duration_ts,time_base:format=duration",
            "-of",
            "json",
        ])
        .arg(input_path);
    let output = run_video_command_output(command, "ffprobe").await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(MediaConnectorError::VideoDecode(format!(
            "ffprobe failed: {stderr}"
        )));
    }

    parse_ffprobe_video_info(&output.stdout)
}

fn parse_ffprobe_video_info(stdout: &[u8]) -> Result<ProbedVideoInfo, MediaConnectorError> {
    let probe: serde_json::Value = serde_json::from_slice(stdout).map_err(|error| {
        MediaConnectorError::VideoDecode(format!("failed to parse ffprobe output: {error}"))
    })?;
    let video_stream = probe
        .get("streams")
        .and_then(serde_json::Value::as_array)
        .and_then(|streams| streams.first());

    let width = video_stream
        .and_then(|stream| stream.get("width"))
        .and_then(json_u32);
    let height = video_stream
        .and_then(|stream| stream.get("height"))
        .and_then(json_u32);
    let stream_duration = video_stream
        .and_then(|stream| stream.get("duration"))
        .and_then(json_positive_f64);
    let stream_time_base_duration = video_stream.and_then(|stream| {
        let duration_ts = stream.get("duration_ts").and_then(json_positive_f64)?;
        let time_base = stream
            .get("time_base")
            .and_then(serde_json::Value::as_str)
            .and_then(parse_time_base)?;
        let duration = duration_ts * time_base;
        (duration.is_finite() && duration > 0.0).then_some(duration)
    });
    let format_duration = probe
        .get("format")
        .and_then(|format| format.get("duration"))
        .and_then(json_positive_f64);

    Ok(ProbedVideoInfo {
        width,
        height,
        duration_seconds: stream_duration
            .or(stream_time_base_duration)
            .or(format_duration),
    })
}

fn json_u32(value: &serde_json::Value) -> Option<u32> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .or_else(|| value.as_str()?.parse::<u32>().ok())
}

fn json_positive_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn parse_time_base(value: &str) -> Option<f64> {
    let (numerator, denominator) = value.split_once('/')?;
    let numerator = numerator.parse::<f64>().ok()?;
    let denominator = denominator.parse::<f64>().ok()?;
    let time_base = numerator / denominator;
    (time_base.is_finite() && time_base > 0.0).then_some(time_base)
}

fn fps_filter_for_metadata(metadata: VideoMetadata, cfg: VideoFetchConfig) -> String {
    if let Some(duration) = metadata.duration_seconds {
        if let Some(filter) = fps_filter_for_duration(duration, cfg) {
            return filter;
        }
    }

    format!("fps={}", cfg.sample_fps)
}

fn expected_sampled_frame_count(metadata: VideoMetadata, cfg: VideoFetchConfig) -> usize {
    if let Some(duration) = metadata.duration_seconds {
        if duration.is_finite() && duration > 0.0 {
            return (duration * cfg.sample_fps as f64)
                .round()
                .clamp(cfg.min_frames as f64, cfg.max_frames as f64) as usize;
        }
    }
    cfg.max_frames
}

fn effective_sample_fps(duration_seconds: Option<f64>, cfg: VideoFetchConfig) -> f32 {
    duration_seconds
        .filter(|duration| duration.is_finite() && *duration > 0.0)
        .map(|duration| {
            let target_frames = (duration * cfg.sample_fps as f64)
                .round()
                .clamp(cfg.min_frames as f64, cfg.max_frames as f64);
            (target_frames / duration) as f32
        })
        .filter(|fps| fps.is_finite() && *fps > 0.0)
        .unwrap_or(cfg.sample_fps)
}

fn fps_filter_for_duration(duration: f64, cfg: VideoFetchConfig) -> Option<String> {
    if !duration.is_finite() || duration <= 0.0 {
        return None;
    }
    let target_frames = (duration * cfg.sample_fps as f64)
        .round()
        .clamp(cfg.min_frames as f64, cfg.max_frames as f64);
    let fps = (target_frames / duration).max(f64::EPSILON);
    Some(format!("fps={fps:.6}"))
}

async fn sampling_filter_for_video(
    input_path: &std::path::Path,
    cfg: VideoFetchConfig,
) -> (String, f32) {
    if let Ok(duration) = probe_video_duration_seconds(input_path).await {
        if let Some(filter) = fps_filter_for_duration(duration, cfg) {
            return (filter, effective_sample_fps(Some(duration), cfg));
        }
    }

    (format!("fps={}", cfg.sample_fps), cfg.sample_fps)
}

async fn probe_video_duration_seconds(
    input_path: &std::path::Path,
) -> Result<f64, MediaConnectorError> {
    match probe_video_info(input_path).await {
        Ok(ProbedVideoInfo {
            duration_seconds: Some(duration),
            ..
        }) => Ok(duration),
        Ok(_) | Err(_) => probe_video_duration_seconds_with_ffmpeg(input_path).await,
    }
}

async fn probe_video_duration_seconds_with_ffmpeg(
    input_path: &std::path::Path,
) -> Result<f64, MediaConnectorError> {
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(input_path);
    let output = run_video_command_output(command, "ffmpeg").await?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    parse_ffmpeg_duration_seconds(&stderr).ok_or_else(|| {
        MediaConnectorError::VideoDecode("failed to parse ffmpeg duration".to_string())
    })
}

fn parse_ffmpeg_duration_seconds(stderr: &str) -> Option<f64> {
    let marker = "Duration:";
    let start = stderr.find(marker)? + marker.len();
    let duration = stderr[start..].trim_start().split(',').next()?.trim();
    let mut parts = duration.split(':');
    let hours = parts.next()?.parse::<f64>().ok()?;
    let minutes = parts.next()?.parse::<f64>().ok()?;
    let seconds = parts.next()?.parse::<f64>().ok()?;
    Some(hours * 3600.0 + minutes * 60.0 + seconds)
}

fn split_png_stream(bytes: &[u8]) -> Result<Vec<&[u8]>, MediaConnectorError> {
    const PNG_SIG: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    const IEND: &[u8; 4] = b"IEND";

    let mut frames = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let Some(rel_start) = bytes[pos..]
            .windows(PNG_SIG.len())
            .position(|w| w == PNG_SIG)
        else {
            break;
        };
        let start = pos + rel_start;
        let mut cursor = start + PNG_SIG.len();

        loop {
            let remaining = bytes.len() - cursor;
            if remaining < 12 {
                return Err(MediaConnectorError::VideoDecode(
                    "truncated PNG frame in ffmpeg output".to_string(),
                ));
            }
            let mut len_bytes = [0_u8; 4];
            len_bytes.copy_from_slice(&bytes[cursor..cursor + 4]);
            let len = u32::from_be_bytes(len_bytes) as usize;
            let chunk_type = &bytes[cursor + 4..cursor + 8];
            if remaining - 12 < len {
                return Err(MediaConnectorError::VideoDecode(
                    "truncated PNG chunk in ffmpeg output".to_string(),
                ));
            }
            cursor += 12 + len;
            if chunk_type == IEND {
                frames.push(&bytes[start..cursor]);
                pos = cursor;
                break;
            }
        }
    }

    Ok(frames)
}

#[cfg(test)]
fn parse_ppm_stream(bytes: &[u8]) -> Result<Vec<image::DynamicImage>, MediaConnectorError> {
    let layouts = parse_ppm_frame_layout(bytes)?;
    let mut frames = Vec::with_capacity(layouts.len());
    for layout in layouts {
        let end = layout.offset.checked_add(layout.len).ok_or_else(|| {
            MediaConnectorError::VideoDecode("PPM frame size overflow".to_string())
        })?;
        let image = image::RgbImage::from_raw(
            layout.width,
            layout.height,
            bytes[layout.offset..end].to_vec(),
        )
        .ok_or_else(|| {
            MediaConnectorError::VideoDecode(format!(
                "failed to build RGB frame from {} bytes for {}x{} video",
                layout.len, layout.width, layout.height
            ))
        })?;
        frames.push(image::DynamicImage::ImageRgb8(image));
    }
    Ok(frames)
}

fn parse_ppm_rgb_video(bytes: Bytes) -> Result<DecodedRgbVideo, MediaConnectorError> {
    let layouts = parse_ppm_frame_layout(&bytes)?;
    let decoded_bytes = layouts.iter().try_fold(0usize, |total, frame| {
        total.checked_add(frame.len).ok_or_else(|| {
            MediaConnectorError::VideoDecode("PPM decoded byte size overflow".to_string())
        })
    })?;
    ensure_decoded_byte_limit(decoded_bytes)?;
    Ok(DecodedRgbVideo::new(bytes, layouts))
}

fn parse_ppm_frame_layout(bytes: &[u8]) -> Result<Vec<DecodedRgbFrame>, MediaConnectorError> {
    let mut frames = Vec::new();
    let mut pos = 0;

    while pos < bytes.len() {
        skip_ppm_whitespace_and_comments(bytes, &mut pos);
        if pos >= bytes.len() {
            break;
        }

        let magic = read_ppm_token(bytes, &mut pos)?.ok_or_else(|| {
            MediaConnectorError::VideoDecode("truncated PPM frame header".to_string())
        })?;
        if magic != b"P6" {
            return Err(MediaConnectorError::VideoDecode(format!(
                "unsupported PPM magic: {}",
                String::from_utf8_lossy(magic)
            )));
        }
        let width = parse_ppm_u32(bytes, &mut pos, "width")?;
        let height = parse_ppm_u32(bytes, &mut pos, "height")?;
        let max_value = parse_ppm_u32(bytes, &mut pos, "max value")?;
        if width == 0 || height == 0 {
            return Err(MediaConnectorError::VideoDecode(
                "PPM frame dimensions must be non-zero".to_string(),
            ));
        }
        if max_value != 255 {
            return Err(MediaConnectorError::VideoDecode(format!(
                "unsupported PPM max value: {max_value}"
            )));
        }
        if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
            return Err(MediaConnectorError::VideoDecode(
                "PPM header is not followed by pixel data".to_string(),
            ));
        }
        pos += 1;

        let frame_size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| {
                MediaConnectorError::VideoDecode(format!(
                    "PPM frame dimensions are too large: {width}x{height}"
                ))
            })?;
        let end = pos.checked_add(frame_size).ok_or_else(|| {
            MediaConnectorError::VideoDecode("PPM frame size overflow".to_string())
        })?;
        if end > bytes.len() {
            return Err(MediaConnectorError::VideoDecode(
                "truncated PPM frame pixel data".to_string(),
            ));
        }
        frames.push(DecodedRgbFrame {
            width,
            height,
            offset: pos,
            len: frame_size,
        });
        pos = end;
    }

    if frames.is_empty() {
        return Err(MediaConnectorError::VideoDecode(
            "ffmpeg produced no frames".to_string(),
        ));
    }

    Ok(frames)
}

fn rawvideo_frame_size(width: u32, height: u32) -> Result<usize, MediaConnectorError> {
    let frame_size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            MediaConnectorError::VideoDecode(format!(
                "video frame dimensions are too large: {width}x{height}"
            ))
        })?;
    if frame_size == 0 {
        return Err(MediaConnectorError::VideoDecode(
            "video frame dimensions must be non-zero".to_string(),
        ));
    }
    Ok(frame_size)
}

fn parse_ppm_u32(bytes: &[u8], pos: &mut usize, field: &str) -> Result<u32, MediaConnectorError> {
    let token = read_ppm_token(bytes, pos)?
        .ok_or_else(|| MediaConnectorError::VideoDecode(format!("truncated PPM {field} header")))?;
    std::str::from_utf8(token)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| {
            MediaConnectorError::VideoDecode(format!(
                "invalid PPM {field}: {}",
                String::from_utf8_lossy(token)
            ))
        })
}

fn read_ppm_token<'a>(
    bytes: &'a [u8],
    pos: &mut usize,
) -> Result<Option<&'a [u8]>, MediaConnectorError> {
    skip_ppm_whitespace_and_comments(bytes, pos);
    if *pos >= bytes.len() {
        return Ok(None);
    }

    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
        if bytes[*pos] == b'#' {
            return Err(MediaConnectorError::VideoDecode(
                "unexpected PPM comment inside token".to_string(),
            ));
        }
        *pos += 1;
    }
    Ok(Some(&bytes[start..*pos]))
}

fn skip_ppm_whitespace_and_comments(bytes: &[u8], pos: &mut usize) {
    loop {
        while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos < bytes.len() && bytes[*pos] == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
            continue;
        }
        break;
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use bytes::Bytes;
    use futures::stream;

    use super::{
        checked_payload_length, collect_http_body_with_limit, decode_base64_with_limit,
        effective_sample_fps, ensure_input_byte_limit, expected_sampled_frame_count,
        fps_filter_for_metadata, parse_ffmpeg_duration_seconds, parse_ffprobe_video_info,
        parse_ppm_stream, read_file_with_limit, split_png_stream, video_temp_suffix,
        MediaConnectorError, VideoFetchConfig, VideoMetadata,
    };

    const TINY_PNG: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 4,
        0, 0, 0, 181, 28, 12, 2, 0, 0, 0, 11, 73, 68, 65, 84, 120, 218, 99, 96, 96, 0, 0, 0, 3, 0,
        1, 43, 9, 141, 84, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];

    #[test]
    fn splits_concatenated_png_stream() {
        let mut stream = Vec::new();
        stream.extend_from_slice(TINY_PNG);
        stream.extend_from_slice(TINY_PNG);

        let frames = match split_png_stream(&stream) {
            Ok(frames) => frames,
            Err(err) => panic!("split png stream failed: {err}"),
        };
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], TINY_PNG);
        assert_eq!(frames[1], TINY_PNG);
    }

    #[test]
    fn parses_ffmpeg_duration() {
        let stderr = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'video.mp4':\n  Duration: 00:01:23.45, start: 0.000000, bitrate: 123 kb/s";
        assert_eq!(parse_ffmpeg_duration_seconds(stderr), Some(83.45));
    }

    #[test]
    fn ffprobe_metadata_prefers_short_video_stream_over_long_container() {
        let output = br#"{
            "streams": [{
                "width": 320,
                "height": 240,
                "duration": "1.000000",
                "duration_ts": 30,
                "time_base": "1/30"
            }],
            "format": {"duration": "120.000000"}
        }"#;
        let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
        assert_eq!(info.duration_seconds, Some(1.0));

        let cfg = VideoFetchConfig {
            min_frames: 4,
            max_frames: 8,
            sample_fps: 2.0,
        };
        let metadata = VideoMetadata {
            width: info.width.expect("video width"),
            height: info.height.expect("video height"),
            duration_seconds: info.duration_seconds,
        };
        assert_eq!(expected_sampled_frame_count(metadata, cfg), 4);
        assert_eq!(fps_filter_for_metadata(metadata, cfg), "fps=4.000000");
    }

    #[test]
    fn ffprobe_metadata_uses_stream_time_base_before_container_duration() {
        let output = br#"{
            "streams": [{
                "width": "640",
                "height": "360",
                "duration": "N/A",
                "duration_ts": 45,
                "time_base": "1/30"
            }],
            "format": {"duration": "90.000000"}
        }"#;
        let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
        assert_eq!(info.width, Some(640));
        assert_eq!(info.height, Some(360));
        assert_eq!(info.duration_seconds, Some(1.5));
    }

    #[test]
    fn detects_video_temp_suffix_from_container_header() {
        let mut mp4 = vec![0; 12];
        mp4[4..8].copy_from_slice(b"ftyp");
        assert_eq!(video_temp_suffix(&mp4), ".mp4");
        assert_eq!(video_temp_suffix(&[0x1a, 0x45, 0xdf, 0xa3]), ".webm");
        assert_eq!(video_temp_suffix(b"RIFF....AVI "), ".avi");
        assert_eq!(video_temp_suffix(b"OggS"), ".ogv");
        assert_eq!(video_temp_suffix(&[0x00, 0x00, 0x01, 0xba]), ".mpg");
        assert_eq!(video_temp_suffix(b"unknown"), ".video");
    }

    #[test]
    fn parses_concatenated_ppm_stream() {
        let stream = b"P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06P6\n# comment\n1 2\n255\n\x07\x08\x09\x0a\x0b\x0c";

        let frames = match parse_ppm_stream(stream) {
            Ok(frames) => frames,
            Err(err) => panic!("parse ppm stream failed: {err}"),
        };
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].width(), 2);
        assert_eq!(frames[0].height(), 1);
        assert_eq!(frames[1].width(), 1);
        assert_eq!(frames[1].height(), 2);
    }

    #[test]
    fn effective_sample_fps_tracks_min_and_max_frame_clamps() {
        let cfg = VideoFetchConfig {
            min_frames: 4,
            max_frames: 8,
            sample_fps: 2.0,
        };

        assert_eq!(effective_sample_fps(Some(1.0), cfg), 4.0);
        assert!((effective_sample_fps(Some(10.0), cfg) - 0.8).abs() < 1e-6);
        assert_eq!(effective_sample_fps(Some(3.0), cfg), 2.0);
        assert_eq!(effective_sample_fps(None, cfg), 2.0);
    }

    #[test]
    fn rejects_truncated_ppm_stream() {
        assert!(parse_ppm_stream(b"P6\n2 1\n255\n\x01\x02").is_err());
    }

    #[test]
    fn rejects_invalid_ppm_header() {
        assert!(parse_ppm_stream(b"P3\n1 1\n255\n\x01\x02\x03").is_err());
        assert!(parse_ppm_stream(b"P6\n1 1\n65535\n\x01\x02\x03").is_err());
    }

    #[test]
    fn rejects_zero_dimension_ppm_stream() {
        assert!(parse_ppm_stream(b"P6\n0 1\n255\n").is_err());
        assert!(parse_ppm_stream(b"P6\n1 0\n255\n").is_err());
    }

    #[test]
    fn rejects_overflowing_ppm_frame_size() {
        assert!(parse_ppm_stream(b"P6\n4294967295 4294967295\n255\n").is_err());
    }

    #[test]
    fn enforces_media_payload_limit_for_each_label() {
        for media in ["image", "video", "audio"] {
            assert!(ensure_input_byte_limit(4, 4, media).is_ok());
            assert!(matches!(
                ensure_input_byte_limit(5, 4, media),
                Err(MediaConnectorError::PayloadTooLarge { media: actual, limit: 4 })
                    if actual == media
            ));
        }

        assert!(checked_payload_length(usize::MAX, 1, usize::MAX, "audio").is_err());
    }

    #[test]
    fn enforces_base64_payload_limit_before_and_after_decode() {
        for media in ["image", "video", "audio"] {
            assert_eq!(decode_base64_with_limit("AAAA", 3, media).unwrap().len(), 3);
            assert!(matches!(
                decode_base64_with_limit("AAAA", 2, media),
                Err(MediaConnectorError::PayloadTooLarge { media: actual, limit: 2 })
                    if actual == media
            ));
        }

        assert!(matches!(
            decode_base64_with_limit("AAAAAAAAAAAAAAAA", 8, "audio"),
            Err(MediaConnectorError::PayloadTooLarge {
                media: "audio",
                limit: 8
            })
        ));
        assert!(matches!(
            decode_base64_with_limit("AAAAAAAAAAAA", 8, "audio"),
            Err(MediaConnectorError::PayloadTooLarge {
                media: "audio",
                limit: 8
            })
        ));
    }

    #[tokio::test]
    async fn reads_files_at_limit_and_rejects_limit_plus_one() -> Result<(), MediaConnectorError> {
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(b"12345")?;
        file.flush()?;

        for media in ["image", "video", "audio"] {
            let bytes = read_file_with_limit(file.path(), 5, media).await?;
            assert_eq!(bytes, Bytes::from_static(b"12345"));

            assert!(matches!(
                read_file_with_limit(file.path(), 4, media).await,
                Err(MediaConnectorError::PayloadTooLarge { media: actual, limit: 4 })
                    if actual == media
            ));
        }
        Ok(())
    }

    #[tokio::test]
    async fn collects_http_body_with_content_and_streaming_limits(
    ) -> Result<(), MediaConnectorError> {
        let known_oversized = reqwest::Response::from(http::Response::new(reqwest::Body::from(
            Bytes::from_static(b"12345"),
        )));
        assert!(matches!(
            collect_http_body_with_limit(known_oversized, 4, "image").await,
            Err(MediaConnectorError::PayloadTooLarge {
                media: "image",
                limit: 4
            })
        ));

        let oversized_stream = stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(b"123")),
            Ok(Bytes::from_static(b"45")),
        ]);
        let unknown_oversized = reqwest::Response::from(http::Response::new(
            reqwest::Body::wrap_stream(oversized_stream),
        ));
        assert!(matches!(
            collect_http_body_with_limit(unknown_oversized, 4, "video").await,
            Err(MediaConnectorError::PayloadTooLarge {
                media: "video",
                limit: 4
            })
        ));

        let within_limit_stream = stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(b"12")),
            Ok(Bytes::from_static(b"34")),
        ]);
        let within_limit = reqwest::Response::from(http::Response::new(
            reqwest::Body::wrap_stream(within_limit_stream),
        ));
        let body = collect_http_body_with_limit(within_limit, 4, "audio").await?;
        assert_eq!(body, Bytes::from_static(b"1234"));
        Ok(())
    }

    #[cfg(feature = "opencv-video")]
    #[test]
    fn opencv_sampling_preserves_min_frames_for_short_clips() {
        let cfg = VideoFetchConfig {
            min_frames: 4,
            max_frames: 8,
            sample_fps: 2.0,
        };
        let indices = super::opencv_frame_indices(1, 30.0, cfg);
        assert_eq!(indices, vec![0, 0, 0, 0]);
        assert_eq!(super::counted_frame_indices(&indices), vec![(0, 4)]);
    }

    #[cfg(feature = "opencv-video")]
    #[test]
    fn opencv_decoder_threads_share_cpu_budget_across_active_decodes() {
        assert_eq!(super::adaptive_opencv_decoder_threads(224, 1), 16);
        assert_eq!(super::adaptive_opencv_decoder_threads(2, 1), 4);
        assert_eq!(super::adaptive_opencv_decoder_threads(4, 2), 4);
        assert_eq!(super::adaptive_opencv_decoder_threads(8, 4), 4);
        assert_eq!(super::adaptive_opencv_decoder_threads(8, 8), 1);
        assert_eq!(super::adaptive_opencv_decoder_threads(8, 9), 1);
        assert_eq!(super::adaptive_opencv_decoder_threads(16, 8), 4);
        assert_eq!(super::adaptive_opencv_decoder_threads(16, 16), 1);
        assert_eq!(super::adaptive_opencv_decoder_threads(224, 8), 8);
        assert_eq!(super::adaptive_opencv_decoder_threads(224, 32), 6);
        assert_eq!(super::adaptive_opencv_decoder_threads(8, 32), 1);
        assert_eq!(super::adaptive_opencv_decoder_threads(1, 0), 2);
    }
}
