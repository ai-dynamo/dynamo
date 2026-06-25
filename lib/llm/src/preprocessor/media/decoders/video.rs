// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::os::fd::AsRawFd;
use std::sync::Once;

use anyhow::Result;
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::media::Type as MediaType;
use ffmpeg_next::software::scaling::{context::Context as Scaler, flag::Flags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use memfile::{CreateOptions, MemFile, Seal};
use ndarray::Array4;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::Decoder;
use crate::preprocessor::media::{
    DecodedMediaData, EncodedMediaData, decoders::DecodedMediaMetadata,
};

const DEFAULT_MAX_ALLOC: u64 = 512 * 1024 * 1024; // 512 MB

/// Environment override for the ffmpeg decoder thread count (frame threading).
/// Unset / 0 => min(available_parallelism, 8).
const DECODE_THREADS_ENV: &str = "DYN_MM_VIDEO_DECODE_THREADS";

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoderLimits {
    /// Maximum allowed total allocation of decoded frames in bytes
    #[serde(default)]
    pub max_alloc: Option<u64>,
}

impl Default for VideoDecoderLimits {
    fn default() -> Self {
        Self {
            max_alloc: Some(DEFAULT_MAX_ALLOC),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoder {
    #[serde(default)]
    pub(crate) limits: VideoDecoderLimits,

    /// sample N frames per second
    #[serde(default)]
    pub(crate) fps: Option<f64>,
    /// sample at most N frames (used with fps)
    #[serde(default)]
    pub(crate) max_frames: Option<u64>,
    /// sample N frames in total (linspace)
    #[serde(default)]
    pub(crate) num_frames: Option<u64>,
    /// fail if some frames fail to decode
    #[serde(default)]
    pub(crate) strict: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VideoMetadata {
    pub(crate) source_fps: f64,
    pub(crate) source_duration: f64,
    pub(crate) sampled_timestamps: Vec<f64>,
    /// Total frame count of the source clip (NOT the number of sampled
    /// frames). HF Qwen3-VL consumes this as `total_num_frames` when
    /// expanding video timestamp tokens under `do_sample_frames=False`.
    pub(crate) source_total_frames: i64,
    /// Integer frame indices (into the source clip's full frame sequence)
    /// of each sampled frame, aligned 1:1 with `sampled_timestamps`. HF
    /// consumes these as `frames_indices` so it can recover per-frame
    /// timestamps without re-sampling.
    pub(crate) frames_indices: Vec<i64>,
}

/// ffmpeg global init (codec registration). Idempotent + guarded so concurrent
/// decodes don't race the C-side registration.
fn ensure_ffmpeg_init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = ffmpeg::init();
    });
}

/// Frame-threading thread count for the ffmpeg decoder. Capped because frame
/// threading has diminishing returns and a per-thread memory cost.
fn decode_thread_count() -> usize {
    if let Ok(v) = std::env::var(DECODE_THREADS_ENV)
        && let Ok(n) = v.parse::<usize>()
        && n > 0
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8)
}

/// Number of frames to sample (count only) from config + source stats.
/// fps-based, explicit num_frames, or all frames; bounded by max_frames.
fn get_num_requested_frames(
    config: &VideoDecoder,
    total_frames: u64,
    duration_secs: f64,
) -> Result<u64> {
    anyhow::ensure!(total_frames > 0, "Cannot determine the video frame count");

    let requested_frames = if let Some(target_fps) = config.fps {
        anyhow::ensure!(duration_secs > 0.0, "Cannot determine the video duration");
        (duration_secs * target_fps) as u64
    } else {
        config.num_frames.unwrap_or(total_frames)
    };

    let requested_frames = requested_frames
        .min(config.max_frames.unwrap_or(requested_frames))
        .max(1);

    anyhow::ensure!(
        requested_frames > 0 && requested_frames <= total_frames,
        "Cannot decode {requested_frames} frames from {total_frames} total frames",
    );

    Ok(requested_frames)
}

/// Source-frame indices to sample, matching HF/decord
/// `np.unique(np.linspace(0, total_frames - 1, num=requested).round())`.
/// Sorted + deduplicated (may be fewer than `requested` when it exceeds total).
fn get_target_frame_indices(requested: u64, total_frames: u64) -> Vec<i64> {
    let total = total_frames.max(1) as i64;
    let n = requested.max(1);
    let mut indices: Vec<i64> = if n == 1 {
        vec![0]
    } else {
        (0..n)
            .map(|i| ((i as f64) * ((total - 1) as f64) / ((n - 1) as f64)).round() as i64)
            .collect()
    };
    indices.dedup(); // monotonic non-decreasing => adjacent dedup == np.unique
    indices
}

/// Copy a decoded RGB24 frame into a tightly-packed `width*height*3` buffer,
/// dropping the scaler's per-row stride padding. The scaler is configured to
/// emit exactly `width`x`height` RGB24, so `stride >= width*3` always holds;
/// the guard makes a violated invariant a clear error instead of a panic.
fn copy_rgb_tight(frame: &VideoFrame, dst: &mut [u8], width: usize, height: usize) -> Result<()> {
    let src = frame.data(0);
    let stride = frame.stride(0);
    let row_bytes = width * 3;
    anyhow::ensure!(
        stride >= row_bytes && src.len() >= height * stride && dst.len() >= height * row_bytes,
        "unexpected RGB frame layout: stride={stride}, src_len={}, need {height}x{row_bytes}",
        src.len()
    );
    for y in 0..height {
        dst[y * row_bytes..(y + 1) * row_bytes]
            .copy_from_slice(&src[y * stride..y * stride + row_bytes]);
    }
    Ok(())
}

/// Accumulates the RGB24 frames sampled at the target indices. The scaler and
/// output buffer are built lazily from the first decoded frame: several codecs
/// only resolve `pix_fmt` after the first `receive_frame`, and the decoded
/// frame's display size can differ from the pre-decode codec context (cropped
/// streams). Sizing everything off the first frame keeps both authoritative.
struct FrameSink {
    num_targets: usize,
    max_alloc: u64,
    source_fps: f64,
    target_pos: usize,
    out_width: usize,
    out_height: usize,
    frame_size: usize,
    scaler: Option<Scaler>,
    rgb: VideoFrame,
    all_frames: Vec<u8>,
    sampled_timestamps: Vec<f64>,
    frames_indices: Vec<i64>,
}

impl FrameSink {
    fn new(num_targets: usize, max_alloc: u64, source_fps: f64) -> Self {
        Self {
            num_targets,
            max_alloc,
            source_fps,
            target_pos: 0,
            out_width: 0,
            out_height: 0,
            frame_size: 0,
            scaler: None,
            rgb: VideoFrame::empty(),
            all_frames: Vec::new(),
            sampled_timestamps: Vec::with_capacity(num_targets),
            frames_indices: Vec::with_capacity(num_targets),
        }
    }

    /// Scale + copy one decoded frame into the next target slot.
    fn take(&mut self, decoded: &VideoFrame, frame_idx: i64) -> Result<()> {
        if self.scaler.is_none() {
            let (w, h) = (decoded.width(), decoded.height());
            anyhow::ensure!(w > 0 && h > 0, "Invalid decoded frame dimensions {w}x{h}");
            anyhow::ensure!(
                (w as u64) * (h as u64) * (self.num_targets as u64) * 3 <= self.max_alloc,
                "Video dimensions {}x{w}x{h}x3 exceed max alloc {}",
                self.num_targets,
                self.max_alloc
            );
            self.out_width = w as usize;
            self.out_height = h as usize;
            self.frame_size = self.out_width * self.out_height * 3;
            self.all_frames = vec![0u8; self.num_targets * self.frame_size];
            self.scaler = Some(Scaler::get(
                decoded.format(),
                w,
                h,
                Pixel::RGB24,
                w,
                h,
                Flags::BILINEAR,
            )?);
        }

        let scaler = self
            .scaler
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("scaler not initialized"))?;
        scaler.run(decoded, &mut self.rgb)?;
        let dst = &mut self.all_frames
            [self.target_pos * self.frame_size..(self.target_pos + 1) * self.frame_size];
        copy_rgb_tight(&self.rgb, dst, self.out_width, self.out_height)?;
        self.sampled_timestamps
            .push(frame_idx as f64 / self.source_fps);
        self.frames_indices.push(frame_idx);
        self.target_pos += 1;
        Ok(())
    }
}

impl Decoder for VideoDecoder {
    fn with_runtime(&self, runtime: Option<&Self>) -> Self {
        match runtime {
            Some(r) => {
                let mut d = r.clone();
                d.limits.clone_from(&self.limits);
                d
            }
            None => self.clone(),
        }
    }

    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        anyhow::ensure!(
            self.fps.is_none() || self.num_frames.is_none(),
            "fps and num_frames cannot be specified at the same time"
        );

        anyhow::ensure!(
            self.max_frames.is_none() || self.num_frames.is_none(),
            "max_frames and num_frames cannot be specified at the same time"
        );

        ensure_ffmpeg_init();

        // ffmpeg wants a file path; use a sealed in-memory file via /proc/self/fd.
        let mut mem_file = MemFile::create("video", CreateOptions::new().allow_sealing(true))?;
        mem_file.write_all(&data.into_bytes()?)?;
        mem_file.add_seals(Seal::Write | Seal::Shrink | Seal::Grow)?;
        let fd_path = format!("/proc/self/fd/{}", mem_file.as_raw_fd());

        let mut input = ffmpeg::format::input(&fd_path)?;

        // Pull stream metadata while the immutable borrow of `input` is alive,
        // then drop it before the mutable packet iteration below.
        let (stream_index, parameters, source_fps, source_duration, source_total_frames) = {
            let stream = input
                .streams()
                .best(MediaType::Video)
                .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;
            let time_base = stream.time_base();
            let rate = {
                let avg = stream.avg_frame_rate();
                if avg.numerator() != 0 {
                    avg
                } else {
                    stream.rate()
                }
            };
            let source_fps = rate.numerator() as f64 / (rate.denominator().max(1)) as f64;
            let duration_ts = stream.duration();
            let source_duration = if duration_ts > 0 {
                duration_ts as f64 * time_base.numerator() as f64
                    / (time_base.denominator().max(1)) as f64
            } else if input.duration() > 0 {
                input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
            } else {
                0.0
            };
            let nb = stream.frames();
            let mut total = if nb > 0 { nb as u64 } else { 0 };
            if total == 0 && source_fps > 0.0 && source_duration > 0.0 {
                total = (source_duration * source_fps) as u64;
            }
            (
                stream.index(),
                stream.parameters(),
                source_fps,
                source_duration,
                total,
            )
        };
        anyhow::ensure!(
            source_total_frames > 0,
            "Cannot determine the video frame count"
        );
        anyhow::ensure!(source_fps > 0.0, "Cannot determine the video frame rate");

        let requested_frames =
            get_num_requested_frames(self, source_total_frames, source_duration)?;
        let target_indices = get_target_frame_indices(requested_frames, source_total_frames);
        let num_targets = target_indices.len();

        // Frame-threaded decode: drive the ffmpeg decoder context directly so
        // we can set the codec thread count.
        let mut codec_ctx = ffmpeg::codec::context::Context::from_parameters(parameters)?;
        codec_ctx.set_threading(ffmpeg::codec::threading::Config {
            kind: ffmpeg::codec::threading::Type::Frame,
            count: decode_thread_count(),
            ..Default::default()
        });
        let mut decoder = codec_ctx.decoder().video()?;

        let max_alloc = self.limits.max_alloc.unwrap_or(u64::MAX);

        // Decode in presentation order (avcodec_receive_frame reorders B-frames
        // to display order, so `frame_idx` is the presentation index decord/HF
        // sample against). Copy the frame at-or-past each target; `>=` (not `==`)
        // means a dropped/corrupt frame can't stall a target. target_indices is
        // sorted ascending, so we stop once the last target is collected.
        let mut sink = FrameSink::new(num_targets, max_alloc, source_fps);
        let mut frame_idx: i64 = 0;
        let mut decoded = VideoFrame::empty();

        'outer: for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            // Tolerate an undecodable packet in non-strict mode (matches the
            // old per-frame leniency); fail fast under strict.
            if let Err(e) = decoder.send_packet(&packet) {
                if self.strict {
                    anyhow::bail!("video packet decode error: {e}");
                }
                continue;
            }
            while decoder.receive_frame(&mut decoded).is_ok() {
                if sink.target_pos < num_targets && frame_idx >= target_indices[sink.target_pos] {
                    sink.take(&decoded, frame_idx)?;
                }
                frame_idx += 1;
                if sink.target_pos >= num_targets {
                    break 'outer;
                }
            }
        }

        // Flush decoder-buffered frames only if we still need targets.
        if sink.target_pos < num_targets {
            decoder.send_eof()?;
            while decoder.receive_frame(&mut decoded).is_ok() {
                if sink.target_pos < num_targets && frame_idx >= target_indices[sink.target_pos] {
                    sink.take(&decoded, frame_idx)?;
                }
                frame_idx += 1;
            }
        }

        let FrameSink {
            mut all_frames,
            sampled_timestamps,
            frames_indices,
            frame_size,
            out_width,
            out_height,
            ..
        } = sink;

        let num_frames_decoded = sampled_timestamps.len();
        anyhow::ensure!(
            num_frames_decoded > 0,
            "Failed to decode any frames, check for video corruption"
        );
        if self.strict {
            anyhow::ensure!(
                num_frames_decoded == num_targets,
                "Decoded {num_frames_decoded} of {num_targets} requested frames (strict mode)"
            );
        }

        // Truncate to frames actually decoded (defensive: container nb_frames
        // can exceed the real frame count).
        all_frames.truncate(num_frames_decoded * frame_size);

        let shape = (num_frames_decoded, out_height, out_width, 3);
        let array = Array4::from_shape_vec(shape, all_frames)?;
        let mut decoded_media: DecodedMediaData = array.try_into()?;
        decoded_media.tensor_info.metadata = Some(DecodedMediaMetadata::Video(VideoMetadata {
            source_fps,
            source_duration,
            sampled_timestamps,
            source_total_frames: source_total_frames as i64,
            frames_indices,
        }));
        Ok(decoded_media)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::rdma::DataType;
    use super::*;
    use rstest::rstest;

    /// Load test video and parse expected dimensions from filename.
    /// Filename format: "{resolution}_{frames}.mp4" (e.g., "240p_10.mp4" -> 320x240, 10 frames)
    fn load_test_video(filename: &str) -> (EncodedMediaData, u32, u32, u32) {
        let path = format!(
            "{}/tests/data/media/{}",
            env!("CARGO_MANIFEST_DIR"),
            filename
        );
        let bytes =
            std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read test video: {}", path));

        let parts: Vec<&str> = filename.strip_suffix(".mp4").unwrap().split('_').collect();
        let resolution = parts[0];
        let frames = parts[1].parse::<u32>().unwrap();

        let (width, height) = match resolution {
            "2p" => (2, 2),
            "240p" => (320, 240),
            "2160p" => (3840, 2160),
            _ => panic!("Unknown resolution: {}", resolution),
        };

        let encoded = EncodedMediaData {
            bytes,
            b64_encoded: false,
        };

        (encoded, width, height, frames)
    }

    #[test]
    fn test_decode_video_num_frames() {
        let (encoded_data, width, height, _total_frames) = load_test_video("240p_10.mp4");

        let requested_frames = 5u64;
        let decoder = VideoDecoder {
            limits: VideoDecoderLimits::default(),
            fps: None,
            max_frames: None,
            num_frames: Some(requested_frames),
            strict: false,
        };

        let decoded = decoder.decode(encoded_data).unwrap();

        assert_eq!(decoded.tensor_info.shape[0], requested_frames as usize);
        assert_eq!(decoded.tensor_info.shape[1], height as usize);
        assert_eq!(decoded.tensor_info.shape[2], width as usize);
        assert_eq!(decoded.tensor_info.shape[3], 3);
        assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
    }

    #[test]
    fn test_decode_video_fps_sampling() {
        let (encoded_data, width, height, _total_frames) = load_test_video("240p_100.mp4");

        let target_fps = 0.5f64;
        let decoder = VideoDecoder {
            limits: VideoDecoderLimits::default(),
            fps: Some(target_fps),
            max_frames: None,
            num_frames: None,
            strict: false,
        };

        let decoded = decoder.decode(encoded_data).unwrap();

        // fps * duration calculation - video decoder uses duration from file
        // Source file is at 1fps, should get exactly 50 frames
        assert_eq!(decoded.tensor_info.shape[0], 50);
        assert_eq!(decoded.tensor_info.shape[1], height as usize);
        assert_eq!(decoded.tensor_info.shape[2], width as usize);
        assert_eq!(decoded.tensor_info.shape[3], 3);
        assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
    }

    #[rstest]
    #[case(Some(320 * 240 * 5 * 3), "240p_10.mp4", 5, true, "within limit")]
    #[case(Some(320 * 240 * 2 * 3), "240p_10.mp4", 5, false, "exceeds limit")]
    #[case(Some(2 * 2 * 10 * 3), "2p_10.mp4", 10, true, "exactly at limit")]
    #[case(None, "2160p_10.mp4", 10, true, "no limit")]
    fn test_max_alloc(
        #[case] max_alloc: Option<u64>,
        #[case] video_file: &str,
        #[case] num_frames: u64,
        #[case] should_succeed: bool,
        #[case] test_case: &str,
    ) {
        let (encoded_data, width, height, _) = load_test_video(video_file);

        let decoder = VideoDecoder {
            limits: VideoDecoderLimits { max_alloc },
            fps: None,
            max_frames: None,
            num_frames: Some(num_frames),
            strict: false,
        };

        let result = decoder.decode(encoded_data);

        if should_succeed {
            assert!(
                result.is_ok(),
                "Should decode successfully for case: {test_case}",
            );
            let decoded = result.unwrap();
            assert_eq!(decoded.tensor_info.shape[1], height as usize);
            assert_eq!(decoded.tensor_info.shape[2], width as usize);
            assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
        } else {
            assert!(result.is_err(), "Should fail for case: {}", test_case);
        }
    }

    #[test]
    fn test_conflicting_fps_and_num_frames() {
        let (encoded_data, ..) = load_test_video("240p_10.mp4");

        let decoder = VideoDecoder {
            limits: VideoDecoderLimits::default(),
            fps: Some(2.0f64),
            max_frames: None,
            num_frames: Some(5u64),
            strict: false,
        };

        let result = decoder.decode(encoded_data);
        assert!(
            result.is_err(),
            "Should fail when both fps and num_frames are specified"
        );
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("cannot be specified at the same time"));
    }

    // Unit tests for get_target_frame_indices (np.unique(np.linspace) parity)

    #[test]
    fn test_get_target_frame_indices() {
        // 10 of 100 frames -> evenly spaced over [0, 99], endpoints included.
        let idx = get_target_frame_indices(10, 100);
        assert_eq!(idx.len(), 10);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[9], 99);
        // linspace(0, 99, 10).round() == [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
        assert_eq!(idx, vec![0, 11, 22, 33, 44, 55, 66, 77, 88, 99]);

        // Single frame -> index 0 (matches np.linspace(0, T-1, 1) == [0]).
        assert_eq!(get_target_frame_indices(1, 50), vec![0]);

        // Sampling all frames -> every index.
        assert_eq!(get_target_frame_indices(5, 5), vec![0, 1, 2, 3, 4]);

        // More requested than available -> deduped (np.unique), fewer returned.
        let dedup = get_target_frame_indices(8, 3);
        assert_eq!(dedup, vec![0, 1, 2]);
    }

    #[test]
    fn test_with_runtime_limit_enforcement() {
        let server_limits = VideoDecoderLimits {
            max_alloc: Some(1024),
        };
        let server_config = VideoDecoder {
            limits: server_limits,
            fps: Some(1.0),
            ..Default::default()
        };

        // Runtime config tries to override limits (should be ignored)
        // And sets different FPS (should be accepted)
        let runtime_limits = VideoDecoderLimits {
            max_alloc: Some(999999),
        };
        let runtime_config = VideoDecoder {
            limits: runtime_limits,
            fps: Some(60.0),
            ..Default::default()
        };

        let merged = server_config.with_runtime(Some(&runtime_config));

        // Check that server limits are preserved
        assert_eq!(merged.limits.max_alloc, Some(1024));

        // Check that other fields are overridden
        assert_eq!(merged.fps, Some(60.0));
    }
}
