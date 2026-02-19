// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::path::Path;

use anyhow::{Result, bail};
use memfile::{CreateOptions, MemFile};
use ndarray::Array3;
use video_rs::encode::{EncoderBuilder, Settings};
use video_rs::time::Time;

/// Encode NHWC RGB24 frames into an H.264 MP4 in memory.
///
/// # Arguments
/// * `data` - Contiguous NHWC RGB24 u8 buffer (all frames concatenated)
/// * `width` - Frame width in pixels (must be even for yuv420p)
/// * `height` - Frame height in pixels (must be even for yuv420p)
/// * `num_frames` - Number of frames in the buffer
/// * `fps` - Output frames per second
///
/// # Returns
/// MP4 bytes with H.264 + yuv420p encoding.
pub fn encode_video(
    data: &[u8],
    width: u32,
    height: u32,
    num_frames: u32,
    fps: u32,
) -> Result<Vec<u8>> {
    // Validate inputs
    if width == 0 || height == 0 || num_frames == 0 || fps == 0 {
        bail!("All parameters must be non-zero: width={width}, height={height}, num_frames={num_frames}, fps={fps}");
    }
    if width % 2 != 0 || height % 2 != 0 {
        bail!("Dimensions must be even for yuv420p: {width}x{height}");
    }

    let frame_size = width as usize * height as usize * 3;
    let expected_len = num_frames as usize * frame_size;
    if data.len() != expected_len {
        bail!(
            "Data length mismatch: expected {} bytes ({}x{}x{}x3), got {}",
            expected_len,
            num_frames,
            height,
            width,
            data.len()
        );
    }

    // Create memfile for in-memory MP4 output (seekable for moov atom)
    let mem_file = MemFile::create("video_enc", CreateOptions::new().allow_sealing(true))?;
    let fd_path = format!("/proc/self/fd/{}", mem_file.as_raw_fd());

    let settings = Settings::preset_h264_yuv420p(width as usize, height as usize, false);
    let mut encoder = EncoderBuilder::new(Path::new(&fd_path), settings)
        .with_format("mp4")
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create encoder: {e}"))?;

    for i in 0..num_frames as usize {
        let offset = i * frame_size;
        let frame_data = &data[offset..offset + frame_size];

        // Create Array3 view (H, W, 3) then convert to owned for encoder
        let frame = Array3::from_shape_vec(
            (height as usize, width as usize, 3),
            frame_data.to_vec(),
        )?;

        let timestamp = Time::from_secs(i as f32 / fps as f32);
        encoder
            .encode(&frame, timestamp)
            .map_err(|e| anyhow::anyhow!("Failed to encode frame {i}: {e}"))?;
    }

    encoder
        .finish()
        .map_err(|e| anyhow::anyhow!("Failed to finish encoding: {e}"))?;

    // Drop encoder to release file handle before reading
    drop(encoder);

    // Read back the MP4 bytes from memfile
    let mut file = mem_file.into_file();
    file.seek(SeekFrom::Start(0))?;
    let mut output = Vec::new();
    file.read_to_end(&mut output)?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_video_basic() {
        // 4 frames of 64x64 black
        let width = 64u32;
        let height = 64u32;
        let num_frames = 4u32;
        let fps = 24u32;
        let data = vec![0u8; (num_frames * height * width * 3) as usize];

        let result = encode_video(&data, width, height, num_frames, fps);
        assert!(result.is_ok(), "Encoding should succeed: {:?}", result.err());

        let mp4 = result.unwrap();
        assert!(!mp4.is_empty(), "MP4 output should not be empty");
        // MP4 ftyp box check: starts with a size field then "ftyp"
        assert!(
            mp4.len() >= 8 && &mp4[4..8] == b"ftyp",
            "Output should be a valid MP4 (ftyp box expected)"
        );
    }

    #[test]
    fn test_encode_video_data_mismatch() {
        let result = encode_video(&[0u8; 100], 64, 64, 4, 24);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Data length mismatch"), "Got: {err}");
    }

    #[test]
    fn test_encode_video_odd_dimensions() {
        let result = encode_video(&[0u8; 63 * 63 * 3], 63, 63, 1, 24);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("even"), "Got: {err}");
    }

    #[test]
    fn test_encode_video_zero_params() {
        assert!(encode_video(&[], 0, 64, 1, 24).is_err());
        assert!(encode_video(&[], 64, 0, 1, 24).is_err());
        assert!(encode_video(&[], 64, 64, 0, 24).is_err());
        assert!(encode_video(&[], 64, 64, 1, 0).is_err());
    }

    #[test]
    fn test_encode_video_colored_frames() {
        // Encode frames with actual color data to verify no corruption
        let width = 64u32;
        let height = 64u32;
        let num_frames = 2u32;
        let fps = 30u32;
        let frame_size = (width * height * 3) as usize;

        let mut data = vec![0u8; num_frames as usize * frame_size];
        // Frame 0: red, Frame 1: blue
        for pixel in data[..frame_size].chunks_exact_mut(3) {
            pixel[0] = 255; // R
        }
        for pixel in data[frame_size..].chunks_exact_mut(3) {
            pixel[2] = 255; // B
        }

        let result = encode_video(&data, width, height, num_frames, fps);
        assert!(result.is_ok(), "Colored frame encoding failed: {:?}", result.err());

        let mp4 = result.unwrap();
        assert!(mp4.len() >= 8 && &mp4[4..8] == b"ftyp");
    }
}
