use std::io::Write;

use anyhow::Result;
use ndarray::Array4;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tempfile::NamedTempFile;
use video_rs::Location;

use super::Decoder;
use crate::preprocessor::media::{
    DecodedMediaData, EncodedMediaData, decoders::DecodedMediaMetadata,
};

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoder {
    // sample N frames per second
    #[serde(default)]
    pub(crate) fps: Option<f32>,
    // sample at most N frames (used with fps)
    #[serde(default)]
    pub(crate) max_frames: Option<u32>,
    // sample N frames in total (linspace)
    #[serde(default)]
    pub(crate) num_frames: Option<u32>,
    // fail if some frames fail to decode
    #[serde(default)]
    pub(crate) strict: bool,
    // maximum total size of the sampled frames in pixels
    #[serde(default)]
    pub(crate) max_pixels: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct VideoMetadata {
    pub(crate) source_fps: f32,
    pub(crate) source_frames: u32,
}

impl Decoder for VideoDecoder {
    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        anyhow::ensure!(
            self.fps.is_none() || self.num_frames.is_none(),
            "fps and num_frames cannot be specified at the same time"
        );

        anyhow::ensure!(
            self.max_frames.is_none() || self.num_frames.is_none(),
            "max_frames and num_frames cannot be specified at the same time"
        );

        let bytes = data.into_bytes()?;

        // video-rs wants a file path, we use tmpfs / ramdisk
        let mut temp_file = NamedTempFile::with_prefix("video")?;
        temp_file.write_all(&bytes)?;
        temp_file.flush()?;

        let location = Location::File(temp_file.path().to_path_buf());
        let mut decoder = video_rs::decode::Decoder::new(location)?;

        let total_frames = decoder.frames()? as u32; // note: this comes from the metadata and might not be exact
        anyhow::ensure!(total_frames > 0, "Cannot determine the video frame count");

        let requested_frames = if let Some(target_fps) = self.fps {
            // fps based sampling
            let duration = decoder.duration()?.as_secs();
            anyhow::ensure!(duration > 0.0, "Cannot determine the video duration");
            (duration * target_fps) as u32
        } else {
            // frame count based sampling
            // last fallback is to decode all frames
            self.num_frames.unwrap_or(total_frames)
        };

        let requested_frames = requested_frames
            .min(self.max_frames.unwrap_or(requested_frames))
            .max(1);

        anyhow::ensure!(
            requested_frames > 0 && requested_frames <= total_frames,
            "Cannot decode {requested_frames} frames from {total_frames} total frames",
        );

        let (width, height) = decoder.size();
        anyhow::ensure!(
            width > 0 && height > 0,
            "Invalid video dimensions {width}x{height}"
        );

        let max_pixels = self.max_pixels.unwrap_or(usize::MAX);
        anyhow::ensure!(
            (width as usize) * (height as usize) * (requested_frames as usize) <= max_pixels,
            "Video dimensions {requested_frames}x{width}x{height} exceed max pixels {max_pixels}"
        );

        let mut all_frames =
            Vec::with_capacity(requested_frames as usize * width as usize * height as usize * 3);
        let mut num_frames_decoded = 0;

        // uniform sampling
        let target_indices: HashSet<u32> = if requested_frames == 1 {
            HashSet::from([total_frames / 2])
        } else {
            (0..requested_frames)
                .map(|i| (i * (total_frames - 1)) / (requested_frames - 1))
                .collect()
        };

        // Decode all frames sequentially (required for P/B-frames), but only keep target frames
        // The loop will go in frame display order
        // TODO: smarter seek-based decoding for better sparse sampling
        for (current_frame_idx, result) in decoder.decode_iter().enumerate() {
            match result {
                Ok((_ts, frame)) => {
                    // Only keep frames at our target indices
                    if target_indices.contains(&(current_frame_idx as u32))
                        && let Some(slice) = frame.as_slice()
                    {
                        all_frames.extend_from_slice(slice);
                        num_frames_decoded += 1;
                        if num_frames_decoded >= requested_frames {
                            break;
                        }
                    }
                }
                Err(video_rs::Error::ReadExhausted | video_rs::Error::DecodeExhausted) => {
                    break;
                }
                Err(_) => {
                    continue;
                }
            }
        }

        anyhow::ensure!(
            num_frames_decoded > 0,
            "Failed to decode any frames, check for video corruption"
        );

        if self.strict {
            anyhow::ensure!(
                num_frames_decoded == requested_frames,
                "Failed to decode all requested frames (strict mode), check for video corruption"
            );
        }

        let shape = (
            num_frames_decoded as usize,
            height as usize,
            width as usize,
            3,
        );
        let array = Array4::from_shape_vec(shape, all_frames)?;
        let mut decoded: DecodedMediaData = array.try_into()?;
        decoded.tensor_info.metadata = Some(DecodedMediaMetadata::Video(VideoMetadata {
            source_fps: decoder.frame_rate(),
            source_frames: total_frames,
        }));
        Ok(decoded)
    }
}
