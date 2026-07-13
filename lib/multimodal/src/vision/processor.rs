// SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vision processor trait and encoder-output types.
//!
//! This module defines the interface for model-specific vision processors
//! and the common output format for preprocessed encoder inputs.

use std::{borrow::Cow, collections::HashMap};

use anyhow::{Context, Result as AnyhowResult};
use image::DynamicImage;
use ndarray::{Array4, ArrayD};

use super::{preprocessor_config::PreProcessorConfig, transforms::TransformError};
use crate::types::{FieldLayout, RgbFrameRef};

/// Helper to extract a dimension from encoder_input given an ndim-dependent axis index.
/// Returns `Err` if the ndim is not 4 or 5.
fn dim_for_ndim(
    ndim: usize,
    axis_4d: usize,
    axis_5d: usize,
    shape: &[usize],
) -> Result<usize, TransformError> {
    match ndim {
        4 => Ok(shape[axis_4d]),
        5 => Ok(shape[axis_5d]),
        _ => Err(TransformError::InvalidShape {
            expected: format!("4D or 5D encoder_input tensor, got {ndim}D"),
            actual: shape.to_vec(),
        }),
    }
}

/// Model-specific output values that vary by architecture.
///
/// Different vision models require different auxiliary outputs beyond encoder_input.
/// This enum captures the common types of such outputs.
#[derive(Debug, Clone)]
pub enum ModelSpecificValue {
    /// A tensor with shape information (data as flat vec, shape as dims)
    Tensor { data: Vec<f32>, shape: Vec<usize> },

    /// A tensor of integers (e.g., aspect_ratio_ids)
    IntTensor { data: Vec<i64>, shape: Vec<usize> },

    /// A tensor of unsigned integers (e.g., image_grid_thw)
    UintTensor { data: Vec<u32>, shape: Vec<usize> },

    /// Simple integer value
    Int(i64),

    /// Simple float value
    Float(f64),

    /// List of integers
    IntVec(Vec<i64>),

    /// List of unsigned integers
    UintVec(Vec<u32>),

    /// List of floats
    FloatVec(Vec<f32>),

    /// List of tuples (e.g., image sizes)
    TupleVec(Vec<(u32, u32)>),

    /// Boolean flag
    Bool(bool),
}

impl ModelSpecificValue {
    /// Create a 1D uint tensor from a vector.
    pub fn uint_1d(data: Vec<u32>) -> Self {
        let len = data.len();
        Self::UintTensor {
            data,
            shape: vec![len],
        }
    }

    /// Create a 2D uint tensor.
    pub fn uint_2d(data: Vec<u32>, rows: usize, cols: usize) -> Self {
        Self::UintTensor {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Create a 1D int tensor from a vector.
    pub fn int_1d(data: Vec<i64>) -> Self {
        let len = data.len();
        Self::IntTensor {
            data,
            shape: vec![len],
        }
    }

    /// Create a 2D int tensor.
    pub fn int_2d(data: Vec<i64>, rows: usize, cols: usize) -> Self {
        Self::IntTensor {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Interpret this value as per-item flat sizes.
    pub fn as_flat_sizes(&self) -> AnyhowResult<Vec<usize>> {
        match self {
            Self::IntTensor { data, .. } => data
                .iter()
                .map(|&v| usize::try_from(v).context("negative flat size"))
                .collect(),
            Self::UintTensor { data, .. } => Ok(data.iter().map(|&v| v as usize).collect()),
            Self::IntVec(values) => values
                .iter()
                .map(|&v| usize::try_from(v).context("negative flat size"))
                .collect(),
            Self::UintVec(values) => Ok(values.iter().map(|&v| v as usize).collect()),
            _ => Err(anyhow::anyhow!("unsupported flat sizes value type")),
        }
    }

    /// Slice item-batched metadata along the first dimension.
    pub fn slice_first_dim(&self, start: usize, len: usize) -> AnyhowResult<Self> {
        match self {
            Self::Tensor { data, shape } => {
                let (data, shape) = slice_tensor_first_dim(data, shape, start, len)?;
                Ok(Self::Tensor { data, shape })
            }
            Self::IntTensor { data, shape } => {
                let (data, shape) = slice_tensor_first_dim(data, shape, start, len)?;
                Ok(Self::IntTensor { data, shape })
            }
            Self::UintTensor { data, shape } => {
                let (data, shape) = slice_tensor_first_dim(data, shape, start, len)?;
                Ok(Self::UintTensor { data, shape })
            }
            Self::IntVec(values) => Ok(Self::IntVec(slice_1d(values, start, len)?.to_vec())),
            Self::UintVec(values) => Ok(Self::UintVec(slice_1d(values, start, len)?.to_vec())),
            Self::FloatVec(values) => Ok(Self::FloatVec(slice_1d(values, start, len)?.to_vec())),
            Self::TupleVec(values) => Ok(Self::TupleVec(slice_1d(values, start, len)?.to_vec())),
            _ => Ok(self.clone()),
        }
    }
}

fn slice_tensor_first_dim<T: Clone>(
    data: &[T],
    shape: &[usize],
    start: usize,
    len: usize,
) -> AnyhowResult<(Vec<T>, Vec<usize>)> {
    let first_dim = *shape
        .first()
        .ok_or_else(|| anyhow::anyhow!("cannot slice scalar tensor"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("tensor slice range overflow"))?;
    anyhow::ensure!(
        end <= first_dim,
        "tensor first-dimension slice {start}..{end} exceeds {first_dim}"
    );
    let row_width = shape[1..]
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| anyhow::anyhow!("tensor row width overflow"))?;
    let data_start = start
        .checked_mul(row_width)
        .ok_or_else(|| anyhow::anyhow!("tensor data start overflow"))?;
    let data_len = len
        .checked_mul(row_width)
        .ok_or_else(|| anyhow::anyhow!("tensor data length overflow"))?;
    let data_end = data_start
        .checked_add(data_len)
        .ok_or_else(|| anyhow::anyhow!("tensor data end overflow"))?;
    anyhow::ensure!(
        data_end <= data.len(),
        "tensor slice data range {data_start}..{data_end} exceeds {}",
        data.len()
    );
    let mut new_shape = shape.to_vec();
    new_shape[0] = len;
    Ok((data[data_start..data_end].to_vec(), new_shape))
}

fn slice_1d<T>(values: &[T], start: usize, len: usize) -> AnyhowResult<&[T]> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("slice range overflow"))?;
    values
        .get(start..end)
        .ok_or_else(|| anyhow::anyhow!("slice range {start}..{end} exceeds {}", values.len()))
}

/// Preprocessed encoder inputs ready for model consumption.
///
/// This struct contains the processor outputs needed by serving backends to
/// construct `MultimodalInputs` for the model. Vision processors currently
/// produce this from images or sampled video frames; future modality processors
/// can reuse the same output contract for audio features or other encoder inputs.
#[derive(Debug, Clone)]
pub struct PreprocessedEncoderInputs {
    /// Primary encoder input as a dynamic-dimensional float32 tensor.
    ///
    /// For vision models this is typically the preprocessed image/video tensor.
    /// Shape varies by model and modality:
    /// - Standard: [B, C, H, W] (4D)
    /// - Phi3-Vision: [B, num_crops+1, C, H, W] (5D)
    pub encoder_input: ArrayD<f32>,

    /// Number of encoder feature tokens per media item in the batch.
    ///
    /// Used to expand placeholder tokens in the text input. For vision this is
    /// usually an image patch or video patch count; for audio this could be an
    /// audio feature-frame count.
    /// For example, LLaVA with 336x336 and patch_size=14 produces 576 tokens.
    pub feature_token_counts: Vec<usize>,

    /// Modality-specific item size metadata before preprocessing.
    ///
    /// Vision processors use this for image/frame dimensions, but the exact
    /// tuple order follows each processor/model contract. Model-specific shape
    /// tensors that need a fixed order should also be emitted in `model_specific`.
    pub item_sizes: Vec<(u32, u32)>,

    /// Model-specific auxiliary outputs.
    ///
    /// Examples:
    /// - Qwen-VL: `image_grid_thw` for rotary position encoding
    /// - LLaMA-Vision: `aspect_ratio_ids`, `aspect_ratio_mask`
    /// - Phi3-Vision: `num_img_tokens` auxiliary metadata
    pub model_specific: HashMap<String, ModelSpecificValue>,
}

impl PreprocessedEncoderInputs {
    /// Create a new PreprocessedEncoderInputs with required fields (4D encoder input).
    pub fn new(
        encoder_input: Array4<f32>,
        feature_token_counts: Vec<usize>,
        item_sizes: Vec<(u32, u32)>,
    ) -> Self {
        Self {
            encoder_input: encoder_input.into_dyn(),
            feature_token_counts,
            item_sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Create a new PreprocessedEncoderInputs with dynamic-dimensional encoder input.
    ///
    /// Use this for models like Phi3-Vision that have 5D tensors.
    pub fn new_dynamic(
        encoder_input: ArrayD<f32>,
        feature_token_counts: Vec<usize>,
        item_sizes: Vec<(u32, u32)>,
    ) -> Self {
        Self {
            encoder_input,
            feature_token_counts,
            item_sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Add a model-specific value.
    pub fn with_extra(mut self, key: impl Into<String>, value: ModelSpecificValue) -> Self {
        self.model_specific.insert(key.into(), value);
        self
    }

    /// Get the number of media items represented by this preprocessed batch.
    pub fn batch_size(&self) -> usize {
        self.item_sizes.len()
    }

    /// Get the number of channels.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[1].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[2].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn channels(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 1, 2, self.encoder_input.shape())
    }

    /// Get the height of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[2].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[3].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn height(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 2, 3, self.encoder_input.shape())
    }

    /// Get the width of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[3].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[4].
    ///
    /// # Errors
    /// Returns `TransformError::InvalidShape` if encoder_input is not 4D or 5D.
    pub fn width(&self) -> Result<usize, TransformError> {
        dim_for_ndim(self.encoder_input.ndim(), 3, 4, self.encoder_input.shape())
    }

    /// Get the number of dimensions of encoder_input.
    pub fn ndim(&self) -> usize {
        self.encoder_input.ndim()
    }

    /// Get total number of encoder feature tokens across all media items.
    pub fn total_feature_tokens(&self) -> usize {
        self.feature_token_counts.iter().sum()
    }

    /// Get the primary encoder input as a flat f32 slice without copying if possible.
    pub fn encoder_input_flat(&self) -> Cow<'_, [f32]> {
        match self.encoder_input.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(self.encoder_input.iter().copied().collect()),
        }
    }

    /// Get the shape of the primary encoder input as a vector.
    pub fn encoder_input_shape(&self) -> Vec<usize> {
        self.encoder_input.shape().to_vec()
    }

    /// Number of media items in this batch.
    pub fn num_media_items(&self) -> usize {
        self.item_sizes.len()
    }

    /// Extract batched tensor keys from explicit field layout declarations.
    pub fn batched_keys(layouts: &HashMap<String, FieldLayout>) -> Vec<String> {
        layouts
            .iter()
            .filter(|(_, l)| matches!(l, FieldLayout::Batched))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Extract flat-slicing tensor keys from explicit field layout declarations.
    ///
    /// Returns a map of tensor name → sizes tensor name.
    pub fn flat_keys(layouts: &HashMap<String, FieldLayout>) -> HashMap<String, String> {
        layouts
            .iter()
            .filter_map(|(k, l)| match l {
                FieldLayout::Flat { sizes_key } => Some((k.clone(), sizes_key.clone())),
                FieldLayout::Batched | FieldLayout::Shared => None,
            })
            .collect()
    }
}

/// Trait for model-specific vision preprocessors.
///
/// Each vision model (LLaVA, Qwen-VL, Phi3-Vision, etc.) implements this trait
/// to provide the correct preprocessing pipeline.
pub trait VisionPreProcessor: Send + Sync {
    /// Default normalization mean for this model family.
    fn default_mean(&self) -> [f64; 3];

    /// Default normalization std for this model family.
    fn default_std(&self) -> [f64; 3];

    /// Preprocess a batch of images.
    ///
    /// # Arguments
    /// * `images` - Input images to preprocess
    /// * `config` - Preprocessor configuration from HuggingFace
    ///
    /// # Returns
    /// Preprocessed encoder inputs ready for the model, or an error.
    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError>;

    /// Preprocess one decoded video clip represented as sampled frames.
    ///
    /// Implementations that support video should emit the same primary
    /// `encoder_input` tensor shape used by the image path, plus video-specific
    /// model metadata such as `video_grid_thw`.
    fn preprocess_video(
        &self,
        _frames: &[DynamicImage],
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support video preprocessing",
            self.model_name()
        )))
    }

    /// Preprocess one decoded video clip represented as borrowed RGB frame
    /// buffers. Implementations can override this to avoid materializing
    /// `DynamicImage` objects after media decode.
    fn preprocess_video_rgb(
        &self,
        _frames: &[RgbFrameRef<'_>],
        _config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(format!(
            "{} does not support RGB video preprocessing",
            self.model_name()
        )))
    }

    /// Calculate the number of vision tokens for a given image size.
    ///
    /// This is used to determine how many placeholder tokens to insert
    /// in the text input before the image has been fully processed.
    ///
    /// # Arguments
    /// * `width` - Image width after preprocessing
    /// * `height` - Image height after preprocessing
    /// * `config` - Preprocessor configuration
    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize;

    /// Get the model family name for identification.
    fn model_name(&self) -> &'static str;

    /// Get the expected image size after preprocessing.
    ///
    /// Some models have fixed sizes, others are dynamic.
    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        config.get_target_size()
    }
}
