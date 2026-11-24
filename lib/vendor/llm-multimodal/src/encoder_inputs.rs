//! Shared encoder-input types for all encoder-backed modalities.

use std::{borrow::Cow, collections::HashMap};

use anyhow::{Context, Result as AnyhowResult};
use ndarray::{Array, ArrayD, Dimension};

use crate::types::FieldLayout;

/// Model-specific auxiliary output values.
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

    /// List of tuples (e.g., media item sizes)
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
#[derive(Debug, Clone)]
pub struct PreprocessedEncoderInputs {
    /// Primary encoder input as a dynamic-dimensional float32 tensor.
    pub encoder_input: ArrayD<f32>,

    /// Number of encoder feature tokens per media item in the batch.
    pub feature_token_counts: Vec<usize>,

    /// Modality-specific item size metadata before preprocessing.
    ///
    /// The exact tuple order follows each processor/model contract. Auxiliary
    /// shape tensors that need a fixed order should be emitted in
    /// `model_specific`.
    pub item_sizes: Vec<(u32, u32)>,

    /// Model-specific auxiliary outputs.
    pub model_specific: HashMap<String, ModelSpecificValue>,
}

impl PreprocessedEncoderInputs {
    /// Create encoder inputs backed by a tensor of any dimensionality.
    pub fn new<D: Dimension>(
        encoder_input: Array<f32, D>,
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

    /// Add a model-specific value.
    pub fn with_extra(mut self, key: impl Into<String>, value: ModelSpecificValue) -> Self {
        self.model_specific.insert(key.into(), value);
        self
    }

    /// Get the number of media items represented by this preprocessed batch.
    pub fn batch_size(&self) -> usize {
        self.item_sizes.len()
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

    /// Extract batched tensor keys from explicit field layout declarations.
    pub fn batched_keys(layouts: &HashMap<String, FieldLayout>) -> Vec<String> {
        layouts
            .iter()
            .filter(|(_, layout)| matches!(layout, FieldLayout::Batched))
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Extract flat-slicing tensor keys from explicit field layout declarations.
    ///
    /// Returns a map of tensor name to sizes tensor name.
    pub fn flat_keys(layouts: &HashMap<String, FieldLayout>) -> HashMap<String, String> {
        layouts
            .iter()
            .filter_map(|(key, layout)| match layout {
                FieldLayout::Flat { sizes_key } => Some((key.clone(), sizes_key.clone())),
                FieldLayout::Batched => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;

    use super::*;

    #[test]
    fn encoder_input_accessors_are_modality_neutral() {
        let inputs = PreprocessedEncoderInputs::new(
            Array4::<f32>::zeros((2, 3, 4, 5)),
            vec![6, 7],
            vec![(4, 5), (8, 9)],
        );

        assert_eq!(inputs.batch_size(), 2);
        assert_eq!(inputs.ndim(), 4);
        assert_eq!(inputs.total_feature_tokens(), 13);
        assert_eq!(inputs.encoder_input_shape(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn encoder_inputs_accept_model_specific_values() {
        let inputs = PreprocessedEncoderInputs::new(
            Array4::<f32>::zeros((1, 3, 224, 224)),
            vec![196],
            vec![(224, 224)],
        )
        .with_extra(
            "image_grid_thw",
            ModelSpecificValue::uint_1d(vec![1, 16, 16]),
        )
        .with_extra("aspect_ratio_id", ModelSpecificValue::Int(0));

        assert!(inputs.model_specific.contains_key("image_grid_thw"));
        assert!(inputs.model_specific.contains_key("aspect_ratio_id"));
    }

    #[test]
    fn model_specific_value_tensor_constructors_set_shapes() {
        assert!(matches!(
            ModelSpecificValue::uint_1d(vec![1, 2, 3]),
            ModelSpecificValue::UintTensor { data, shape }
                if data == vec![1, 2, 3] && shape == vec![3]
        ));
        assert!(matches!(
            ModelSpecificValue::uint_2d(vec![1, 2, 3, 4], 2, 2),
            ModelSpecificValue::UintTensor { data, shape }
                if data == vec![1, 2, 3, 4] && shape == vec![2, 2]
        ));
        assert!(matches!(
            ModelSpecificValue::int_1d(vec![1, 2, 3]),
            ModelSpecificValue::IntTensor { data, shape }
                if data == vec![1, 2, 3] && shape == vec![3]
        ));
        assert!(matches!(
            ModelSpecificValue::int_2d(vec![1, 2, 3, 4], 2, 2),
            ModelSpecificValue::IntTensor { data, shape }
                if data == vec![1, 2, 3, 4] && shape == vec![2, 2]
        ));
    }

    #[test]
    fn encoder_input_flat_preserves_values() {
        let encoder_input = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let inputs = PreprocessedEncoderInputs::new(encoder_input, vec![4], vec![(2, 2)]);

        assert_eq!(inputs.encoder_input_flat(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
