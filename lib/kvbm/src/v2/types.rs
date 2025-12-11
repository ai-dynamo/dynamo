// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core data types for block manager v2.
//!
//! This module defines framework-agnostic data structures used throughout
//! the block manager implementation.

use std::fmt;
use std::str::FromStr;

/// KV cache memory layout format.
///
/// Defines how key-value cache tensors are organized in memory.
/// This is important for memory access patterns and tensor parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheLayout {
    /// NHD format: [N, H, D] where N=block_size, H=heads, D=hidden_dim
    ///
    /// Memory layout: tokens are contiguous, then heads, then hidden dimension.
    /// Common in vLLM and many transformer implementations.
    NHD,

    /// HND format: [H, N, D] where H=heads, N=block_size, D=hidden_dim
    ///
    /// Memory layout: heads are contiguous, then tokens, then hidden dimension.
    /// Used in some optimized implementations for better cache locality.
    HND,

    /// Unknown or ambiguous format.
    ///
    /// Used when the layout cannot be determined from available information.
    Unknown,
}

impl CacheLayout {
    /// Parse layout from string representation.
    ///
    /// # Examples
    /// ```
    /// # use dynamo_kvbm::v2::types::CacheLayout;
    /// # use std::str::FromStr;
    /// assert_eq!(CacheLayout::parse("NHD"), CacheLayout::NHD);
    /// assert_eq!(CacheLayout::parse("HND"), CacheLayout::HND);
    /// assert_eq!(CacheLayout::parse("nhd"), CacheLayout::NHD);
    /// assert_eq!(CacheLayout::parse("invalid"), CacheLayout::Unknown);
    /// // Also available via FromStr trait
    /// assert_eq!(CacheLayout::from_str("NHD").unwrap(), CacheLayout::NHD);
    /// ```
    pub fn parse(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "NHD" => Self::NHD,
            "HND" => Self::HND,
            _ => Self::Unknown,
        }
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NHD => "NHD",
            Self::HND => "HND",
            Self::Unknown => "Unknown",
        }
    }

    /// Check if this layout is fully contiguous.
    ///
    /// NHD layout is typically not fully contiguous in vLLM implementations,
    /// as different layers may be separated in memory.
    pub fn is_fully_contiguous(&self) -> bool {
        match self {
            Self::NHD => false,
            Self::HND => true,
            Self::Unknown => false,
        }
    }
}

impl fmt::Display for CacheLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for CacheLayout {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::parse(s))
    }
}

/// Cache data type.
///
/// Defines the numerical precision used for KV cache storage.
/// Different dtypes trade off memory usage vs. precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheDtype {
    /// 16-bit floating point (IEEE 754 half precision)
    ///
    /// Provides good balance of memory efficiency and precision.
    /// Most common choice for KV cache storage.
    FP16,

    /// 16-bit brain floating point
    ///
    /// Same memory footprint as FP16 but with FP32's exponent range.
    /// Better for training stability, increasingly common in inference.
    BF16,

    /// 32-bit floating point (IEEE 754 single precision)
    ///
    /// Full precision, higher memory usage.
    /// Used when precision is critical or hardware doesn't support FP16.
    FP32,

    /// 8-bit integer quantization
    ///
    /// Reduces memory by 4x compared to FP16.
    /// Requires careful calibration to maintain quality.
    INT8,

    /// 8-bit floating point
    ///
    /// Newer format with better dynamic range than INT8.
    /// Supported on recent NVIDIA GPUs (H100+).
    FP8,
}

impl CacheDtype {
    /// Get the size in bytes of this dtype.
    ///
    /// # Examples
    /// ```
    /// # use dynamo_kvbm::v2::types::CacheDtype;
    /// assert_eq!(CacheDtype::FP16.bytes_per_element(), 2);
    /// assert_eq!(CacheDtype::FP32.bytes_per_element(), 4);
    /// assert_eq!(CacheDtype::INT8.bytes_per_element(), 1);
    /// ```
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::FP16 => 2,
            Self::BF16 => 2,
            Self::FP32 => 4,
            Self::INT8 => 1,
            Self::FP8 => 1,
        }
    }

    /// Parse dtype from string representation.
    ///
    /// Handles various common string formats used by different frameworks.
    ///
    /// # Examples
    /// ```
    /// # use dynamo_kvbm::v2::types::CacheDtype;
    /// # use std::str::FromStr;
    /// assert_eq!(CacheDtype::parse("float16"), Some(CacheDtype::FP16));
    /// assert_eq!(CacheDtype::parse("fp16"), Some(CacheDtype::FP16));
    /// assert_eq!(CacheDtype::parse("half"), Some(CacheDtype::FP16));
    /// assert_eq!(CacheDtype::parse("bfloat16"), Some(CacheDtype::BF16));
    /// assert_eq!(CacheDtype::parse("invalid"), None);
    /// // Also available via FromStr trait
    /// assert_eq!(CacheDtype::from_str("fp16").unwrap(), CacheDtype::FP16);
    /// assert!(CacheDtype::from_str("invalid").is_err());
    /// ```
    pub fn parse(s: &str) -> Option<Self> {
        let s_lower = s.to_lowercase();
        let s_trimmed = s_lower.trim();

        // Check exact matches first for precision
        match s_trimmed {
            "fp16" | "float16" | "half" => Some(Self::FP16),
            "bf16" | "bfloat16" => Some(Self::BF16),
            "fp32" | "float32" | "float" => Some(Self::FP32),
            "int8" => Some(Self::INT8),
            "fp8" => Some(Self::FP8),
            _ => {
                // Fall back to contains for partial matches, but check more specific patterns first
                if s_trimmed.contains("bfloat16") || s_trimmed.contains("bf16") {
                    Some(Self::BF16)
                } else if s_trimmed.contains("float16")
                    || s_trimmed.contains("fp16")
                    || s_trimmed.contains("half")
                {
                    Some(Self::FP16)
                } else if s_trimmed.contains("float32") || s_trimmed.contains("fp32") {
                    Some(Self::FP32)
                } else if s_trimmed.contains("int8") {
                    Some(Self::INT8)
                } else if s_trimmed.contains("fp8") {
                    Some(Self::FP8)
                } else {
                    None
                }
            }
        }
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FP16 => "fp16",
            Self::BF16 => "bf16",
            Self::FP32 => "fp32",
            Self::INT8 => "int8",
            Self::FP8 => "fp8",
        }
    }
}

impl fmt::Display for CacheDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Error type for parsing CacheDtype from string.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseCacheDtypeError {
    input: String,
}

impl fmt::Display for ParseCacheDtypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid cache dtype: '{}'", self.input)
    }
}

impl std::error::Error for ParseCacheDtypeError {}

impl FromStr for CacheDtype {
    type Err = ParseCacheDtypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| ParseCacheDtypeError {
            input: s.to_string(),
        })
    }
}

impl Default for CacheDtype {
    fn default() -> Self {
        Self::FP16
    }
}

/// Model executor backend type.
///
/// Defines the distributed execution backend used for running the model.
/// These values are extracted from vLLM's configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelExecutorBackend {
    /// Ray backend for distributed execution
    Ray,

    /// Multi-process backend (e.g., torch.multiprocessing)
    MultiProcessor,

    /// Unknown or unsupported backend
    ///
    /// This includes backends like "uni" (single process) and "external_launcher"
    /// that are not specifically handled.
    Unknown,
}

impl ModelExecutorBackend {
    /// Parse backend from vLLM string representation.
    ///
    /// # Examples
    /// ```
    /// # use dynamo_kvbm::v2::types::ModelExecutorBackend;
    /// assert_eq!(ModelExecutorBackend::parse("ray"), ModelExecutorBackend::Ray);
    /// assert_eq!(ModelExecutorBackend::parse("mp"), ModelExecutorBackend::MultiProcessor);
    /// assert_eq!(ModelExecutorBackend::parse("uni"), ModelExecutorBackend::Unknown);
    /// assert_eq!(ModelExecutorBackend::parse("external_launcher"), ModelExecutorBackend::Unknown);
    /// ```
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "ray" => Self::Ray,
            "mp" => Self::MultiProcessor,
            _ => Self::Unknown,
        }
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ray => "ray",
            Self::MultiProcessor => "mp",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for ModelExecutorBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ModelExecutorBackend {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::parse(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_layout_from_str() {
        assert_eq!(CacheLayout::from_str("NHD").unwrap(), CacheLayout::NHD);
        assert_eq!(CacheLayout::from_str("HND").unwrap(), CacheLayout::HND);
        assert_eq!(CacheLayout::from_str("nhd").unwrap(), CacheLayout::NHD);
        assert_eq!(CacheLayout::from_str("hnd").unwrap(), CacheLayout::HND);
        assert_eq!(
            CacheLayout::from_str("invalid").unwrap(),
            CacheLayout::Unknown
        );
    }

    #[test]
    fn test_cache_layout_contiguity() {
        assert!(!CacheLayout::NHD.is_fully_contiguous());
        assert!(CacheLayout::HND.is_fully_contiguous());
        assert!(!CacheLayout::Unknown.is_fully_contiguous());
    }

    #[test]
    fn test_cache_dtype_bytes() {
        assert_eq!(CacheDtype::FP16.bytes_per_element(), 2);
        assert_eq!(CacheDtype::BF16.bytes_per_element(), 2);
        assert_eq!(CacheDtype::FP32.bytes_per_element(), 4);
        assert_eq!(CacheDtype::INT8.bytes_per_element(), 1);
        assert_eq!(CacheDtype::FP8.bytes_per_element(), 1);
    }

    #[test]
    fn test_cache_dtype_parse() {
        assert_eq!(CacheDtype::parse("float16"), Some(CacheDtype::FP16));
        assert_eq!(CacheDtype::parse("fp16"), Some(CacheDtype::FP16));
        assert_eq!(CacheDtype::parse("half"), Some(CacheDtype::FP16));
        assert_eq!(CacheDtype::parse("bfloat16"), Some(CacheDtype::BF16));
        assert_eq!(CacheDtype::parse("bf16"), Some(CacheDtype::BF16));
        assert_eq!(CacheDtype::parse("float32"), Some(CacheDtype::FP32));
        assert_eq!(CacheDtype::parse("fp32"), Some(CacheDtype::FP32));
        assert_eq!(CacheDtype::parse("int8"), Some(CacheDtype::INT8));
        assert_eq!(CacheDtype::parse("fp8"), Some(CacheDtype::FP8));
        assert_eq!(CacheDtype::parse("invalid"), None);
    }

    #[test]
    fn test_cache_dtype_from_str() {
        use std::str::FromStr;

        assert_eq!(CacheDtype::from_str("fp16").unwrap(), CacheDtype::FP16);
        assert_eq!(CacheDtype::from_str("bfloat16").unwrap(), CacheDtype::BF16);
        assert!(CacheDtype::from_str("invalid").is_err());
    }

    #[test]
    fn test_cache_dtype_default() {
        assert_eq!(CacheDtype::default(), CacheDtype::FP16);
    }
}
