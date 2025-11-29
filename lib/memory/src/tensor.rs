// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tensor abstraction built on top of MemoryDescriptor.
//!
//! A tensor is memory with shape, stride, and element size metadata.
//! The underlying memory could be externally owned, self-owned, or a view.

use super::nixl::{self, NixlDescriptor};
use super::{MemoryDescriptor, StorageKind};
use std::any::Any;
use std::sync::Arc;

/// A tensor is memory with shape, stride, and element size metadata.
///
/// This trait extends [`MemoryDescriptor`] with tensor-specific metadata.
/// The underlying memory could be externally owned, self-owned, or a view.
///
/// # Shape and Stride
///
/// - `shape()` returns the number of elements in each dimension
/// - `stride()` returns the number of elements to skip when incrementing each dimension
/// - `element_size()` returns the number of bytes per element
///
/// For a contiguous tensor with shape `[2, 3, 4]`:
/// - stride would be `[12, 4, 1]` (row-major/C order)
/// - total elements = 2 * 3 * 4 = 24
/// - total bytes = 24 * element_size()
pub trait TensorDescriptor: MemoryDescriptor {
    /// Shape of the tensor (number of elements per dimension).
    fn shape(&self) -> &[usize];

    /// Stride of the tensor (elements to skip per dimension).
    ///
    /// `stride[i]` indicates how many elements to skip when incrementing dimension `i`.
    fn stride(&self) -> &[usize];

    /// Number of bytes per element.
    fn element_size(&self) -> usize;
}

// =============================================================================
// Helper methods for TensorDescriptor
// =============================================================================

/// Extension trait providing helper methods for tensor descriptors.
pub trait TensorDescriptorExt: TensorDescriptor {
    /// Total number of elements in the tensor (product of shape).
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Number of dimensions (rank).
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Check if tensor is contiguous in memory (row-major/C order).
    ///
    /// A tensor is contiguous if its strides follow the pattern where
    /// the last dimension has stride 1, and each preceding dimension
    /// has stride equal to the product of all following dimensions.
    fn is_contiguous(&self) -> bool {
        let shape = self.shape();
        let stride = self.stride();

        if shape.is_empty() {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..shape.len()).rev() {
            if stride[i] != expected_stride {
                return false;
            }
            expected_stride *= shape[i];
        }
        true
    }

    /// Compute the contiguous stride for the current shape.
    ///
    /// Returns the stride that would make this tensor contiguous
    /// (row-major/C order).
    fn contiguous_stride(&self) -> Vec<usize> {
        let shape = self.shape();
        if shape.is_empty() {
            return vec![];
        }

        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        stride
    }
}

// Blanket impl for all TensorDescriptor types
impl<T: TensorDescriptor + ?Sized> TensorDescriptorExt for T {}

// =============================================================================
// Arc<dyn TensorDescriptor> support for NixlRegisterExt
// =============================================================================

impl nixl::NixlCompatible for Arc<dyn TensorDescriptor> {
    fn nixl_params(&self) -> (*const u8, usize, nixl::MemType, u64) {
        let storage = self.storage_kind();
        let (mem_type, device_id) = match storage {
            StorageKind::Device(idx) => (nixl::MemType::Vram, idx as u64),
            StorageKind::System => (nixl::MemType::Dram, 0),
            StorageKind::Pinned => (nixl::MemType::Dram, 0),
            StorageKind::Disk(fd) => (nixl::MemType::File, fd),
        };
        (self.addr() as *const u8, self.size(), mem_type, device_id)
    }
}

impl MemoryDescriptor for Arc<dyn TensorDescriptor> {
    fn addr(&self) -> usize {
        (**self).addr()
    }

    fn size(&self) -> usize {
        (**self).size()
    }

    fn storage_kind(&self) -> StorageKind {
        (**self).storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl TensorDescriptor for Arc<dyn TensorDescriptor> {
    fn shape(&self) -> &[usize] {
        (**self).shape()
    }

    fn stride(&self) -> &[usize] {
        (**self).stride()
    }

    fn element_size(&self) -> usize {
        (**self).element_size()
    }
}

// =============================================================================
// Arc<dyn TensorDescriptor + Send + Sync> support
// =============================================================================

impl nixl::NixlCompatible for Arc<dyn TensorDescriptor + Send + Sync> {
    fn nixl_params(&self) -> (*const u8, usize, nixl::MemType, u64) {
        let storage = self.storage_kind();
        let (mem_type, device_id) = match storage {
            StorageKind::Device(idx) => (nixl::MemType::Vram, idx as u64),
            StorageKind::System => (nixl::MemType::Dram, 0),
            StorageKind::Pinned => (nixl::MemType::Dram, 0),
            StorageKind::Disk(fd) => (nixl::MemType::File, fd),
        };
        (self.addr() as *const u8, self.size(), mem_type, device_id)
    }
}

impl MemoryDescriptor for Arc<dyn TensorDescriptor + Send + Sync> {
    fn addr(&self) -> usize {
        (**self).addr()
    }

    fn size(&self) -> usize {
        (**self).size()
    }

    fn storage_kind(&self) -> StorageKind {
        (**self).storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl TensorDescriptor for Arc<dyn TensorDescriptor + Send + Sync> {
    fn shape(&self) -> &[usize] {
        (**self).shape()
    }

    fn stride(&self) -> &[usize] {
        (**self).stride()
    }

    fn element_size(&self) -> usize {
        (**self).element_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test tensor for unit tests
    #[derive(Debug)]
    struct TestTensor {
        addr: usize,
        size: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
        element_size: usize,
    }

    impl MemoryDescriptor for TestTensor {
        fn addr(&self) -> usize {
            self.addr
        }

        fn size(&self) -> usize {
            self.size
        }

        fn storage_kind(&self) -> StorageKind {
            StorageKind::System
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
            None
        }
    }

    impl TensorDescriptor for TestTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn stride(&self) -> &[usize] {
            &self.stride
        }

        fn element_size(&self) -> usize {
            self.element_size
        }
    }

    #[test]
    fn test_numel() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 24 * 4, // 24 elements * 4 bytes
            shape: vec![2, 3, 4],
            stride: vec![12, 4, 1],
            element_size: 4,
        };
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn test_ndim() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 24 * 4,
            shape: vec![2, 3, 4],
            stride: vec![12, 4, 1],
            element_size: 4,
        };
        assert_eq!(tensor.ndim(), 3);
    }

    #[test]
    fn test_is_contiguous_true() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 24 * 4,
            shape: vec![2, 3, 4],
            stride: vec![12, 4, 1], // Contiguous stride
            element_size: 4,
        };
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_false() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 24 * 4,
            shape: vec![2, 3, 4],
            stride: vec![24, 4, 1], // Non-contiguous (gap between first dim)
            element_size: 4,
        };
        assert!(!tensor.is_contiguous());
    }

    #[test]
    fn test_contiguous_stride() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 24 * 4,
            shape: vec![2, 3, 4],
            stride: vec![24, 4, 1], // Non-contiguous
            element_size: 4,
        };
        assert_eq!(tensor.contiguous_stride(), vec![12, 4, 1]);
    }

    #[test]
    fn test_empty_tensor() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 0,
            shape: vec![],
            stride: vec![],
            element_size: 4,
        };
        assert_eq!(tensor.numel(), 1); // Empty product is 1
        assert_eq!(tensor.ndim(), 0);
        assert!(tensor.is_contiguous());
    }
}
