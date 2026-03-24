// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    /// Returns the CUDA device ID if the tensor is on a CUDA device.
    fn cuda_device_id(&self) -> Option<usize> {
        match self.storage_kind() {
            StorageKind::Device(idx) => Some(idx as usize),
            _ => None,
        }
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

    #[test]
    fn test_1d_tensor_contiguous() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 10 * 4,
            shape: vec![10],
            stride: vec![1],
            element_size: 4,
        };
        assert_eq!(tensor.numel(), 10);
        assert_eq!(tensor.ndim(), 1);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.contiguous_stride(), vec![1]);
    }

    #[test]
    fn test_1d_tensor_non_contiguous() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 10 * 4,
            shape: vec![10],
            stride: vec![2], // Strided access (every other element)
            element_size: 4,
        };
        assert!(!tensor.is_contiguous());
    }

    #[test]
    fn test_2d_tensor() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 6 * 4,
            shape: vec![2, 3],
            stride: vec![3, 1],
            element_size: 4,
        };
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.ndim(), 2);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_high_dimensional_tensor() {
        // 5D tensor: [2, 3, 4, 5, 6]
        let shape = vec![2, 3, 4, 5, 6];
        // Contiguous stride: [360, 120, 30, 6, 1]
        let stride = vec![360, 120, 30, 6, 1];
        let numel: usize = shape.iter().product();
        let tensor = TestTensor {
            addr: 0x1000,
            size: numel * 4,
            shape,
            stride,
            element_size: 4,
        };
        assert_eq!(tensor.numel(), 720);
        assert_eq!(tensor.ndim(), 5);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.contiguous_stride(), vec![360, 120, 30, 6, 1]);
    }

    #[test]
    fn test_tensor_with_size_1_dimensions() {
        // Shape with singleton dimensions: [1, 3, 1, 4]
        let tensor = TestTensor {
            addr: 0x1000,
            size: 12 * 4,
            shape: vec![1, 3, 1, 4],
            stride: vec![12, 4, 4, 1], // Contiguous for this shape
            element_size: 4,
        };
        assert_eq!(tensor.numel(), 12);
        assert_eq!(tensor.ndim(), 4);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_contiguous_stride_empty() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 0,
            shape: vec![],
            stride: vec![],
            element_size: 4,
        };
        assert!(tensor.contiguous_stride().is_empty());
    }

    #[test]
    fn test_contiguous_stride_1d() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 5 * 4,
            shape: vec![5],
            stride: vec![1],
            element_size: 4,
        };
        assert_eq!(tensor.contiguous_stride(), vec![1]);
    }

    #[test]
    fn test_cuda_device_id_system() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 100,
            shape: vec![10],
            stride: vec![1],
            element_size: 4,
        };
        assert_eq!(tensor.cuda_device_id(), None);
    }

    /// Test tensor that reports Device storage kind
    #[derive(Debug)]
    struct DeviceTensor {
        addr: usize,
        size: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
        element_size: usize,
        device_id: u32,
    }

    impl MemoryDescriptor for DeviceTensor {
        fn addr(&self) -> usize {
            self.addr
        }

        fn size(&self) -> usize {
            self.size
        }

        fn storage_kind(&self) -> StorageKind {
            StorageKind::Device(self.device_id)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
            None
        }
    }

    impl TensorDescriptor for DeviceTensor {
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
    fn test_cuda_device_id_device() {
        let tensor = DeviceTensor {
            addr: 0x1000,
            size: 100,
            shape: vec![10],
            stride: vec![1],
            element_size: 4,
            device_id: 2,
        };
        assert_eq!(tensor.cuda_device_id(), Some(2));
    }

    #[test]
    fn test_arc_tensor_descriptor() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 24 * 4,
            shape: vec![2, 3, 4],
            stride: vec![12, 4, 1],
            element_size: 4,
        };
        let arc: Arc<dyn TensorDescriptor> = Arc::new(tensor);

        assert_eq!(arc.addr(), 0x1000);
        assert_eq!(arc.size(), 24 * 4);
        assert_eq!(arc.shape(), &[2, 3, 4]);
        assert_eq!(arc.stride(), &[12, 4, 1]);
        assert_eq!(arc.element_size(), 4);
        assert_eq!(arc.storage_kind(), StorageKind::System);
        assert!(arc.nixl_descriptor().is_none());
    }

    #[test]
    fn test_arc_tensor_send_sync() {
        // TestTensor doesn't impl Send+Sync, so we need a type that does
        struct SendSyncTensor {
            addr: usize,
            size: usize,
            shape: Vec<usize>,
            stride: Vec<usize>,
            element_size: usize,
        }

        unsafe impl Send for SendSyncTensor {}
        unsafe impl Sync for SendSyncTensor {}

        impl std::fmt::Debug for SendSyncTensor {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("SendSyncTensor").finish()
            }
        }

        impl MemoryDescriptor for SendSyncTensor {
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

        impl TensorDescriptor for SendSyncTensor {
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

        let tensor = SendSyncTensor {
            addr: 0x2000,
            size: 100,
            shape: vec![10],
            stride: vec![1],
            element_size: 4,
        };
        let arc: Arc<dyn TensorDescriptor + Send + Sync> = Arc::new(tensor);

        assert_eq!(arc.addr(), 0x2000);
        assert_eq!(arc.size(), 100);
        assert_eq!(arc.shape(), &[10]);
        assert_eq!(arc.stride(), &[1]);
        assert_eq!(arc.element_size(), 4);
    }

    #[test]
    fn test_tensor_shape_stride_element_size() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 48,
            shape: vec![3, 4],
            stride: vec![4, 1],
            element_size: 4,
        };
        assert_eq!(tensor.shape(), &[3, 4]);
        assert_eq!(tensor.stride(), &[4, 1]);
        assert_eq!(tensor.element_size(), 4);
    }

    #[test]
    fn test_tensor_numel_single_element() {
        let tensor = TestTensor {
            addr: 0x1000,
            size: 4,
            shape: vec![1, 1, 1],
            stride: vec![1, 1, 1],
            element_size: 4,
        };
        assert_eq!(tensor.numel(), 1);
    }
}
