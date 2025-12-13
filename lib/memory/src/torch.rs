// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::nixl::{self, NixlDescriptor};
use super::{MemoryDescriptor, StorageKind};
use std::any::Any;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TorchDevice {
    Cuda(usize),
    Other(String),
}

impl TorchDevice {
    pub fn is_cuda(&self) -> bool {
        matches!(self, TorchDevice::Cuda(_))
    }

    pub fn cuda_device_index(&self) -> Option<usize> {
        match self {
            TorchDevice::Cuda(index) => Some(*index),
            TorchDevice::Other(_) => None,
        }
    }

    fn to_storage_kind(&self) -> StorageKind {
        match self {
            TorchDevice::Cuda(index) => StorageKind::Device(*index as u32),
            TorchDevice::Other(_) => panic!("TorchDevice::Other is not supported for StorageKind"),
        }
    }

    fn to_mem_type(&self) -> nixl::MemType {
        match self {
            TorchDevice::Cuda(_) => nixl::MemType::Vram,
            TorchDevice::Other(_) => panic!("TorchDevice::Other is not supported for MemType"),
        }
    }

    fn device_id(&self) -> u64 {
        match self {
            TorchDevice::Cuda(index) => *index as u64,
            TorchDevice::Other(_) => 0,
        }
    }
}

pub trait TorchTensor: std::fmt::Debug + Send + Sync {
    fn device(&self) -> TorchDevice;
    fn data_ptr(&self) -> u64;
    fn size_bytes(&self) -> usize;
    fn shape(&self) -> Vec<usize>;
    fn stride(&self) -> Vec<usize>;
}

// =============================================================================
// Arc<dyn TorchTensor> support for NixlRegisterExt
// =============================================================================

impl nixl::NixlCompatible for Arc<dyn TorchTensor> {
    fn nixl_params(&self) -> (*const u8, usize, nixl::MemType, u64) {
        let device = self.device();
        (
            self.data_ptr() as *const u8,
            self.size_bytes(),
            device.to_mem_type(),
            device.device_id(),
        )
    }
}

impl MemoryDescriptor for Arc<dyn TorchTensor> {
    fn addr(&self) -> usize {
        self.data_ptr() as usize
    }

    fn size(&self) -> usize {
        self.size_bytes()
    }

    fn storage_kind(&self) -> StorageKind {
        self.device().to_storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}
