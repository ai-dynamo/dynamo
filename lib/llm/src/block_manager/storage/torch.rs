// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TorchDevice {
    Cuda(usize),
    Other(String),
}

pub trait TorchTensor: std::fmt::Debug + Send + Sync {
    fn device(&self) -> TorchDevice;
    fn data_ptr(&self) -> u64;
    fn size_bytes(&self) -> usize;
    fn shape(&self) -> Vec<usize>;
    fn stride(&self) -> Vec<usize>;
}

/// Check if a tensor is on a Ze/XPU/SYCL device
pub fn is_ze(tensor: &dyn TorchTensor) -> bool {
    match tensor.device() {
        TorchDevice::Other(ref kind) => {
            let device = kind.to_ascii_lowercase();
            device.contains("xpu") || device.contains("ze") || device.contains("sycl")
        }
        TorchDevice::Cuda(_) => false,
    }
}

/// Check if a tensor is on a CUDA device
pub fn is_cuda(tensor: &dyn TorchTensor) -> bool {
    matches!(tensor.device(), TorchDevice::Cuda(_))
}

/// Check if all tensors in a slice are on Ze/XPU/SYCL devices
pub fn is_ze_tensors(tensors: &[Arc<dyn TorchTensor>]) -> bool {
    !tensors.is_empty() && tensors.iter().all(|t| is_ze(t.as_ref()))
}

/// Check if all tensors in a slice are on CUDA devices
pub fn is_cuda_tensors(tensors: &[Arc<dyn TorchTensor>]) -> bool {
    !tensors.is_empty() && tensors.iter().all(|t| is_cuda(t.as_ref()))
}
