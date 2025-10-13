// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
