// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::DeviceBackend;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TorchDevice {
    Cuda(usize),
    Other(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceBackendKind {
    Cuda,
    Hpu,
    Xpu,
}

impl DeviceBackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Hpu => "hpu",
            Self::Xpu => "xpu",
        }
    }
}

pub fn parse_device_backend_kind(device: &TorchDevice) -> Option<DeviceBackendKind> {
    match device {
        TorchDevice::Cuda(_) => Some(DeviceBackendKind::Cuda),
        TorchDevice::Other(device_type) => {
            let normalized = device_type.to_ascii_lowercase();
            if normalized.starts_with("cuda") {
                Some(DeviceBackendKind::Cuda)
            } else if normalized.starts_with("hpu") {
                Some(DeviceBackendKind::Hpu)
            } else if normalized.starts_with("xpu") {
                Some(DeviceBackendKind::Xpu)
            } else {
                None
            }
        }
    }
}

pub fn infer_tensor_device_family(tensors: &[Arc<dyn TorchTensor>]) -> anyhow::Result<DeviceBackendKind> {
    if tensors.is_empty() {
        return Err(anyhow::anyhow!("No tensors provided; cannot infer tensor backend family"));
    }

    let first_device = tensors[0].device();
    let first_family = parse_device_backend_kind(&first_device).ok_or_else(|| {
        anyhow::anyhow!(
            "Unsupported tensor device type '{:?}' for tensor[0]; supported backends: cuda, xpu, hpu",
            first_device
        )
    })?;

    for (idx, tensor) in tensors.iter().enumerate().skip(1) {
        let family = parse_device_backend_kind(&tensor.device()).ok_or_else(|| {
            anyhow::anyhow!(
                "Unsupported tensor device type '{:?}' for tensor[{}]; supported backends: cuda, xpu, hpu",
                tensor.device(),
                idx
            )
        })?;

        if family != first_family {
            return Err(anyhow::anyhow!(
                "Mixed tensor device families are not supported: tensor[0]={:?} ({}) vs tensor[{}]={:?} ({})",
                first_device,
                first_family.as_str(),
                idx,
                tensor.device(),
                family.as_str()
            ));
        }
    }

    Ok(first_family)
}

pub fn map_backend_kind_to_legacy_backend(kind: DeviceBackendKind) -> anyhow::Result<DeviceBackend> {
    match kind {
        DeviceBackendKind::Cuda => Ok(DeviceBackend::Cuda),
        DeviceBackendKind::Xpu => Ok(DeviceBackend::Ze),
        DeviceBackendKind::Hpu => Ok(DeviceBackend::Hpu),
    }
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

/// Check if a tensor is on an HPU device
pub fn is_hpu(tensor: &dyn TorchTensor) -> bool {
    match tensor.device() {
        TorchDevice::Other(ref kind) => {
            let device = kind.to_ascii_lowercase();
            device.contains("hpu") || device.contains("habana")
        }
        TorchDevice::Cuda(_) => false,
    }
}

/// Check if all tensors in a slice are on Ze/XPU/SYCL devices
pub fn is_ze_tensors(tensors: &[Arc<dyn TorchTensor>]) -> bool {
    !tensors.is_empty() && tensors.iter().all(|t| is_ze(t.as_ref()))
}

/// Check if all tensors in a slice are on CUDA devices
pub fn is_cuda_tensors(tensors: &[Arc<dyn TorchTensor>]) -> bool {
    !tensors.is_empty() && tensors.iter().all(|t| is_cuda(t.as_ref()))
}

/// Check if all tensors in a slice are on HPU devices
pub fn is_hpu_tensors(tensors: &[Arc<dyn TorchTensor>]) -> bool {
    !tensors.is_empty() && tensors.iter().all(|t| is_hpu(t.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct MockTensor {
        device: TorchDevice,
    }

    impl TorchTensor for MockTensor {
        fn device(&self) -> TorchDevice {
            self.device.clone()
        }

        fn data_ptr(&self) -> u64 {
            0
        }

        fn size_bytes(&self) -> usize {
            0
        }

        fn shape(&self) -> Vec<usize> {
            vec![1]
        }

        fn stride(&self) -> Vec<usize> {
            vec![1]
        }
    }

    #[test]
    fn parse_device_backend_kind_accepts_known_backends() {
        assert_eq!(
            parse_device_backend_kind(&TorchDevice::Cuda(0)),
            Some(DeviceBackendKind::Cuda)
        );
        assert_eq!(
            parse_device_backend_kind(&TorchDevice::Other("hpu:0".to_string())),
            Some(DeviceBackendKind::Hpu)
        );
        assert_eq!(
            parse_device_backend_kind(&TorchDevice::Other("xpu".to_string())),
            Some(DeviceBackendKind::Xpu)
        );
    }

    #[test]
    fn parse_device_backend_kind_returns_none_for_unknown_backend() {
        assert_eq!(
            parse_device_backend_kind(&TorchDevice::Other("cpu".to_string())),
            None
        );
    }

    #[test]
    fn infer_tensor_device_family_rejects_mixed_backends() {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(MockTensor {
                device: TorchDevice::Cuda(0),
            }),
            Arc::new(MockTensor {
                device: TorchDevice::Other("xpu:0".to_string()),
            }),
        ];

        let result = infer_tensor_device_family(&tensors);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Mixed tensor device families"));
    }

    #[test]
    fn infer_tensor_device_family_accepts_single_family() {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(MockTensor {
                device: TorchDevice::Other("xpu".to_string()),
            }),
            Arc::new(MockTensor {
                device: TorchDevice::Other("xpu:1".to_string()),
            }),
        ];

        let family = infer_tensor_device_family(&tensors).expect("xpu family should be inferred");
        assert_eq!(family, DeviceBackendKind::Xpu);
    }

    #[test]
    fn map_backend_kind_to_legacy_backend_supports_hpu() {
        let result = map_backend_kind_to_legacy_backend(DeviceBackendKind::Hpu);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), DeviceBackend::Hpu);
    }
}
