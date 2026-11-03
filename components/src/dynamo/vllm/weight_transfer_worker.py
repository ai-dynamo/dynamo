# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal vLLM worker extension for physical-device transfer qualification."""

import torch


class StrictRLWorkerExtension:
    """Expose only read-only device identity from the owning vLLM worker."""

    device: torch.device

    def get_weight_transfer_device_info(self) -> dict[str, str]:
        """Report the worker-assigned CUDA device, independent of ambient state."""
        properties = torch.cuda.get_device_properties(self.device)
        return {"device_uuid": str(properties.uuid)}
