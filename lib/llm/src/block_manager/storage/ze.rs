// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZE-specific extensions for DeviceStorage (torch tensor integration).

// This file previously contained the deprecated `new_from_torch_ze` function.
// It has been removed as it was never called and the functionality is now
// handled by the unified `DeviceStorage::new_from_torch()` method.
