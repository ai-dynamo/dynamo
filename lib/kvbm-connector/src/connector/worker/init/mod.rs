// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Initialization module for the [`ConnectorWorker`]
//!
//! Initializatio happens in the following steps:
//!
//! -

pub(crate) mod pending;

pub(crate) use pending::{DeviceLayoutKind, PendingWorkerState};
