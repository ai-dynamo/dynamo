// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for UDS transport

#![cfg(unix)]

#[macro_use]
mod common;

use common::{UdsFactory, scenarios};

transport_integration_tests!(UdsFactory);
