// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for gRPC transport

#![cfg(feature = "grpc")]

#[macro_use]
mod common;

use common::{GrpcFactory, scenarios};

transport_integration_tests!(GrpcFactory);
