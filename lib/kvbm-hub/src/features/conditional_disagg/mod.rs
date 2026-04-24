// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ConditionalDisagg feature — hub-side manager and client-side wrapper.

pub mod client;
pub mod manager;

pub use client::ConditionalDisaggClient;
pub use manager::ConditionalDisaggManager;
