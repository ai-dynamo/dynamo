// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core event storage engine backed by a generational slot system.

pub(crate) mod system;

pub use system::EventSystemBase;
