// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local event implementation backed by a generational slot system.

mod event;
pub(crate) mod system;

pub use event::LocalEvent;
pub use system::LocalEventSystem;
