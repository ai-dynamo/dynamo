// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Re-exports upstream async-openai realtime event types. Per the ownership
// rubric in `lib/protocols/CLAUDE.md`, no Dynamo-side overrides are needed
// today: upstream covers the full event surface (`RealtimeClientEvent`,
// `RealtimeServerEvent`, and their per-variant payloads) and no real client
// has been observed to send a shape upstream rejects.

pub use async_openai::types::realtime::*;
