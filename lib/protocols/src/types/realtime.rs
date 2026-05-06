// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Re-exports upstream async-openai realtime event types. Per the ownership
// rubric in `lib/protocols/CLAUDE.md`, no Dynamo-side overrides are needed
// today: upstream covers the full event surface (`RealtimeClientEvent`,
// `RealtimeServerEvent`, and their per-variant payloads) and no real client
// has been observed to send a shape upstream rejects.

pub use async_openai::types::realtime::*;

/// Returns the `type` wire-tag string for a realtime event variant — useful
/// for logging, error messages, and metric labels that need a stable name
/// without reserializing the value.
///
/// `async-openai` ships an equivalent `crate::traits::EventType` trait, but it
/// is gated on the `_api` feature, which pulls reqwest / tokio / secrecy /
/// eventsource-stream into the build. `dynamo-protocols` is types-only by
/// design (see the Cargo.toml banner), so we mirror the trait shape locally.
/// If `_api` ever becomes affordable for this crate, swap `pub use
/// async_openai::traits::EventType;` in here and remove the impls below; call
/// sites need no changes.
///
/// Implemented today only for `RealtimeClientEvent`. The `RealtimeServerEvent`
/// impl can be added when a consumer needs it.
pub trait EventType {
    fn event_type(&self) -> &'static str;
}

impl EventType for RealtimeClientEvent {
    fn event_type(&self) -> &'static str {
        // `RealtimeClientEvent` is not `#[non_exhaustive]`, so a future upstream
        // variant breaks this match at compile time rather than silently
        // returning a stale label.
        match self {
            RealtimeClientEvent::SessionUpdate(_) => "session.update",
            RealtimeClientEvent::InputAudioBufferAppend(_) => "input_audio_buffer.append",
            RealtimeClientEvent::InputAudioBufferCommit(_) => "input_audio_buffer.commit",
            RealtimeClientEvent::InputAudioBufferClear(_) => "input_audio_buffer.clear",
            RealtimeClientEvent::ConversationItemCreate(_) => "conversation.item.create",
            RealtimeClientEvent::ConversationItemRetrieve(_) => "conversation.item.retrieve",
            RealtimeClientEvent::ConversationItemTruncate(_) => "conversation.item.truncate",
            RealtimeClientEvent::ConversationItemDelete(_) => "conversation.item.delete",
            RealtimeClientEvent::ResponseCreate(_) => "response.create",
            RealtimeClientEvent::ResponseCancel(_) => "response.cancel",
            RealtimeClientEvent::OutputAudioBufferClear(_) => "output_audio_buffer.clear",
        }
    }
}
