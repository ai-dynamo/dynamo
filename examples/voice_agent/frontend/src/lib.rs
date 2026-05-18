// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::realtime::RealtimeServerEvent;
use dynamo_runtime::protocols::annotated::Annotated;

pub fn server_event_finishes_turn(chunk: &Annotated<RealtimeServerEvent>) -> bool {
    matches!(
        chunk.data.as_ref(),
        Some(
            RealtimeServerEvent::SessionUpdated(_)
                | RealtimeServerEvent::ResponseDone(_)
                | RealtimeServerEvent::Error(_)
        )
    )
}
