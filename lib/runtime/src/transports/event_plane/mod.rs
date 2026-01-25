// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event Plane: Transport-agnostic pub/sub communication layer.

mod codec;
mod frame;
mod nats_transport;
mod traits;
mod transport;

pub use codec::MsgpackCodec;
pub use frame::{Frame, FrameError, FrameHeader};
pub use nats_transport::NatsTransport;
pub use traits::{EventEnvelope, EventStream, TypedEventStream};
pub use transport::{EventTransportRx, EventTransportTx, WireStream};
