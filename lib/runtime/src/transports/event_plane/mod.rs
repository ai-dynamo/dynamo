// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event Plane: Transport-agnostic pub/sub communication layer.

mod codec;
mod dynamic_subscriber;
mod frame;
mod nats_transport;
mod traits;
mod transport;
mod zmq_transport;

pub use codec::MsgpackCodec;
pub use dynamic_subscriber::DynamicSubscriber;
pub use frame::{Frame, FrameError, FrameHeader};
pub use nats_transport::NatsTransport;
pub use traits::{EventEnvelope, EventStream, TypedEventStream};
pub use transport::{EventTransportRx, EventTransportTx, WireStream};
pub use zmq_transport::{ZmqPubTransport, ZmqSubTransport};
