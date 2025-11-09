// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Transport Module
//!
//! This module provides a high-performance TCP transport implementation with:
//! - Zero-copy frame codec for minimal overhead
//! - CPU pinning support for predictable latency
//! - Frame type routing (Message, Response, Ack, Event)
//! - Graceful shutdown with proper FIN handling
//! - Keep-alive for dead connection detection

mod framing;
mod listener;
mod transport;

pub use framing::TcpFrameCodec;
pub use listener::{RuntimeConfig, TcpListener, TcpListenerBuilder};
pub use transport::{TcpTransport, TcpTransportBuilder};

// we will need to define a connection pool
// there will typically be three sockets per instance:
// - one for receiving active messages to marshall to handlers
// - one for sending responses to the originator
// - one for sending ack/events to the originator (skip head-of-line blocking on larger responses)
//
// that means we have:
// - three bound sockets accepting connections
// - per-remote instance three connections (one for each socket) to send data
//   - we will need a connection pool to manage the connections
//   - the lru for the connection pool will keep all three connections for each instance
//   - the connection pool needs to be generic over the value type to support other transport types (should be in utils module
//
//
// - to start let's simplify, so we need to put a small indicator in our frame to indicate the type of frame it is:
//   - u16 for schema version
//   - v0.1 - use a u8 to indicate the type of frame it is:
//     - message - from send message call
//     - response - from send response call
//     - ack - from send ack call
//     - event - from trigger event call

// - framing is simple:
//   - we have a header and a payload both already encoded as bytes
//   - 2x u32 for the length of the header and payload
//   - on the send side, i would like an extremely efficient way of sending with &[u8] slices without out copying for header and payload
//     - if this is not possible, we need to discuss

// on the rx side - in the first version wil use use the single socket connection and dispatch to the appropriate flume sender
// - in the future, we will have one socket per type (possibly)
// - the rx side never sends any data, but does issue the proper close protocol

// on the tx side when we initiate a shutdown, we want to stop accepting new messages on our bound socket.
// we need to close teh flume channel and drain it,
// then for each open connection, use an into_iter() over the lru pool, issue the FIN / shutdown protocol and await the recv close

// - in tcp world, will also need a way to signify that one side or the other is shutting down

// Client initiates shutdown: The client calls shutdown() (with the SD_SEND or equivalent flag) to indicate it has no more data to send and sends a FIN segment.
// Server receives FIN: The server receives the FIN, acknowledges it, and enters the CLOSE_WAIT state.
// Server finishes sending data: The server continues to send any remaining data it has.
// Server closes socket: Once the server has no more data, it calls close() (or equivalent), sending its own FIN segment and entering the LAST_ACK state.
// Client receives FIN and enters TIME_WAIT: The client receives the server's FIN, acknowledges it, and enters the TIME_WAIT state for a duration of 2 Maximum Segment Lifetimes (2MSL).
// Server closes fully: The server receives the final ACK and closes fully.

// note - if our lru evicts a connections, then we need to issue the same shutdown protocol, but we do not drain/close the flume channel
// the remote side simply follows the same protocol and drops its sender
