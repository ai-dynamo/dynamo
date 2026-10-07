// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phantom publisher sockets.
//!
//! Each phantom binds one ZMQ XPUB socket and sends the same 4-frame multipart as the runtime's
//! `ZmqPubTransport` (topic, big-endian publisher ID, big-endian sequence, encoded `Frame`), so
//! the indexer's SUB side and the wire are format-identical to a live worker's PUB socket. The
//! XPUB socket adds what a PUB socket hides:
//!
//! - subscription messages, so the publisher starts only after the indexer has subscribed (a PUB
//!   socket silently drops everything sent before the subscriber joins);
//! - `ZMQ_XPUB_NODROP`, so a send that hits the high-water mark returns `EAGAIN`. The message is
//!   then dropped, as a PUB socket would drop it, but counted.

use std::os::raw::c_int;

use anyhow::{Context, Result, ensure};
use bytes::Bytes;
use dynamo_runtime::transports::event_plane::Frame;

/// The runtime's ZMQ send high-water mark (`zmq_transport::ZMQ_SNDHWM`).
pub const SNDHWM: i32 = 100_000;
/// Bounded flush of queued messages when a socket closes, so exit cannot hang on a lost peer.
const LINGER_MS: i32 = 10_000;
const SUBSCRIBE: u8 = 1;
const UNSUBSCRIBE: u8 = 0;

/// A ZMQ context with the runtime's I/O-thread setting (`DYN_ZMQ_IO_THREADS`, default 4).
pub fn context() -> Result<zmq::Context> {
    let io_threads = match std::env::var("DYN_ZMQ_IO_THREADS") {
        Ok(value) => value
            .parse::<i32>()
            .ok()
            .filter(|threads| *threads > 0)
            .context("DYN_ZMQ_IO_THREADS must be a positive integer")?,
        Err(_) => 4,
    };
    let context = zmq::Context::new();
    context.set_io_threads(io_threads)?;
    Ok(context)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendOutcome {
    Sent,
    /// The subscriber's pipe was at the high-water mark; the message was dropped.
    HwmDrop,
}

pub struct PhantomSocket {
    socket: zmq::Socket,
    subscribes: u64,
    unsubscribes: u64,
}

impl PhantomSocket {
    /// Bind an XPUB socket; returns it with the bound endpoint.
    pub fn bind(context: &zmq::Context, endpoint: &str) -> Result<(Self, String)> {
        let mut socket = context.socket(zmq::XPUB)?;
        socket.set_ipv6(endpoint.contains('['))?;
        socket.set_sndhwm(SNDHWM)?;
        socket.set_linger(LINGER_MS)?;
        set_xpub_nodrop(&mut socket)?;
        socket
            .bind(endpoint)
            .with_context(|| format!("binding XPUB {endpoint}"))?;
        let bound = socket
            .get_last_endpoint()?
            .map_err(|_| anyhow::anyhow!("bound endpoint is not UTF-8"))?;
        Ok((
            Self {
                socket,
                subscribes: 0,
                unsubscribes: 0,
            },
            bound,
        ))
    }

    /// Drain pending subscription messages; returns whether a subscriber is attached.
    pub fn poll_subscriptions(&mut self) -> Result<bool> {
        loop {
            match self.socket.recv_bytes(zmq::DONTWAIT) {
                Ok(message) => match message.first() {
                    Some(&SUBSCRIBE) => self.subscribes += 1,
                    Some(&UNSUBSCRIBE) => self.unsubscribes += 1,
                    _ => {}
                },
                Err(zmq::Error::EAGAIN) => return Ok(self.subscribed()),
                Err(error) => return Err(error.into()),
            }
        }
    }

    pub fn subscribed(&self) -> bool {
        self.subscribes > self.unsubscribes
    }

    /// Subscriptions seen so far; more than one means the subscriber reconnected.
    pub fn subscribes(&self) -> u64 {
        self.subscribes
    }

    pub fn unsubscribes(&self) -> u64 {
        self.unsubscribes
    }

    /// Send one event-plane envelope as `ZmqPubTransport::publish` frames it. Never blocks.
    pub fn send(
        &self,
        topic: &str,
        publisher_id: u64,
        sequence: u64,
        envelope: Bytes,
    ) -> Result<SendOutcome> {
        let frame = Frame::new(envelope).encode();
        let publisher = publisher_id.to_be_bytes();
        let sequence = sequence.to_be_bytes();
        let parts: [&[u8]; 4] = [topic.as_bytes(), &publisher, &sequence, &frame];
        match self.socket.send_multipart(parts, zmq::DONTWAIT) {
            Ok(()) => Ok(SendOutcome::Sent),
            Err(zmq::Error::EAGAIN) => Ok(SendOutcome::HwmDrop),
            Err(error) => Err(error.into()),
        }
    }
}

fn set_xpub_nodrop(socket: &mut zmq::Socket) -> Result<()> {
    let enabled: c_int = 1;
    // SAFETY: the socket pointer is live for this call, and ZMQ_XPUB_NODROP takes an int.
    let rc = unsafe {
        zmq_sys::zmq_setsockopt(
            socket.as_mut_ptr(),
            zmq_sys::ZMQ_XPUB_NODROP as c_int,
            (&enabled as *const c_int).cast(),
            std::mem::size_of::<c_int>(),
        )
    };
    ensure!(
        rc == 0,
        "setting ZMQ_XPUB_NODROP: {}",
        zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use dynamo_runtime::transports::event_plane::{Codec, EventTransportTx, ZmqPubTransport};

    use super::*;

    const TOPIC: &str = "kv-events";

    fn receive(sub: &zmq::Socket, timeout: Duration) -> Option<Vec<Vec<u8>>> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            match sub.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => return Some(frames),
                Err(zmq::Error::EAGAIN) => std::thread::sleep(Duration::from_millis(5)),
                Err(error) => panic!("receive failed: {error}"),
            }
        }
        None
    }

    /// The gate observes the subscription, and the frames match the runtime PUB transport's.
    #[tokio::test]
    async fn gate_sees_the_subscriber_and_frames_match_the_runtime_transport() {
        let context = context().unwrap();
        let (mut phantom, phantom_endpoint) =
            PhantomSocket::bind(&context, "tcp://127.0.0.1:*").unwrap();
        let (runtime_pub, runtime_endpoint) = ZmqPubTransport::bind("tcp://127.0.0.1:0", TOPIC)
            .await
            .unwrap();
        assert!(!phantom.poll_subscriptions().unwrap());

        let sub = context.socket(zmq::SUB).unwrap();
        sub.set_subscribe(TOPIC.as_bytes()).unwrap();
        sub.connect(&phantom_endpoint).unwrap();
        sub.connect(&runtime_endpoint).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !phantom.poll_subscriptions().unwrap() {
            assert!(
                Instant::now() < deadline,
                "the XPUB never saw the subscription"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!((phantom.subscribes(), phantom.unsubscribes()), (1, 0));

        let codec = Codec::default();
        let payload = codec.encode_payload(&vec![1u64, 2, 3]).unwrap();
        let envelope = codec
            .encode_envelope_parts(77, 5, 1234, TOPIC, &payload)
            .unwrap();

        // The runtime PUB socket has no gate: resend until the subscriber has joined it.
        let expected = loop {
            runtime_pub.publish(TOPIC, envelope.clone()).await.unwrap();
            if let Some(frames) = receive(&sub, Duration::from_millis(100)) {
                break frames;
            }
            assert!(Instant::now() < deadline, "the runtime PUB never delivered");
        };
        while receive(&sub, Duration::from_millis(100)).is_some() {}

        assert_eq!(
            phantom.send(TOPIC, 77, 5, envelope).unwrap(),
            SendOutcome::Sent
        );
        let frames = receive(&sub, Duration::from_secs(5)).expect("XPUB delivery");
        assert_eq!(frames, expected);
        assert_eq!(frames.len(), 4);
    }
}
