// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A listener for the plain HTTP server whose connections can be closed when
//! shutdown has waited long enough: `axum::serve` waits for every connection
//! to finish, with no deadline of its own.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};

/// Accepts connections whose IO fails, wherever it would wait, once `close`
/// is cancelled. A connection the server is blocked on (its client stopped
/// reading, so the response cannot flush and its body is never polled again)
/// would otherwise never finish; failing its IO ends it, dropping its
/// response and whatever that response held. One waiting on its handler
/// rather than on IO still ends only when the handler does.
pub(super) struct ClosingListener {
    listener: TcpListener,
    close: CancellationToken,
}

impl ClosingListener {
    pub(super) fn new(listener: TcpListener, close: CancellationToken) -> Self {
        Self { listener, close }
    }
}

impl axum::serve::Listener for ClosingListener {
    type Io = ClosingStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        let (stream, addr) = axum::serve::Listener::accept(&mut self.listener).await;
        // A token per connection: waiting IO then contends on no lock that
        // other connections share.
        let close = self.close.child_token();
        let stream = ClosingStream {
            stream,
            read_closed: Box::pin(close.clone().cancelled_owned()),
            write_closed: Box::pin(close.cancelled_owned()),
        };
        (stream, addr)
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

/// A connection from [`ClosingListener`]. Each direction waits on its own
/// future, so a reader and a writer in different tasks (an upgraded
/// connection, split) are both woken.
pub(super) struct ClosingStream {
    stream: TcpStream,
    read_closed: Pin<Box<WaitForCancellationFutureOwned>>,
    write_closed: Pin<Box<WaitForCancellationFutureOwned>>,
}

/// `polled`, unless it is pending and `closed` has fired. Only IO that would
/// wait polls `closed`, so IO that is ready costs nothing extra.
fn unless_closed<T>(
    polled: Poll<io::Result<T>>,
    closed: Pin<&mut WaitForCancellationFutureOwned>,
    cx: &mut Context<'_>,
) -> Poll<io::Result<T>> {
    if polled.is_pending() && closed.poll(cx).is_ready() {
        return Poll::Ready(Err(io::Error::new(
            io::ErrorKind::ConnectionAborted,
            "connection closed at the shutdown deadline",
        )));
    }
    polled
}

impl AsyncRead for ClosingStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.stream).poll_read(cx, buf);
        unless_closed(polled, this.read_closed.as_mut(), cx)
    }
}

impl AsyncWrite for ClosingStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.stream).poll_write(cx, buf);
        unless_closed(polled, this.write_closed.as_mut(), cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.stream).poll_write_vectored(cx, bufs);
        unless_closed(polled, this.write_closed.as_mut(), cx)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.stream).poll_flush(cx);
        unless_closed(polled, this.write_closed.as_mut(), cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.stream).poll_shutdown(cx);
        unless_closed(polled, this.write_closed.as_mut(), cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::serve::Listener;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn accept_one(close: &CancellationToken) -> (ClosingStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut listener = ClosingListener::new(listener, close.clone());
        let client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (stream, _) = listener.accept().await;
        (stream, client)
    }

    #[tokio::test]
    async fn waiting_write_fails_once_closed() {
        let close = CancellationToken::new();
        let (mut stream, client) = accept_one(&close).await;
        // The client never reads, so writing soon waits for room.
        let writer = tokio::spawn(async move {
            let chunk = vec![0u8; 64 * 1024];
            loop {
                if let Err(err) = stream.write_all(&chunk).await {
                    return err;
                }
            }
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!writer.is_finished(), "a full socket makes writing wait");

        close.cancel();
        let err = tokio::time::timeout(Duration::from_secs(5), writer)
            .await
            .expect("closing ends a waiting write")
            .unwrap();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionAborted);
        drop(client);
    }

    #[tokio::test]
    async fn waiting_read_fails_once_closed_and_ready_io_passes() {
        let close = CancellationToken::new();
        let (mut stream, mut client) = accept_one(&close).await;
        client.write_all(b"sent").await.unwrap();
        let mut sent = [0u8; 4];
        stream.read_exact(&mut sent).await.unwrap();
        assert_eq!(&sent, b"sent");

        // The client sends nothing more, so reading waits.
        let reader = tokio::spawn(async move { stream.read(&mut [0u8; 1]).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        close.cancel();
        let err = tokio::time::timeout(Duration::from_secs(5), reader)
            .await
            .expect("closing ends a waiting read")
            .unwrap()
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionAborted);
    }
}
