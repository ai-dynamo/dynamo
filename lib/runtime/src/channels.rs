// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Channel facades used by Dynamo internals.
//!
//! The default implementation is Tokio's channel types. Building
//! `dynamo-runtime` with `--features flume-channels` swaps the facade to flume
//! for call sites that import this module.

pub mod mpsc {
    #[cfg(feature = "flume-channels")]
    mod imp {
        use std::fmt;

        pub fn backend_name() -> &'static str {
            "flume"
        }

        pub mod error {
            use std::fmt;

            #[derive(Debug, Clone, PartialEq, Eq)]
            pub struct SendError<T>(pub T);

            impl<T> SendError<T> {
                pub fn into_inner(self) -> T {
                    self.0
                }
            }

            impl<T> fmt::Display for SendError<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("channel closed")
                }
            }

            impl<T: fmt::Debug> std::error::Error for SendError<T> {}

            #[derive(Debug, Clone, PartialEq, Eq)]
            pub enum TrySendError<T> {
                Full(T),
                Closed(T),
            }

            impl<T> TrySendError<T> {
                pub fn into_inner(self) -> T {
                    match self {
                        Self::Full(value) | Self::Closed(value) => value,
                    }
                }
            }

            impl<T> fmt::Display for TrySendError<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    match self {
                        Self::Full(_) => f.write_str("channel full"),
                        Self::Closed(_) => f.write_str("channel closed"),
                    }
                }
            }

            impl<T: fmt::Debug> std::error::Error for TrySendError<T> {}

            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub enum TryRecvError {
                Empty,
                Disconnected,
            }

            impl fmt::Display for TryRecvError {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    match self {
                        Self::Empty => f.write_str("channel empty"),
                        Self::Disconnected => f.write_str("channel closed"),
                    }
                }
            }

            impl std::error::Error for TryRecvError {}
        }

        impl<T> From<flume::SendError<T>> for error::SendError<T> {
            fn from(value: flume::SendError<T>) -> Self {
                Self(value.0)
            }
        }

        impl<T> From<flume::TrySendError<T>> for error::TrySendError<T> {
            fn from(value: flume::TrySendError<T>) -> Self {
                match value {
                    flume::TrySendError::Full(value) => Self::Full(value),
                    flume::TrySendError::Disconnected(value) => Self::Closed(value),
                }
            }
        }

        impl From<flume::TryRecvError> for error::TryRecvError {
            fn from(value: flume::TryRecvError) -> Self {
                match value {
                    flume::TryRecvError::Empty => Self::Empty,
                    flume::TryRecvError::Disconnected => Self::Disconnected,
                }
            }
        }

        pub struct Sender<T> {
            inner: flume::Sender<T>,
        }

        impl<T> Clone for Sender<T> {
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl<T> fmt::Debug for Sender<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("Sender").finish_non_exhaustive()
            }
        }

        impl<T> Sender<T> {
            pub async fn send(&self, value: T) -> Result<(), error::SendError<T>> {
                self.inner.send_async(value).await.map_err(Into::into)
            }

            pub fn try_send(&self, value: T) -> Result<(), error::TrySendError<T>> {
                self.inner.try_send(value).map_err(Into::into)
            }

            pub fn is_closed(&self) -> bool {
                self.inner.is_disconnected()
            }
        }

        pub struct Receiver<T> {
            inner: flume::Receiver<T>,
        }

        impl<T> fmt::Debug for Receiver<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("Receiver").finish_non_exhaustive()
            }
        }

        impl<T> Receiver<T> {
            pub async fn recv(&mut self) -> Option<T> {
                self.inner.recv_async().await.ok()
            }

            pub fn try_recv(&mut self) -> Result<T, error::TryRecvError> {
                self.inner.try_recv().map_err(Into::into)
            }
        }

        pub struct UnboundedSender<T> {
            inner: flume::Sender<T>,
        }

        impl<T> Clone for UnboundedSender<T> {
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl<T> fmt::Debug for UnboundedSender<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("UnboundedSender").finish_non_exhaustive()
            }
        }

        impl<T> UnboundedSender<T> {
            pub fn send(&self, value: T) -> Result<(), error::SendError<T>> {
                self.inner.send(value).map_err(Into::into)
            }

            pub fn is_closed(&self) -> bool {
                self.inner.is_disconnected()
            }
        }

        pub struct UnboundedReceiver<T> {
            inner: flume::Receiver<T>,
        }

        impl<T> fmt::Debug for UnboundedReceiver<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("UnboundedReceiver").finish_non_exhaustive()
            }
        }

        impl<T> UnboundedReceiver<T> {
            pub async fn recv(&mut self) -> Option<T> {
                self.inner.recv_async().await.ok()
            }

            pub fn try_recv(&mut self) -> Result<T, error::TryRecvError> {
                self.inner.try_recv().map_err(Into::into)
            }
        }

        pub fn channel<T>(buffer: usize) -> (Sender<T>, Receiver<T>) {
            let (tx, rx) = flume::bounded(buffer);
            (Sender { inner: tx }, Receiver { inner: rx })
        }

        pub fn unbounded_channel<T>() -> (UnboundedSender<T>, UnboundedReceiver<T>) {
            let (tx, rx) = flume::unbounded();
            (
                UnboundedSender { inner: tx },
                UnboundedReceiver { inner: rx },
            )
        }
    }

    #[cfg(not(feature = "flume-channels"))]
    mod imp {
        use std::fmt;

        pub fn backend_name() -> &'static str {
            "tokio"
        }

        pub mod error {
            use std::fmt;

            #[derive(Debug, Clone, PartialEq, Eq)]
            pub struct SendError<T>(pub T);

            impl<T> SendError<T> {
                pub fn into_inner(self) -> T {
                    self.0
                }
            }

            impl<T> fmt::Display for SendError<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("channel closed")
                }
            }

            impl<T: fmt::Debug> std::error::Error for SendError<T> {}

            #[derive(Debug, Clone, PartialEq, Eq)]
            pub enum TrySendError<T> {
                Full(T),
                Closed(T),
            }

            impl<T> TrySendError<T> {
                pub fn into_inner(self) -> T {
                    match self {
                        Self::Full(value) | Self::Closed(value) => value,
                    }
                }
            }

            impl<T> fmt::Display for TrySendError<T> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    match self {
                        Self::Full(_) => f.write_str("channel full"),
                        Self::Closed(_) => f.write_str("channel closed"),
                    }
                }
            }

            impl<T: fmt::Debug> std::error::Error for TrySendError<T> {}

            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub enum TryRecvError {
                Empty,
                Disconnected,
            }

            impl fmt::Display for TryRecvError {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    match self {
                        Self::Empty => f.write_str("channel empty"),
                        Self::Disconnected => f.write_str("channel closed"),
                    }
                }
            }

            impl std::error::Error for TryRecvError {}
        }

        impl<T> From<tokio::sync::mpsc::error::SendError<T>> for error::SendError<T> {
            fn from(value: tokio::sync::mpsc::error::SendError<T>) -> Self {
                Self(value.0)
            }
        }

        impl<T> From<tokio::sync::mpsc::error::TrySendError<T>> for error::TrySendError<T> {
            fn from(value: tokio::sync::mpsc::error::TrySendError<T>) -> Self {
                match value {
                    tokio::sync::mpsc::error::TrySendError::Full(value) => Self::Full(value),
                    tokio::sync::mpsc::error::TrySendError::Closed(value) => Self::Closed(value),
                }
            }
        }

        impl From<tokio::sync::mpsc::error::TryRecvError> for error::TryRecvError {
            fn from(value: tokio::sync::mpsc::error::TryRecvError) -> Self {
                match value {
                    tokio::sync::mpsc::error::TryRecvError::Empty => Self::Empty,
                    tokio::sync::mpsc::error::TryRecvError::Disconnected => Self::Disconnected,
                }
            }
        }

        pub struct Sender<T> {
            inner: tokio::sync::mpsc::Sender<T>,
        }

        impl<T> Clone for Sender<T> {
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl<T> fmt::Debug for Sender<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("Sender").finish_non_exhaustive()
            }
        }

        impl<T> Sender<T> {
            pub async fn send(&self, value: T) -> Result<(), error::SendError<T>> {
                self.inner.send(value).await.map_err(Into::into)
            }

            pub fn try_send(&self, value: T) -> Result<(), error::TrySendError<T>> {
                self.inner.try_send(value).map_err(Into::into)
            }

            pub fn is_closed(&self) -> bool {
                self.inner.is_closed()
            }
        }

        pub struct Receiver<T> {
            inner: tokio::sync::mpsc::Receiver<T>,
        }

        impl<T> fmt::Debug for Receiver<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("Receiver").finish_non_exhaustive()
            }
        }

        impl<T> Receiver<T> {
            pub async fn recv(&mut self) -> Option<T> {
                self.inner.recv().await
            }

            pub fn try_recv(&mut self) -> Result<T, error::TryRecvError> {
                self.inner.try_recv().map_err(Into::into)
            }
        }

        pub struct UnboundedSender<T> {
            inner: tokio::sync::mpsc::UnboundedSender<T>,
        }

        impl<T> Clone for UnboundedSender<T> {
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl<T> fmt::Debug for UnboundedSender<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("UnboundedSender").finish_non_exhaustive()
            }
        }

        impl<T> UnboundedSender<T> {
            pub fn send(&self, value: T) -> Result<(), error::SendError<T>> {
                self.inner.send(value).map_err(Into::into)
            }

            pub fn is_closed(&self) -> bool {
                self.inner.is_closed()
            }
        }

        pub struct UnboundedReceiver<T> {
            inner: tokio::sync::mpsc::UnboundedReceiver<T>,
        }

        impl<T> fmt::Debug for UnboundedReceiver<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("UnboundedReceiver").finish_non_exhaustive()
            }
        }

        impl<T> UnboundedReceiver<T> {
            pub async fn recv(&mut self) -> Option<T> {
                self.inner.recv().await
            }

            pub fn try_recv(&mut self) -> Result<T, error::TryRecvError> {
                self.inner.try_recv().map_err(Into::into)
            }
        }

        pub fn channel<T>(buffer: usize) -> (Sender<T>, Receiver<T>) {
            let (tx, rx) = tokio::sync::mpsc::channel(buffer);
            (Sender { inner: tx }, Receiver { inner: rx })
        }

        pub fn unbounded_channel<T>() -> (UnboundedSender<T>, UnboundedReceiver<T>) {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            (
                UnboundedSender { inner: tx },
                UnboundedReceiver { inner: rx },
            )
        }
    }

    pub use imp::*;
}

#[cfg(test)]
mod tests {
    use super::mpsc;

    #[tokio::test]
    async fn bounded_channel_round_trips() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.send(7).await.unwrap();
        assert_eq!(rx.recv().await, Some(7));
    }

    #[tokio::test]
    async fn unbounded_channel_round_trips() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tx.send(11).unwrap();
        assert_eq!(rx.recv().await, Some(11));
    }
}
