// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`KvbmServiceClient`] — typed gRPC client over a unix domain socket.

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures::StreamExt;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;

use crate::error::{ClientError, ClientResult};
use crate::handle::RegistrationHandle;
use crate::proto::{self, RawClient, RegisterRequest};

/// Typed gRPC client over a unix domain socket.
pub struct KvbmServiceClient {
    inner: RawClient<Channel>,
}

impl KvbmServiceClient {
    /// Connect to a `kvbm-service` UDS endpoint.
    ///
    /// The `path` must point to a bound unix domain socket. The connection is
    /// established eagerly; returns an error if the socket is unreachable within
    /// the 5-second connect timeout.
    pub async fn connect_uds(path: impl AsRef<Path>) -> ClientResult<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        // tonic over UDS: the HTTP URI is a dummy (never dialed); the connector
        // ignores it and opens a UnixStream to `path` instead.
        let channel = Endpoint::try_from("http://[::1]:50051")
            .map_err(ClientError::Connect)?
            .connect_timeout(Duration::from_secs(5))
            .connect_with_connector(service_fn(move |_: Uri| {
                let p = path.clone();
                async move {
                    let stream = tokio::net::UnixStream::connect(p).await?;
                    Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
                }
            }))
            .await?;
        Ok(Self {
            inner: RawClient::new(channel),
        })
    }

    /// Call `Register` and wait for the first `Accepted` event.
    ///
    /// Returns a [`RegistrationHandle`] owning the open server stream. Drop the
    /// handle to ungracefully detach (the server will unregister the client).
    ///
    /// `instance` is the tagged-union description of what the client wants to
    /// register (today: only [`proto::KvbmInstance`]). The server derives the
    /// tenancy key as a hash of the active arm.
    ///
    /// Returns an error if:
    /// - the RPC itself fails (transport or server-side `Status`),
    /// - the stream closes before emitting any event ([`ClientError::NoAcceptedEvent`]),
    /// - the first event is not `Accepted` ([`ClientError::UnexpectedFirstEvent`]).
    pub async fn register(
        &mut self,
        client_id: impl Into<String>,
        instance: proto::RegistrationInstance,
    ) -> ClientResult<RegistrationHandle> {
        let req = RegisterRequest {
            client_id: client_id.into(),
            instance: Some(instance),
        };
        let response = self.inner.register(req).await?;
        let mut stream = response.into_inner();

        let first = stream
            .next()
            .await
            .ok_or(ClientError::NoAcceptedEvent)?
            .map_err(ClientError::Rpc)?;

        let accepted = match first.kind {
            Some(proto::event::Kind::Accepted(a)) => a,
            Some(other) => {
                return Err(ClientError::UnexpectedFirstEvent(format!("{other:?}")));
            }
            None => {
                return Err(ClientError::UnexpectedFirstEvent("empty event".into()));
            }
        };

        Ok(RegistrationHandle::new(accepted, stream))
    }
}
