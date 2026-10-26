// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::{SinkExt, StreamExt, stream::BoxStream};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream,
    tungstenite::{Message, client::IntoClientRequest, protocol::WebSocketConfig},
};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use url::Url;

use crate::{Error, Limits, Protocol, Session};

pub(crate) type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone)]
pub struct Client {
    pub(crate) inner: Arc<Inner>,
}

pub(crate) struct Inner {
    base: Url,
    authorization: Option<reqwest::header::HeaderValue>,
    pub(crate) http: reqwest::Client,
    pub(crate) model: String,
    pub(crate) limits: Limits,
    pub(crate) shutdown: CancellationToken,
    pub(crate) tasks: TaskTracker,
}

impl Client {
    /// `base` may contain a reverse-proxy path prefix. Credentials are carried
    /// in an authorization header, never in URLs or diagnostics.
    pub fn new(
        base: Url,
        model: String,
        limits: Limits,
        api_key: Option<&str>,
    ) -> Result<Self, Error> {
        limits.validate()?;
        if !matches!(base.scheme(), "http" | "https")
            || base.host().is_none()
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
            || model.is_empty()
        {
            return Err(Error::Invalid("expected an HTTP base URL without credentials, query, or fragment, and a nonempty model".into()));
        }
        let authorization = api_key
            .map(|key| {
                let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                    .map_err(|_| Error::Invalid("invalid upstream API key".into()))?;
                value.set_sensitive(true);
                Ok::<_, Error>(value)
            })
            .transpose()?;
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(value) = &authorization {
            headers.insert(reqwest::header::AUTHORIZATION, value.clone());
        }
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(limits.connect_timeout)
            .build()?;
        Ok(Self {
            inner: Arc::new(Inner {
                base,
                authorization,
                http,
                model,
                limits,
                shutdown: CancellationToken::new(),
                tasks: TaskTracker::new(),
            }),
        })
    }

    pub(crate) fn url(&self, path: &str) -> Url {
        let mut url = self.inner.base.clone();
        url.set_path(&format!(
            "{}/{}",
            url.path().trim_end_matches('/'),
            path.trim_start_matches('/')
        ));
        url
    }

    /// Check both engine health and model registration before serving traffic.
    pub async fn health(&self) -> Result<(), Error> {
        tokio::time::timeout(self.inner.limits.connect_timeout, async {
            self.inner
                .http
                .get(self.url("health"))
                .send()
                .await?
                .error_for_status()?;
            let response = self
                .inner
                .http
                .get(self.url("v1/models"))
                .send()
                .await?
                .error_for_status()?;
            let models: Value = response.json().await?;
            let present = models["data"].as_array().is_some_and(|models| {
                models
                    .iter()
                    .any(|entry| entry["id"].as_str() == Some(&self.inner.model))
            });
            if !present {
                return Err(Error::Protocol(
                    "configured model is absent from upstream /v1/models".into(),
                ));
            }
            Ok(())
        })
        .await
        .map_err(|_| Error::Timeout)?
    }

    /// Handshake probe for realtime deployments. This does not synthesize an
    /// inference result; it checks upstream health, model identity, and WS routing.
    pub async fn realtime_health(&self, protocol: Protocol) -> Result<(), Error> {
        self.health().await?;
        if protocol == Protocol::OmniDuplex {
            return self.duplex_health().await;
        }
        let mut socket = self.connect().await?;
        let event = tokio::time::timeout(self.inner.limits.connect_timeout, receive(&mut socket))
            .await
            .map_err(|_| Error::Timeout)??;
        let result = async {
            if event["type"] != "session.created"
                || event.get("session").is_some() != (protocol == Protocol::Omni)
            {
                return Err(Error::Protocol(
                    "upstream realtime wire contract does not match the configured protocol".into(),
                ));
            }
            if protocol == Protocol::Omni {
                self.update_omni_session(
                    &mut socket,
                    serde_json::json!({
                        "type": "realtime", "model": self.inner.model
                    }),
                )
                .await?;
            }
            Ok(())
        }
        .await;
        self.close(&mut socket).await;
        result
    }

    pub(crate) async fn update_omni_session(
        &self,
        socket: &mut Socket,
        session: Value,
    ) -> Result<(), Error> {
        tokio::time::timeout(self.inner.limits.connect_timeout, async {
            send(
                socket,
                serde_json::json!({"type": "session.update", "session": session}),
            )
            .await?;
            let mut updated = receive(socket).await?;
            // vLLM-Omni sends one conversation greeting after session.created.
            // It may already be queued when we send session.update. Only that
            // greeting may precede the acknowledgement, within one deadline.
            if updated["type"] == "conversation.created" {
                updated = receive(socket).await?;
            }
            if updated["type"] != "session.updated" {
                return Err(Error::Protocol("upstream rejected the Omni session".into()));
            }
            Ok(())
        })
        .await
        .map_err(|_| Error::Timeout)?
    }

    pub fn session(&self, protocol: Protocol, requests: BoxStream<'static, Value>) -> Session {
        self.session_with_cancellation(protocol, requests, std::future::pending())
    }

    /// Interrupt background turn work independently of response-stream polling.
    /// Input EOF still drains accepted turns; cancellation closes them.
    pub fn session_with_cancellation(
        &self,
        protocol: Protocol,
        requests: BoxStream<'static, Value>,
        cancellation: impl Future<Output = ()> + Send + 'static,
    ) -> Session {
        Session::start(self.clone(), protocol, requests, cancellation)
    }

    pub async fn shutdown(&self) {
        self.inner.shutdown.cancel();
        self.inner.tasks.close();
        self.inner.tasks.wait().await;
    }

    pub(crate) async fn connect(&self) -> Result<Socket, Error> {
        self.connect_realtime(false).await
    }

    pub(crate) async fn connect_duplex(&self) -> Result<Socket, Error> {
        self.connect_realtime(true).await
    }

    async fn connect_realtime(&self, duplex: bool) -> Result<Socket, Error> {
        let mut url = self.url("v1/realtime");
        let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
        url.set_scheme(scheme)
            .map_err(|_| Error::Invalid("invalid WebSocket scheme".into()))?;
        url.query_pairs_mut()
            .append_pair("model", &self.inner.model);
        if duplex {
            url.query_pairs_mut()
                .append_pair("duplex", "1")
                .append_pair("autostart", "0");
        }
        let mut request = url.as_str().into_client_request()?;
        if let Some(value) = &self.inner.authorization {
            request
                .headers_mut()
                .insert(reqwest::header::AUTHORIZATION, value.clone());
        }
        let config = WebSocketConfig::default()
            .max_message_size(Some(self.inner.limits.max_buffered_output_bytes.get()))
            .max_frame_size(Some(self.inner.limits.max_buffered_output_bytes.get()));
        let (socket, _) = tokio::time::timeout(
            self.inner.limits.connect_timeout,
            tokio_tungstenite::connect_async_with_config(request, Some(config), false),
        )
        .await
        .map_err(|_| Error::Timeout)??;
        Ok(socket)
    }

    async fn duplex_health(&self) -> Result<(), Error> {
        let mut socket = self.connect_duplex().await?;
        let result = tokio::time::timeout(self.inner.limits.connect_timeout, async {
            send(
                &mut socket,
                serde_json::json!({
                    "type": "session.update",
                    "session": {"model": self.inner.model, "modalities": ["text"]}
                }),
            )
            .await?;
            let event = receive(&mut socket).await?;
            if event["type"] != "session.created"
                || event["session"]["capabilities"]["supports_independent_io_streams"] != true
            {
                return Err(Error::Protocol(
                    "upstream does not advertise native duplex sessions".into(),
                ));
            }
            send(&mut socket, serde_json::json!({"type": "session.close"})).await?;
            loop {
                let event = receive(&mut socket).await?;
                match event["type"].as_str() {
                    Some("session.closed") => return Ok(()),
                    Some("error") => {
                        return Err(Error::Protocol(
                            "upstream rejected duplex health probe".into(),
                        ));
                    }
                    _ => {}
                }
            }
        })
        .await
        .map_err(|_| Error::Timeout)
        .and_then(|result| result);
        if result.is_err() {
            // A failed probe must not leave a resumable session consuming a slot.
            let _ = tokio::time::timeout(
                self.inner.limits.close_timeout,
                send(&mut socket, serde_json::json!({"type": "session.close"})),
            )
            .await;
        }
        self.close(&mut socket).await;
        result
    }

    pub(crate) async fn close(&self, socket: &mut Socket) {
        let _ = tokio::time::timeout(self.inner.limits.close_timeout, socket.close(None)).await;
    }
}

pub(crate) async fn send(socket: &mut Socket, event: Value) -> Result<(), Error> {
    socket
        .send(Message::Text(serde_json::to_string(&event)?.into()))
        .await?;
    Ok(())
}

pub(crate) async fn receive(socket: &mut Socket) -> Result<Value, Error> {
    loop {
        match socket.next().await {
            Some(Ok(Message::Text(text))) => return Ok(serde_json::from_str(&text)?),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Close(_))) | None => return Err(Error::Closed),
            Some(Ok(_)) => return Err(Error::Protocol("expected a JSON text frame".into())),
            Some(Err(error)) => return Err(error.into()),
        }
    }
}
