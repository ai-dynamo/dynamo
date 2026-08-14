// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use futures::{SinkExt, StreamExt};
use prometheus::IntCounter;
use rustls::pki_types::ServerName;
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tokio_util::codec::{FramedRead, FramedWrite};

use super::{CallHomeHandshake, TcpStreamConnectionInfo};
use crate::{
    engine::AsyncEngineContext,
    pipeline::network::{
        ConnectionInfo, ControlMessage, StreamReceiver, StreamType, TwoPartCodec,
        codec::{TwoPartMessage, TwoPartMessageType},
    },
};

pub struct TcpClient;

type BoxRead = Box<dyn tokio::io::AsyncRead + Unpin + Send>;
type BoxWrite = Box<dyn tokio::io::AsyncWrite + Unpin + Send>;

impl TcpClient {
    async fn connect(address: &str) -> std::io::Result<TcpStream> {
        let backoff = std::time::Duration::from_millis(200);
        loop {
            match TcpStream::connect(address).await {
                Ok(socket) => {
                    socket.set_nodelay(true)?;
                    return Ok(socket);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AddrNotAvailable => {
                    tracing::warn!(%error, "TCP request callback connect retry");
                    tokio::time::sleep(backoff).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn connect_and_split(address: &str) -> Result<(BoxRead, BoxWrite)> {
        let stream = Self::connect(address).await?;
        if let Some(connector) = get_tls_connector()? {
            let server_name = tls_server_name(address)?;
            let tls_stream = tokio::time::timeout(
                crate::tls_utils::handshake_timeout(),
                connector.connect(server_name, stream),
            )
            .await
            .with_context(|| format!("TLS handshake timed out connecting to {address}"))?
            .with_context(|| format!("TLS handshake failed connecting to {address}"))?;
            let (read, write) = tokio::io::split(tls_stream);
            Ok((Box::new(read), Box::new(write)))
        } else {
            let (read, write) = tokio::io::split(stream);
            Ok((Box::new(read), Box::new(write)))
        }
    }

    pub async fn create_request_stream(
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
        cancellation_counter: Option<IntCounter>,
    ) -> Result<StreamReceiver> {
        let info = TcpStreamConnectionInfo::try_from(info)
            .context("tcp request callback connection info")?;
        if info.stream_type != StreamType::Request {
            return Err(anyhow!("TCP callback connection is not a request stream"));
        }
        if info.context != context.id() {
            return Err(anyhow!(
                "TCP callback context mismatch: expected {}, got {}",
                context.id(),
                info.context
            ));
        }

        let (read_half, write_half) = Self::connect_and_split(&info.address).await?;
        let reader = FramedRead::new(read_half, TwoPartCodec::default());
        let mut writer = FramedWrite::new(write_half, TwoPartCodec::default());
        let handshake = CallHomeHandshake {
            subject: info.subject,
            stream_type: StreamType::Request,
        };
        writer
            .send(TwoPartMessage::from_header(
                serde_json::to_vec(&handshake)?.into(),
            ))
            .await?;
        drop(writer);

        let (bytes_tx, bytes_rx) =
            tokio::sync::mpsc::channel(crate::pipeline::network::DEFAULT_SEND_BUFFER_COUNT);
        tokio::spawn(read_request_stream(
            reader,
            bytes_tx,
            context,
            cancellation_counter,
        ));
        Ok(StreamReceiver { rx: bytes_rx })
    }
}

static TCP_TLS_CONNECTOR: once_cell::sync::OnceCell<Option<TlsConnector>> =
    once_cell::sync::OnceCell::new();

fn get_tls_connector() -> Result<&'static Option<TlsConnector>> {
    TCP_TLS_CONNECTOR.get_or_try_init(build_tls_connector_from_env)
}

fn build_tls_connector_from_env() -> Result<Option<TlsConnector>> {
    use crate::config::environment_names::tcp_response_stream::tls as env;
    let ca_cert_path = std::env::var(env::DYN_TCP_TLS_CA_CERT_PATH).ok();
    let insecure = crate::config::env_is_truthy(env::DYN_TCP_TLS_INSECURE);

    if ca_cert_path.is_none() && !insecure {
        if std::env::var(env::DYN_TCP_TLS_CERT_PATH).is_ok() {
            tracing::warn!(
                "TCP client is running in plaintext mode but {} is set. Set {} (or {} for dev) to enable client-side TLS.",
                env::DYN_TCP_TLS_CERT_PATH,
                env::DYN_TCP_TLS_CA_CERT_PATH,
                env::DYN_TCP_TLS_INSECURE,
            );
        }
        return Ok(None);
    }
    if !insecure && ca_cert_path.is_none() {
        anyhow::bail!(
            "TCP TLS is enabled but {} is not set and {} is not true; provide a CA cert or set insecure mode for development",
            env::DYN_TCP_TLS_CA_CERT_PATH,
            env::DYN_TCP_TLS_INSECURE,
        );
    }

    let tls_config = crate::tls_utils::client_tls_config(
        ca_cert_path.as_deref().map(std::path::Path::new),
        insecure,
    )?;
    Ok(Some(TlsConnector::from(Arc::new(tls_config))))
}

fn tls_server_name(address: &str) -> Result<ServerName<'static>> {
    use crate::config::environment_names::tcp_response_stream::tls as env;
    let name = std::env::var(env::DYN_TCP_TLS_SERVER_NAME).unwrap_or_else(|_| {
        address
            .parse::<std::net::SocketAddr>()
            .map(|socket| socket.ip().to_string())
            .unwrap_or_else(|_| {
                address.rfind(':').map_or_else(
                    || address.to_owned(),
                    |position| address[..position].to_owned(),
                )
            })
    });
    ServerName::try_from(name).map_err(|error| anyhow!("invalid TLS server name: {error}"))
}

async fn read_request_stream(
    mut reader: FramedRead<BoxRead, TwoPartCodec>,
    bytes_tx: tokio::sync::mpsc::Sender<bytes::Bytes>,
    context: Arc<dyn AsyncEngineContext>,
    cancellation_counter: Option<IntCounter>,
) {
    let mut cancelled = false;
    loop {
        tokio::select! {
            biased;
            _ = context.killed() => break,
            _ = context.stopped() => break,
            _ = bytes_tx.closed() => break,
            message = reader.next() => match message {
                Some(Ok(message)) => match message.into_message_type() {
                    TwoPartMessageType::DataOnly(data) => {
                        if bytes_tx.send(data).await.is_err() { break; }
                    }
                    TwoPartMessageType::HeaderOnly(header) => {
                        match serde_json::from_slice::<ControlMessage>(&header) {
                            Ok(ControlMessage::Sentinel) => break,
                            Ok(ControlMessage::Stop) => {
                                cancelled = true;
                                context.stop();
                                break;
                            }
                            Ok(ControlMessage::Kill) => {
                                cancelled = true;
                                context.kill();
                                break;
                            }
                            Err(error) => {
                                tracing::warn!(%error, "invalid TCP request callback control");
                                cancelled = true;
                                context.kill();
                                break;
                            }
                        }
                    }
                    _ => {
                        cancelled = true;
                        context.kill();
                        break;
                    }
                },
                Some(Err(error)) => {
                    tracing::warn!(%error, "TCP request callback read failed");
                    cancelled = true;
                    context.kill();
                    break;
                }
                None => {
                    cancelled = true;
                    context.kill();
                    break;
                }
            }
        }
    }
    if cancelled && let Some(counter) = cancellation_counter {
        counter.inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::AsyncEngineContextProvider, pipeline::Context};

    fn make_ca_file() -> tempfile::NamedTempFile {
        use std::io::Write;
        let key_pair = rcgen::KeyPair::generate().unwrap();
        let cert = rcgen::CertificateParams::new(vec!["localhost".to_string()])
            .unwrap()
            .self_signed(&key_pair)
            .unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(cert.pem().as_bytes()).unwrap();
        file
    }

    #[test]
    fn tcp_tls_connector_supports_plaintext_insecure_and_ca_modes() {
        temp_env::with_vars_unset(["DYN_TCP_TLS_CA_CERT_PATH", "DYN_TCP_TLS_INSECURE"], || {
            assert!(build_tls_connector_from_env().unwrap().is_none());
        });
        temp_env::with_vars(
            [
                ("DYN_TCP_TLS_INSECURE", Some("true")),
                ("DYN_TCP_TLS_CA_CERT_PATH", None),
            ],
            || assert!(build_tls_connector_from_env().unwrap().is_some()),
        );
        let ca = make_ca_file();
        temp_env::with_vars(
            [
                (
                    "DYN_TCP_TLS_CA_CERT_PATH",
                    Some(ca.path().to_str().unwrap()),
                ),
                ("DYN_TCP_TLS_INSECURE", None),
            ],
            || assert!(build_tls_connector_from_env().unwrap().is_some()),
        );
    }

    #[tokio::test]
    async fn rejects_wrong_stream_type_and_context_before_connecting() {
        let context = Context::new(());
        let response_info = TcpStreamConnectionInfo {
            address: "127.0.0.1:1".to_string(),
            subject: "subject".to_string(),
            context: context.id().to_string(),
            stream_type: StreamType::Response,
        }
        .into();
        let error =
            match TcpClient::create_request_stream(context.context(), response_info, None).await {
                Err(error) => error,
                Ok(_) => panic!("response stream type unexpectedly opened a request stream"),
            };
        assert!(error.to_string().contains("not a request stream"));

        let wrong_context = TcpStreamConnectionInfo {
            address: "127.0.0.1:1".to_string(),
            subject: "subject".to_string(),
            context: "different".to_string(),
            stream_type: StreamType::Request,
        }
        .into();
        let error =
            match TcpClient::create_request_stream(context.context(), wrong_context, None).await {
                Err(error) => error,
                Ok(_) => panic!("wrong context unexpectedly opened a request stream"),
            };
        assert!(error.to_string().contains("context mismatch"));
    }
}
