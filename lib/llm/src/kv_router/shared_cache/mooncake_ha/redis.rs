// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis-backed Mooncake master leader discovery.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use url::Url;

use super::{MooncakeLeaderResolver, MooncakeLeaderUnavailable};

const MAX_REDIS_LINE_BYTES: usize = 4096;

struct RedisMooncakeLeaderResolver {
    host: String,
    port: u16,
    master_view_key: String,
    db_index: u8,
    username: Option<String>,
    password: Option<String>,
    connection: Mutex<Option<BufReader<TcpStream>>>,
}

#[derive(Debug, PartialEq, Eq)]
enum RedisResponse {
    Simple(String),
    Bulk(Option<Vec<u8>>),
    Integer(i64),
}

pub(super) fn parse_mooncake_redis_endpoint(address: &str) -> anyhow::Result<(String, u16)> {
    let url = Url::parse(address)?;
    anyhow::ensure!(
        url.scheme() == "redis",
        "Mooncake Redis locator must use redis://"
    );
    anyhow::ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && matches!(url.path(), "" | "/"),
        "Mooncake Redis locator only supports redis://<host>[:<port>]/"
    );
    let host = url
        .host_str()
        .filter(|host| !host.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Mooncake Redis locator has no host"))?;
    Ok((host.to_string(), url.port().unwrap_or(6379)))
}

pub(super) fn sanitize_redis_hash_tag(cluster_id: &str) -> String {
    cluster_id
        .strip_suffix('/')
        .unwrap_or(cluster_id)
        .replace(['{', '}'], "_")
}

fn mooncake_redis_master_view_key(cluster_id: &str) -> String {
    format!("mooncake-store/{{{cluster_id}}}/master_view")
}

pub(super) fn build_redis_leader_resolver(
    host: &str,
    port: u16,
    cluster_id: &str,
    db_index: u8,
) -> Arc<dyn MooncakeLeaderResolver> {
    Arc::new(RedisMooncakeLeaderResolver {
        host: host.to_string(),
        port,
        master_view_key: mooncake_redis_master_view_key(cluster_id),
        db_index,
        username: std::env::var("MC_REDIS_USERNAME")
            .ok()
            .filter(|value| !value.is_empty()),
        password: std::env::var("MC_REDIS_PASSWORD")
            .ok()
            .filter(|value| !value.is_empty()),
        connection: Mutex::new(None),
    })
}

impl RedisMooncakeLeaderResolver {
    async fn connect(&self) -> anyhow::Result<BufReader<TcpStream>> {
        let stream = TcpStream::connect((self.host.as_str(), self.port))
            .await
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to connect to Mooncake Redis HA backend {}:{}: {error}",
                    self.host,
                    self.port
                )
            })?;
        let mut connection = BufReader::new(stream);

        if let Some(password) = self.password.as_deref() {
            let response = if let Some(username) = self.username.as_deref() {
                redis_request(&mut connection, &["AUTH", username, password]).await?
            } else {
                redis_request(&mut connection, &["AUTH", password]).await?
            };
            anyhow::ensure!(
                matches!(&response, RedisResponse::Simple(value) if value == "OK"),
                "Mooncake Redis AUTH returned unexpected response: {response:?}"
            );
        }

        if self.db_index != 0 {
            let db_index = self.db_index.to_string();
            match redis_request(&mut connection, &["SELECT", db_index.as_str()]).await? {
                RedisResponse::Simple(response) if response == "OK" => {}
                response => anyhow::bail!(
                    "Mooncake Redis SELECT {} returned unexpected response: {response:?}",
                    self.db_index
                ),
            }
        }

        Ok(connection)
    }

    async fn query_current_leader(
        &self,
        connection: &mut BufReader<TcpStream>,
    ) -> anyhow::Result<Option<String>> {
        match redis_request(
            connection,
            &["HGET", self.master_view_key.as_str(), "leader_address"],
        )
        .await?
        {
            RedisResponse::Bulk(Some(value)) => {
                let leader_address = std::str::from_utf8(&value)?.trim();
                Ok((!leader_address.is_empty()).then(|| leader_address.to_string()))
            }
            RedisResponse::Bulk(None) => Ok(None),
            response => anyhow::bail!(
                "Mooncake Redis HGET {} leader_address returned unexpected response: {response:?}",
                self.master_view_key
            ),
        }
    }
}

#[async_trait]
impl MooncakeLeaderResolver for RedisMooncakeLeaderResolver {
    async fn current_leader(&self) -> anyhow::Result<String> {
        let mut connection_slot = self.connection.lock().await;
        let reused_connection = connection_slot.is_some();
        let mut connection = match connection_slot.take() {
            Some(connection) => connection,
            None => self.connect().await?,
        };

        // Keep the socket out of the shared slot while a request is in flight. If the
        // outer leader-resolution timeout cancels this future, the possibly desynchronized
        // connection is dropped instead of being reused by the next refresh.
        let mut result = self.query_current_leader(&mut connection).await;
        if result.is_err() && reused_connection {
            connection = self.connect().await?;
            result = self.query_current_leader(&mut connection).await;
        }

        let leader = result?;
        *connection_slot = Some(connection);
        leader.ok_or_else(|| {
            MooncakeLeaderUnavailable {
                message: format!(
                    "Mooncake Redis master view {} has no leader address",
                    self.master_view_key
                ),
            }
            .into()
        })
    }
}

async fn redis_request(
    connection: &mut BufReader<TcpStream>,
    args: &[&str],
) -> anyhow::Result<RedisResponse> {
    let mut command = format!("*{}\r\n", args.len()).into_bytes();
    for arg in args {
        command.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        command.extend_from_slice(arg.as_bytes());
        command.extend_from_slice(b"\r\n");
    }
    connection.get_mut().write_all(&command).await?;
    connection.get_mut().flush().await?;
    redis_read_response(connection).await
}

async fn redis_read_response(
    connection: &mut BufReader<TcpStream>,
) -> anyhow::Result<RedisResponse> {
    let mut prefix = [0_u8; 1];
    connection.read_exact(&mut prefix).await?;
    match prefix[0] {
        b'+' => Ok(RedisResponse::Simple(redis_read_line(connection).await?)),
        b':' => Ok(RedisResponse::Integer(
            redis_read_line(connection).await?.parse()?,
        )),
        b'$' => {
            let length: i64 = redis_read_line(connection).await?.parse()?;
            if length == -1 {
                return Ok(RedisResponse::Bulk(None));
            }
            anyhow::ensure!(length >= 0, "Redis returned invalid bulk length {length}");
            let length = usize::try_from(length)?;
            anyhow::ensure!(length <= 64 * 1024, "Redis bulk response is too large");
            let mut value = vec![0_u8; length];
            connection.read_exact(&mut value).await?;
            let mut crlf = [0_u8; 2];
            connection.read_exact(&mut crlf).await?;
            anyhow::ensure!(crlf == *b"\r\n", "Redis bulk response is missing CRLF");
            Ok(RedisResponse::Bulk(Some(value)))
        }
        b'-' => anyhow::bail!(
            "Redis returned error: {}",
            redis_read_line(connection).await?
        ),
        prefix => anyhow::bail!("Redis returned unsupported response prefix 0x{prefix:02x}"),
    }
}

async fn redis_read_line(connection: &mut BufReader<TcpStream>) -> anyhow::Result<String> {
    let mut line = Vec::new();
    loop {
        let (bytes_to_copy, found_newline) = {
            let buffered = connection.fill_buf().await?;
            anyhow::ensure!(
                !buffered.is_empty(),
                "unexpected EOF in Redis response line"
            );
            let bytes_to_copy = buffered
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(buffered.len(), |index| index + 1);
            let next_len = line
                .len()
                .checked_add(bytes_to_copy)
                .ok_or_else(|| anyhow::anyhow!("Redis response line is too large"))?;
            anyhow::ensure!(
                next_len <= MAX_REDIS_LINE_BYTES,
                "Redis response line is too large"
            );
            line.extend_from_slice(&buffered[..bytes_to_copy]);
            (bytes_to_copy, buffered[bytes_to_copy - 1] == b'\n')
        };
        connection.consume(bytes_to_copy);
        if found_newline {
            break;
        }
    }
    anyhow::ensure!(
        line.len() >= 2 && line.ends_with(b"\r\n"),
        "invalid Redis line"
    );
    line.truncate(line.len() - 2);
    Ok(String::from_utf8(line)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    async fn read_mock_redis_request(connection: &mut BufReader<TcpStream>) -> Vec<String> {
        let mut prefix = [0_u8; 1];
        connection.read_exact(&mut prefix).await.unwrap();
        assert_eq!(prefix[0], b'*');
        let count: usize = redis_read_line(connection).await.unwrap().parse().unwrap();
        let mut args = Vec::with_capacity(count);
        for _ in 0..count {
            connection.read_exact(&mut prefix).await.unwrap();
            assert_eq!(prefix[0], b'$');
            let length: usize = redis_read_line(connection).await.unwrap().parse().unwrap();
            let mut value = vec![0_u8; length];
            connection.read_exact(&mut value).await.unwrap();
            let mut crlf = [0_u8; 2];
            connection.read_exact(&mut crlf).await.unwrap();
            assert_eq!(&crlf, b"\r\n");
            args.push(String::from_utf8(value).unwrap());
        }
        args
    }

    #[test]
    fn test_mooncake_redis_master_view_key_matches_mooncake() {
        assert_eq!(
            mooncake_redis_master_view_key(&sanitize_redis_hash_tag("cluster/{a}/")),
            "mooncake-store/{cluster/_a_}/master_view"
        );
    }

    #[tokio::test]
    async fn test_redis_leader_resolver_reads_selected_db() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = BufReader::new(socket);
            assert_eq!(
                read_mock_redis_request(&mut connection).await,
                vec!["AUTH", "dynamo", "secret"]
            );
            connection.get_mut().write_all(b"+OK\r\n").await.unwrap();
            connection.get_mut().flush().await.unwrap();

            assert_eq!(
                read_mock_redis_request(&mut connection).await,
                vec!["SELECT", "7"]
            );
            connection.get_mut().write_all(b"+OK\r\n").await.unwrap();
            connection.get_mut().flush().await.unwrap();

            for leader in ["10.0.0.8:50051", "10.0.0.9:50051"] {
                assert_eq!(
                    read_mock_redis_request(&mut connection).await,
                    vec![
                        "HGET",
                        "mooncake-store/{cluster-a}/master_view",
                        "leader_address",
                    ]
                );
                connection
                    .get_mut()
                    .write_all(format!("${}\r\n{leader}\r\n", leader.len()).as_bytes())
                    .await
                    .unwrap();
                connection.get_mut().flush().await.unwrap();
            }
        });
        let resolver = RedisMooncakeLeaderResolver {
            host: "127.0.0.1".to_string(),
            port,
            master_view_key: "mooncake-store/{cluster-a}/master_view".to_string(),
            db_index: 7,
            username: Some("dynamo".to_string()),
            password: Some("secret".to_string()),
            connection: Mutex::new(None),
        };

        assert_eq!(resolver.current_leader().await.unwrap(), "10.0.0.8:50051");
        assert_eq!(resolver.current_leader().await.unwrap(), "10.0.0.9:50051");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn test_redis_leader_resolver_reconnects_after_stale_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            for leader in ["10.0.0.8:50051", "10.0.0.9:50051"] {
                let (socket, _) = listener.accept().await.unwrap();
                let mut connection = BufReader::new(socket);
                assert_eq!(
                    read_mock_redis_request(&mut connection).await,
                    vec![
                        "HGET",
                        "mooncake-store/{cluster-a}/master_view",
                        "leader_address",
                    ]
                );
                connection
                    .get_mut()
                    .write_all(format!("${}\r\n{leader}\r\n", leader.len()).as_bytes())
                    .await
                    .unwrap();
                connection.get_mut().flush().await.unwrap();
            }
        });
        let resolver = RedisMooncakeLeaderResolver {
            host: "127.0.0.1".to_string(),
            port,
            master_view_key: "mooncake-store/{cluster-a}/master_view".to_string(),
            db_index: 0,
            username: None,
            password: None,
            connection: Mutex::new(None),
        };

        assert_eq!(resolver.current_leader().await.unwrap(), "10.0.0.8:50051");
        assert_eq!(resolver.current_leader().await.unwrap(), "10.0.0.9:50051");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn test_redis_response_line_is_bounded_before_newline() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            socket
                .write_all(&vec![b'x'; MAX_REDIS_LINE_BYTES + 1])
                .await
                .unwrap();
            socket.flush().await.unwrap();
        });
        let mut connection = BufReader::new(TcpStream::connect(address).await.unwrap());

        let error = redis_read_line(&mut connection).await.unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Redis response line is too large")
        );
        server.await.unwrap();
    }
}
