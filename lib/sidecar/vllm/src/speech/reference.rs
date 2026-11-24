// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{DynamoError, ErrorType};
use dynamo_llm::preprocessor::media::MediaFetcher;
use dynamo_sidecar_common::{connection_timeout, invalid_argument};
use hyper_util::client::proxy::matcher::Matcher;
use reqwest::{Client, Response, StatusCode, redirect::Policy};
use url::Url;

pub(super) fn policy_from_env() -> MediaFetcher {
    let mut policy = MediaFetcher::from_env();
    // Match the Python speech handler; the private-address blocklist still applies.
    policy.allow_direct_ip = true;
    policy.allow_direct_port = true;
    policy
}

pub(super) fn fetch_error(error: reqwest::Error) -> DynamoError {
    if error.is_timeout() {
        connection_timeout("reference-audio fetch timed out")
    } else {
        invalid_argument("could not fetch reference audio")
    }
}

pub(super) struct ReferenceClient {
    direct: Client,
    proxied: Client,
    proxies: Matcher,
    trust_egress_proxy: bool,
}

impl ReferenceClient {
    pub(super) fn new(policy: &MediaFetcher) -> Result<Self, DynamoError> {
        // Snapshot the same system matcher as reqwest when constructing its client.
        let proxies = Matcher::from_system();
        let direct = policy
            .http_client_builder()
            .no_proxy()
            .redirect(Policy::none())
            .https_only(!policy.allow_private_ips)
            .build();
        let mut proxy_policy = policy.clone();
        // Only this client's connect-time resolver may reach a private proxy.
        // Every origin still passes the original policy in get().
        proxy_policy.allow_private_ips = true;
        let proxied = proxy_policy
            .http_client_builder()
            .redirect(Policy::none())
            .https_only(!policy.allow_private_ips)
            .build();
        let client_error = |_| {
            DynamoError::builder()
                .error_type(ErrorType::Internal)
                .message("could not initialize reference-audio HTTP client")
                .build()
        };
        Ok(Self {
            direct: direct.map_err(client_error)?,
            proxied: proxied.map_err(client_error)?,
            proxies,
            trust_egress_proxy: std::env::var("DYN_MM_TRUST_EGRESS_PROXY")
                .is_ok_and(|value| value.trim() == "1"),
        })
    }

    fn client_for(&self, url: &Url, policy: &MediaFetcher) -> Result<&Client, DynamoError> {
        let uri = url
            .as_str()
            .parse()
            .map_err(|_| invalid_argument("invalid reference-audio URL"))?;
        if self.proxies.intercept(&uri).is_none() {
            return Ok(&self.direct);
        }
        if !policy.allow_private_ips && !self.trust_egress_proxy {
            return Err(DynamoError::builder()
                .error_type(ErrorType::Internal)
                .message("reference audio would use an egress proxy outside the connect-time address check; set DYN_MM_TRUST_EGRESS_PROXY=1 to assert that the proxy enforces destination policy, or unset the proxy")
                .build());
        }
        Ok(&self.proxied)
    }

    pub(super) async fn get(
        &self,
        mut url: Url,
        policy: &MediaFetcher,
    ) -> Result<Response, DynamoError> {
        // resolve_reference's caller bounds the entire fetch, including all hops/body.
        let mut redirects_remaining = 3;
        loop {
            if !matches!(url.scheme(), "http" | "https") {
                return Err(invalid_argument("reference audio requires an HTTP(S) URL"));
            }
            if url.scheme() == "http" && !policy.allow_private_ips {
                return Err(invalid_argument(
                    "HTTP reference audio requires DYN_MM_ALLOW_INTERNAL=1",
                ));
            }
            policy
                .check_if_url_allowed_with_dns(&url)
                .await
                .map_err(|_| {
                    invalid_argument("reference-audio URL was rejected by media policy")
                })?;
            let response = self
                .client_for(&url, policy)?
                .get(url.clone())
                .send()
                .await
                .map_err(fetch_error)?;
            let location = response.headers().get(reqwest::header::LOCATION);
            if matches!(
                response.status(),
                StatusCode::MOVED_PERMANENTLY
                    | StatusCode::FOUND
                    | StatusCode::SEE_OTHER
                    | StatusCode::TEMPORARY_REDIRECT
                    | StatusCode::PERMANENT_REDIRECT
            ) && let Some(location) = location
            {
                if redirects_remaining == 0 {
                    return Err(invalid_argument("too many reference-audio redirects"));
                }
                redirects_remaining -= 1;
                let mut next = url
                    .join(
                        location
                            .to_str()
                            .map_err(|_| invalid_argument("invalid reference-audio redirect"))?,
                    )
                    .map_err(|_| invalid_argument("invalid reference-audio redirect"))?;
                if next.origin() == url.origin()
                    && next.username().is_empty()
                    && next.password().is_none()
                {
                    next.set_username(url.username())
                        .expect("HTTP(S) URLs have an authority");
                    next.set_password(url.password())
                        .expect("HTTP(S) URLs have an authority");
                }
                url = next;
            } else {
                return response.error_for_status().map_err(fetch_error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, extract::Path, routing::get};
    use dynamo_backend_common::{BackendError, RawEngine};
    use futures::{StreamExt, TryStreamExt};
    use serde_json::json;
    use std::{sync::Arc, time::Duration};
    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

    fn protected_policy() -> MediaFetcher {
        let mut policy = policy_from_env();
        policy.allow_private_ips = false;
        policy.allow_remote_urls = true;
        policy.timeout = Some(Duration::from_secs(3));
        policy
    }

    fn internal_policy(timeout: Duration) -> MediaFetcher {
        let mut policy = protected_policy();
        policy.allow_private_ips = true;
        policy.timeout = Some(timeout);
        policy
    }

    #[tokio::test]
    async fn reference_audio_follows_three_redirects_but_rejects_four() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let visited = Arc::new(std::sync::Mutex::new(Vec::new()));
            let seen = visited.clone();
            let app = Router::new().route(
                "/{remaining}",
                get(move |Path(remaining): Path<usize>| {
                    let seen = seen.clone();
                    async move {
                        seen.lock().unwrap().push(remaining);
                        if remaining == 0 {
                            axum::http::Response::builder()
                                .body(Body::from("audio"))
                                .unwrap()
                        } else {
                            axum::http::Response::builder()
                                .status(StatusCode::FOUND)
                                .header("location", format!("/{}", remaining - 1))
                                .body(Body::empty())
                                .unwrap()
                        }
                    }
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let policy = internal_policy(Duration::from_secs(2));
            let mut client = ReferenceClient::new(&policy).unwrap();
            client.proxies = Matcher::builder().build();
            let audio = super::super::resolve_reference(
                &format!("http://{address}/3"),
                &policy,
                &client,
                1024,
            )
            .await
            .unwrap();
            assert_eq!(audio, "data:application/octet-stream;base64,YXVkaW8=");
            assert_eq!(*visited.lock().unwrap(), [3, 2, 1, 0]);
            visited.lock().unwrap().clear();
            let error = super::super::resolve_reference(
                &format!("http://{address}/4"),
                &policy,
                &client,
                1024,
            )
            .await
            .unwrap_err();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
            assert!(
                error
                    .to_string()
                    .contains("too many reference-audio redirects")
            );
            assert_eq!(*visited.lock().unwrap(), [4, 3, 2, 1]);
            server.abort();
        })
        .await
        .expect("reference redirect test must finish");
    }

    async fn assert_timeout_classification(phase: &str) {
        tokio::time::timeout(Duration::from_secs(5), async {
            let app = Router::new()
                .route(
                    "/headers",
                    get(|| async { std::future::pending::<axum::http::Response<Body>>().await }),
                )
                .route(
                    "/body",
                    get(|| async {
                        Body::from_stream(
                            futures::stream::once(async { Ok::<_, std::io::Error>("audio") })
                                .chain(futures::stream::pending()),
                        )
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let (mut engine, upstream) = super::super::tests::engine().await;
            // Exercise both possible timer winners deterministically.
            for (http_timeout, outer_timeout) in [(20, 250), (250, 20)] {
                let policy = internal_policy(Duration::from_millis(http_timeout));
                let mut client = ReferenceClient::new(&policy).unwrap();
                client.proxies = Matcher::builder().build();
                engine.reference_client = Arc::new(client);
                engine.reference_policy = internal_policy(Duration::from_millis(outer_timeout));
                let error = engine
                    .generate(
                        json!({
                            "input": "hello", "ref_audio": format!("http://{address}/{phase}"),
                        }),
                        super::super::tests::context(),
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap_err();
                assert_eq!(
                    error.error_type(),
                    ErrorType::Backend(BackendError::ConnectionTimeout)
                );
                assert!(
                    error
                        .to_string()
                        .contains("reference-audio fetch timed out")
                );
            }
            engine.cleanup().await.unwrap();
            upstream.abort();
            server.abort();
        })
        .await
        .expect("reference timeout test must finish");
    }

    #[tokio::test]
    async fn reference_header_timeout_is_a_gateway_timeout() {
        assert_timeout_classification("headers").await;
    }

    #[tokio::test]
    async fn reference_body_timeout_is_a_gateway_timeout() {
        assert_timeout_classification("body").await;
    }

    #[test]
    fn proxy_selection_honors_no_proxy_and_explicit_trust() {
        let policy = protected_policy();
        let mut client = ReferenceClient::new(&policy).unwrap();
        client.proxies = Matcher::builder()
            .https("http://proxy.test:3128")
            .no(".bypass.test,9.9.9.9")
            .build();
        client.trust_egress_proxy = false;
        for url in [
            "https://8.8.8.8:8443/ref.wav",
            "https://public.test/ref.wav",
        ] {
            let error = client
                .client_for(&url.parse().unwrap(), &policy)
                .unwrap_err();
            assert_eq!(error.error_type(), ErrorType::Internal);
            assert!(error.to_string().contains("DYN_MM_TRUST_EGRESS_PROXY"));
        }
        for url in [
            "https://9.9.9.9/ref.wav",
            "https://media.bypass.test/ref.wav",
        ] {
            assert!(std::ptr::eq(
                client.client_for(&url.parse().unwrap(), &policy).unwrap(),
                &client.direct,
            ));
        }
        client.trust_egress_proxy = true;
        assert!(std::ptr::eq(
            client
                .client_for(&"https://8.8.8.8/ref.wav".parse().unwrap(), &policy)
                .unwrap(),
            &client.proxied,
        ));
        // A later NO_PROXY hop must not inherit the trusted proxy's resolver.
        assert!(std::ptr::eq(
            client
                .client_for(&"https://9.9.9.9/ref.wav".parse().unwrap(), &policy)
                .unwrap(),
            &client.direct,
        ));
        let mut internal_policy = policy.clone();
        internal_policy.allow_private_ips = true;
        client.trust_egress_proxy = false;
        assert!(
            client
                .client_for(
                    &"https://8.8.8.8/ref.wav".parse().unwrap(),
                    &internal_policy
                )
                .is_ok()
        );
    }

    #[test]
    fn public_reference_addresses_and_ports_keep_private_address_protection() {
        let policy = protected_policy();
        for url in [
            "https://8.8.8.8/ref.wav",
            "https://media.example:8443/ref.wav",
        ] {
            policy.check_if_url_allowed(&url.parse().unwrap()).unwrap();
        }
        for url in [
            "https://127.0.0.1/ref.wav",
            "https://169.254.169.254/ref.wav",
            "https://[::1]/ref.wav",
        ] {
            assert!(policy.check_if_url_allowed(&url.parse().unwrap()).is_err());
        }
    }

    async fn read_headers(stream: &mut (impl AsyncRead + Unpin)) -> String {
        let mut header = Vec::new();
        while !header.ends_with(b"\r\n\r\n") {
            assert!(header.len() < 8192);
            header.push(stream.read_u8().await.unwrap());
        }
        String::from_utf8(header).unwrap()
    }

    #[tokio::test]
    async fn trusted_private_proxy_cannot_redirect_to_private_origin() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let certificate = rcgen::generate_simple_self_signed(vec!["8.8.8.8".into()]).unwrap();
            let key = rustls::pki_types::PrivatePkcs8KeyDer::from(certificate.key_pair.serialize_der());
            let tls = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_safe_default_protocol_versions().unwrap()
                .with_no_client_auth()
                .with_single_cert(vec![certificate.cert.der().clone()], key.into()).unwrap();
            let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let proxy_url = format!("http://localhost:{}", listener.local_addr().unwrap().port());
            let proxy = tokio::spawn(async move {
                for (path, location) in [
                    ("/ref.wav", "https://8.8.8.8:8443/next.wav"),
                    ("/next.wav", "https://127.0.0.1/private.wav"),
                ] {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    assert!(read_headers(&mut socket).await.starts_with("CONNECT 8.8.8.8:8443 "));
                    socket.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n").await.unwrap();
                    let mut tls = acceptor.accept(socket).await.unwrap();
                    let headers = read_headers(&mut tls).await;
                    assert!(headers.starts_with(&format!("GET {path} ")));
                    assert!(headers.lines().any(|line| line.split_once(':').is_some_and(|(name, value)|
                        name.eq_ignore_ascii_case("authorization") && value.trim() == "Basic dXNlcjpwYXNz"
                    )));
                    tls.write_all(format!("HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
                    tls.shutdown().await.unwrap();
                }
            });
            let policy = protected_policy();
            let mut client = ReferenceClient::new(&policy).unwrap();
            client.proxies = Matcher::builder().https(&proxy_url).build();
            client.trust_egress_proxy = true;
            let mut proxy_policy = policy.clone();
            proxy_policy.allow_private_ips = true;
            client.proxied = proxy_policy.http_client_builder()
                .redirect(Policy::none())
                .https_only(true)
                .proxy(reqwest::Proxy::all(proxy_url).unwrap())
                .add_root_certificate(reqwest::Certificate::from_der(certificate.cert.der()).unwrap())
                .build().unwrap();
            let error = client.get("https://user:pass@8.8.8.8:8443/ref.wav".parse().unwrap(), &policy).await.unwrap_err();
            assert_eq!(error.error_type(), ErrorType::Backend(BackendError::InvalidArgument));
            assert!(error.to_string().contains("rejected by media policy"));
            proxy.await.unwrap();
        }).await.expect("local TLS proxy must finish within the test deadline");
    }
}
