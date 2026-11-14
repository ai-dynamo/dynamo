// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use dynamo_backend_common::RuntimeConfig;
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportArgs, HttpEndpoint};

const GRPC_ENDPOINT_ENV: &str = "DYN_SIDECAR_GRPC_ENDPOINT";
const HTTP_ENDPOINT_ENV: &str = "DYN_SIDECAR_HTTP_ENDPOINT";

/// Expand a bare port to `localhost:<port>`; any other value passes through unchanged.
fn with_default_host(raw: &str, argument: &str) -> Result<String, String> {
    let raw = raw.trim();
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Ok(raw.to_string());
    }
    match raw.parse::<u16>() {
        Ok(port) if port != 0 => Ok(format!("localhost:{port}")),
        _ => Err(format!(
            "`{argument}` port must be between 1 and 65535; got {raw}"
        )),
    }
}

fn parse_grpc_endpoint(raw: &str) -> Result<GrpcEndpoint, String> {
    let raw = with_default_host(raw, "--grpc")?;
    GrpcEndpoint::parse(&raw, "--grpc").map_err(|error| error.to_string())
}

fn parse_http_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    let raw = with_default_host(raw, "--http")?;
    parse_http_url(&raw, "--http")
}

/// Parse `host:port` or an HTTP(S) URL.
fn parse_http_url(raw: &str, argument: &str) -> Result<HttpEndpoint, String> {
    let raw = raw.trim();
    // Accept `host:port` like the gRPC flags do.
    let raw = if raw.is_empty() || raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    HttpEndpoint::parse(&raw, argument).map_err(|error| error.to_string())
}

/// Reject a bare port, which a URL parser would otherwise read as an IPv4 address (`8000` is `0.0.31.64`).
fn reject_bare_port(raw: &str, argument: &str, short: &str) -> Result<(), String> {
    let raw = raw.trim();
    if !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "`{argument}` needs a host, such as `localhost:{raw}`; use `{short} {raw}` for a local port"
        ));
    }
    Ok(())
}

/// `--grpc-endpoint` parser shared with the other Dynamo sidecars.
fn parse_compat_grpc_endpoint(raw: &str) -> Result<GrpcEndpoint, String> {
    reject_bare_port(raw, "--grpc-endpoint", "--grpc")?;
    GrpcEndpoint::parse(raw, "--grpc-endpoint").map_err(|error| error.to_string())
}

fn parse_compat_http_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    reject_bare_port(raw, "--http-endpoint", "--http")?;
    parse_http_url(raw, "--http-endpoint")
}

/// `raw` with its scheme, host, and port spelled out, such as `http://localhost:8000`, and its path when it has one.
///
/// A URL parser leaves out a scheme's default port and adds a `/` root path; this writes the port and drops that `/`.
/// A value that is not a host URL, such as a `unix:` socket, is returned unchanged.
fn complete_url(raw: &str) -> String {
    let Ok(url) = url::Url::parse(raw) else {
        return raw.to_string();
    };
    let (Some(host), Some(port)) = (url.host_str(), url.port_or_known_default()) else {
        return raw.to_string();
    };
    let path = url.path().trim_end_matches('/');
    format!("{}://{host}:{port}{path}", url.scheme())
}

/// The `host:port` both endpoints connect to, if they name the same one.
///
/// Compares spellings only: `localhost` and `127.0.0.1` are treated as different hosts.
fn shared_authority(grpc: &GrpcEndpoint, http: &HttpEndpoint) -> Option<String> {
    let authority = |raw: &str| {
        let url = url::Url::parse(raw).ok()?;
        Some((url.host_str()?.to_string(), url.port_or_known_default()?))
    };
    // A Unix-socket gRPC endpoint has no host, so it never matches.
    let (host, port) = authority(grpc.as_str())?;
    (authority(http.as_str())? == (host.clone(), port)).then(|| format!("{host}:{port}"))
}

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-triton-sidecar",
    about = "Dynamo Sidecar for NVIDIA Triton Inference Server hosted tensor, classify, and embeddings models."
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub runtime: RuntimeConfig,

    /// Dynamo namespace used for model discovery.
    #[arg(long, default_value = "dynamo", env = "DYN_NAMESPACE")]
    pub namespace: String,

    /// Dynamo component containing the per-model endpoints.
    #[arg(long, default_value = "triton", env = "DYN_COMPONENT")]
    pub component: String,

    /// Triton KServe v2 gRPC endpoint as a port, `host:port`, or a plaintext gRPC/HTTP URL. A bare port means `localhost:<port>`; `--grpc` with no value means `localhost:8001`.
    ///
    /// Enables the gRPC transport. Without `--grpc` or `--grpc-endpoint`, `DYN_SIDECAR_GRPC_ENDPOINT` is read in the same format.
    /// At least one of `--grpc` and `--http` is required; when both are set, inference requests alternate between the two protocols.
    #[arg(
        long = "grpc",
        value_name = "ENDPOINT",
        num_args = 0..=1,
        default_missing_value = "localhost:8001",
        value_parser = parse_grpc_endpoint,
        overrides_with = "compat_grpc_endpoint",
    )]
    pub grpc_endpoint: Option<GrpcEndpoint>,

    /// Same as `--grpc`, in the format the other Dynamo sidecars accept: `host:port` or a plaintext gRPC/HTTP URL, with no bare-port or no-value default.
    ///
    /// When both are passed, the last one wins.
    #[arg(
        long = "grpc-endpoint",
        value_name = "ENDPOINT",
        value_parser = parse_compat_grpc_endpoint,
        overrides_with = "grpc_endpoint",
    )]
    pub compat_grpc_endpoint: Option<GrpcEndpoint>,

    /// Triton KServe v2 HTTP/REST endpoint as a port, `host:port`, or an HTTP(S) URL. A bare port means `localhost:<port>`; `--http` with no value means `localhost:8000`.
    ///
    /// Enables the HTTP transport. Without `--http` or `--http-endpoint`, `DYN_SIDECAR_HTTP_ENDPOINT` is read in the same format.
    /// At least one of `--grpc` and `--http` is required; when both are set, inference requests alternate between the two protocols.
    #[arg(
        long = "http",
        value_name = "ENDPOINT",
        num_args = 0..=1,
        default_missing_value = "localhost:8000",
        value_parser = parse_http_endpoint,
        overrides_with = "compat_http_endpoint",
    )]
    pub http_endpoint: Option<HttpEndpoint>,

    /// Same as `--http`, as `host:port` or an HTTP(S) URL, with no bare-port or no-value default.
    ///
    /// When both are passed, the last one wins.
    #[arg(
        long = "http-endpoint",
        value_name = "ENDPOINT",
        value_parser = parse_compat_http_endpoint,
        overrides_with = "http_endpoint",
    )]
    pub compat_http_endpoint: Option<HttpEndpoint>,

    #[command(flatten)]
    pub grpc: GrpcTransportArgs,

    /// Triton model to expose. Repeat to expose multiple models.
    ///
    /// With no explicit model names, the sidecar registers every READY model returned by Triton's RepositoryIndex API.
    #[arg(
        long = "export",
        value_name = "MODEL",
        env = "DYN_TRITON_MODELS",
        value_delimiter = ','
    )]
    pub model_names: Vec<String>,

    /// Triton model to serve on the frontend's OpenAI `/v1/classify` endpoint instead of as a KServe tensor model. Repeat to serve multiple models.
    ///
    /// The model must declare exactly one TYPE_STRING input with dims [1] or [-1] and exactly one TYPE_FP32 output.
    #[arg(
        long = "classify",
        value_name = "MODEL",
        env = "DYN_TRITON_CLASSIFY_MODELS",
        value_delimiter = ','
    )]
    pub classify_models: Vec<String>,

    /// Triton model to serve on the frontend's OpenAI `/v1/embeddings` endpoint instead of as a KServe tensor model. Repeat to serve multiple models.
    ///
    /// The model must declare exactly one TYPE_STRING input with dims [1] or [-1] and exactly one TYPE_FP32 output.
    #[arg(
        long = "embeddings",
        value_name = "MODEL",
        env = "DYN_TRITON_EMBEDDINGS_MODELS",
        value_delimiter = ','
    )]
    pub embeddings_models: Vec<String>,

    /// Option or environment variable that set the gRPC endpoint, recorded by [`Args::validate`].
    #[arg(skip)]
    grpc_endpoint_source: Option<&'static str>,

    /// Option or environment variable that set the HTTP endpoint, recorded by [`Args::validate`].
    #[arg(skip)]
    http_endpoint_source: Option<&'static str>,
}

/// Endpoint the sidecar registers for a Triton model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Task {
    Tensor,
    Classify,
    Embeddings,
}

impl Task {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Task::Tensor => "tensor",
            Task::Classify => "classify",
            Task::Embeddings => "embeddings",
        }
    }
}

impl Args {
    /// Fold `--grpc-endpoint` and `--http-endpoint` into `--grpc` and `--http`, trim model names, and reject a model passed to both `--classify` and `--embeddings`.
    pub(crate) fn validate(&mut self) -> Result<(), String> {
        self.validate_with_env(|name| std::env::var(name).ok())
    }

    /// [`Args::validate`], reading environment variables through `env`.
    ///
    /// The endpoint variables are applied here rather than by clap, which reports a conflict between `--grpc` and `--grpc-endpoint` when either one also reads a variable.
    fn validate_with_env(&mut self, env: impl Fn(&str) -> Option<String>) -> Result<(), String> {
        let env = |name: &str| env(name).filter(|value| !value.trim().is_empty());
        // A compat flag is set only when it came last on the command line.
        self.grpc_endpoint_source = self.grpc_endpoint.as_ref().map(|_| "--grpc");
        if let Some(endpoint) = self.compat_grpc_endpoint.take() {
            self.grpc_endpoint = Some(endpoint);
            self.grpc_endpoint_source = Some("--grpc-endpoint");
        }
        self.http_endpoint_source = self.http_endpoint.as_ref().map(|_| "--http");
        if let Some(endpoint) = self.compat_http_endpoint.take() {
            self.http_endpoint = Some(endpoint);
            self.http_endpoint_source = Some("--http-endpoint");
        }
        // A flag on the command line beats its environment variable.
        if self.grpc_endpoint.is_none()
            && let Some(raw) = env(GRPC_ENDPOINT_ENV)
        {
            self.grpc_endpoint = Some(
                parse_grpc_endpoint(&raw)
                    .map_err(|error| format!("{GRPC_ENDPOINT_ENV}: {error}"))?,
            );
            self.grpc_endpoint_source = Some(GRPC_ENDPOINT_ENV);
        }
        if self.http_endpoint.is_none()
            && let Some(raw) = env(HTTP_ENDPOINT_ENV)
        {
            self.http_endpoint = Some(
                parse_http_endpoint(&raw)
                    .map_err(|error| format!("{HTTP_ENDPOINT_ENV}: {error}"))?,
            );
            self.http_endpoint_source = Some(HTTP_ENDPOINT_ENV);
        }
        if self.grpc_endpoint.is_none() && self.http_endpoint.is_none() {
            return Err(format!(
                "pass --grpc or --http, or set {GRPC_ENDPOINT_ENV} or {HTTP_ENDPOINT_ENV}"
            ));
        }
        if let (Some(grpc), Some(http)) = (&self.grpc_endpoint, &self.http_endpoint)
            && let Some(authority) = shared_authority(grpc, http)
        {
            return Err(format!(
                "--grpc and --http both name {authority}; Triton serves gRPC and HTTP on different ports (8001 and 8000 by default)"
            ));
        }
        for names in [
            &mut self.model_names,
            &mut self.classify_models,
            &mut self.embeddings_models,
        ] {
            *names = names
                .iter()
                .map(|name| name.trim())
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
        }
        if let Some(model) = self
            .classify_models
            .iter()
            .find(|model| self.embeddings_models.contains(model))
        {
            return Err(format!(
                "Triton model '{model}' is passed to both --classify and --embeddings; serve it through one"
            ));
        }
        Ok(())
    }

    /// The endpoint environment variables, set to every resolved endpoint in its complete form, so subprocesses see the endpoints the sidecar uses.
    pub(crate) fn endpoint_env(&self) -> Vec<(&'static str, String)> {
        let grpc = self
            .grpc_endpoint
            .as_ref()
            .map(|endpoint| (GRPC_ENDPOINT_ENV, complete_url(endpoint.as_str())));
        let http = self
            .http_endpoint
            .as_ref()
            .map(|endpoint| (HTTP_ENDPOINT_ENV, complete_url(endpoint.as_str())));
        grpc.into_iter().chain(http).collect()
    }

    /// Each endpoint and model name the sidecar was given, with the option or environment variable that set it.
    ///
    /// `matches` tells a model flag from its environment variable; call this after [`Args::validate`], which records the endpoint sources and trims the model names.
    pub(crate) fn parsed_values(&self, matches: &clap::ArgMatches) -> Vec<(&'static str, String)> {
        let endpoints = [
            (
                self.grpc_endpoint_source,
                self.grpc_endpoint
                    .as_ref()
                    .map(|endpoint| endpoint.to_string()),
            ),
            (
                self.http_endpoint_source,
                self.http_endpoint
                    .as_ref()
                    .map(|endpoint| endpoint.to_string()),
            ),
        ]
        .into_iter()
        .filter_map(|(source, endpoint)| Some((source?, endpoint?)));
        let models = [
            (
                "model_names",
                "--export",
                "DYN_TRITON_MODELS",
                &self.model_names,
            ),
            (
                "classify_models",
                "--classify",
                "DYN_TRITON_CLASSIFY_MODELS",
                &self.classify_models,
            ),
            (
                "embeddings_models",
                "--embeddings",
                "DYN_TRITON_EMBEDDINGS_MODELS",
                &self.embeddings_models,
            ),
        ]
        .into_iter()
        .flat_map(|(id, flag, env, names)| {
            let source = match matches.value_source(id) {
                Some(clap::parser::ValueSource::EnvVariable) => env,
                _ => flag,
            };
            names.iter().map(move |name| (source, name.clone()))
        });
        endpoints.chain(models).collect()
    }

    /// Endpoint `model` is served through.
    pub(crate) fn task(&self, model: &str) -> Task {
        if self.classify_models.iter().any(|name| name == model) {
            Task::Classify
        } else if self.embeddings_models.iter().any(|name| name == model) {
            Task::Embeddings
        } else {
            Task::Tensor
        }
    }

    /// Models passed to `--classify` or `--embeddings`.
    pub(crate) fn text_models(&self) -> impl Iterator<Item = &String> {
        self.classify_models.iter().chain(&self.embeddings_models)
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Args, Task};

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn parses_explicit_models() {
        let args = Args::try_parse_from([
            "dynamo-triton-sidecar",
            "--grpc",
            "localhost:8001",
            "--export",
            "a",
            "--export",
            "b",
        ])
        .unwrap();
        assert_eq!(args.model_names, ["a", "b"]);
        assert_eq!(
            args.grpc_endpoint.unwrap().as_str(),
            "http://localhost:8001"
        );
        assert!(args.http_endpoint.is_none());
    }

    #[test]
    fn parses_http_only() {
        let args =
            Args::try_parse_from(["dynamo-triton-sidecar", "--http", "localhost:8000"]).unwrap();
        assert!(args.grpc_endpoint.is_none());
        assert_eq!(
            args.http_endpoint.unwrap().as_str(),
            "http://localhost:8000/"
        );
    }

    #[test]
    fn parses_both_protocols() {
        let args = Args::try_parse_from([
            "dynamo-triton-sidecar",
            "--grpc",
            "localhost:8001",
            "--http",
            "https://triton.example.com/prefix",
        ])
        .unwrap();
        assert!(args.grpc_endpoint.is_some());
        assert_eq!(
            args.http_endpoint.unwrap().as_str(),
            "https://triton.example.com/prefix"
        );
    }

    #[test]
    fn requires_an_endpoint() {
        let mut args = Args::try_parse_from(["dynamo-triton-sidecar"]).unwrap();
        let error = args.validate_with_env(no_env).unwrap_err();
        assert!(error.contains("pass --grpc or --http"), "{error}");
        assert!(Args::try_parse_from(["dynamo-triton-sidecar", "--http", ""]).is_err());
        assert!(
            Args::try_parse_from(["dynamo-triton-sidecar", "--http", "grpc://localhost:8000"])
                .is_err()
        );
    }

    #[test]
    fn expands_ports_and_bare_flags_to_localhost() {
        let args =
            Args::try_parse_from(["dynamo-triton-sidecar", "--grpc", "9001", "--http"]).unwrap();
        assert_eq!(
            args.grpc_endpoint.unwrap().as_str(),
            "http://localhost:9001"
        );
        assert_eq!(
            args.http_endpoint.unwrap().as_str(),
            "http://localhost:8000/"
        );

        let args = Args::try_parse_from([
            "dynamo-triton-sidecar",
            "--grpc",
            "--classify",
            "classifier",
        ])
        .unwrap();
        assert_eq!(
            args.grpc_endpoint.unwrap().as_str(),
            "http://localhost:8001"
        );
        assert_eq!(args.classify_models, ["classifier"]);

        let args = Args::try_parse_from(["dynamo-triton-sidecar", "--http", "8080"]).unwrap();
        assert_eq!(
            args.http_endpoint.unwrap().as_str(),
            "http://localhost:8080/"
        );

        for port in ["0", "65536", "99999"] {
            assert!(
                Args::try_parse_from(["dynamo-triton-sidecar", "--grpc", port]).is_err(),
                "--grpc {port} accepted"
            );
        }
    }

    fn validated(argv: &[&str]) -> Args {
        let mut args = Args::try_parse_from(argv).unwrap();
        args.validate_with_env(no_env).unwrap();
        args
    }

    #[test]
    fn rejects_the_same_endpoint_for_both_protocols() {
        for argv in [
            &["dynamo-triton-sidecar", "--grpc", "8001", "--http", "8001"][..],
            &[
                "dynamo-triton-sidecar",
                "--grpc",
                "Triton:8000",
                "--http",
                "http://triton:8000/v2-prefix",
            ],
            &[
                "dynamo-triton-sidecar",
                "--grpc-endpoint",
                "grpc://triton",
                "--http-endpoint",
                "triton:80",
            ],
        ] {
            let mut args = Args::try_parse_from(argv).unwrap();
            let error = args
                .validate_with_env(no_env)
                .expect_err(&format!("{argv:?} accepted"));
            assert!(error.contains("--grpc and --http both name"), "{error}");
        }

        for argv in [
            &["dynamo-triton-sidecar", "--grpc", "--http"][..],
            &[
                "dynamo-triton-sidecar",
                "--grpc",
                "8001",
                "--http",
                "triton:8001",
            ],
            &[
                "dynamo-triton-sidecar",
                "--grpc",
                "unix:///run/triton.sock",
                "--http",
                "localhost:80",
            ],
        ] {
            validated(argv);
        }
    }

    #[test]
    fn command_line_flags_beat_environment_variables() {
        let env = |name: &str| match name {
            "DYN_SIDECAR_GRPC_ENDPOINT" => Some("triton:9001".to_string()),
            "DYN_SIDECAR_HTTP_ENDPOINT" => Some("9000".to_string()),
            _ => None,
        };
        let endpoints = |argv: &[&str]| {
            let mut args = Args::try_parse_from(argv).unwrap();
            args.validate_with_env(env).unwrap();
            (
                args.grpc_endpoint.unwrap().to_string(),
                args.http_endpoint.unwrap().to_string(),
            )
        };
        assert_eq!(
            endpoints(&["dynamo-triton-sidecar"]),
            ("http://triton:9001".into(), "http://localhost:9000/".into())
        );
        for argv in [
            &["dynamo-triton-sidecar", "--grpc"][..],
            &["dynamo-triton-sidecar", "--grpc-endpoint", "localhost:8001"],
            &[
                "dynamo-triton-sidecar",
                "--grpc-endpoint",
                "other:1",
                "--grpc",
                "8001",
            ],
        ] {
            assert_eq!(endpoints(argv).0, "http://localhost:8001", "{argv:?}");
        }
        assert_eq!(
            endpoints(&["dynamo-triton-sidecar", "--http-endpoint", "triton:8000"]).1,
            "http://triton:8000/"
        );

        let mut args = Args::try_parse_from(["dynamo-triton-sidecar"]).unwrap();
        let error = args
            .validate_with_env(|name| {
                (name == "DYN_SIDECAR_GRPC_ENDPOINT").then(|| "0".to_string())
            })
            .unwrap_err();
        assert!(error.starts_with("DYN_SIDECAR_GRPC_ENDPOINT:"), "{error}");
        let error = args
            .validate_with_env(|name| {
                (name == "DYN_SIDECAR_HTTP_ENDPOINT").then(|| " ".to_string())
            })
            .unwrap_err();
        assert!(error.contains("pass --grpc or --http"), "{error}");
    }

    #[test]
    fn writes_complete_endpoints_to_the_environment() {
        let env = |argv: &[&str]| validated(argv).endpoint_env();
        assert_eq!(
            env(&["dynamo-triton-sidecar", "--grpc", "--http"]),
            [
                (
                    "DYN_SIDECAR_GRPC_ENDPOINT",
                    "http://localhost:8001".to_string()
                ),
                (
                    "DYN_SIDECAR_HTTP_ENDPOINT",
                    "http://localhost:8000".to_string()
                ),
            ]
        );
        assert_eq!(
            env(&[
                "dynamo-triton-sidecar",
                "--grpc-endpoint",
                "grpc://Triton",
                "--http",
                "https://triton.example.com/prefix/",
            ]),
            [
                ("DYN_SIDECAR_GRPC_ENDPOINT", "http://triton:80".to_string()),
                (
                    "DYN_SIDECAR_HTTP_ENDPOINT",
                    "https://triton.example.com:443/prefix".to_string()
                ),
            ]
        );
        assert_eq!(
            env(&["dynamo-triton-sidecar", "--grpc", "unix:///run/triton.sock"]),
            [(
                "DYN_SIDECAR_GRPC_ENDPOINT",
                "unix:///run/triton.sock".to_string()
            )]
        );

        // Each written value parses back to the same endpoint.
        let args = validated(&[
            "dynamo-triton-sidecar",
            "--grpc",
            "9001",
            "--http",
            "triton:80/v2",
        ]);
        let written: std::collections::HashMap<_, _> = args.endpoint_env().into_iter().collect();
        let mut reparsed = Args::try_parse_from(["dynamo-triton-sidecar"]).unwrap();
        reparsed
            .validate_with_env(|name| written.get(name).cloned())
            .unwrap();
        assert_eq!(reparsed.grpc_endpoint, args.grpc_endpoint);
        assert_eq!(reparsed.http_endpoint, args.http_endpoint);
    }

    #[test]
    fn records_where_each_value_came_from() {
        use clap::{CommandFactory, FromArgMatches};

        let parsed = |argv: &[&str], env: &dyn Fn(&str) -> Option<String>| {
            let matches = Args::command().try_get_matches_from(argv).unwrap();
            let mut args = Args::from_arg_matches(&matches).unwrap();
            args.validate_with_env(env).unwrap();
            args.parsed_values(&matches)
        };
        assert_eq!(
            parsed(
                &[
                    "dynamo-triton-sidecar",
                    "--grpc-endpoint",
                    "triton:8001",
                    "--http",
                    "--export",
                    "a",
                    "--classify",
                    " b ,c",
                    "--embeddings",
                    "d",
                ],
                &no_env,
            ),
            [
                ("--grpc-endpoint", "http://triton:8001".to_string()),
                ("--http", "http://localhost:8000/".to_string()),
                ("--export", "a".to_string()),
                ("--classify", "b".to_string()),
                ("--classify", "c".to_string()),
                ("--embeddings", "d".to_string()),
            ]
        );
        assert_eq!(
            parsed(&["dynamo-triton-sidecar"], &|name| {
                (name == "DYN_SIDECAR_GRPC_ENDPOINT").then(|| "9001".to_string())
            }),
            [(
                "DYN_SIDECAR_GRPC_ENDPOINT",
                "http://localhost:9001".to_string()
            )]
        );
    }

    #[test]
    fn accepts_the_sidecar_endpoint_flags() {
        let args = validated(&[
            "dynamo-triton-sidecar",
            "--grpc-endpoint",
            "triton:8001",
            "--http-endpoint",
            "triton:8000",
        ]);
        assert_eq!(args.grpc_endpoint.unwrap().as_str(), "http://triton:8001");
        assert_eq!(args.http_endpoint.unwrap().as_str(), "http://triton:8000/");

        // The last of `--grpc` and `--grpc-endpoint` wins.
        let args = validated(&[
            "dynamo-triton-sidecar",
            "--grpc",
            "--grpc-endpoint",
            "triton:9001",
        ]);
        assert_eq!(args.grpc_endpoint.unwrap().as_str(), "http://triton:9001");
        let args = validated(&[
            "dynamo-triton-sidecar",
            "--grpc-endpoint",
            "triton:9001",
            "--grpc",
        ]);
        assert_eq!(
            args.grpc_endpoint.unwrap().as_str(),
            "http://localhost:8001"
        );

        // The sidecar flags require a value and do not expand a bare port.
        for argv in [
            &["dynamo-triton-sidecar", "--grpc-endpoint"][..],
            &["dynamo-triton-sidecar", "--http-endpoint"],
            &["dynamo-triton-sidecar", "--grpc-endpoint", "8001"],
            &["dynamo-triton-sidecar", "--http-endpoint", "8000"],
        ] {
            assert!(Args::try_parse_from(argv).is_err(), "{argv:?} accepted");
        }
    }

    #[test]
    fn assigns_tasks_per_model() {
        let mut args = Args::try_parse_from([
            "dynamo-triton-sidecar",
            "--grpc",
            "localhost:8001",
            "--classify",
            "classifier",
            "--embeddings",
            "embedder, other_embedder",
            "--embeddings",
            " embedder ",
        ])
        .unwrap();
        args.validate_with_env(no_env).unwrap();
        assert_eq!(args.classify_models, ["classifier"]);
        assert_eq!(args.embeddings_models, ["embedder", "other_embedder"]);
        assert_eq!(args.task("classifier"), Task::Classify);
        assert_eq!(args.task("other_embedder"), Task::Embeddings);
        assert_eq!(args.task("tensor_model"), Task::Tensor);
        assert_eq!(
            args.text_models().collect::<Vec<_>>(),
            ["classifier", "embedder", "other_embedder"]
        );
    }

    #[test]
    fn rejects_a_model_with_two_tasks() {
        let mut args = Args::try_parse_from([
            "dynamo-triton-sidecar",
            "--grpc",
            "localhost:8001",
            "--classify",
            "a,b",
            "--embeddings",
            "b",
        ])
        .unwrap();
        let error = args.validate_with_env(no_env).unwrap_err();
        assert!(error.contains("'b'"), "{error}");
        for removed in ["--task", "--classify-input-name"] {
            assert!(
                Args::try_parse_from([
                    "dynamo-triton-sidecar",
                    "--grpc",
                    "x:1",
                    removed,
                    "classify"
                ])
                .is_err(),
                "{removed} still accepted"
            );
        }
    }
}
