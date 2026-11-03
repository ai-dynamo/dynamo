// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A pool frontend that does no inference, for Global Frontend stress tests.
//!
//! `POST /v1/chat/completions` answers with `max_tokens` streamed
//! chat-completion chunks, in the format Dynamo's HTTP frontend sends: one
//! SSE event per token, `role` only in the first, `finish_reason: "length"`
//! on the last, a usage event when the request asks for one, then
//! `data: [DONE]`. The tokens go out at once, or one `--itl-ms` apart. The
//! request JSON is not parsed; the few fields it needs are found by substring
//! search.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, ready};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::{BodyExt, Either, Full};
use hyper::body::{Body, Frame, Incoming, SizeHint};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode, header};
use hyper_util::rt::TokioIo;
use memchr::memmem;
use tokio::net::TcpSocket;
use tokio::time::{Instant, Interval};

/// The text of every token.
const TOKEN: &str = " a";
/// Tokens sent when the request names no limit.
const DEFAULT_MAX_TOKENS: u64 = 16;
const USAGE: &str = "\
Usage: mock-pool [--host H] [--port P] [--threads N] [--model M]

  --host     address to listen on (default 0.0.0.0)
  --port     port to listen on (default 9000)
  --threads  tokio worker threads (default: CPUs this process may run on)
  --model    model id GET /v1/models lists (default Qwen/Qwen3-0.6B)
  --itl-ms   milliseconds between a response's tokens (default 0: all at once).
             Tokens come due on schedule, as an engine makes them, and go out
             back to back when the reader falls behind. Tokio timers have
             millisecond resolution.";

type ResponseBody = Either<Full<Bytes>, SseBody>;

fn main() {
    let mut host = "0.0.0.0".to_owned();
    let mut port = 9000u16;
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut model = "Qwen/Qwen3-0.6B".to_owned();
    let mut itl_ms = 0u64;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("{USAGE}");
            return;
        }
        let value = args
            .next()
            .unwrap_or_else(|| panic!("{flag} needs a value\n\n{USAGE}"));
        match flag.as_str() {
            "--host" => host = value,
            "--port" => port = value.parse().expect("--port"),
            "--threads" => threads = value.parse().expect("--threads"),
            "--model" => model = value,
            "--itl-ms" => itl_ms = value.parse().expect("--itl-ms"),
            _ => panic!("unknown flag {flag}\n\n{USAGE}"),
        }
    }
    let address: SocketAddr = format!("{host}:{port}").parse().expect("listen address");
    let models: &'static [u8] = format!(
        r#"{{"object":"list","data":[{{"id":"{model}","object":"model","created":0,"owned_by":"mock-pool"}}]}}"#
    )
    .into_bytes()
    .leak();
    let itl = (itl_ms > 0).then(|| Duration::from_millis(itl_ms));
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(serve(address, models, itl));
}

async fn serve(address: SocketAddr, models: &'static [u8], itl: Option<Duration>) {
    let socket = if address.is_ipv4() {
        TcpSocket::new_v4()
    } else {
        TcpSocket::new_v6()
    }
    .expect("socket");
    socket.set_reuseaddr(true).expect("SO_REUSEADDR");
    socket.bind(address).expect("bind");
    let listener = socket.listen(65_535).expect("listen");
    eprintln!("mock-pool listening on {address}");
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _)) => stream,
            Err(error) => {
                // Out of file descriptors, most likely: back off and retry.
                eprintln!("accept failed: {error}");
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        tokio::spawn(async move {
            let service = service_fn(move |request| handle(request, models, itl));
            let _ = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(TokioIo::new(stream), service)
                .await;
        });
    }
}

async fn handle(
    request: Request<Incoming>,
    models: &'static [u8],
    itl: Option<Duration>,
) -> Result<Response<ResponseBody>, Infallible> {
    let fixed = |status: StatusCode, content_type: &'static str, body: &'static [u8]| {
        Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .body(Either::Left(Full::new(Bytes::from_static(body))))
            .unwrap()
    };
    let response = match (request.method(), request.uri().path()) {
        (&Method::POST, "/v1/chat/completions") => match request.into_body().collect().await {
            Ok(body) => {
                let body = body.to_bytes();
                Response::builder()
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .header(header::CACHE_CONTROL, "no-cache")
                    .body(Either::Right(SseBody::new(
                        &body,
                        &next_id(),
                        unix_seconds(),
                        itl,
                    )))
                    .unwrap()
            }
            Err(_) => fixed(StatusCode::BAD_REQUEST, "text/plain", b"unreadable body"),
        },
        (&Method::GET, "/v1/models") => fixed(StatusCode::OK, "application/json", models),
        (&Method::GET, "/health" | "/live") => fixed(
            StatusCode::OK,
            "application/json",
            br#"{"status":"healthy"}"#,
        ),
        _ => fixed(StatusCode::NOT_FOUND, "text/plain", b"not found"),
    };
    Ok(response)
}

/// The SSE events of one streamed response, sent one frame each.
struct SseBody {
    /// The first event; carries `role` (and the finish reason when it is the
    /// only token).
    first: Option<Bytes>,
    /// Every token event between the first and the last.
    middle: Bytes,
    middle_left: u64,
    last: Option<Bytes>,
    usage: Option<Bytes>,
    done: bool,
    /// Ticks once per inter-token latency, starting one after the first token.
    /// A missed tick fires at once (tokio's default burst behaviour), so the
    /// tokens that came due while the reader was behind go out back to back.
    pace: Option<Interval>,
}

impl SseBody {
    fn new(request: &[u8], id: &str, created: u64, itl: Option<Duration>) -> Self {
        let tokens = max_tokens(request).max(1);
        let model = model(request).unwrap_or(b"mock");
        let model = std::str::from_utf8(model).unwrap_or("mock");
        let event = |delta: &str, finish_reason: &str| {
            Bytes::from(
                format!(
                    r#"data: {{"id":"chatcmpl-{id}","choices":[{{"index":0,"delta":{delta},"finish_reason":{finish_reason},"logprobs":null}}],"created":{created},"model":"{model}","object":"chat.completion.chunk"}}"#
                ) + "\n\n",
            )
        };
        let first_delta = format!(r#"{{"content":"{TOKEN}","role":"assistant"}}"#);
        let delta = format!(r#"{{"content":"{TOKEN}"}}"#);
        let (first, last) = if tokens == 1 {
            (event(&first_delta, r#""length""#), None)
        } else {
            (
                event(&first_delta, "null"),
                Some(event(&delta, r#""length""#)),
            )
        };
        let usage = include_usage(request).then(|| {
            // The mock does not tokenize; about four bytes of JSON per token.
            let prompt = request.len() as u64 / 4;
            Bytes::from(format!(
                r#"data: {{"id":"chatcmpl-{id}","choices":[],"created":{created},"model":"{model}","object":"chat.completion.chunk","usage":{{"prompt_tokens":{prompt},"completion_tokens":{tokens},"total_tokens":{}}}}}"#,
                prompt + tokens
            ) + "\n\n")
        });
        Self {
            first: Some(first),
            middle: event(&delta, "null"),
            middle_left: tokens.saturating_sub(2),
            last,
            usage,
            done: false,
            pace: itl.map(|itl| tokio::time::interval_at(Instant::now() + itl, itl)),
        }
    }

    /// Whether the next event is a token after the first, which waits for its tick.
    fn paced_next(&self) -> bool {
        self.first.is_none() && (self.middle_left > 0 || self.last.is_some())
    }

    fn next_event(&mut self) -> Option<Bytes> {
        if let Some(first) = self.first.take() {
            return Some(first);
        }
        if self.middle_left > 0 {
            self.middle_left -= 1;
            return Some(self.middle.clone());
        }
        if let Some(event) = self.last.take().or_else(|| self.usage.take()) {
            return Some(event);
        }
        if !self.done {
            self.done = true;
            return Some(Bytes::from_static(b"data: [DONE]\n\n"));
        }
        None
    }
}

impl Body for SseBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        let this = self.get_mut();
        if this.paced_next()
            && let Some(pace) = &mut this.pace
        {
            ready!(pace.poll_tick(cx));
        }
        Poll::Ready(this.next_event().map(|event| Ok(Frame::data(event))))
    }

    fn is_end_stream(&self) -> bool {
        self.done
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

/// The request's `max_completion_tokens`, else its `max_tokens`. JSON keys
/// inside strings are escaped, so a match is a real key; the last one wins,
/// since the top-level fields follow the messages.
fn max_tokens(request: &[u8]) -> u64 {
    [&b"\"max_completion_tokens\":"[..], &b"\"max_tokens\":"[..]]
        .iter()
        .find_map(|key| {
            let at = memmem::rfind(request, key)? + key.len();
            request[at..]
                .iter()
                .skip_while(|byte| byte.is_ascii_whitespace())
                .take_while(|byte| byte.is_ascii_digit())
                .fold(None, |value: Option<u64>, byte| {
                    Some(value.unwrap_or(0) * 10 + u64::from(byte - b'0'))
                })
        })
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

/// The request's `model`, as written (already JSON-escaped).
fn model(request: &[u8]) -> Option<&[u8]> {
    let key = b"\"model\":";
    let at = memmem::rfind(request, key)? + key.len();
    let rest = &request[at..];
    let start = rest.iter().position(|&byte| byte == b'"')? + 1;
    let end = start + rest[start..].iter().position(|&byte| byte == b'"')?;
    Some(&rest[start..end])
}

fn include_usage(request: &[u8]) -> bool {
    memmem::find(request, b"\"include_usage\":true").is_some()
        || memmem::find(request, b"\"include_usage\": true").is_some()
}

/// A random-looking UUID v4 string, unique within the process.
fn next_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seed = u64::from(std::process::id()) << 32 ^ unix_seconds();
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let high = splitmix64(seed ^ count);
    let low = splitmix64(high ^ count);
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        high >> 32,
        (high >> 16) & 0xffff,
        high & 0xfff,
        0x8000 | (low >> 48) & 0x3fff,
        low & 0xffff_ffff_ffff
    )
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(request: &str) -> Vec<String> {
        let mut body = SseBody::new(request.as_bytes(), "id", 7, None);
        std::iter::from_fn(|| body.next_event())
            .map(|event| String::from_utf8(event.to_vec()).unwrap())
            .collect()
    }

    #[test]
    fn finds_the_token_limit_model_and_usage_flag() {
        let request = r#"{"messages":[{"role":"user","content":"say \"max_tokens\": 3"}],"model":"m/x","max_tokens": 42,"stream":true,"stream_options":{"include_usage":true}}"#;
        assert_eq!(max_tokens(request.as_bytes()), 42);
        assert_eq!(model(request.as_bytes()), Some(&b"m/x"[..]));
        assert!(include_usage(request.as_bytes()));
        assert_eq!(
            max_tokens(br#"{"max_completion_tokens":5,"max_tokens":9}"#),
            5
        );
        assert_eq!(max_tokens(br#"{"model":"m"}"#), DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn streams_one_event_per_token_in_dynamos_format() {
        let events =
            events(r#"{"model":"m","max_tokens":3,"stream_options":{"include_usage":true}}"#);
        let chunk = |delta: &str, finish: &str| {
            format!(
                "data: {{\"id\":\"chatcmpl-id\",\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{finish},\"logprobs\":null}}],\"created\":7,\"model\":\"m\",\"object\":\"chat.completion.chunk\"}}\n\n"
            )
        };
        assert_eq!(
            events,
            [
                chunk(r#"{"content":" a","role":"assistant"}"#, "null"),
                chunk(r#"{"content":" a"}"#, "null"),
                chunk(r#"{"content":" a"}"#, r#""length""#),
                "data: {\"id\":\"chatcmpl-id\",\"choices\":[],\"created\":7,\"model\":\"m\",\"object\":\"chat.completion.chunk\",\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":3,\"total_tokens\":20}}\n\n".to_owned(),
                "data: [DONE]\n\n".to_owned(),
            ]
        );
        for event in &events[..4] {
            serde_json::from_str::<serde_json::Value>(&event[6..]).unwrap();
        }
    }

    #[tokio::test(start_paused = true)]
    async fn paced_tokens_come_one_inter_token_latency_apart() {
        let request = br#"{"model":"m","max_tokens":3,"stream_options":{"include_usage":true}}"#;
        let mut body = SseBody::new(request, "id", 7, Some(Duration::from_millis(10)));
        let start = Instant::now();
        let mut sent_at_ms = Vec::new();
        while let Some(frame) = body.frame().await {
            frame.unwrap();
            sent_at_ms.push(start.elapsed().as_millis());
        }
        // Three tokens 10 ms apart; the usage event and [DONE] follow the last.
        assert_eq!(sent_at_ms, [0, 10, 20, 20, 20]);
    }

    #[test]
    fn a_single_token_carries_role_and_finish_reason_and_usage_is_opt_in() {
        let events = events(r#"{"model":"m","max_tokens":1}"#);
        assert_eq!(events.len(), 2);
        assert!(events[0].contains(r#""role":"assistant"},"finish_reason":"length""#));
        assert_eq!(events[1], "data: [DONE]\n\n");
    }
}
