// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::extract::MatchedPath;
use axum::http::Request;
use axum::http::header::HeaderName;
use axum::middleware::Next;
use axum::response::Response;
use serde::Serialize;

#[derive(Clone)]
pub struct AccessLogModel(pub String);

#[derive(Serialize)]
struct AccessLogEntry {
    ts: String,
    trace_id: String,
    method: String,
    path: String,
    model: String,
    status: u16,
    duration_ms: f64,
}

// ---------------------------------------------------------------------------
// AccessLogSink — wraps tracing_appender::non_blocking with reopen support
// ---------------------------------------------------------------------------

static IO_ERRORS: AtomicU64 = AtomicU64::new(0);

pub struct AccessLogSink {
    path: PathBuf,
    inner: parking_lot::Mutex<SinkInner>,
    trace_id_header: HeaderName,
    use_local_time: bool,
}

struct SinkInner {
    writer: tracing_appender::non_blocking::NonBlocking,
    _guard: tracing_appender::non_blocking::WorkerGuard,
}

impl AccessLogSink {
    pub fn new(path: &Path, trace_id_header: HeaderName, use_local_time: bool) -> io::Result<Self> {
        let file = File::options().create(true).append(true).open(path)?;
        let (writer, guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
            .lossy(false)
            .finish(file);
        Ok(Self {
            path: path.to_path_buf(),
            inner: parking_lot::Mutex::new(SinkInner {
                writer,
                _guard: guard,
            }),
            trace_id_header,
            use_local_time,
        })
    }

    fn write_line(&self, line: &str) {
        let mut record = line.to_owned();
        record.push('\n');
        let mut inner = self.inner.lock();
        if let Err(e) = inner.writer.write_all(record.as_bytes()) {
            let prev = IO_ERRORS.fetch_add(1, Ordering::Relaxed);
            if prev % 100 == 0 {
                tracing::warn!(error = %e, total = prev + 1, "access log write failed");
            }
        }
    }

    pub fn reopen(&self) -> io::Result<()> {
        let new_file = File::options().create(true).append(true).open(&self.path)?;
        let (new_writer, new_guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
            .lossy(false)
            .finish(new_file);
        let old = {
            let mut inner = self.inner.lock();
            std::mem::replace(
                &mut *inner,
                SinkInner {
                    writer: new_writer,
                    _guard: new_guard,
                },
            )
        };
        drop(old);
        Ok(())
    }

    fn format_timestamp(&self) -> String {
        if self.use_local_time {
            chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, false)
        } else {
            chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        }
    }
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

pub async fn access_log_middleware(
    axum::extract::State(sink): axum::extract::State<Option<Arc<AccessLogSink>>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let Some(ref sink) = sink else {
        return next.run(req).await;
    };

    let trace_id = req
        .headers()
        .get(&sink.trace_id_header)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_owned();
    let method = req.method().to_string();
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());

    let ts = sink.format_timestamp();
    let start = Instant::now();
    let response = next.run(req).await;
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    let status = response.status().as_u16();
    let model = response
        .extensions()
        .get::<AccessLogModel>()
        .map(|m| m.0.as_str())
        .unwrap_or("-")
        .to_owned();

    let entry = AccessLogEntry {
        ts,
        trace_id,
        method,
        path,
        model,
        status,
        duration_ms,
    };

    if let Ok(line) = serde_json::to_string(&entry) {
        sink.write_line(&line);
    }

    response
}

/// Parse and validate an HTTP header name at startup.
pub fn parse_header_name(name: &str) -> Result<HeaderName, axum::http::Error> {
    HeaderName::from_bytes(name.as_bytes()).map_err(|e| axum::http::Error::from(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_log_sink_write_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("access.log");
        let sink = AccessLogSink::new(&path, HeaderName::from_static("x-trace-id"), false).unwrap();

        sink.write_line(r#"{"ts":"t1","trace_id":"-","method":"GET","path":"/health","model":"-","status":200,"duration_ms":0.1}"#);
        drop(sink.inner.lock());
        std::thread::sleep(std::time::Duration::from_millis(100));

        let rotated = dir.path().join("access.log.1");
        std::fs::rename(&path, &rotated).unwrap();
        sink.reopen().unwrap();

        sink.write_line(r#"{"ts":"t2","trace_id":"-","method":"POST","path":"/query","model":"llama","status":200,"duration_ms":1.0}"#);
        drop(sink);
        std::thread::sleep(std::time::Duration::from_millis(100));

        let old_content = std::fs::read_to_string(&rotated).unwrap();
        assert!(old_content.contains("\"ts\":\"t1\""));
        assert!(!old_content.contains("\"ts\":\"t2\""));

        let new_content = std::fs::read_to_string(&path).unwrap();
        assert!(new_content.contains("\"ts\":\"t2\""));
        assert!(!new_content.contains("\"ts\":\"t1\""));
    }

    #[test]
    fn access_log_sink_drop_flushes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("access.log");
        let sink = AccessLogSink::new(&path, HeaderName::from_static("x-trace-id"), false).unwrap();

        for i in 0..5 {
            sink.write_line(&format!(r#"{{"n":{i}}}"#));
        }

        drop(sink);
        std::thread::sleep(std::time::Duration::from_millis(100));

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 5);
        assert!(content.contains(r#"{"n":0}"#));
        assert!(content.contains(r#"{"n":4}"#));
    }

    #[test]
    fn parse_header_name_validates() {
        assert!(parse_header_name("x-trace-id").is_ok());
        assert!(parse_header_name("x-request-id").is_ok());
        assert!(parse_header_name("invalid header\n").is_err());
    }
}
