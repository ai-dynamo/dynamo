// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP response headers a chat engine sets through its request context.

use std::sync::Arc;

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use parking_lot::Mutex;

/// Request-context key of the [`ResponseHeaders`] a chat completions request
/// carries.
pub const RESPONSE_HEADERS_CONTEXT_KEY: &str = "dynamo.llm.response_headers";

/// Headers the HTTP layer owns; an engine may not set them.
const RESERVED: &[&str] = &[
    "connection",
    "content-encoding",
    "content-length",
    "content-type",
    "keep-alive",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "x-request-id",
];

/// The HTTP response headers a chat engine sets for one request.
///
/// The chat completions handler puts one on the request context when it reads
/// the engine's stream before it commits the response, and takes the headers
/// when it commits. The engine reaches it through that context
/// (`Context.set_response_header` in Python).
#[derive(Clone)]
pub struct ResponseHeaders(Arc<Mutex<Option<HeaderMap>>>);

impl Default for ResponseHeaders {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Some(HeaderMap::new()))))
    }
}

impl ResponseHeaders {
    /// Set a response header, replacing an earlier value for the same name.
    ///
    /// `Ok(false)` when the response is already committed and cannot carry the
    /// header. `Err` for a name or value that is not a valid header, or for a
    /// header the HTTP layer owns.
    pub fn set(&self, name: &str, value: &str) -> Result<bool, String> {
        let header = HeaderName::try_from(name)
            .map_err(|_| format!("invalid response header name '{name}'"))?;
        let value = HeaderValue::try_from(value)
            .map_err(|_| format!("invalid value for response header '{name}'"))?;
        if RESERVED.contains(&header.as_str()) {
            return Err(format!(
                "response header '{header}' is set by the HTTP layer"
            ));
        }
        match self.0.lock().as_mut() {
            Some(headers) => {
                headers.insert(header, value);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Take the headers set so far. Every later [`Self::set`] returns `Ok(false)`.
    pub fn take(&self) -> HeaderMap {
        self.0.lock().take().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_validates_headers_and_stops_at_take() {
        let headers = ResponseHeaders::default();

        assert_eq!(headers.set("X-Capture-Refused", "call-1"), Ok(true));
        // The HTTP layer's headers, in any case, and malformed ones are errors.
        assert!(headers.set("Content-Type", "text/plain").is_err());
        assert!(headers.set("bad name", "v").is_err());
        assert!(headers.set("x-ok", "bad\r\nvalue").is_err());

        let taken = headers.take();
        assert_eq!(taken.len(), 1);
        assert_eq!(taken.get("x-capture-refused").unwrap(), "call-1");

        // The response is committed: the header is not applied, and says so.
        assert_eq!(headers.set("x-late", "v"), Ok(false));
        assert!(headers.take().is_empty());
    }
}
