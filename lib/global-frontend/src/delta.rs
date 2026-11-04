// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Requests sent to pool frontends as deltas (see
//! [`dynamo_global_common::delta`]).
//!
//! For each conversation, the Global Frontend remembers the last request a
//! pool frontend kept: its [`Parent`], not the request. The conversation's
//! next request goes to that frontend as a delta against it, and every body
//! sent to a frontend that keeps requests is zstd-compressed. A frontend that
//! no longer has the parent says so and gets the whole request. Sessions do
//! not move between pools with their kept requests: a request whose session
//! is placed on another pool goes there whole.

use std::collections::HashSet;
use std::sync::{Arc, PoisonError, RwLock};
use std::time::Duration;

use axum::body::Bytes;
use axum::http::{HeaderMap, HeaderName, HeaderValue, header};
use dynamo_global_common::delta::{self, Digest, Parent};
use dynamo_global_router::{Pool, PrefixKey, continued};
use moka::future::Cache;
use url::Url;

/// Conversations remembered at most; the least used go first.
const MAX_CONVERSATIONS: u64 = 1 << 20;
/// Bodies at least this large are compressed off the async runtime.
const MAX_INLINE_COMPRESSION: usize = 64 * 1024;

/// What the Global Frontend remembers of the requests pool frontends keep.
#[derive(Debug)]
pub struct Deltas {
    /// Each conversation's last kept request, by [`Conversation::key`].
    sent: Cache<String, Arc<Sent>>,
    /// Frontends that keep requests, and so take deltas and zstd bodies.
    keeping: RwLock<HashSet<Url>>,
    /// zstd level, or 0 to send bodies uncompressed.
    level: i32,
}

#[derive(Debug)]
struct Sent {
    frontend: Url,
    parent: Parent,
}

/// A request's conversation: the key its request is remembered under, and
/// the keys of the requests it may continue, deepest first.
#[derive(Debug)]
pub(crate) struct Conversation {
    key: String,
    continues: Vec<String>,
}

impl Conversation {
    /// The conversation of a request that names `session`, or else whose
    /// input has the `prefix` keys; `None` for neither.
    pub(crate) fn new(session: Option<&str>, prefix: &[PrefixKey]) -> Option<Self> {
        match session {
            Some(session) => Some(Self {
                key: session.to_owned(),
                continues: vec![session.to_owned()],
            }),
            None => Some(Self {
                key: prefix.last()?.session_id(),
                continues: continued(prefix).map(PrefixKey::session_id).collect(),
            }),
        }
    }
}

/// One request's body on its way to a pool frontend.
#[derive(Debug)]
pub(crate) struct Upload<'p> {
    pub(crate) frontend: &'p Url,
    /// Taken when sent, so it is freed once written.
    pub(crate) body: Bytes,
    /// Headers to add to the client's.
    pub(crate) headers: Vec<(HeaderName, HeaderValue)>,
    /// `delta` or `whole`, for metrics.
    pub(crate) kind: &'static str,
    /// The whole request, kept for a delta until it is answered.
    whole: Option<Bytes>,
    remember: Option<Remember>,
}

impl<'p> Upload<'p> {
    /// `body` to `frontend` as it is.
    pub(crate) fn whole(frontend: &'p Url, body: Bytes) -> Self {
        Self {
            frontend,
            body,
            headers: Vec::new(),
            kind: "whole",
            whole: None,
            remember: None,
        }
    }

    pub(crate) fn is_delta(&self) -> bool {
        self.whole.is_some()
    }
}

/// What to remember once the frontend answers.
#[derive(Debug)]
struct Remember {
    key: String,
    /// The key of the remembered request this one continues, if another.
    continued: Option<String>,
    frontend: Url,
    parent: Parent,
}

impl Deltas {
    /// Remembers conversations for `idle` without a request, and compresses
    /// bodies at zstd `level`, or not at all for 0.
    pub fn new(idle: Duration, level: i32) -> Self {
        Self {
            sent: Cache::builder()
                .max_capacity(MAX_CONVERSATIONS)
                .time_to_idle(idle)
                .build(),
            keeping: RwLock::default(),
            level,
        }
    }

    /// How to send `body`, a request in `conversation` sent with
    /// `authorization`, to `pool`, whose selector chose `frontend`. A delta
    /// goes to the frontend that keeps its parent instead.
    pub(crate) async fn upload<'p>(
        &self,
        conversation: Conversation,
        body: Bytes,
        authorization: Option<&[u8]>,
        pool: &'p Pool,
        frontend: &'p Url,
    ) -> Upload<'p> {
        let Some(request) = delta::Request::parse(&body) else {
            return Upload::whole(frontend, body);
        };
        let mut previous = None;
        for key in conversation.continues {
            if let Some(sent) = self.sent.get(&key).await {
                previous = Some((key, sent));
                break;
            }
        }
        let keeper = previous.as_ref().and_then(|(_, sent)| {
            pool.frontends
                .iter()
                .find(|frontend| **frontend == sent.frontend && self.keeps(frontend))
        });
        let delta = match (keeper, &previous) {
            (Some(keeper), Some((_, sent))) => {
                sent.parent.delta(&request).map(|delta| (keeper, delta))
            }
            _ => None,
        };
        let frontend = delta.as_ref().map_or(frontend, |(keeper, _)| *keeper);
        let continued = previous
            .map(|(key, _)| key)
            .filter(|key| *key != conversation.key);
        let remember = Remember {
            key: conversation.key,
            continued,
            frontend: frontend.clone(),
            parent: Parent::new(request.state(authorization), &request),
        };
        drop(request);
        let mut upload = match delta {
            Some((_, (delta, delta_body))) => {
                let mut upload = Upload::whole(frontend, delta_body.into());
                upload
                    .headers
                    .push(header_pair(delta::DELTA_HEADER, &delta.to_string()));
                upload.kind = "delta";
                upload.whole = Some(body);
                upload
            }
            None => {
                let mut upload = Upload::whole(frontend, body);
                self.store(&mut upload);
                upload
            }
        };
        upload.remember = Some(remember);
        self.compress(&mut upload).await;
        upload
    }

    /// Makes `upload` the whole request, after its frontend refused its
    /// delta. False if it was not a delta.
    pub(crate) async fn resend(&self, upload: &mut Upload<'_>) -> bool {
        let Some(whole) = upload.whole.take() else {
            return false;
        };
        upload.body = whole;
        upload.headers.clear();
        upload.kind = "whole";
        self.store(upload);
        self.compress(upload).await;
        true
    }

    /// Learns from the headers the frontend answered `upload` with: whether
    /// it kept the request, and so whether it keeps requests.
    pub(crate) async fn answered(&self, upload: Upload<'_>, headers: &HeaderMap) {
        let was_delta = upload.is_delta();
        let Some(remember) = upload.remember else {
            return;
        };
        let kept = headers
            .get(delta::STATE_HEADER)
            .and_then(|kept| kept.to_str().ok())
            .and_then(|kept| kept.parse::<Digest>().ok());
        if kept != Some(remember.parent.state) {
            // A frontend that answers a request to keep without keeping it
            // keeps none.
            if !was_delta && kept.is_none() && self.keeps(&remember.frontend) {
                self.keeping
                    .write()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(&remember.frontend);
            }
            return;
        }
        if !self.keeps(&remember.frontend) {
            self.keeping
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(remember.frontend.clone());
        }
        let sent = Sent {
            frontend: remember.frontend,
            parent: remember.parent,
        };
        self.sent.insert(remember.key, Arc::new(sent)).await;
        // The frontend kept this request in place of its parent.
        if let Some(continued) = remember.continued {
            self.sent.invalidate(&continued).await;
        }
    }

    fn keeps(&self, frontend: &Url) -> bool {
        let keeping = self.keeping.read().unwrap_or_else(PoisonError::into_inner);
        keeping.contains(frontend)
    }

    /// Asks `upload`'s frontend to keep it.
    fn store(&self, upload: &mut Upload<'_>) {
        upload.headers.push(header_pair(delta::STORE_HEADER, "1"));
    }

    /// Compresses `upload` if its frontend keeps requests.
    async fn compress(&self, upload: &mut Upload<'_>) {
        if self.level == 0 || !self.keeps(upload.frontend) {
            return;
        }
        let level = self.level;
        let body = upload.body.clone();
        let compressed = if body.len() < MAX_INLINE_COMPRESSION {
            zstd::bulk::compress(&body, level).ok()
        } else {
            let compressing =
                tokio::task::spawn_blocking(move || zstd::bulk::compress(&body, level));
            compressing.await.ok().and_then(Result::ok)
        };
        if let Some(compressed) = compressed {
            upload.body = compressed.into();
            let zstd = HeaderValue::from_static("zstd");
            upload.headers.push((header::CONTENT_ENCODING, zstd));
        }
    }
}

fn header_pair(name: &'static str, value: &str) -> (HeaderName, HeaderValue) {
    let value = HeaderValue::from_str(value).expect("delta headers are ASCII");
    (HeaderName::from_static(name), value)
}
