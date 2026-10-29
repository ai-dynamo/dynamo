// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owner forward routes (`PREFIX=owner`), for backends whose resources
//! (sessions, for example) keep state on the one worker that created them.
//! Each request goes to the worker owning the resource it names. A worker
//! takes part by advertising the base URL of its HTTP API as `forward_url`
//! in its model card's runtime data.
//!
//! - `POST PREFIX` creates a resource. The body's `id` names it; when the body
//!   names none, the frontend mints one and adds it. An id some worker already
//!   holds goes to that worker, whose answer is its own (typically a
//!   conflict). Otherwise rendezvous hashing picks the worker, so concurrent
//!   creates of one id through different frontends meet on one worker.
//! - `PREFIX/{id}`, and everything under it, goes to the worker holding
//!   `{id}`. Owners are cached; on a miss, every worker is asked
//!   `GET PREFIX/{id}`, and the one answering 2xx holds it. Workers stay the
//!   source of truth, so any frontend finds any resource, also after a restart.
//! - A resource no worker holds is a 404, and nothing is forwarded. When a
//!   worker cannot be asked, the answer is a 503 instead: a client that hears
//!   404 recreates the resource, possibly beside the original.
//! - Other requests to `PREFIX` itself, such as a listing, would span every
//!   worker, and are refused.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
};
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::{ForwardRoutes, error_response, read_upload, scoped_url, upstream_base};
use crate::namespace::{NamespaceFilter, NamespacePrefixMode};

/// The runtime-data key under which a worker advertises its HTTP API's base
/// URL.
pub(crate) const FORWARD_URL_KEY: &str = "forward_url";
/// How long a worker may take to say whether it holds a resource.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
/// Owners cached at most. Past it, owners unused for [`CACHE_IDLE`] go first.
const MAX_CACHED_OWNERS: usize = 65_536;
const CACHE_IDLE: Duration = Duration::from_secs(3600);
const MAX_ID_BYTES: usize = 256;

/// Where owner routes find workers: the frontend's discovery, scoped
/// to the namespaces it serves. Watching ends when `cancel` fires.
pub(crate) struct WorkerDiscovery {
    pub(crate) discovery: Arc<dyn Discovery>,
    pub(crate) namespaces: NamespaceFilter,
    pub(crate) cancel: CancellationToken,
}

/// The base URL each live worker advertises.
pub(super) struct WorkerUrls(watch::Receiver<HashMap<WorkerId, reqwest::Url>>);

impl WorkerUrls {
    /// Follows the model cards `source` discovers. A worker is dropped with its
    /// last card, when it leaves discovery.
    pub(super) fn watch(source: WorkerDiscovery) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());
        tokio::spawn(async move {
            let WorkerDiscovery {
                discovery,
                namespaces,
                cancel,
            } = source;
            let mut events = match discovery
                .list_and_watch(DiscoveryQuery::AllModels, Some(cancel.clone()))
                .await
            {
                Ok(events) => events,
                Err(err) => {
                    tracing::error!(error = %err, "owner forward routes cannot watch workers");
                    return;
                }
            };
            let mode = NamespacePrefixMode::from_env();
            let advertised = |card| advertised_url(card, &namespaces, mode);
            let mut cards: HashMap<DiscoveryInstanceId, (WorkerId, reqwest::Url)> = HashMap::new();
            loop {
                let event = tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = tx.closed() => break,
                    event = events.next() => match event {
                        Some(event) => event,
                        None => break,
                    },
                };
                match event {
                    Ok(DiscoveryEvent::Added(card)) => {
                        let id = card.id();
                        match advertised(card) {
                            Some(url) => cards.insert(id, url),
                            None => cards.remove(&id),
                        };
                    }
                    Ok(DiscoveryEvent::Removed(id)) => {
                        cards.remove(&id);
                    }
                    Ok(DiscoveryEvent::Resync(all)) => {
                        cards = all
                            .into_iter()
                            .filter_map(|card| Some((card.id(), advertised(card)?)))
                            .collect();
                    }
                    Ok(DiscoveryEvent::ModelTaintsUpdated(_)) => continue,
                    Err(err) => {
                        tracing::warn!(error = %err, "owner forward routes: discovery error");
                        continue;
                    }
                }
                let urls: HashMap<_, _> = cards.values().cloned().collect();
                tx.send_if_modified(|current| {
                    let changed = *current != urls;
                    if changed {
                        *current = urls;
                    }
                    changed
                });
            }
        });
        Self(rx)
    }

    fn get(&self, worker: WorkerId) -> Option<reqwest::Url> {
        self.0.borrow().get(&worker).cloned()
    }

    fn all(&self) -> Vec<(WorkerId, reqwest::Url)> {
        self.0
            .borrow()
            .iter()
            .map(|(worker, url)| (*worker, url.clone()))
            .collect()
    }

    #[cfg(test)]
    pub(super) async fn wait_for_count(&self, count: usize) {
        let mut rx = self.0.clone();
        rx.wait_for(|urls| urls.len() == count).await.unwrap();
    }
}

/// The URL a model card in `namespaces` advertises, with its worker.
fn advertised_url(
    card: DiscoveryInstance,
    namespaces: &NamespaceFilter,
    mode: NamespacePrefixMode,
) -> Option<(WorkerId, reqwest::Url)> {
    let DiscoveryInstance::Model {
        namespace,
        instance_id,
        card_json,
        ..
    } = card
    else {
        return None;
    };
    if !namespaces.matches_with_prefix_mode(&namespace, mode) {
        return None;
    }
    let url = card_json
        .get("runtime_config")?
        .get("runtime_data")?
        .get(FORWARD_URL_KEY)?
        .as_str()?;
    match upstream_base(url) {
        Ok(url) => Some((instance_id, url)),
        Err(err) => {
            tracing::warn!(instance_id, "ignoring worker {FORWARD_URL_KEY} that {err}");
            None
        }
    }
}

/// Which worker holds each resource.
pub(super) struct Owners {
    workers: WorkerUrls,
    /// Owners found so far. Only a cache: an entry is dropped when its worker
    /// leaves or answers 404, and a missing one is looked up again, so a
    /// resource deleted and recreated elsewhere costs one 404.
    cached: Mutex<HashMap<String, Cached>>,
}

struct Cached {
    worker: WorkerId,
    used: Instant,
}

impl Owners {
    pub(super) fn new(workers: WorkerUrls) -> Self {
        Self {
            workers,
            cached: Mutex::new(HashMap::new()),
        }
    }

    #[cfg(test)]
    pub(super) fn workers(&self) -> &WorkerUrls {
        &self.workers
    }

    /// The worker holding `id` and its base URL; `None` when every
    /// worker says it holds no such resource; otherwise the response to send
    /// instead. `auth` goes with each question, for workers that check it.
    async fn locate(
        &self,
        client: &reqwest::Client,
        prefix: &str,
        id: &str,
        auth: Option<&HeaderValue>,
    ) -> Result<Option<(WorkerId, reqwest::Url)>, Response> {
        if let Some(worker) = self.cached(id) {
            match self.workers.get(worker) {
                Some(url) => return Ok(Some((worker, url))),
                // The resource left with its worker; the id may since name another.
                None => self.forget(id),
            }
        }
        let workers = self.workers.all();
        if workers.is_empty() {
            return Err(no_workers());
        }
        let path = format!("{prefix}/{id}");
        let mut asked: FuturesUnordered<_> = workers
            .into_iter()
            .map(|(worker, base)| {
                let url = scoped_url(&base, prefix, &path, None);
                async move { (worker, base, holds(client, url, auth).await) }
            })
            .collect();
        let mut unanswered = false;
        while let Some((worker, base, held)) = asked.next().await {
            match held {
                Some(true) => {
                    self.remember(id, worker);
                    return Ok(Some((worker, base)));
                }
                Some(false) => {}
                None => unanswered = true,
            }
        }
        if unanswered {
            return Err(error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "could not ask every worker for this resource",
            ));
        }
        Ok(None)
    }

    /// The worker a new resource goes to: rendezvous hashing of its id over
    /// the live workers, so frontends agree while membership holds.
    fn place(&self, id: &str) -> Option<(WorkerId, reqwest::Url)> {
        self.workers
            .all()
            .into_iter()
            .max_by_key(|(worker, _)| xxhash_rust::xxh3::xxh3_64_with_seed(id.as_bytes(), *worker))
    }

    fn cached(&self, id: &str) -> Option<WorkerId> {
        let mut cached = self.cached.lock();
        let entry = cached.get_mut(id)?;
        entry.used = Instant::now();
        Some(entry.worker)
    }

    fn remember(&self, id: &str, worker: WorkerId) {
        let mut cached = self.cached.lock();
        let now = Instant::now();
        if cached.len() >= MAX_CACHED_OWNERS && !cached.contains_key(id) {
            cached.retain(|_, entry| now.duration_since(entry.used) < CACHE_IDLE);
            if cached.len() >= MAX_CACHED_OWNERS {
                return;
            }
        }
        cached.insert(id.to_string(), Cached { worker, used: now });
    }

    fn forget(&self, id: &str) {
        self.cached.lock().remove(id);
    }
}

/// Whether the worker at `url` holds the resource: `None` when it gave no
/// answer to go by.
async fn holds(
    client: &reqwest::Client,
    url: Option<reqwest::Url>,
    auth: Option<&HeaderValue>,
) -> Option<bool> {
    let url = url?;
    let mut request = client.get(url.clone()).timeout(PROBE_TIMEOUT);
    if let Some(auth) = auth {
        request = request.header(header::AUTHORIZATION, auth.clone());
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => Some(true),
        Ok(response) if response.status() == StatusCode::NOT_FOUND => Some(false),
        Ok(response) => {
            tracing::warn!(%url, status = %response.status(), "owner lookup got no answer");
            None
        }
        Err(err) => {
            tracing::warn!(%url, error = %err, "owner lookup failed");
            None
        }
    }
}

impl ForwardRoutes {
    /// Forwards `request`, under the owner route at `prefix`, as the
    /// module docs describe; see [`Self::forward`] for the rest.
    pub(super) async fn forward_owned<P: Send + Sync + 'static>(
        &self,
        owners: &Owners,
        prefix: &str,
        request: Request,
        cancel: CancellationToken,
        permit: P,
    ) -> Response {
        let path = request.uri().path();
        // The route matched, so the rest is empty or starts with '/'.
        let rest = &path[prefix.len()..];
        if matches!(rest, "" | "/") {
            if request.method() != Method::POST {
                return error_response(
                    StatusCode::NOT_IMPLEMENTED,
                    "a listing spans workers; only creating a resource is forwarded here",
                );
            }
            return self
                .create_resource(owners, prefix, request, cancel, permit)
                .await;
        }
        let id = rest[1..].split('/').next().unwrap_or_default().to_string();
        if id.is_empty() {
            return no_such_resource();
        }
        let deletes =
            request.method() == Method::DELETE && rest[1..].trim_end_matches('/') == id.as_str();
        let auth = request.headers().get(header::AUTHORIZATION).cloned();
        let located = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                return super::super::openai::ErrorMessage::_service_unavailable().into_response();
            }
            located = owners.locate(&self.client, prefix, &id, auth.as_ref()) => located,
        };
        let base = match located {
            Ok(Some((_, base))) => base,
            Ok(None) => return no_such_resource(),
            Err(refused) => return refused,
        };
        let Some(url) = scoped_url(&base, prefix, path, request.uri().query()) else {
            return no_such_resource();
        };
        let response = self.forward_to(request, url, cancel, permit).await;
        let status = response.status();
        if status == StatusCode::NOT_FOUND || (deletes && status.is_success()) {
            owners.forget(&id);
        }
        response
    }

    /// Creates a resource on its worker, and remembers the worker once it has.
    async fn create_resource<P: Send + Sync + 'static>(
        &self,
        owners: &Owners,
        prefix: &str,
        request: Request,
        cancel: CancellationToken,
        permit: P,
    ) -> Response {
        let (mut parts, body) = request.into_parts();
        let body = match read_upload(&parts, body, &cancel).await {
            Ok(body) => body,
            Err(refused) => return refused,
        };
        let mut fields = if body.iter().all(u8::is_ascii_whitespace) {
            serde_json::Map::new()
        } else {
            match serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(&body) {
                Ok(fields) => fields,
                Err(_) => {
                    return error_response(
                        StatusCode::BAD_REQUEST,
                        "a resource is created from a JSON object",
                    );
                }
            }
        };
        let (id, body, minted) = match fields.get("id") {
            Some(serde_json::Value::String(id)) => (id.clone(), body, false),
            None | Some(serde_json::Value::Null) => {
                // Minted here, so the frontend knows the id before any worker does.
                let id = uuid::Uuid::new_v4().simple().to_string();
                fields.insert("id".to_string(), id.clone().into());
                let body = Bytes::from(serde_json::Value::Object(fields).to_string());
                parts.headers.insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                (id, body, true)
            }
            Some(_) => return error_response(StatusCode::BAD_REQUEST, "'id' must be a string"),
        };
        if !valid_id(&id) {
            return error_response(
                StatusCode::BAD_REQUEST,
                "'id' must be 1 to 256 characters of A-Z a-z 0-9 - . _ ~",
            );
        }
        let located = if minted {
            Ok(None)
        } else {
            let auth = parts.headers.get(header::AUTHORIZATION);
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    return super::super::openai::ErrorMessage::_service_unavailable().into_response();
                }
                located = owners.locate(&self.client, prefix, &id, auth) => located,
            }
        };
        let (worker, base) = match located {
            Ok(Some(held)) => held,
            Ok(None) => match owners.place(&id) {
                Some(placed) => placed,
                None => return no_workers(),
            },
            Err(refused) => return refused,
        };
        let Some(url) = scoped_url(&base, prefix, parts.uri.path(), parts.uri.query()) else {
            return no_workers();
        };
        parts
            .headers
            .insert(header::CONTENT_LENGTH, HeaderValue::from(body.len()));
        let request = Request::from_parts(parts, Body::from(body));
        let response = self.forward_to(request, url, cancel, permit).await;
        if response.status().is_success() {
            owners.remember(&id, worker);
        }
        response
    }
}

/// An id usable verbatim as one path segment.
fn valid_id(id: &str) -> bool {
    (1..=MAX_ID_BYTES).contains(&id.len())
        && !matches!(id, "." | "..")
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~'))
}

pub(super) fn no_workers() -> Response {
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "no worker serves this forwarded route",
    )
}

fn no_such_resource() -> Response {
    error_response(StatusCode::NOT_FOUND, "no worker holds this resource")
}
