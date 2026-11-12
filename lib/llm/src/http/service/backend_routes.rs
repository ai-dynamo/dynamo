// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend routes: HTTP APIs a backend serves itself, for resources such as
//! sessions that keep state on the one worker that created them. A worker
//! declares them in its model card's runtime data as `http_routes`, the base
//! URL of its HTTP server and the path prefixes it serves there. A request no
//! frontend route matches goes to a worker declaring a prefix that covers it:
//!
//! - `POST PREFIX` creates a resource. The JSON body's `id` names it; when the
//!   body names none, the frontend mints one and adds it. An id some worker
//!   already holds goes to that worker, which answers for itself (typically
//!   with a conflict); a new one goes to a worker picked at random.
//! - `PREFIX/{id}`, and every path under it, goes to the worker holding
//!   `{id}`. The router's session affinity table keeps that worker while
//!   requests use the resource, and for an hour after. Without a binding (after
//!   that hour, a restart, or for a resource created through another frontend),
//!   every worker declaring the prefix is asked `GET PREFIX/{id}`, and the one
//!   answering 2xx holds it.
//! - A resource no worker holds is a 404. When a worker cannot be asked, the
//!   answer is a 503 instead: a client that hears 404 recreates the resource,
//!   possibly beside the original.
//! - Other requests to `PREFIX` itself, such as a listing, would span every
//!   worker, and are refused.

mod proxy;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{HeaderValue, Method, StatusCode, Uri, header};
use axum::response::Response;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::discovery::{
    Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
};
use dynamo_session_affinity::host::{AffinityLease, MAX_SESSION_AFFINITY_ID_BYTES};
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::namespace::{NamespaceFilter, NamespacePrefixMode};
use crate::protocols::common::extensions::SessionAffinityId;
use crate::session_affinity::{AffinityCoordinator, AffinityTarget, Hold, SessionAffinityMode};
use proxy::{error_response, read_upload, scoped_url, upstream_base};

/// The runtime-data key under which a worker declares its routes, as
/// `{"url": BASE_URL, "prefixes": [PREFIX, ...]}`.
const HTTP_ROUTES_KEY: &str = "http_routes";
/// How long the worker of an idle resource is kept; after that, the workers
/// are asked again.
const BINDING_TTL: Duration = Duration::from_secs(3600);
/// How long a worker may take to say whether it holds a resource.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Where backend routes find workers: the frontend's discovery, scoped to the
/// namespaces it serves. Watching ends when `cancel` fires.
pub(crate) struct WorkerDiscovery {
    pub(crate) discovery: Arc<dyn Discovery>,
    pub(crate) namespaces: NamespaceFilter,
    pub(crate) cancel: CancellationToken,
}

/// The workers declaring each prefix, with their base URLs.
type Routes = HashMap<String, HashMap<WorkerId, reqwest::Url>>;

/// The routes live workers declare, and the worker holding each resource.
pub(crate) struct BackendRoutes {
    routes: watch::Receiver<Routes>,
    /// Keyed by the resource's path, `PREFIX/{id}`.
    owners: AffinityCoordinator,
    client: reqwest::Client,
}

/// A request a declared route covers, with that route's prefix. Only
/// [`BackendRoutes::resolve`] makes one.
pub(crate) struct ForwardTarget(String);

/// Where a resource lives.
enum Owner {
    /// On the worker at this base URL, with a lease on its binding when the
    /// table holds one.
    Held(reqwest::Url, Option<AffinityLease>),
    /// On no worker yet, with the table's hold on it when there is one, to
    /// commit once a worker has it.
    Unheld(Option<Hold>),
}

impl BackendRoutes {
    /// Follows the routes declared by the workers `source` discovers.
    pub(crate) fn new(source: WorkerDiscovery) -> anyhow::Result<Self> {
        let owners = AffinityCoordinator::new(BINDING_TTL, SessionAffinityMode::Hard)?;
        let client = proxy::client()?;
        Ok(Self {
            routes: watch_routes(source),
            owners,
            client,
        })
    }

    /// The route covering `uri`, or `None` when no declared prefix covers its
    /// path. Prefixes match whole segments, the longest first: `/v1/sessions`
    /// covers `/v1/sessions` and `/v1/sessions/abc`, not `/v1/sessionsx`.
    pub(crate) fn resolve(&self, uri: &Uri) -> Option<ForwardTarget> {
        let path = uri.path();
        if !forwardable(path) {
            return None;
        }
        self.routes
            .borrow()
            .keys()
            .filter(|prefix| under(path, prefix))
            .max_by_key(|prefix| prefix.len())
            .map(|prefix| ForwardTarget(prefix.clone()))
    }

    /// Forwards `request` under `target`'s prefix, as the module docs
    /// describe. `permit` lives as long as the forwarded exchange (the
    /// response body, or the WebSocket tunnel), so callers can count it as
    /// inflight. Tunnels end when `cancel` fires.
    pub(crate) async fn forward<P: Send + 'static>(
        &self,
        request: Request,
        target: ForwardTarget,
        cancel: CancellationToken,
        permit: P,
    ) -> Response {
        let prefix = target.0;
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
            return self.create(&prefix, request, cancel, permit).await;
        }
        let id = rest[1..].split('/').next().unwrap_or_default();
        if !valid_id(&prefix, id) {
            return no_such_resource();
        }
        let key = format!("{prefix}/{id}");
        let deletes = request.method() == Method::DELETE && rest[1..].trim_end_matches('/') == id;
        let auth = request.headers().get(header::AUTHORIZATION).cloned();
        let (base, mut lease) = match self.owner(&prefix, &key, auth.as_ref(), true).await {
            Ok(Owner::Held(base, lease)) => (base, lease),
            Ok(Owner::Unheld(_)) => return no_such_resource(),
            Err(refused) => return refused,
        };
        let Some(url) = scoped_url(&base, &prefix, path, request.uri().query()) else {
            return no_such_resource();
        };
        let (upstream, upgrade) = match proxy::send(&self.client, request, &url).await {
            Ok(sent) => sent,
            Err(refused) => return refused,
        };
        let status = upstream.status();
        if (status == StatusCode::NOT_FOUND || (deletes && status.is_success()))
            && let Some(lease) = lease.as_mut()
        {
            // The worker holds it no longer.
            lease.invalidate();
        }
        proxy::client_response(upstream, upgrade, url, cancel, (permit, lease))
    }

    /// Creates a resource on its worker, and binds it there once it exists.
    async fn create<P: Send + 'static>(
        &self,
        prefix: &str,
        request: Request,
        cancel: CancellationToken,
        permit: P,
    ) -> Response {
        let (mut parts, body) = request.into_parts();
        let body = match read_upload(&parts, body).await {
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
        if !valid_id(prefix, &id) {
            let message = format!(
                "'id' must be 1 to {} characters of A-Z a-z 0-9 - . _ ~",
                max_id_len(prefix)
            );
            return error_response(StatusCode::BAD_REQUEST, &message);
        }
        let key = format!("{prefix}/{id}");
        let auth = parts.headers.get(header::AUTHORIZATION).cloned();
        // A minted id is new, so no worker is asked for it.
        let (base, lease, placed) = match self.owner(prefix, &key, auth.as_ref(), !minted).await {
            Ok(Owner::Held(base, lease)) => (base, lease, None),
            Ok(Owner::Unheld(hold)) => match self.place(prefix) {
                Some((worker, base)) => (base, None, Some((worker, hold))),
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
        let (upstream, upgrade) = match proxy::send(&self.client, request, &url).await {
            Ok(sent) => sent,
            Err(refused) => return refused,
        };
        let lease = match placed {
            Some((worker, Some(hold))) if upstream.status().is_success() => {
                self.commit(hold, worker)
            }
            _ => lease,
        };
        proxy::client_response(upstream, upgrade, url, cancel, (permit, lease))
    }

    /// Where the resource at `key`, under `prefix`, lives: on the worker the
    /// table binds it to, or, when `ask` is set and the table binds none, on
    /// the worker that says it holds it.
    async fn owner(
        &self,
        prefix: &str,
        key: &str,
        auth: Option<&HeaderValue>,
        ask: bool,
    ) -> Result<Owner, Response> {
        loop {
            let hold = match self
                .owners
                .acquire(&SessionAffinityId::new(key), None)
                .await
            {
                Ok(Hold::Bound { target, mut lease }) => {
                    if let Some(base) = self.worker(prefix, target.worker_id) {
                        return Ok(Owner::Held(base, Some(lease)));
                    }
                    // Its worker left; the id may since name a resource elsewhere.
                    lease.invalidate();
                    continue;
                }
                Ok(hold) => Some(hold),
                // A full table costs only the binding: the workers still know.
                Err(err) => {
                    tracing::warn!(error = %err, "backend route resource not bound");
                    None
                }
            };
            if ask && let Some((worker, base)) = self.ask(prefix, key, auth).await? {
                let lease = hold.and_then(|hold| self.commit(hold, worker));
                return Ok(Owner::Held(base, lease));
            }
            return Ok(Owner::Unheld(hold));
        }
    }

    /// Asks every worker declaring `prefix` whether it holds the resource at
    /// `key`, with `auth` for workers that check it: the one answering 2xx
    /// does. `None` when every worker answers 404; a 503 when one gives no
    /// answer, as it might hold the resource.
    async fn ask(
        &self,
        prefix: &str,
        key: &str,
        auth: Option<&HeaderValue>,
    ) -> Result<Option<(WorkerId, reqwest::Url)>, Response> {
        let mut asked: FuturesUnordered<_> = self
            .workers(prefix)
            .into_iter()
            .map(|(worker, base)| {
                let url = scoped_url(&base, prefix, key, None);
                async move { (worker, base, holds(&self.client, url, auth).await) }
            })
            .collect();
        let mut unanswered = false;
        while let Some((worker, base, held)) = asked.next().await {
            match held {
                Some(true) => return Ok(Some((worker, base))),
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

    /// A worker declaring `prefix`, picked at random, with its base URL.
    fn place(&self, prefix: &str) -> Option<(WorkerId, reqwest::Url)> {
        let mut workers = self.workers(prefix);
        if workers.is_empty() {
            return None;
        }
        Some(workers.swap_remove(rand::random_range(0..workers.len())))
    }

    /// Binds the resource `hold` holds to `worker`.
    fn commit(&self, hold: Hold, worker: WorkerId) -> Option<AffinityLease> {
        self.owners
            .commit(hold, AffinityTarget::worker(worker))
            .inspect_err(|err| tracing::warn!(error = %err, "backend route resource not bound"))
            .ok()
    }

    /// The workers declaring `prefix`, with their base URLs.
    fn workers(&self, prefix: &str) -> Vec<(WorkerId, reqwest::Url)> {
        self.routes
            .borrow()
            .get(prefix)
            .map_or_else(Vec::new, |workers| {
                workers
                    .iter()
                    .map(|(worker, base)| (*worker, base.clone()))
                    .collect()
            })
    }

    /// The base URL of `worker`, while it declares `prefix`.
    fn worker(&self, prefix: &str, worker: WorkerId) -> Option<reqwest::Url> {
        self.routes.borrow().get(prefix)?.get(&worker).cloned()
    }

    #[cfg(test)]
    async fn wait_for_workers(&self, prefix: &str, count: usize) {
        let mut routes = self.routes.clone();
        routes
            .wait_for(|routes| routes.get(prefix).map_or(0, HashMap::len) == count)
            .await
            .unwrap();
    }
}

/// What a model card declares.
struct Declared {
    worker: WorkerId,
    url: reqwest::Url,
    prefixes: Vec<String>,
}

/// Follows the routes declared by the model cards `source` discovers. A
/// worker's routes go with its last card, when it leaves discovery.
fn watch_routes(source: WorkerDiscovery) -> watch::Receiver<Routes> {
    let (tx, rx) = watch::channel(Routes::new());
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
                tracing::error!(error = %err, "backend routes cannot watch workers");
                return;
            }
        };
        let mode = NamespacePrefixMode::from_env();
        let declared = |card| declared_routes(card, &namespaces, mode);
        let mut cards: HashMap<DiscoveryInstanceId, Declared> = HashMap::new();
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
                    match declared(card) {
                        Some(routes) => cards.insert(id, routes),
                        None => cards.remove(&id),
                    };
                }
                Ok(DiscoveryEvent::Removed(id)) => {
                    cards.remove(&id);
                }
                Ok(DiscoveryEvent::Resync(all)) => {
                    cards = all
                        .into_iter()
                        .filter_map(|card| Some((card.id(), declared(card)?)))
                        .collect();
                }
                Ok(DiscoveryEvent::ModelTaintsUpdated(_)) => continue,
                Err(err) => {
                    tracing::warn!(error = %err, "backend routes: discovery error");
                    continue;
                }
            }
            let mut routes = Routes::new();
            for card in cards.values() {
                for prefix in &card.prefixes {
                    routes
                        .entry(prefix.clone())
                        .or_default()
                        .insert(card.worker, card.url.clone());
                }
            }
            tx.send_if_modified(|current| {
                let changed = *current != routes;
                if changed {
                    *current = routes;
                }
                changed
            });
        }
    });
    rx
}

/// The routes a model card in `namespaces` declares, with its worker.
fn declared_routes(
    card: DiscoveryInstance,
    namespaces: &NamespaceFilter,
    mode: NamespacePrefixMode,
) -> Option<Declared> {
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
    let routes = card_json
        .get("runtime_config")?
        .get("runtime_data")?
        .get(HTTP_ROUTES_KEY)?;
    let url = match upstream_base(routes.get("url")?.as_str()?) {
        Ok(url) => url,
        Err(err) => {
            tracing::warn!(instance_id, "ignoring {HTTP_ROUTES_KEY} whose url {err}");
            return None;
        }
    };
    let mut prefixes = Vec::new();
    for prefix in routes.get("prefixes")?.as_array()? {
        match prefix.as_str().and_then(declared_prefix) {
            Some(prefix) => prefixes.push(prefix),
            None => {
                tracing::warn!(instance_id, %prefix, "ignoring invalid {HTTP_ROUTES_KEY} prefix")
            }
        }
    }
    Some(Declared {
        worker: instance_id,
        url,
        prefixes,
    })
}

/// `prefix` without its trailing `/`, when it is one or more segments of
/// `A-Z a-z 0-9 - . _ ~`, none of them `.` or `..`.
fn declared_prefix(prefix: &str) -> Option<String> {
    let prefix = prefix.trim_end_matches('/');
    prefix
        .strip_prefix('/')?
        .split('/')
        .all(|segment| !matches!(segment, "" | "." | "..") && segment.bytes().all(unreserved))
        .then(|| prefix.to_string())
}

/// Whether `path` may be forwarded: paths with backslashes or dot segments,
/// including those separated by an encoded `/` or `\`, are not, since
/// normalization here or on the worker would let them escape their prefix.
/// Other encoded bytes are forwarded as sent.
fn forwardable(path: &str) -> bool {
    let separated = path
        .to_ascii_lowercase()
        .replace("%2f", "/")
        .replace("%5c", "/");
    !path.contains('\\')
        && !separated
            .split('/')
            .any(|segment| matches!(segment, "." | ".." | "%2e" | "%2e%2e" | ".%2e" | "%2e."))
}

/// Whether `path` is `prefix` or under it, by whole segments.
fn under(path: &str, prefix: &str) -> bool {
    path.strip_prefix(prefix)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// The longest id of a resource under `prefix`: its key, `PREFIX/{id}`, must
/// fit the session affinity table.
fn max_id_len(prefix: &str) -> usize {
    MAX_SESSION_AFFINITY_ID_BYTES.saturating_sub(prefix.len() + 1)
}

/// An id usable verbatim as one path segment, whose key fits the table.
fn valid_id(prefix: &str, id: &str) -> bool {
    (1..=max_id_len(prefix)).contains(&id.len())
        && !matches!(id, "." | "..")
        && id.bytes().all(unreserved)
}

fn unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
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
            tracing::warn!(%url, status = %response.status(), "backend route lookup got no answer");
            None
        }
        Err(err) => {
            tracing::warn!(%url, error = %err, "backend route lookup failed");
            None
        }
    }
}

fn no_workers() -> Response {
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "no worker serves this route",
    )
}

fn no_such_resource() -> Response {
    error_response(StatusCode::NOT_FOUND, "no worker holds this resource")
}

#[cfg(test)]
mod tests;
