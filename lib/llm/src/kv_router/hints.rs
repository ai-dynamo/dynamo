// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-call pins: when a response ends in a tool call, soft-pin the session's pages in the
//! Mooncake master (`DYN_KVM_PIN_ENDPOINT`, its admin HTTP address) so the store keeps them
//! until the session's next turn.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::http::HeaderMap;
use dynamo_kv_hints::pin::{
    MooncakeObjectLayout, ToolCallPinSlot, parse_sim_tool_call, pin_ttl_and_priority,
    soft_pin_requests,
};
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::pipeline::SingleIn;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::discovery::RuntimeConfigWatch;
use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

pub const PIN_SLOT_CONTEXT_KEY: &str = "kv_hint_pin_slot";
pub const SIM_TOOL_CALL_CONTEXT_KEY: &str = "kv_hint_sim_tool_call";
/// Runtime-config key of a worker's Mooncake objects, published by the SGLang sidecar.
pub const MOONCAKE_OBJECTS_RUNTIME_KEY: &str = "sglang_mooncake_objects";

/// What a worker stores a page as in Mooncake: SGLang's `hicache_object_layout`, and what
/// names its pages.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MooncakeObjects {
    pub page_size: u32,
    pub is_eagle: bool,
    pub layout: MooncakeObjectLayout,
}

/// Request-context value: a tool call follows the response, expected to take this long.
pub struct SimToolCall(pub Option<Duration>);

pub fn pin_endpoint() -> Option<&'static str> {
    static ENDPOINT: OnceLock<Option<String>> = OnceLock::new();
    ENDPOINT
        .get_or_init(|| std::env::var(env_llm::DYN_KVM_PIN_ENDPOINT).ok())
        .as_deref()
        .filter(|endpoint| !endpoint.is_empty())
}

pub fn sim_tool_call_from_headers(headers: &HeaderMap) -> Option<SimToolCall> {
    let value = headers
        .get("x-dynamo-experimental-sim-tool-call")?
        .to_str()
        .ok()?;
    pin_endpoint().map(|_| SimToolCall(parse_sim_tool_call(value)))
}

/// One router's pins: its workers' object layouts and the master that holds their pages.
pub struct KvHints {
    store: Option<StorePins>,
    configs: RuntimeConfigWatch,
}

/// The Mooncake master's `/api/v1/soft_pin`.
struct StorePins {
    client: reqwest::Client,
    url: String,
}

impl StorePins {
    /// Sends each request from a background task, so a pin never delays a response; a pin
    /// that fails is only logged.
    fn pin(&self, tenant: &str, keys: &[String], ttl: Duration, priority: u32) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        for request in soft_pin_requests(tenant, keys, ttl, priority) {
            let Ok(body) = serde_json::to_vec(&request) else {
                continue;
            };
            let send = self
                .client
                .post(&self.url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body)
                .send();
            runtime.spawn(async move {
                if let Err(error) = send.await.and_then(|response| response.error_for_status()) {
                    tracing::debug!(%error, "pin dropped");
                }
            });
        }
    }
}

impl KvHints {
    pub fn new(configs: RuntimeConfigWatch) -> Arc<Self> {
        let store = pin_endpoint().map(|endpoint| StorePins {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
            url: format!("{}/api/v1/soft_pin", endpoint.trim_end_matches('/')),
        });
        Arc::new(Self { store, configs })
    }

    /// `request` was routed to `worker`: arm its pin.
    pub(crate) fn on_route(
        self: &Arc<Self>,
        request: &SingleIn<PreprocessedRequest>,
        worker: WorkerWithDpRank,
    ) {
        let Ok(slot) = request.get::<ToolCallPinSlot>(PIN_SLOT_CONTEXT_KEY) else {
            return;
        };
        let hints = Arc::clone(self);
        let prompt = request.token_ids.clone();
        slot.arm(move |output, expected| {
            let (Some(store), Some(objects)) = (&hints.store, hints.objects(worker)) else {
                return;
            };
            let tokens: Vec<u32> = prompt.iter().chain(output).copied().collect();
            let pages = super::shared_cache::logical_page_hashes(
                &tokens,
                objects.page_size,
                objects.is_eagle,
            );
            let (ttl, priority) = pin_ttl_and_priority(expected, tokens.len());
            // SGLang's default tenant: it does not publish the one it stores under.
            store.pin(
                "default",
                &objects.layout.object_keys(&pages),
                ttl,
                priority,
            );
        });
    }

    fn objects(&self, worker: WorkerWithDpRank) -> Option<MooncakeObjects> {
        let configs = self.configs.borrow();
        let config = configs.get(&worker.worker_id)?;
        config
            .get_engine_specific(MOONCAKE_OBJECTS_RUNTIME_KEY)
            .ok()
            .flatten()
    }
}

type Boxed<T> = std::pin::Pin<Box<dyn Stream<Item = Annotated<T>> + Send>>;

/// Record the engine's output tokens in `slot`.
pub fn record_output<S>(stream: S, slot: Option<Arc<ToolCallPinSlot>>) -> Boxed<BackendOutput>
where
    S: Stream<Item = Annotated<BackendOutput>> + Send + 'static,
{
    let Some(slot) = slot else {
        return Box::pin(stream);
    };
    Box::pin(stream.inspect(move |response| {
        if let Some(output) = &response.data {
            slot.push_output(&output.token_ids);
        }
    }))
}

/// Record tool calls in `slot` and finish it when the response ends.
pub fn pin_on_tool_call<S>(
    stream: S,
    slot: Option<Arc<ToolCallPinSlot>>,
    announced: Option<Option<Duration>>,
) -> Boxed<NvCreateChatCompletionStreamResponse>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let Some(slot) = slot else {
        return Box::pin(stream);
    };
    let seen = Arc::clone(&slot);
    let chunks = stream.inspect(move |response| {
        let choices = response.data.iter().flat_map(|chunk| &chunk.inner.choices);
        if choices
            .into_iter()
            .any(|c| matches!(&c.delta.tool_calls, Some(t) if !t.is_empty()))
        {
            seen.tool_called();
        }
    });
    let end = futures::stream::once(async move { slot.finish(announced) });
    Box::pin(chunks.chain(end.filter_map(|()| async { None })))
}
