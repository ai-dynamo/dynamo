// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateful `/v1/responses` support for the in-process Dynamo endpoint.

use std::env;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use dynamo_protocols::types::responses::InputParam;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::{
    key_value_store::{KeyValueStore, KeyValueStoreConfig},
    protocols::openai::responses::NvCreateResponse,
};

const DEFAULT_TTL_SECS: u64 = 30 * 24 * 60 * 60;
const DEFAULT_TTL_SWEEP_SECS: u64 = 60 * 60;
const DEFAULT_STORE_NAMESPACE: &str = "dynamo/stateful-responses";

const DEFAULT_TTL_SECS_ENV: &str = "DYN_STATEFUL_RESPONSES_DEFAULT_TTL_SECS";
const TTL_SWEEP_SECS_ENV: &str = "DYN_STATEFUL_RESPONSES_TTL_SWEEP_SECS";
const STORE_URL_ENV: &str = "DYN_STATEFUL_RESPONSES_STORE_URL";
const STORE_NAMESPACE_ENV: &str = "DYN_STATEFUL_RESPONSES_STORE_NAMESPACE";

fn is_falsey(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn duration_secs_from_env(
    name: &str,
    default: Option<Duration>,
) -> anyhow::Result<Option<Duration>> {
    let Ok(value) = env::var(name) else {
        return Ok(default);
    };

    parse_optional_duration_secs(name, &value)
}

fn store_settings_from_env() -> anyhow::Result<(Option<Duration>, Option<Duration>)> {
    Ok((
        duration_secs_from_env(
            DEFAULT_TTL_SECS_ENV,
            Some(Duration::from_secs(DEFAULT_TTL_SECS)),
        )?,
        duration_secs_from_env(
            TTL_SWEEP_SECS_ENV,
            Some(Duration::from_secs(DEFAULT_TTL_SWEEP_SECS)),
        )?,
    ))
}

fn parse_optional_duration_secs(name: &str, value: &str) -> anyhow::Result<Option<Duration>> {
    let value = value.trim();
    if value.is_empty()
        || is_falsey(value)
        || matches!(value.to_ascii_lowercase().as_str(), "none" | "never")
    {
        return Ok(None);
    }

    let secs = value
        .parse::<u64>()
        .map_err(|err| anyhow::anyhow!("invalid {name}: {err}"))?;
    Ok(Some(Duration::from_secs(secs)))
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct StoredResponseContext {
    pub input_items: Vec<Value>,
}

#[derive(Clone, Debug)]
pub struct ResponseContextStoreConfig {
    pub store_config: KeyValueStoreConfig,
    pub namespace: String,
    pub default_ttl: Option<Duration>,
    pub ttl_sweep_interval: Option<Duration>,
}

impl ResponseContextStoreConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let store_config = KeyValueStoreConfig::parse(
            &env::var(STORE_URL_ENV).unwrap_or_else(|_| "memory".to_string()),
        )?;
        let namespace =
            env::var(STORE_NAMESPACE_ENV).unwrap_or_else(|_| DEFAULT_STORE_NAMESPACE.to_string());
        if namespace.trim().is_empty() {
            anyhow::bail!("{STORE_NAMESPACE_ENV} must not be empty");
        }
        let (default_ttl, ttl_sweep_interval) = store_settings_from_env()?;

        Ok(Self {
            store_config,
            namespace,
            default_ttl,
            ttl_sweep_interval,
        })
    }
}

pub struct ResponseContextStoreManager {
    config: ResponseContextStoreConfig,
    store: OnceCell<Arc<dyn KeyValueStore>>,
    sweeper_started: AtomicBool,
    shutdown: CancellationToken,
}

impl ResponseContextStoreManager {
    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_env_with_shutdown(CancellationToken::new())
    }

    pub fn from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<Self> {
        Ok(Self::with_shutdown(
            ResponseContextStoreConfig::from_env()?,
            shutdown,
        ))
    }

    pub fn new(config: ResponseContextStoreConfig) -> Self {
        Self::with_shutdown(config, CancellationToken::new())
    }

    pub fn with_shutdown(config: ResponseContextStoreConfig, shutdown: CancellationToken) -> Self {
        Self {
            config,
            store: OnceCell::new(),
            sweeper_started: AtomicBool::new(false),
            shutdown: shutdown.child_token(),
        }
    }

    pub fn from_store(
        store: Arc<dyn KeyValueStore>,
        default_ttl: Option<Duration>,
        ttl_sweep_interval: Option<Duration>,
        shutdown: CancellationToken,
    ) -> Self {
        Self {
            config: ResponseContextStoreConfig {
                store_config: KeyValueStoreConfig::Memory,
                namespace: DEFAULT_STORE_NAMESPACE.to_string(),
                default_ttl,
                ttl_sweep_interval,
            },
            store: OnceCell::new_with(Some(store)),
            sweeper_started: AtomicBool::new(false),
            shutdown: shutdown.child_token(),
        }
    }

    pub async fn store(&self) -> anyhow::Result<Arc<dyn KeyValueStore>> {
        let store = self
            .store
            .get_or_try_init(|| async {
                self.config.store_config.open(&self.config.namespace).await
            })
            .await?
            .clone();

        if let Some(interval) = self.config.ttl_sweep_interval
            && self
                .sweeper_started
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            spawn_store_ttl_sweeper(store.clone(), interval, self.shutdown.child_token());
        }

        Ok(store)
    }

    pub async fn warmup(&self) -> anyhow::Result<()> {
        self.store().await.map(|_| ())
    }

    pub async fn prepare_request(
        &self,
        request: &mut NvCreateResponse,
    ) -> anyhow::Result<ExpandedResponseContext> {
        let previous_context = match request.inner.previous_response_id.as_deref() {
            Some(previous_response_id) => {
                let store = self.store().await?;
                Some(serde_json::from_slice::<StoredResponseContext>(
                    &store.get(previous_response_id).await?.ok_or_else(|| {
                        PreviousResponseNotFound(previous_response_id.to_string())
                    })?,
                )?)
            }
            None => None,
        };

        expand_typed_request(request, previous_context.as_ref())
    }

    pub async fn persist_response<T: Serialize>(
        &self,
        expanded: &ExpandedResponseContext,
        response_id: &str,
        output_items: &[T],
    ) -> anyhow::Result<()> {
        if !expanded.should_store {
            return Ok(());
        }

        let mut input_items = expanded.input_items.clone();
        for item in output_items {
            input_items.push(serde_json::to_value(item)?);
        }
        let context = serde_json::to_vec(&StoredResponseContext { input_items })?;
        self.store()
            .await?
            .put(response_id, &context, self.config.default_ttl)
            .await?;
        Ok(())
    }

    pub async fn delete(&self, response_id: &str) -> anyhow::Result<bool> {
        self.store().await?.delete(response_id).await
    }
}

impl Drop for ResponseContextStoreManager {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedResponseContext {
    pub input_items: Vec<Value>,
    pub should_store: bool,
}

#[derive(Debug, Error)]
#[error("`previous_response_id` `{0}` was not found")]
pub struct PreviousResponseNotFound(pub String);

fn spawn_store_ttl_sweeper(
    store: Arc<dyn KeyValueStore>,
    interval: Duration,
    shutdown: CancellationToken,
) {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = tokio::time::sleep(interval) => {
                    if let Err(err) = store.purge_expired().await {
                        tracing::warn!(%err, "failed to purge expired stateful Responses contexts");
                    }
                }
            }
        }
    });
}

fn expand_typed_request(
    request: &mut NvCreateResponse,
    previous_context: Option<&StoredResponseContext>,
) -> anyhow::Result<ExpandedResponseContext> {
    let current_items = typed_input_as_items(&request.inner.input)?;
    let mut input_items = Vec::new();
    if let Some(context) = previous_context {
        input_items.extend(context.input_items.iter().cloned());
    }
    input_items.extend(current_items);

    request.inner.input = serde_json::from_value(Value::Array(input_items.clone()))?;

    let should_store = request.inner.store.unwrap_or(true);
    request.inner.store = Some(should_store);

    Ok(ExpandedResponseContext {
        input_items,
        should_store,
    })
}

fn typed_input_as_items(input: &InputParam) -> Result<Vec<Value>, serde_json::Error> {
    match input {
        InputParam::Text(text) => Ok(vec![message_item("user", "input_text", text)]),
        InputParam::Items(items) => items
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>(),
    }
}

fn message_item(role: &str, content_type: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": [{ "type": content_type, "text": text }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        key_value_store::{MemoryKeyValueStore, test_store_contract},
        protocols::common::extensions::NvExt,
    };
    use async_trait::async_trait;
    use dynamo_protocols::types::responses::{
        AssistantRole, FunctionToolCall, InputItem, Item, MessageItem, OutputItem, OutputMessage,
        OutputMessageContent, OutputStatus, OutputTextContent,
    };

    fn manager() -> ResponseContextStoreManager {
        ResponseContextStoreManager::new(ResponseContextStoreConfig {
            store_config: KeyValueStoreConfig::Memory,
            namespace: DEFAULT_STORE_NAMESPACE.to_string(),
            default_ttl: None,
            ttl_sweep_interval: None,
        })
    }

    fn context(input_items: Vec<Value>) -> StoredResponseContext {
        StoredResponseContext { input_items }
    }

    fn context_bytes(input_items: Vec<Value>) -> Vec<u8> {
        serde_json::to_vec(&context(input_items)).unwrap()
    }

    fn text_request(text: &str) -> NvCreateResponse {
        NvCreateResponse {
            extensions: Default::default(),
            inner: dynamo_protocols::types::responses::CreateResponse {
                input: InputParam::Text(text.into()),
                model: Some("m".into()),
                ..Default::default()
            },
            nvext: None,
        }
    }

    #[tokio::test]
    async fn memory_store_round_trips_expires_and_deletes_bytes() {
        assert_eq!(
            parse_optional_duration_secs("ttl", "60").unwrap(),
            Some(Duration::from_secs(60))
        );
        assert_eq!(parse_optional_duration_secs("ttl", "0").unwrap(), None);
        assert_eq!(parse_optional_duration_secs("ttl", "never").unwrap(), None);

        test_store_contract(&MemoryKeyValueStore::new()).await;
    }

    #[tokio::test]
    async fn manager_expands_typed_responses_request_without_opening_store() {
        let manager = manager();
        let mut request = text_request("hello");
        request.nvext = Some(
            NvExt::builder()
                .token_data(vec![1, 2, 3])
                .extra_fields(vec!["worker_id".to_string()])
                .build()
                .unwrap(),
        );

        let expanded = manager.prepare_request(&mut request).await.unwrap();

        assert!(expanded.should_store);
        assert_eq!(expanded.input_items[0]["role"], "user");
        assert_eq!(request.inner.store, Some(true));
        let nvext = request.nvext.as_ref().unwrap();
        assert_eq!(nvext.token_data.as_deref(), Some(&[1, 2, 3][..]));
        assert_eq!(
            nvext.extra_fields.as_ref().unwrap(),
            &vec!["worker_id".to_string()]
        );
        assert!(manager.store.get().is_none());
    }

    #[test]
    fn stored_output_items_round_trip_as_followup_input_items() {
        let output_items = vec![
            OutputItem::Message(OutputMessage {
                id: "msg_1".into(),
                role: AssistantRole::Assistant,
                status: OutputStatus::Completed,
                phase: None,
                content: vec![OutputMessageContent::OutputText(OutputTextContent {
                    text: "hello".into(),
                    annotations: vec![],
                    logprobs: Some(vec![]),
                })],
            }),
            OutputItem::FunctionCall(FunctionToolCall {
                arguments: "{}".into(),
                call_id: "call_1".into(),
                namespace: None,
                name: "tool".into(),
                id: Some("fc_1".into()),
                status: Some(OutputStatus::Completed),
            }),
        ];
        let stored = output_items
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let InputParam::Items(items) = serde_json::from_value(Value::Array(stored)).unwrap() else {
            panic!("expected item input");
        };
        assert!(matches!(
            &items[0],
            InputItem::Item(Item::Message(MessageItem::Output(_)))
        ));
        assert!(matches!(&items[1], InputItem::Item(Item::FunctionCall(_))));
    }

    #[tokio::test]
    async fn manager_expands_previous_response_and_honors_store_false() {
        let manager = manager();
        let store = manager.store().await.unwrap();
        store
            .put(
                "resp_1",
                &context_bytes(vec![message_item("assistant", "output_text", "old")]),
                None,
            )
            .await
            .unwrap();

        let mut request = text_request("new");
        request.inner.previous_response_id = Some("resp_1".into());
        request.inner.store = Some(false);

        let expanded = manager.prepare_request(&mut request).await.unwrap();
        manager
            .persist_response(&expanded, "resp_2", &[] as &[Value])
            .await
            .unwrap();

        assert!(!expanded.should_store);
        assert_eq!(expanded.input_items.len(), 2);
        assert_eq!(store.get("resp_2").await.unwrap(), None);

        store
            .put(
                "resp_expired",
                &context_bytes(vec![json!({"type": "message"})]),
                Some(Duration::ZERO),
            )
            .await
            .unwrap();
        request.inner.previous_response_id = Some("resp_expired".into());
        let err = manager.prepare_request(&mut request).await.unwrap_err();
        assert!(err.downcast_ref::<PreviousResponseNotFound>().is_some());
    }

    struct FailingStore;

    #[async_trait]
    impl KeyValueStore for FailingStore {
        async fn get(&self, _key: &str) -> anyhow::Result<Option<Vec<u8>>> {
            Ok(None)
        }

        async fn put(
            &self,
            _key: &str,
            _value: &[u8],
            _ttl: Option<Duration>,
        ) -> anyhow::Result<()> {
            anyhow::bail!("store failed")
        }

        async fn delete(&self, _key: &str) -> anyhow::Result<bool> {
            Ok(false)
        }
    }

    #[tokio::test]
    async fn manager_skips_store_failures_when_store_is_false() {
        let manager = ResponseContextStoreManager::from_store(
            Arc::new(FailingStore),
            None,
            None,
            CancellationToken::new(),
        );

        assert!(
            manager
                .persist_response(
                    &ExpandedResponseContext {
                        input_items: Vec::new(),
                        should_store: true,
                    },
                    "resp_fail",
                    &[] as &[Value],
                )
                .await
                .is_err()
        );
        manager
            .persist_response(
                &ExpandedResponseContext {
                    input_items: Vec::new(),
                    should_store: false,
                },
                "resp_skip",
                &[] as &[Value],
            )
            .await
            .unwrap();
    }
}
