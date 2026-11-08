// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use parking_lot::Mutex;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Context as _;
use tokio::sync::OnceCell;

use super::{
    DecisionDispatch, DecisionEngine, SystemOneSelectionError, TokenExecutor,
    allocator::AllocatorTrimOnDrop, decision_policy::DecisionCapabilities,
    worker_set::retain_teardown_until_requests_finish,
};
use crate::{
    decision_profile::{DecisionProfileKind, NativeExecutorKind},
    kv_router::RoutingLoadContext,
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::{
        decision_head::DecisionHeadStreamingEngine, decision_read::DecisionReadStreamingEngine,
    },
    tokenizers::Tokenizer,
    types::openai::generate::GenerateStreamingEngine,
};
use dynamo_runtime::{
    engine::Data,
    pipeline::{
        AsyncEngine, Error, ManyOut, PushRouterOptions, RouterMode, SingleIn,
        network::egress::push_router::PushRouter,
    },
    protocols::maybe_error::MaybeError,
};
use serde::{Serialize, de::DeserializeOwned};
use tokio_util::sync::CancellationToken;

type NativeEngine<T, U> = Arc<dyn AsyncEngine<SingleIn<T>, ManyOut<U>, Error>>;

pub(crate) struct TokenResources {
    pub(crate) engine: GenerateStreamingEngine,
    pub(crate) preprocessor: Arc<OpenAIPreprocessor>,
}

#[derive(Default)]
struct TokenResourceCell {
    state: Mutex<TokenInitState>,
}

const TOKEN_RETRY_DELAY: Duration = Duration::from_secs(1);

#[derive(Default)]
enum TokenInitState {
    #[default]
    Cold,
    Ready(Arc<TokenResources>),
    PermanentFailure(Arc<anyhow::Error>),
    Backoff {
        until: Instant,
        error: Arc<anyhow::Error>,
    },
}

pub(crate) fn token_error_is_retryable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .find_map(|cause| {
            let kind = cause
                .downcast_ref::<std::io::Error>()
                .map(std::io::Error::kind)
                .or_else(|| {
                    cause
                        .downcast_ref::<serde_json::Error>()
                        .and_then(serde_json::Error::io_error_kind)
                });
            kind.map(|kind| {
                !matches!(
                    kind,
                    std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput
                )
            })
            .or_else(|| {
                cause
                    .is::<crate::model_card::TokenizerLoadUnavailable>()
                    .then_some(true)
            })
        })
        .unwrap_or(false)
}

impl TokenResourceCell {
    fn get(&self) -> Option<Arc<TokenResources>> {
        match &*self.state.try_lock()? {
            TokenInitState::Ready(token) => Some(token.clone()),
            _ => None,
        }
    }

    #[cfg(test)]
    fn initialized(value: Arc<TokenResources>) -> Self {
        Self {
            state: Mutex::new(TokenInitState::Ready(value)),
        }
    }

    fn get_or_init(
        &self,
        build: impl FnOnce() -> anyhow::Result<Arc<TokenResources>>,
    ) -> Result<Arc<TokenResources>, Arc<anyhow::Error>> {
        // The admitted blocking owner retains its permit until initialization settles.
        let mut state = self.state.lock();
        match &*state {
            TokenInitState::Ready(token) => return Ok(token.clone()),
            TokenInitState::PermanentFailure(error) => return Err(error.clone()),
            TokenInitState::Backoff { until, error } if Instant::now() < *until => {
                return Err(error.clone());
            }
            TokenInitState::Cold | TokenInitState::Backoff { .. } => {}
        }
        match build() {
            Ok(token) => {
                *state = TokenInitState::Ready(token.clone());
                Ok(token)
            }
            Err(cause) => {
                let error = Arc::new(cause);
                *state = if token_error_is_retryable(&error) {
                    TokenInitState::Backoff {
                        until: Instant::now() + TOKEN_RETRY_DELAY,
                        error: error.clone(),
                    }
                } else {
                    TokenInitState::PermanentFailure(error.clone())
                };
                Err(error)
            }
        }
    }

    fn has_failed(&self) -> bool {
        self.state
            .try_lock()
            .is_some_and(|state| matches!(*state, TokenInitState::PermanentFailure(_)))
    }

    fn is_ready(&self) -> bool {
        self.state
            .try_lock()
            .is_some_and(|state| matches!(*state, TokenInitState::Ready(_)))
    }
}

pub(crate) struct ScoreFactory {
    engine: GenerateStreamingEngine,
    source: TokenizerLoad,
}

impl ScoreFactory {
    pub(crate) fn new(engine: GenerateStreamingEngine, source: TokenizerLoad) -> Self {
        Self { engine, source }
    }
}

pub(crate) enum TokenizerLoad {
    Loaded(Tokenizer),
    Card,
}

struct DecisionFactory {
    card: ModelDeploymentCard,
    score: Option<ScoreFactory>,
    load: Option<Arc<RoutingLoadContext>>,
}

impl DecisionFactory {
    fn has_inputs(&self) -> bool {
        self.score.is_some() || self.load.is_some()
    }

    fn token(
        &self,
        build: impl FnOnce(
            &ModelDeploymentCard,
            Tokenizer,
            &CancellationToken,
        ) -> anyhow::Result<Arc<OpenAIPreprocessor>>,
    ) -> anyhow::Result<TokenResources> {
        let score = self
            .score
            .as_ref()
            .context("token scoring route is unavailable")?;
        let tokenizer = match &score.source {
            TokenizerLoad::Loaded(tokenizer) => tokenizer.clone(),
            TokenizerLoad::Card => self.card.tokenizer().context("decision tokenizer")?,
        };
        let load = self.load.as_ref().context("decision load context")?;
        let preprocessor = build(&self.card, tokenizer, &load.cancellation_token())?;
        Ok(TokenResources {
            engine: score.engine.clone(),
            preprocessor,
        })
    }
}

pub(crate) struct DecisionResources {
    capabilities: DecisionCapabilities,
    token: TokenResourceCell,
    head: OnceCell<DecisionHeadStreamingEngine>,
    read: OnceCell<DecisionReadStreamingEngine>,
    factory: Option<DecisionFactory>,
    teardown: Arc<AllocatorTrimOnDrop>,
}

fn retain_token(token: TokenResources, teardown: &Arc<AllocatorTrimOnDrop>) -> TokenResources {
    TokenResources {
        engine: retain_teardown_until_requests_finish(token.engine, teardown),
        preprocessor: token.preprocessor,
    }
}

impl Default for DecisionResources {
    fn default() -> Self {
        Self::new(Arc::new(AllocatorTrimOnDrop::new()))
    }
}

impl DecisionResources {
    pub(crate) fn planned(
        card: &ModelDeploymentCard,
        mode: RouterMode,
        enabled: bool,
        teardown: Arc<AllocatorTrimOnDrop>,
    ) -> Option<Self> {
        let capabilities = DecisionCapabilities::for_card(card, mode, enabled);
        capabilities
            .kinds
            .any()
            .then(|| Self::with_capabilities(card.clone(), capabilities, teardown))
    }

    #[cfg(test)]
    pub(crate) fn for_worker(
        card: ModelDeploymentCard,
        score: Option<ScoreFactory>,
        load: Option<Arc<RoutingLoadContext>>,
        mode: RouterMode,
        teardown: Arc<AllocatorTrimOnDrop>,
    ) -> Self {
        let capabilities = DecisionCapabilities::for_card(&card, mode, true);
        Self::with_capabilities(card, capabilities, teardown).with_inputs(score, load)
    }

    fn with_capabilities(
        card: ModelDeploymentCard,
        capabilities: DecisionCapabilities,
        teardown: Arc<AllocatorTrimOnDrop>,
    ) -> Self {
        Self {
            capabilities,
            factory: Some(DecisionFactory {
                card,
                score: None,
                load: None,
            }),
            ..Self::new(teardown)
        }
    }

    pub(crate) fn new(teardown: Arc<AllocatorTrimOnDrop>) -> Self {
        Self {
            capabilities: Default::default(),
            token: Default::default(),
            head: OnceCell::new(),
            read: OnceCell::new(),
            factory: None,
            teardown,
        }
    }

    pub(crate) fn capabilities(&self) -> DecisionCapabilities {
        self.capabilities
    }

    pub(crate) fn with_inputs(
        mut self,
        score: Option<ScoreFactory>,
        load: Option<Arc<RoutingLoadContext>>,
    ) -> Self {
        self.capabilities = self
            .capabilities
            .with_inputs(score.is_some(), load.is_some());
        if let Some(factory) = &mut self.factory {
            factory.score = score;
            factory.load = load;
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn insert_token(&mut self, token: TokenResources) {
        self.token = TokenResourceCell::initialized(Arc::new(retain_token(token, &self.teardown)));
    }
    #[cfg(test)]
    pub(crate) fn insert_head(&mut self, engine: DecisionHeadStreamingEngine) {
        self.head = OnceCell::new_with(Some(retain_teardown_until_requests_finish(
            engine,
            &self.teardown,
        )));
    }
    #[cfg(test)]
    pub(crate) fn insert_read(&mut self, engine: DecisionReadStreamingEngine) {
        self.read = OnceCell::new_with(Some(retain_teardown_until_requests_finish(
            engine,
            &self.teardown,
        )));
    }
    pub(crate) fn is_empty(&self) -> bool {
        !self.token.is_ready() && self.head.get().is_none() && self.read.get().is_none()
    }

    pub(crate) fn is_prepared(&self, kind: DecisionProfileKind) -> bool {
        match kind {
            DecisionProfileKind::Token(_) => self.token.is_ready(),
            DecisionProfileKind::Native(NativeExecutorKind::Head) => self.head.get().is_some(),
            DecisionProfileKind::Native(NativeExecutorKind::Read) => self.read.get().is_some(),
        }
    }

    pub(crate) fn supports(&self, kind: DecisionProfileKind) -> bool {
        if matches!(kind, DecisionProfileKind::Token(_)) && self.token.has_failed() {
            return false;
        }
        self.is_prepared(kind)
            || self
                .factory
                .as_ref()
                .is_some_and(|factory| factory.has_inputs())
                && self.capabilities.kinds.supports(kind)
    }

    fn has_supported_kind(&self) -> bool {
        !self.is_empty()
            || self
                .factory
                .as_ref()
                .is_some_and(|factory| factory.has_inputs())
                && self.capabilities.kinds.any()
    }

    pub(crate) fn into_shared(self) -> Option<Arc<Self>> {
        self.has_supported_kind().then(|| Arc::new(self))
    }

    pub(crate) fn prepared_token(&self) -> Option<Arc<TokenResources>> {
        self.token.get()
    }

    pub(crate) fn prepare_token(&self) -> anyhow::Result<Arc<TokenResources>> {
        self.prepare_token_with(super::watcher::worker_set_chat_preprocessor)
    }

    fn prepare_token_with(
        &self,
        build: impl FnOnce(
            &ModelDeploymentCard,
            Tokenizer,
            &CancellationToken,
        ) -> anyhow::Result<Arc<OpenAIPreprocessor>>,
    ) -> anyhow::Result<Arc<TokenResources>> {
        Ok(self.token.get_or_init(|| {
            let result = self
                .factory
                .as_ref()
                .context("token resource factory is unavailable")?
                .token(build)
                .map(|token| Arc::new(retain_token(token, &self.teardown)));
            if let Err(cause) = &result {
                tracing::warn!(error = %format!("{cause:#}"), retryable = token_error_is_retryable(cause), "Decision token configuration is unavailable");
            }
            result
        }).map_err(|cause| if token_error_is_retryable(&cause) { SystemOneSelectionError::Unavailable } else { SystemOneSelectionError::Unsupported })?)
    }

    #[cfg(test)]
    pub(crate) fn initialize_token_for_test(
        &self,
        build: impl FnOnce() -> anyhow::Result<Arc<TokenResources>>,
    ) -> Result<Arc<TokenResources>, Arc<anyhow::Error>> {
        self.token.get_or_init(build)
    }

    #[cfg(test)]
    pub(crate) fn token_failed_for_test(&self) -> bool {
        self.token.has_failed()
    }

    #[cfg(test)]
    pub(crate) fn score_engine_for_test(&self) -> Option<&GenerateStreamingEngine> {
        self.factory
            .as_ref()?
            .score
            .as_ref()
            .map(|score| &score.engine)
    }

    #[cfg(test)]
    pub(crate) fn load_context_for_test(&self) -> Option<&Arc<RoutingLoadContext>> {
        self.factory.as_ref()?.load.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn teardown_guard_for_test(&self) -> &Arc<AllocatorTrimOnDrop> {
        &self.teardown
    }

    pub(crate) async fn prepare_native(&self, kind: DecisionProfileKind) -> anyhow::Result<()> {
        if !self.supports(kind) {
            return Err(SystemOneSelectionError::Unsupported.into());
        }
        let DecisionProfileKind::Native(kind) = kind else {
            return Ok(());
        };
        match kind {
            NativeExecutorKind::Head => self.native_router(&self.head).await,
            NativeExecutorKind::Read => self.native_router(&self.read).await,
        }
    }

    async fn native_router<T, U>(&self, cell: &OnceCell<NativeEngine<T, U>>) -> anyhow::Result<()>
    where
        T: Data + Serialize,
        U: Data + DeserializeOwned + MaybeError,
    {
        cell.get_or_try_init(|| async {
            let factory = self.factory.as_ref().context("native resource factory")?;
            let client = factory
                .load
                .as_ref()
                .context("native load context")?
                .client();
            let router = PushRouter::<T, U>::from_client_with_options(
                client.clone(),
                self.capabilities.mode,
                PushRouterOptions {
                    dispatch: Some(
                        DecisionDispatch::for_client(client, super::GuardPolicy::Required).await?,
                    ),
                    ..crate::kv_router::plugins::push_router_options()
                },
            )
            .await?;
            Ok::<_, anyhow::Error>(retain_teardown_until_requests_finish(
                Arc::new(router),
                &self.teardown,
            ))
        })
        .await?;
        Ok(())
    }

    pub(crate) fn prepare(&self, kind: DecisionProfileKind) -> anyhow::Result<DecisionEngine> {
        if !self.supports(kind) {
            return Err(SystemOneSelectionError::Unsupported.into());
        }
        Ok(match kind {
            DecisionProfileKind::Token(backend) => {
                DecisionEngine::Token(TokenExecutor::new(backend, self.prepare_token()?))
            }
            DecisionProfileKind::Native(NativeExecutorKind::Head) => DecisionEngine::Head(
                self.head
                    .get()
                    .context("decision head resource is unavailable")?
                    .clone(),
            ),
            DecisionProfileKind::Native(NativeExecutorKind::Read) => DecisionEngine::Read(
                self.read
                    .get()
                    .context("decision read resource is unavailable")?
                    .clone(),
            ),
        })
    }
}

#[cfg(test)]
#[path = "decision_resources/tests.rs"]
mod tests;
