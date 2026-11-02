// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::decision::DispatchFence;
use dynamo_runtime::{
    component::{Client, Instance},
    discovery::EndpointInstanceId,
    engine::Data,
    pipeline::{
        AddressedPushRouter, AddressedRequest, Context, Error, ManyIn, ManyOut, SingleIn,
        StreamingDispatch,
    },
    protocols::maybe_error::MaybeError,
};
use serde::{Deserialize, Serialize};

const DISPATCH_GUARD: &str = "dynamo.decision.dispatch_guard";

#[derive(Clone, Default)]
pub(crate) struct DispatchReceipt(Arc<AtomicBool>);

impl DispatchReceipt {
    pub(crate) fn is_validated(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

struct GuardedDispatch {
    fence: Arc<DispatchFence>,
    receipt: DispatchReceipt,
}

impl GuardedDispatch {
    fn validate(&self, worker: u64) -> Result<(), Error> {
        self.fence
            .validate(Some(worker))
            .map_err(|_| unavailable_dispatch())?;
        self.receipt.0.store(true, Ordering::Release);
        Ok(())
    }
}

pub(super) fn attach_dispatch_guard<T: Data>(
    request: &mut Context<T>,
    fence: Arc<DispatchFence>,
    receipt: DispatchReceipt,
) {
    request.insert(DISPATCH_GUARD, GuardedDispatch { fence, receipt });
}

#[cfg(test)]
pub(crate) fn validate_fixture_dispatch<T: Data>(request: &Context<T>, worker: u64) {
    request
        .get::<GuardedDispatch>(DISPATCH_GUARD)
        .expect("fixture scoring request must carry a guard")
        .validate(worker)
        .expect("fixture dispatch must be admitted");
}

fn unavailable_dispatch() -> Error {
    dynamo_runtime::error::DynamoError::builder()
        .error_type(dynamo_runtime::error::ErrorType::Unavailable)
        .message("selected decision qualification is unavailable")
        .build()
        .into()
}

#[derive(Clone, Copy)]
pub(crate) enum GuardPolicy {
    Required,
    Optional,
}

pub(crate) struct DecisionDispatch<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    inner: Arc<dyn StreamingDispatch<T, U>>,
    guard: GuardPolicy,
}

impl<T, U> DecisionDispatch<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    pub(crate) fn with_dispatch(
        inner: Arc<dyn StreamingDispatch<T, U>>,
        guard: GuardPolicy,
    ) -> Arc<Self> {
        Arc::new(Self { inner, guard })
    }

    pub(crate) async fn for_client(
        client: &Client,
        guard: GuardPolicy,
    ) -> anyhow::Result<Arc<Self>> {
        Ok(Self::with_dispatch(
            AddressedPushRouter::from_runtime_provider(&client.endpoint).await?,
            guard,
        ))
    }
}

#[async_trait::async_trait]
impl<T, U> StreamingDispatch<T, U> for DecisionDispatch<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
        if matches!(self.guard, GuardPolicy::Optional)
            && request
                .get_optional::<GuardedDispatch>(DISPATCH_GUARD)
                .map_err(|_| unavailable_dispatch())?
                .is_none()
        {
            return self.inner.generate(request).await;
        }
        let timeout = request.establish_timeout();
        let (addressed, envelope) = request.transfer(());
        let (wire, address, instance) = addressed.into_parts();
        let instance = instance.ok_or_else(unavailable_dispatch)?;
        // This runs after worker selection, even when its asynchronous lookup was paused.
        let guard = envelope
            .get::<GuardedDispatch>(DISPATCH_GUARD)
            .map_err(|_| unavailable_dispatch())?;
        guard.validate(instance.id())?;
        let addressed = AddressedRequest::with_instance(wire, address, instance)
            .with_establish_timeout(timeout);
        self.inner.generate(envelope.map(|()| addressed)).await
    }

    async fn generate_bidirectional(
        &self,
        instance: Instance,
        address: String,
        input: ManyIn<T>,
    ) -> Result<ManyOut<U>, Error> {
        if matches!(self.guard, GuardPolicy::Optional)
            && input
                .get_optional::<GuardedDispatch>(DISPATCH_GUARD)
                .map_err(|_| unavailable_dispatch())?
                .is_none()
        {
            return self
                .inner
                .generate_bidirectional(instance, address, input)
                .await;
        }
        Err(unavailable_dispatch())
    }

    async fn on_instance_removed(&self, id: &EndpointInstanceId) {
        self.inner.on_instance_removed(id).await;
    }

    async fn on_instance_added(&self, id: &EndpointInstanceId) {
        self.inner.on_instance_added(id).await;
    }
}
