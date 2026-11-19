// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{
    component::Instance,
    engine::Data,
    pipeline::{AddressedRequest, ManyOut, SingleIn, StreamingDispatch},
    protocols::maybe_error::MaybeError,
};

/// The exact producer transport retained by a successful dispatch. This route
/// only releases existing resources; it must never admit new inference.
pub struct CleanupRoute<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    dispatch: Arc<dyn StreamingDispatch<T, U>>,
    address: String,
    worker_id: u64,
    establish_timeout: Option<Duration>,
}

impl<T, U> Clone for CleanupRoute<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    fn clone(&self) -> Self {
        Self {
            dispatch: self.dispatch.clone(),
            address: self.address.clone(),
            worker_id: self.worker_id,
            establish_timeout: self.establish_timeout,
        }
    }
}

impl<T, U> CleanupRoute<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    pub(super) fn new(
        dispatch: Arc<dyn StreamingDispatch<T, U>>,
        instance: &Instance,
        establish_timeout: Option<Duration>,
    ) -> Self {
        Self {
            dispatch,
            address: instance.transport.address().to_owned(),
            worker_id: instance.id(),
            establish_timeout,
        }
    }

    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub async fn dispatch(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        // No discovery association: removal must not cancel an old allocation's
        // cleanup. The legacy JSON envelope remains accepted by worker ingress.
        self.dispatch
            .generate(request.map(|request| {
                AddressedRequest::new(request, self.address.clone())
                    .with_establish_timeout(self.establish_timeout)
            }))
            .await
    }
}
