// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-shot notification when a request reaches the final worker dispatch hop.

use anyhow::{Error, Result};
use parking_lot::Mutex;

use crate::engine::Data;
use crate::pipeline::Context;

const DISPATCH_OBSERVER_KEY: &str = "dynamo.dispatch_observer";

struct DispatchObserver {
    callback: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

/// Observe the first worker dispatch made by this context or its derived hops.
pub fn attach_dispatch_observer<T: Data>(
    request: &mut Context<T>,
    callback: impl FnOnce() + Send + 'static,
) {
    request.insert(
        DISPATCH_OBSERVER_KEY,
        DispatchObserver {
            callback: Mutex::new(Some(Box::new(callback))),
        },
    );
}

/// Share the notification across independently controlled pipeline hops.
pub fn propagate_dispatch_observer<S: Data, T: Data>(
    source: &Context<S>,
    target: &mut Context<T>,
) -> Result<()> {
    if let Some(observer) = source
        .get_optional::<DispatchObserver>(DISPATCH_OBSERVER_KEY)
        .map_err(Error::msg)?
    {
        target.insert_arc(DISPATCH_OBSERVER_KEY, observer);
    }
    Ok(())
}

/// Fire only after worker selection, transport resolution, and admission succeed.
pub fn notify_dispatch<T: Data>(request: &Context<T>) -> Result<()> {
    if let Some(observer) = request
        .get_optional::<DispatchObserver>(DISPATCH_OBSERVER_KEY)
        .map_err(Error::msg)?
    {
        let callback = observer.callback.lock().take();
        if let Some(callback) = callback {
            callback();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn derived_hops_notify_only_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let mut request = Context::new(());
        attach_dispatch_observer(&mut request, move || {
            observed.fetch_add(1, Ordering::SeqCst);
        });
        let mut encode = Context::new(());
        let mut prefill = Context::new(());
        propagate_dispatch_observer(&request, &mut encode).unwrap();
        propagate_dispatch_observer(&request, &mut prefill).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        notify_dispatch(&encode).unwrap();
        notify_dispatch(&prefill).unwrap();
        notify_dispatch(&request).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
