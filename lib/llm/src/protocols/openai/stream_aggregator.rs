// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{Stream, StreamExt};

use crate::types::Annotated;

/// Response types whose `Annotated<T>` streams can be folded into a single `T`
/// using shared aggregation infrastructure.
pub trait StreamAggregable: Sized {
    /// Empty fallback when the stream yields no data items.
    fn empty() -> Self;
    /// Merge `next` into `self`. Implementors define type-specific
    /// behavior (extending data, summing usage, etc.).
    fn merge(&mut self, next: Self);
}

/// Aggregate a stream of [`Annotated<T>`] into a single `T`. The first error
/// encountered short-circuits further merging and is returned; the remainder
/// of the stream is dropped.
pub async fn aggregate_stream<T, S>(stream: S) -> Result<T, String>
where
    T: StreamAggregable,
    S: Stream<Item = Annotated<T>>,
{
    let mut stream = std::pin::pin!(stream);
    let mut response: Option<T> = None;

    while let Some(delta) = stream.next().await {
        let delta = delta.ok()?;
        if let Some(data) = delta.data {
            match response.as_mut() {
                Some(existing) => existing.merge(data),
                None => response = Some(data),
            }
        }
    }

    Ok(response.unwrap_or_else(T::empty))
}
