// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{Stream, StreamExt};

use crate::types::Annotated;
use dynamo_runtime::error::DynamoError;

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
pub async fn aggregate_stream<T, S>(stream: S) -> Result<T, DynamoError>
where
    T: StreamAggregable,
    S: Stream<Item = Annotated<T>>,
{
    let mut stream = std::pin::pin!(stream);
    let mut response: Option<T> = None;

    while let Some(delta) = stream.next().await {
        if delta.is_error() {
            return Err(delta.error.unwrap_or_else(|| {
                DynamoError::msg(
                    delta
                        .comment
                        .map(|comments| comments.join(", "))
                        .unwrap_or_else(|| "unknown error".to_string()),
                )
            }));
        }
        if let Some(data) = delta.data {
            match response.as_mut() {
                Some(existing) => existing.merge(data),
                None => response = Some(data),
            }
        }
    }

    Ok(response.unwrap_or_else(T::empty))
}
