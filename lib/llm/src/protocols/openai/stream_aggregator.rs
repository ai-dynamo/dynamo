// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::DynamoError;
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

/// Collect exactly one data item while preserving a typed backend error.
///
/// Unary media APIs use Dynamo's streaming request plane internally even
/// though their HTTP response is singular. Treating an empty or multi-item
/// stream as success would hide a broken worker contract.
pub async fn collect_unary_stream<T, S>(stream: S) -> Result<T, DynamoError>
where
    S: Stream<Item = Annotated<T>>,
{
    let mut stream = std::pin::pin!(stream);
    let mut response = None;

    while let Some(delta) = stream.next().await {
        if delta.is_error() {
            let fallback = delta
                .comment
                .unwrap_or_else(|| vec!["unknown backend error".to_string()])
                .join(", ");
            return Err(delta.error.unwrap_or_else(|| DynamoError::msg(fallback)));
        }

        if let Some(data) = delta.data {
            if response.replace(data).is_some() {
                return Err(DynamoError::msg(
                    "unary response stream produced more than one data item",
                ));
            }
        }
    }

    response.ok_or_else(|| DynamoError::msg("unary response stream produced no data"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::error::ErrorType;

    #[tokio::test]
    async fn unary_collector_preserves_backend_error_type() {
        let error = DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message("invalid language")
            .build();
        let stream = futures::stream::iter([Annotated {
            data: None::<usize>,
            id: None,
            event: Some("error".to_string()),
            comment: None,
            error: Some(error),
        }]);

        let error = collect_unary_stream(stream).await.unwrap_err();

        assert!(matches!(error.error_type(), ErrorType::InvalidArgument));
    }

    #[tokio::test]
    async fn unary_collector_rejects_empty_stream() {
        let stream = futures::stream::empty::<Annotated<usize>>();

        let error = collect_unary_stream(stream).await.unwrap_err();

        assert!(error.to_string().contains("produced no data"));
    }
}
