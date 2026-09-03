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
        if let Some(data) = delta.into_data()? {
            match response.as_mut() {
                Some(existing) => existing.merge(data),
                None => response = Some(data),
            }
        }
    }

    Ok(response.unwrap_or_else(T::empty))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{
        error::{BackendError, ErrorType},
        protocols::maybe_error::MaybeError,
    };
    use futures::stream;
    use serde::Deserialize;

    #[derive(Debug, Default, PartialEq, Deserialize)]
    struct Chunks(Vec<u32>);

    impl StreamAggregable for Chunks {
        fn empty() -> Self {
            Self::default()
        }

        fn merge(&mut self, next: Self) {
            self.0.extend(next.0);
        }
    }

    #[tokio::test]
    async fn merges_data_and_falls_back_to_empty() {
        let empty = aggregate_stream::<Chunks, _>(stream::empty())
            .await
            .unwrap();
        assert_eq!(empty, Chunks(vec![]));

        let stream = stream::iter(vec![
            Annotated::from_data(Chunks(vec![1])),
            Annotated::from_data(Chunks(vec![2])),
        ]);
        assert_eq!(aggregate_stream(stream).await.unwrap(), Chunks(vec![1, 2]));
    }

    /// The property every OpenAI endpoint depends on: a backend rejection keeps
    /// its type, so the HTTP layer can answer 4xx rather than a blanket 500.
    #[tokio::test]
    async fn preserves_backend_error_type() {
        let stream = stream::iter(vec![Annotated::<Chunks>::from_err(
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::InvalidArgument))
                .message("invalid argument")
                .build(),
        )]);

        let error = aggregate_stream(stream).await.unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert_eq!(error.message(), "invalid argument");
    }
}
