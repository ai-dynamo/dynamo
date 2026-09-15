// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Native responses, with an in-memory source available to sidecar unit tests.
// Keep tonic inline; boxing it for the smaller test variant would add a production allocation.
#[cfg_attr(feature = "testing", allow(clippy::large_enum_variant))]
pub enum NativeStream<T> {
    Grpc(tonic::Streaming<T>),
    #[cfg(feature = "testing")]
    Scripted(crate::testing::ScriptedStream<T>),
}

impl<T> NativeStream<T> {
    pub async fn message(&mut self) -> Result<Option<T>, tonic::Status> {
        match self {
            Self::Grpc(stream) => stream.message().await,
            #[cfg(feature = "testing")]
            Self::Scripted(stream) => stream.message().await,
        }
    }
}
