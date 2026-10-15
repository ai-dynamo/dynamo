// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use dynamo_runtime::{
    engine::AsyncEngine,
    error::{DynamoError, ErrorClass},
    pipeline::{ManyOut, SingleIn, network::egress::push_router::PushRouter},
};
use tokio::sync::watch;

use crate::{
    model_card::ModelDeploymentCard,
    protocols::openai::images::{NvCreateImageRequest, NvImagesResponse},
    types::{Annotated, openai::images::OpenAIImagesStreamingEngine},
};

pub(crate) const SNAPSHOT_KEY: &str = "image_edit_admission";

pub(crate) fn advertised_limit(card: &ModelDeploymentCard) -> usize {
    let capability = card
        .user_data
        .as_ref()
        .and_then(|data| data.get("image_edits"));
    let Some(capability) = capability else {
        return 1;
    };
    if capability
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
    {
        return 1;
    }
    capability
        .get("max_input_images")
        .and_then(serde_json::Value::as_u64)
        .and_then(|limit| usize::try_from(limit).ok())
        .unwrap_or(1)
}

#[derive(Clone, Debug)]
pub(crate) struct ImageEditSnapshot {
    pub(crate) worker_ids: HashSet<u64>,
    pub(crate) max_input_images: usize,
}

/// Capability records are updated only after the discovery group commits.
/// Admission remains owned by the controller; a missing record contributes 1.
pub(crate) struct ImageEditAdmission {
    limits: ArcSwap<HashMap<u64, usize>>,
    admitted_ids: watch::Receiver<Vec<u64>>,
}

impl ImageEditAdmission {
    pub(crate) fn new(admitted_ids: watch::Receiver<Vec<u64>>) -> Self {
        Self {
            limits: ArcSwap::from_pointee(HashMap::new()),
            admitted_ids,
        }
    }

    pub(crate) fn replace(&self, members: impl Iterator<Item = (u64, usize)>) {
        self.limits.store(Arc::new(members.collect()));
    }

    pub(crate) fn snapshot(&self) -> ImageEditSnapshot {
        let ids = self.admitted_ids.borrow();
        let limits = self.limits.load();
        ImageEditSnapshot {
            max_input_images: ids
                .iter()
                .map(|id| limits.get(id).copied().unwrap_or(1))
                .min()
                .unwrap_or(0),
            worker_ids: ids.iter().copied().collect(),
        }
    }

    fn validate_dispatch(
        &self,
        snapshot: &ImageEditSnapshot,
        worker: u64,
        count: usize,
    ) -> anyhow::Result<()> {
        let current = self.snapshot();
        if !snapshot.worker_ids.contains(&worker)
            || !current.worker_ids.contains(&worker)
            || count > snapshot.max_input_images
            || count > current.max_input_images
        {
            return Err(DynamoError::builder()
                .class(ErrorClass::Unavailable)
                .message("Image edit worker membership changed before dispatch")
                .build()
                .into());
        }
        Ok(())
    }
}

pub(crate) struct ImageEditEngine {
    pub(crate) router: PushRouter<NvCreateImageRequest, Annotated<NvImagesResponse>>,
    pub(crate) admission: Arc<ImageEditAdmission>,
}

#[async_trait]
impl
    AsyncEngine<SingleIn<NvCreateImageRequest>, ManyOut<Annotated<NvImagesResponse>>, anyhow::Error>
    for ImageEditEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateImageRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<NvImagesResponse>>> {
        let count = request.input_reference.as_ref().map_or(0, Vec::len);
        if count == 0 {
            return self.router.generate(request).await;
        }
        let snapshot = request
            .get::<ImageEditSnapshot>(SNAPSHOT_KEY)
            .unwrap_or_else(|_| Arc::new(self.admission.snapshot()));
        let (_, stream) = self
            .router
            .select_and_dispatch(request, |_, worker| {
                self.admission.validate_dispatch(&snapshot, worker, count)
            })
            .await?;
        Ok(stream)
    }
}

pub(crate) struct ImageEngineSelection {
    pub(crate) engine: OpenAIImagesStreamingEngine,
    pub(crate) snapshot: Option<ImageEditSnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn capabilities_are_versioned_conservative_and_do_not_change_identity() {
        let checksum = ModelDeploymentCard::default().mdcsum().to_owned();
        for (record, expected) in [
            (json!({"schema_version": 1, "max_input_images": 4}), 4),
            (json!({"schema_version": 1, "max_input_images": 0}), 0),
            (json!({"schema_version": 2, "max_input_images": 4}), 1),
            (json!({"schema_version": 1, "max_input_images": true}), 1),
            (json!({"schema_version": 1, "max_input_images": -1}), 1),
            (json!(null), 1),
        ] {
            let mut card = ModelDeploymentCard::default();
            card.user_data = Some(json!({"image_edits": record}));
            assert_eq!(advertised_limit(&card), expected);
            assert_eq!(card.mdcsum(), checksum);
        }
    }

    #[test]
    fn snapshot_uses_only_admitted_members_and_fences_membership_changes() {
        let (tx, rx) = watch::channel(vec![1, 2]);
        let admission = ImageEditAdmission::new(rx);
        admission.replace([(1, 4), (2, 3), (99, 0)].into_iter());
        let snapshot = admission.snapshot();
        assert_eq!(snapshot.max_input_images, 3);
        assert!(admission.validate_dispatch(&snapshot, 1, 3).is_ok());
        assert!(admission.validate_dispatch(&snapshot, 99, 1).is_err());
        // A joining legacy worker reduces the whole set's ceiling before dispatch.
        tx.send_replace(vec![1, 2, 3]);
        assert_eq!(admission.snapshot().max_input_images, 1);
        assert!(admission.validate_dispatch(&snapshot, 1, 3).is_err());
        assert!(admission.validate_dispatch(&snapshot, 3, 1).is_err());
        tx.send_replace(vec![1]);
        assert_eq!(admission.snapshot().max_input_images, 4);
        assert!(admission.validate_dispatch(&snapshot, 2, 1).is_err());
        // Retiring the generation empties its channel; old snapshots cannot revive it.
        tx.send_replace(vec![]);
        assert!(admission.validate_dispatch(&snapshot, 1, 1).is_err());
    }
}
