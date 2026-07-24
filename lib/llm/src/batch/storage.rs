// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use bytes::Bytes;
use dynamo_protocols::types::{OpenAIFile, OpenAIFilePurpose};
use futures::{Stream, StreamExt};
use object_store::{GetResult, MultipartUpload, ObjectStore, PutPayloadMut, PutResult, path::Path};
use uuid::Uuid;

const UPLOAD_PART_SIZE: usize = 5 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct BatchFileStore {
    store: Arc<dyn ObjectStore>,
    prefix: Path,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum BatchFileStorageError {
    #[error("batch file was not found")]
    NotFound,
    #[error("batch file is too large")]
    FileTooLarge,
    #[error("batch file upload stream failed: {0}")]
    UploadStream(String),
    #[error("batch file upload is no longer active")]
    UploadNotActive,
    #[error("system clock is before the Unix epoch: {0}")]
    InvalidSystemTime(#[source] std::time::SystemTimeError),
    #[error("batch file metadata is invalid: {0}")]
    InvalidMetadata(#[source] serde_json::Error),
    #[error("batch file storage operation failed: {0}")]
    ObjectStore(#[source] object_store::Error),
}

impl BatchFileStore {
    pub(crate) fn from_url(url: &str) -> anyhow::Result<Self> {
        let url = url::Url::parse(url).context("invalid batch storage URL")?;
        let (store, prefix) = object_store::parse_url_opts(&url, std::env::vars())
            .context("failed to configure batch object store")?;
        Ok(Self {
            store: Arc::from(store),
            prefix,
        })
    }

    pub(crate) async fn upload_file<S>(
        &self,
        filename: String,
        purpose: OpenAIFilePurpose,
        mut chunks: S,
    ) -> Result<PendingFile, BatchFileStorageError>
    where
        S: Stream<Item = Result<Bytes, BatchFileStorageError>> + Unpin,
    {
        let id = format!("file-{}", Uuid::new_v4());
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(BatchFileStorageError::InvalidSystemTime)?
            .as_secs();
        let content_path = self.content_path(&id);
        let upload = self
            .store
            .put_multipart(&content_path)
            .await
            .map_err(BatchFileStorageError::ObjectStore)?;
        let mut upload = PendingUpload::new(upload);
        let mut bytes = 0_u32;

        while let Some(chunk) = chunks.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    upload.abort().await;
                    return Err(error);
                }
            };
            bytes = match u32::try_from(chunk.len())
                .ok()
                .and_then(|chunk_len| bytes.checked_add(chunk_len))
            {
                Some(bytes) => bytes,
                None => {
                    upload.abort().await;
                    return Err(BatchFileStorageError::FileTooLarge);
                }
            };
            if let Err(error) = upload.write(chunk).await {
                upload.abort().await;
                return Err(error);
            }
        }

        upload.finish().await?;

        #[allow(deprecated)]
        let file = OpenAIFile {
            id,
            object: "file".to_string(),
            bytes,
            created_at,
            expires_at: None,
            filename,
            purpose,
            status: None,
            status_details: None,
        };
        let metadata = match serde_json::to_vec(&file) {
            Ok(metadata) => metadata,
            Err(error) => {
                self.delete_content(&file.id).await;
                return Err(BatchFileStorageError::InvalidMetadata(error));
            }
        };
        if let Err(error) = self
            .store
            .put(&self.metadata_path(&file.id), metadata.into())
            .await
        {
            self.delete_content(&file.id).await;
            return Err(BatchFileStorageError::ObjectStore(error));
        }

        Ok(PendingFile::new(self.clone(), file))
    }

    pub(crate) async fn get_file(
        &self,
        file_id: &str,
    ) -> Result<(OpenAIFile, GetResult), BatchFileStorageError> {
        validate_file_id(file_id)?;
        let metadata = self
            .store
            .get(&self.metadata_path(file_id))
            .await
            .map_err(map_get_error)?
            .bytes()
            .await
            .map_err(map_get_error)?;
        let file =
            serde_json::from_slice(&metadata).map_err(BatchFileStorageError::InvalidMetadata)?;
        let content = self
            .store
            .get(&self.content_path(file_id))
            .await
            .map_err(map_get_error)?;
        Ok((file, content))
    }

    pub(crate) async fn delete_file(&self, file_id: &str) {
        if validate_file_id(file_id).is_err() {
            return;
        }
        if let Err(error) = self.store.delete(&self.metadata_path(file_id)).await
            && !matches!(error, object_store::Error::NotFound { .. })
        {
            tracing::warn!(error = %error, "failed to remove batch file metadata");
        }
        self.delete_content(file_id).await;
    }

    fn content_path(&self, file_id: &str) -> Path {
        self.file_path(file_id).child("content")
    }

    fn metadata_path(&self, file_id: &str) -> Path {
        self.file_path(file_id).child("metadata.json")
    }

    fn file_path(&self, file_id: &str) -> Path {
        self.prefix.child("files").child(file_id)
    }

    async fn delete_content(&self, file_id: &str) {
        if let Err(error) = self.store.delete(&self.content_path(file_id)).await
            && !matches!(error, object_store::Error::NotFound { .. })
        {
            tracing::warn!(error = %error, "failed to remove batch file content");
        }
    }
}

pub(crate) struct PendingFile {
    store: Option<BatchFileStore>,
    file: Option<OpenAIFile>,
}

impl PendingFile {
    fn new(store: BatchFileStore, file: OpenAIFile) -> Self {
        Self {
            store: Some(store),
            file: Some(file),
        }
    }

    pub(crate) fn commit(mut self) -> Result<OpenAIFile, BatchFileStorageError> {
        let file = self
            .file
            .take()
            .ok_or(BatchFileStorageError::UploadNotActive)?;
        self.store.take();
        Ok(file)
    }

    pub(crate) async fn abort(mut self) {
        let Some(store) = self.store.take() else {
            return;
        };
        if let Some(file) = self.file.take() {
            store.delete_file(&file.id).await;
        }
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        let (Some(store), Some(file)) = (self.store.take(), self.file.take()) else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            tracing::warn!("unable to remove uncommitted batch file without a Tokio runtime");
            return;
        };
        runtime.spawn(async move {
            store.delete_file(&file.id).await;
        });
    }
}

fn validate_file_id(file_id: &str) -> Result<(), BatchFileStorageError> {
    file_id
        .strip_prefix("file-")
        .and_then(|id| Uuid::parse_str(id).ok())
        .map(|_| ())
        .ok_or(BatchFileStorageError::NotFound)
}

fn map_get_error(error: object_store::Error) -> BatchFileStorageError {
    match error {
        object_store::Error::NotFound { .. } => BatchFileStorageError::NotFound,
        error => BatchFileStorageError::ObjectStore(error),
    }
}

struct PendingUpload {
    upload: Option<Box<dyn MultipartUpload>>,
    buffer: PutPayloadMut,
}

impl PendingUpload {
    fn new(upload: Box<dyn MultipartUpload>) -> Self {
        Self {
            upload: Some(upload),
            buffer: PutPayloadMut::new(),
        }
    }

    async fn write(&mut self, mut chunk: Bytes) -> Result<(), BatchFileStorageError> {
        while !chunk.is_empty() {
            let remaining = UPLOAD_PART_SIZE - self.buffer.content_length();
            let part = chunk.split_to(chunk.len().min(remaining));
            self.buffer.push(part);
            if self.buffer.content_length() == UPLOAD_PART_SIZE {
                self.flush().await?;
            }
        }
        Ok(())
    }

    fn upload_mut(&mut self) -> Result<&mut Box<dyn MultipartUpload>, BatchFileStorageError> {
        self.upload
            .as_mut()
            .ok_or(BatchFileStorageError::UploadNotActive)
    }

    async fn flush(&mut self) -> Result<(), BatchFileStorageError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let payload = std::mem::take(&mut self.buffer).freeze();
        self.upload_mut()?
            .put_part(payload)
            .await
            .map_err(BatchFileStorageError::ObjectStore)
    }

    async fn finish(mut self) -> Result<PutResult, BatchFileStorageError> {
        if let Err(error) = self.flush().await {
            self.abort().await;
            return Err(error);
        }
        let result = self
            .upload_mut()?
            .complete()
            .await
            .map_err(BatchFileStorageError::ObjectStore);
        match result {
            Ok(result) => {
                self.upload.take();
                Ok(result)
            }
            Err(error) => {
                self.abort().await;
                Err(error)
            }
        }
    }

    async fn abort(&mut self) {
        if let Some(mut upload) = self.upload.take()
            && let Err(error) = upload.abort().await
        {
            tracing::warn!(error = %error, "failed to abort batch file upload");
        }
    }
}

impl Drop for PendingUpload {
    fn drop(&mut self) {
        let Some(mut upload) = self.upload.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            tracing::warn!("unable to abort batch file upload without a Tokio runtime");
            return;
        };
        runtime.spawn(async move {
            if let Err(error) = upload.abort().await {
                tracing::warn!(error = %error, "failed to abort cancelled batch file upload");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use object_store::{PutPayload, UploadPart};
    use tokio::sync::oneshot;

    use super::*;

    #[derive(Debug)]
    struct RecordingUpload {
        aborted: Option<oneshot::Sender<()>>,
    }

    #[derive(Debug)]
    struct FailingFinalPartUpload {
        abort_started: Option<oneshot::Sender<()>>,
        abort_release: Option<oneshot::Receiver<()>>,
    }

    #[async_trait::async_trait]
    impl MultipartUpload for RecordingUpload {
        fn put_part(&mut self, _data: PutPayload) -> UploadPart {
            Box::pin(async { Ok(()) })
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            Ok(PutResult {
                e_tag: None,
                version: None,
            })
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            if let Some(aborted) = self.aborted.take() {
                let _ = aborted.send(());
            }
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl MultipartUpload for FailingFinalPartUpload {
        fn put_part(&mut self, _data: PutPayload) -> UploadPart {
            Box::pin(async { Err(object_store::Error::NotImplemented) })
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            Err(object_store::Error::NotImplemented)
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            if let Some(abort_started) = self.abort_started.take() {
                let _ = abort_started.send(());
            }
            if let Some(abort_release) = self.abort_release.take() {
                let _ = abort_release.await;
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn dropping_pending_upload_aborts_multipart_upload() {
        let (aborted_tx, aborted_rx) = oneshot::channel();
        let upload = Box::new(RecordingUpload {
            aborted: Some(aborted_tx),
        });

        drop(PendingUpload::new(upload));

        tokio::time::timeout(std::time::Duration::from_secs(1), aborted_rx)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn final_part_failure_awaits_abort() {
        let (abort_started_tx, abort_started_rx) = oneshot::channel();
        let (abort_release_tx, abort_release_rx) = oneshot::channel();
        let upload = Box::new(FailingFinalPartUpload {
            abort_started: Some(abort_started_tx),
            abort_release: Some(abort_release_rx),
        });
        let mut pending = PendingUpload::new(upload);
        pending
            .write(Bytes::from_static(b"final part"))
            .await
            .unwrap();

        let mut finish = tokio::spawn(pending.finish());
        abort_started_rx.await.unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut finish)
                .await
                .is_err(),
            "finish returned before abort completed"
        );

        abort_release_tx.send(()).unwrap();
        assert!(matches!(
            finish.await.unwrap(),
            Err(BatchFileStorageError::ObjectStore(_))
        ));
    }

    #[tokio::test]
    async fn file_store_persists_batch_output_across_instances() {
        let directory = tempfile::tempdir().unwrap();
        let url = url::Url::from_directory_path(directory.path())
            .unwrap()
            .to_string();
        let content = Bytes::from_static(b"{\"custom_id\":\"request-1\",\"response\":{}}\n");

        let store = BatchFileStore::from_url(&url).unwrap();
        let file = store
            .upload_file(
                "output.jsonl".to_string(),
                OpenAIFilePurpose::BatchOutput,
                stream::iter([Ok::<_, BatchFileStorageError>(content.clone())]),
            )
            .await
            .unwrap()
            .commit()
            .unwrap();
        drop(store);

        let store = BatchFileStore::from_url(&url).unwrap();
        let (stored_file, stored_content) = store.get_file(&file.id).await.unwrap();

        assert_eq!(stored_file, file);
        assert_eq!(stored_content.bytes().await.unwrap(), content);
    }

    #[tokio::test]
    async fn dropping_pending_file_removes_stored_objects() {
        let directory = tempfile::tempdir().unwrap();
        let url = url::Url::from_directory_path(directory.path())
            .unwrap()
            .to_string();
        let store = BatchFileStore::from_url(&url).unwrap();
        let pending = store
            .upload_file(
                "input.jsonl".to_string(),
                OpenAIFilePurpose::Batch,
                stream::iter([Ok::<_, BatchFileStorageError>(Bytes::from_static(b"{}\n"))]),
            )
            .await
            .unwrap();
        let file_id = pending.file.as_ref().unwrap().id.clone();

        drop(pending);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !matches!(
                store.get_file(&file_id).await,
                Err(BatchFileStorageError::NotFound)
            ) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
