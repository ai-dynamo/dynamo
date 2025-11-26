use anyhow::{Context, Result};
use dynamo_runtime::transports::nats;

use crate::kv_router::protocols::{WorkerId, WorkerKvQueryRequest, WorkerKvQueryResponse};
use crate::kv_router::WORKER_KV_INDEXER_QUERY_SUBJECT;

/// Router-side client for querying worker local KV indexers
pub struct WorkerQueryClient {
    nats_client: nats::Client, // TODO is this the right nats abstraction to use?
                               // I think we want to use component/namespace but
                               // I'm not sure how to get the underlying nats client from it
                               // so that we can do request/reply (instead of pub/sub)
    namespace_name: String,
}

impl WorkerQueryClient {
    pub fn new(nats_client: nats::Client, namespace_name: String) -> Self {
        Self { nats_client, namespace_name }
    }

    /// Query a specific worker's local KV indexer and return its buffered events
    pub async fn query_worker(
        &self,
        worker_id: WorkerId,
    ) -> Result<WorkerKvQueryResponse> {
        // Match worker's subscribe format: namespace.{namespace_name}.{SUBJECT}.{worker_id}
        let subject = format!("namespace.{}.{}.{}",
            self.namespace_name,
            WORKER_KV_INDEXER_QUERY_SUBJECT,
            worker_id
        );

        tracing::info!("Router sending request to worker {} on NATS subject: {}", worker_id, subject);

        // Create and serialize request
        let request = WorkerKvQueryRequest { worker_id };
        let request_bytes = serde_json::to_vec(&request)
            .context("Failed to serialize WorkerKvQueryRequest")?;

        // send NATS request with timeout
        let timeout = tokio::time::Duration::from_secs(1);
        let response_msg = tokio::time::timeout(
            timeout,
            self.nats_client.client().request(subject.clone(), request_bytes.into()),
        )
        .await
        .context(format!("Request to worker {} timed out after {:?}", worker_id, timeout))?
        .context(format!("Failed to send request to worker {} on subject {}", worker_id, subject))?;

        // Deserialize response
        let response: WorkerKvQueryResponse = serde_json::from_slice(&response_msg.payload)
            .context("Failed to deserialize WorkerKvQueryResponse")?;

        Ok(response)
    }
}