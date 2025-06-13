use super::*;

use utils::*;
use zmq::*;

use dynamo_runtime::{utils::leader_worker_barrier::LeaderBarrier, DistributedRuntime};

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvbmLeaderData {
    pub zmq_url: String,
    pub broadcast_port: usize,
    pub ack_port: usize,
}

#[pyclass]
pub struct KvbmLeader {
    _drt: DistributedRuntime,
    // The DistributedRuntime only stores a handle, so we need to keep the runtime around.
    _runtime: tokio::runtime::Runtime,
    _zmq_leader: ZmqActiveMessageLeader,
}

#[pymethods]
impl KvbmLeader {
    #[new]
    #[pyo3(signature = (barrier_id, world_size=1))]
    fn new(barrier_id: String, world_size: usize) -> PyResult<Self> {
        let (drt, runtime) = build_drt().map_err(to_pyerr)?;

        tracing::info!(
            "Syncing leader barrier with {} workers on barrier id {}",
            world_size,
            barrier_id
        );

        let zmq_data = Arc::new(KvbmLeaderData {
            zmq_url: "127.0.0.1".to_string(),
            broadcast_port: 5555,
            ack_port: 5556,
        });

        let leader_barrier =
            LeaderBarrier::new(barrier_id, world_size, Some(Duration::from_secs(30)));

        let drt_clone = drt.clone();
        let zmq_data_clone = zmq_data.clone();
        drt.runtime()
            .primary()
            .block_on(async move {
                // TODO: Hardcode for now.
                leader_barrier
                    .sync(&drt_clone, zmq_data_clone.as_ref())
                    .await
            })
            .map_err(|e| {
                to_pyerr(anyhow::anyhow!(format!(
                    "Failed to sync leader barrier: {:?}",
                    e
                )))
            })?;

        tracing::info!("Leader barrier synced with {} workers", world_size);

        let zmq_leader = drt
            .runtime()
            .primary()
            .block_on(async move {
                let cancel_token = CancellationToken::new();
                ZmqActiveMessageLeader::new(
                    &zmq_data.zmq_url,
                    zmq_data.broadcast_port,
                    zmq_data.ack_port,
                    world_size,
                    Duration::from_secs(30),
                    cancel_token.clone(),
                )
                .await
            })
            .map_err(|e| {
                to_pyerr(anyhow::anyhow!(format!(
                    "Failed to create ZmqActiveMessageLeader: {:?}",
                    e
                )))
            })?;

        Ok(Self {
            _drt: drt,
            _runtime: runtime,
            _zmq_leader: zmq_leader,
        })
    }
}
