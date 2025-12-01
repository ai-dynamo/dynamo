// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::v2::InstanceId;

use dynamo_nova::Nova;
use dynamo_nova_backend::{PeerInfo, WorkerAddress};

use anyhow::{Result, bail};
use parking_lot::Mutex;
use std::sync::Arc;

pub trait ConnectorLeaderInterface: Send + Sync {}

pub struct ConnectorLeader {
    nova: Arc<Nova>,
    state: Arc<Mutex<ConnectorLeaderState>>,
}

#[derive(Default)]
struct ConnectorLeaderState {
    worker_instance_ids: Vec<InstanceId>,
}

impl ConnectorLeader {
    pub fn new(nova: Arc<Nova>) -> Self {
        Self {
            nova,
            state: Arc::new(Mutex::new(ConnectorLeaderState::default())),
        }
    }

    pub fn register_worker(
        &self,
        rank: usize,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> Result<()> {
        let mut state = self.state.lock();

        if rank != state.worker_instance_ids.len() {
            bail!("Rank mismatch");
        }

        state.worker_instance_ids.push(instance_id);

        self.nova
            .register_peer(PeerInfo::new(instance_id, worker_address))?;

        Ok(())
    }

    pub fn initialize_workers(&self) -> Result<()> {
        todo!()
    }
}
