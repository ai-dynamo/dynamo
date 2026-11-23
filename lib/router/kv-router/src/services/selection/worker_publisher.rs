// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_scheduler::ScopedReplicaEvent;
use dynamo_slot_tracker::host::{
    ActiveSequenceEvent, ActiveSequenceEventData, ActiveSequenceEventOrigin,
};

use crate::services::common::replica_sync::REPLICA_TOPIC;
use crate::services::common::zmq::create_bound_pub_socket_sync;

/// Publishes worker-origin active-sequence events directly to selector subscribers.
pub struct WorkerSequenceEventPublisher {
    socket: zmq::Socket,
    endpoint: String,
    event: ScopedReplicaEvent,
}

impl WorkerSequenceEventPublisher {
    pub fn new(
        bind_endpoint: &str,
        model_name: String,
        routing_group: String,
        worker: WorkerWithDpRank,
    ) -> Result<Self> {
        let socket = create_bound_pub_socket_sync(bind_endpoint)?;
        let endpoint = socket
            .get_last_endpoint()?
            .map_err(|_| anyhow!("publisher endpoint is not UTF-8"))?;
        Ok(Self {
            socket,
            endpoint,
            event: ScopedReplicaEvent {
                model_name,
                routing_group,
                block_size: 0,
                event: ActiveSequenceEvent {
                    request_id: String::new(),
                    worker,
                    data: ActiveSequenceEventData::Free,
                    router_id: 0,
                    lora_name: None,
                    origin: ActiveSequenceEventOrigin::Worker,
                },
            },
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn mark_prefill_completed(&mut self, request_id: &str) -> Result<()> {
        self.publish(request_id, ActiveSequenceEventData::MarkPrefillCompleted)
    }

    pub fn free(&mut self, request_id: &str) -> Result<()> {
        self.publish(request_id, ActiveSequenceEventData::Free)
    }

    fn publish(&mut self, request_id: &str, data: ActiveSequenceEventData) -> Result<()> {
        self.event.event.request_id.clear();
        self.event.event.request_id.push_str(request_id);
        self.event.event.data = data;
        let payload = rmp_serde::to_vec_named(&self.event)?;
        self.socket
            .send_multipart([REPLICA_TOPIC, payload.as_slice()], zmq::DONTWAIT)?;
        Ok(())
    }
}
