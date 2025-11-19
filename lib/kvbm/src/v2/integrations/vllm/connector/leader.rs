// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod core;

use std::collections::{HashMap, HashSet};

use crate::v2::integrations::connector::ConnectorMetadataBuilder;
use dynamo_tokens::compute_hash_v2;
use serde::Serialize;

/// Minimal representation of a scheduler slot request.
#[derive(Clone, Debug)]
pub struct Request {
    pub request_id: String,
    pub lora_name: Option<String>,
    pub salt_hash: u64,
}

impl Request {
    pub fn new(
        request_id: impl Into<String>,
        lora_name: Option<String>,
        salt: Option<String>,
    ) -> Self {
        #[derive(Serialize)]
        struct SaltPayload<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<&'a str>,
        }

        let request_id = request_id.into();
        let payload = SaltPayload {
            salt: salt.as_deref(),
            lora_name: lora_name.as_deref(),
        };
        let salt_bytes = serde_json::to_vec(&payload).expect("failed to serialize salt payload");
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        Self {
            request_id,
            lora_name,
            salt_hash,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NewRequestData {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub block_ids: Vec<u32>,
    pub num_computed_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CachedRequestData {
    pub request_id: String,
    pub resumed_from_preemption: bool,
    pub new_token_ids: Vec<u32>,
    pub new_block_ids: Vec<u32>,
    pub num_computed_tokens: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SchedulerOutput {
    pub(crate) new_requests: Vec<NewRequestData>,
    pub(crate) cached_requests: Vec<CachedRequestData>,
    pub(crate) num_scheduled_tokens: HashMap<String, usize>,
}

impl SchedulerOutput {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_new_request(
        &mut self,
        request_id: impl Into<String>,
        prompt_token_ids: Vec<u32>,
        block_ids: Vec<u32>,
        num_computed_tokens: usize,
    ) {
        self.new_requests.push(NewRequestData {
            request_id: request_id.into(),
            prompt_token_ids,
            block_ids,
            num_computed_tokens,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_cached_request(
        &mut self,
        request_id: impl Into<String>,
        resumed_from_preemption: bool,
        new_token_ids: Vec<u32>,
        new_block_ids: Vec<u32>,
        num_computed_tokens: usize,
    ) {
        self.cached_requests.push(CachedRequestData {
            request_id: request_id.into(),
            resumed_from_preemption,
            new_token_ids,
            new_block_ids,
            num_computed_tokens,
        });
    }

    pub fn set_num_scheduled_tokens(&mut self, counts: HashMap<String, usize>) {
        self.num_scheduled_tokens = counts;
    }

    pub fn new_requests(&self) -> &[NewRequestData] {
        &self.new_requests
    }

    pub fn cached_requests(&self) -> &[CachedRequestData] {
        &self.cached_requests
    }
}

#[derive(Debug)]
pub struct SchedulerConnectorLeader {
    engine_id: String,
    metadata: ConnectorMetadataBuilder,
    known_slots: HashSet<String>,
    forward_seq: HashMap<String, u64>,
    pending_deletes: Vec<String>,
}

impl SchedulerConnectorLeader {
    pub fn new(engine_id: impl Into<String>) -> Self {
        Self {
            engine_id: engine_id.into(),
            metadata: ConnectorMetadataBuilder::new(1),
            known_slots: HashSet::new(),
            forward_seq: HashMap::new(),
            pending_deletes: Vec::new(),
        }
    }

    pub fn engine_id(&self) -> &str {
        &self.engine_id
    }

    pub fn has_slot(&self, request_id: &str) -> bool {
        self.known_slots.contains(request_id)
    }

    pub fn create_slot(&mut self, request: Request, _all_token_ids: Vec<Vec<i64>>) {
        let inserted = self.known_slots.insert(request.request_id.clone());
        if inserted {
            let event = format!("evt.create.{}", request.request_id);
            self.metadata.queue_slot_create(request.request_id, event);
        }
    }

    pub fn get_num_new_matched_tokens(
        &self,
        _request_id: &str,
        _num_tokens: usize,
        _num_computed_tokens: usize,
    ) -> (usize, bool) {
        (0, false)
    }

    pub fn update_state_after_alloc(
        &mut self,
        _request_id: &str,
        _block_ids: Vec<u32>,
        _num_external_tokens: usize,
    ) {
    }

    pub fn request_finished(&mut self, request_id: &str) -> bool {
        self.pending_deletes.push(request_id.to_string());
        false
    }

    pub fn update_connector_output(&mut self) {}

    pub fn get_finished_count(&self) -> Option<usize> {
        None
    }

    pub fn build_connector_metadata(&mut self, output: &SchedulerOutput) -> Vec<u8> {
        for req in output.new_requests() {
            self.queue_forward_event(&req.request_id);
        }
        for req in output.cached_requests() {
            self.queue_forward_event(&req.request_id);
        }

        for request_id in self.pending_deletes.drain(..) {
            self.metadata.queue_slot_delete(request_id);
        }

        self.metadata.build_bytes()
    }

    fn queue_forward_event(&mut self, request_id: &str) {
        if !self.known_slots.contains(request_id) {
            return;
        }
        let counter = self.forward_seq.entry(request_id.to_string()).or_insert(0);
        let event = format!("evt.forward.{}.{}", request_id, counter);
        self.metadata
            .queue_forward_event(request_id.to_string(), 0, event);
        *counter += 1;
    }
}
