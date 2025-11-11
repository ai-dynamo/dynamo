// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Message Client

pub mod builders;

use anyhow::Result;
use dashmap::DashMap;
use std::{collections::HashSet, sync::Arc};

use crate::am::{
    InstanceId, PeerInfo,
    common::{ActiveMessage, responses::ResponseManager},
};

use dynamo_nova_backend::{NovaBackend, TransportErrorHandler};

#[derive(Debug, thiserror::Error)]
#[error("handler '{handler_name}' not found on instance {instance_id}")]
struct MissingHandlerError {
    instance_id: InstanceId,
    handler_name: String,
}

pub struct ActiveMessageClient {
    instance_id: InstanceId,
    response_manager: ResponseManager,
    backend: Arc<NovaBackend>,
    error_handler: Arc<dyn TransportErrorHandler>,
}

impl ActiveMessageClient {
    pub(crate) fn new(
        instance_id: InstanceId,
        response_manager: ResponseManager,
        backend: Arc<NovaBackend>,
        error_handler: Arc<dyn TransportErrorHandler>,
    ) -> Self {
        Self {
            instance_id,
            response_manager,
            backend,
            error_handler,
        }
    }

    pub(crate) async fn send_message(
        &self,
        target: InstanceId,
        message: ActiveMessage,
    ) -> Result<()> {
        let (header, payload, message_type) = message.encode();
        self.backend.send_message(
            target,
            header.to_vec(),
            payload.to_vec(),
            message_type,
            self.error_handler.clone(),
        )
    }

    pub(crate) async fn connect_to_peer(&self, info: PeerInfo) -> Result<()> {
        unimplemented!()
    }

    async fn evaluate_handler_availability(
        &self,
        instance_id: InstanceId,
        handler_name: &str,
    ) -> Result<(), MissingHandlerError> {
        unimplemented!()

        // if handler_name.starts_with('_') {
        //     return Ok(());
        // }

        // if let Some(handlers) = self.instances.get(&instance_id) {
        //     if handlers.contains(handler_name) {
        //         return Ok(());
        //     }
        // }

        // // TODO: evaluate if we should try to update the handler list or fail

        // Err(MissingHandlerError {
        //     instance_id,
        //     handler_name: handler_name.to_string(),
        // })
    }

    fn check_for_handler_on_instance(
        &self,
        instance_id: InstanceId,
        handler_name: &str,
    ) -> Result<(), MissingHandlerError> {
        unimplemented!()

        // if handler_name.starts_with('_') {
        //     return Ok(());
        // }

        // if let Some(handlers) = self.instances.get(&instance_id) {
        //     if handlers.contains(handler_name) {
        //         return Ok(());
        //     }
        // }

        // Err(MissingHandlerError {
        //     instance_id,
        //     handler_name: handler_name.to_string(),
        // })
    }

    async fn update_instance_handlers(&self, _instance_id: InstanceId) -> Result<()> {
        unimplemented!()
    }

    async fn register_outcome(
        &self,
    ) -> Result<
        crate::am::common::responses::ResponseAwaiter,
        crate::am::common::responses::ResponseRegistrationError,
    > {
        self.response_manager.register_outcome()
    }
}
