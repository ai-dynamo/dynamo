// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use std::collections::VecDeque;
use tmq::{
    AsZmqSocket, Context, Message, Multipart,
    publish::{Publish, publish},
    subscribe::{Subscribe, subscribe},
};

use crate::active_message::{client::Endpoint, handler::ActiveMessage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    Tcp,
    Ipc,
}

pub enum ZmqTransport {
    Publisher {
        socket: Publish,
        endpoint: Option<Endpoint>,
        transport_type: TransportType,
    },
    Subscriber {
        socket: Subscribe,
        endpoint: Option<Endpoint>,
        transport_type: TransportType,
    },
}

impl std::fmt::Debug for ZmqTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Publisher {
                endpoint,
                transport_type,
                ..
            } => f
                .debug_struct("Publisher")
                .field("endpoint", endpoint)
                .field("transport_type", transport_type)
                .finish(),
            Self::Subscriber {
                endpoint,
                transport_type,
                ..
            } => f
                .debug_struct("Subscriber")
                .field("endpoint", endpoint)
                .field("transport_type", transport_type)
                .finish(),
        }
    }
}

impl ZmqTransport {
    pub fn new_publisher_bound(context: &Context, address: &str) -> Result<Self> {
        let transport_type = Self::detect_transport_type(address);
        let socket = publish(context).bind(address)?;

        let bound_endpoint = socket.get_socket().get_last_endpoint().unwrap().unwrap();

        Ok(Self::Publisher {
            socket,
            endpoint: Some(bound_endpoint),
            transport_type,
        })
    }

    pub fn new_subscriber_bound(context: &Context, address: &str) -> Result<Self> {
        let transport_type = Self::detect_transport_type(address);
        let socket = subscribe(context).bind(address)?.subscribe(b"")?;

        let bound_endpoint = socket.get_socket().get_last_endpoint().unwrap().unwrap();

        Ok(Self::Subscriber {
            socket,
            endpoint: Some(bound_endpoint),
            transport_type,
        })
    }

    pub fn new_publisher_connected(context: &Context, endpoint: &str) -> Result<Self> {
        let transport_type = Self::detect_transport_type(endpoint);
        let socket = publish(context).connect(endpoint)?;

        Ok(Self::Publisher {
            socket,
            endpoint: Some(endpoint.to_string()),
            transport_type,
        })
    }

    pub fn new_subscriber_connected(context: &Context, endpoint: &str) -> Result<Self> {
        let transport_type = Self::detect_transport_type(endpoint);
        let socket = subscribe(context).connect(endpoint)?.subscribe(b"")?;

        Ok(Self::Subscriber {
            socket,
            endpoint: Some(endpoint.to_string()),
            transport_type,
        })
    }

    fn detect_transport_type(address: &str) -> TransportType {
        if address.starts_with("ipc://") {
            TransportType::Ipc
        } else {
            TransportType::Tcp
        }
    }

    fn serialize_message(message: &ActiveMessage) -> Result<Multipart> {
        let serialized = serde_json::to_vec(message)?;
        let mut parts = VecDeque::new();
        parts.push_back(Message::from(serialized));
        Ok(Multipart(parts))
    }

    fn deserialize_message(multipart: Multipart) -> Result<ActiveMessage> {
        if multipart.is_empty() {
            anyhow::bail!("Empty multipart message");
        }

        let bytes = &*multipart[0];
        let message: ActiveMessage = serde_json::from_slice(bytes)?;
        Ok(message)
    }

    pub fn local_endpoint(&self) -> Option<&Endpoint> {
        match self {
            Self::Publisher { endpoint, .. } => endpoint.as_ref(),
            Self::Subscriber { endpoint, .. } => endpoint.as_ref(),
        }
    }

    pub async fn receive(&mut self) -> Result<ActiveMessage> {
        match self {
            Self::Subscriber { socket, .. } => {
                if let Some(Ok(multipart)) = socket.next().await {
                    Self::deserialize_message(multipart)
                } else {
                    anyhow::bail!("Failed to receive message")
                }
            }
            Self::Publisher { .. } => {
                anyhow::bail!("Cannot receive on publisher socket")
            }
        }
    }
}
