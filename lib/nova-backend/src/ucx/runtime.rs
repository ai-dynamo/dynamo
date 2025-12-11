// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! UCX runtime with LocalSet for handling Rc-based UCX operations

use std::rc::Rc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use async_ucx::ucp::{Context as UcxContext, Endpoint, Worker, WorkerAddress as UcpWorkerAddress};
use bytes::Bytes;
use dynamo_identity::InstanceId;
use hashbrown::HashMap;
use tokio::runtime::Builder;
use tokio::sync::{mpsc, oneshot};
use tokio::task::LocalSet;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, warn};

use crate::TransportAdapter;

// Fixed lane IDs for the 3 message types
pub const LANE_MESSAGE: u16 = 0;
pub const LANE_RESPONSE: u16 = 1;
pub const LANE_EVENT: u16 = 2;

/// Request to get or create an endpoint
pub enum EndpointRequest {
    GetEndpoint {
        instance_id: InstanceId,
        ucx_blob: Bytes, // decoded UCX WorkerAddress
        reply: oneshot::Sender<Result<Rc<Endpoint>>>,
    },
}

/// Health status for an endpoint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointHealthStatus {
    /// Endpoint exists in cache and is healthy
    Healthy,
    /// Endpoint exists in cache but get_status() returned an error
    Unhealthy,
    /// Endpoint not in cache (never connected)
    NotConnected,
}

/// Control message for endpoint manager
pub enum EndpointControlMessage {
    CheckHealth {
        instance_id: InstanceId,
        reply: oneshot::Sender<EndpointHealthStatus>,
    },
}

/// Spawn the UCX runtime with LocalSet
///
/// This creates a dedicated thread running a LocalSet where all UCX operations happen.
/// The runtime manages:
/// - 3 receiver tasks (one per lane) that route incoming messages
/// - 3 sender tasks (one per message type) that send outgoing messages
/// - 1 endpoint manager task that handles lazy endpoint creation with deduplication
pub fn spawn_ucx_runtime(
    channels: TransportAdapter,
    msg_rx: flume::Receiver<SendTask>,
    resp_rx: flume::Receiver<SendTask>,
    event_rx: flume::Receiver<SendTask>,
    cancel_token: CancellationToken,
) -> Result<(
    mpsc::UnboundedSender<GetWorkerAddressRequest>,
    mpsc::UnboundedSender<EndpointControlMessage>,
    std::thread::JoinHandle<()>,
)> {
    // Channel for requesting worker address
    let (worker_addr_tx, worker_addr_rx) = mpsc::unbounded_channel();

    // Create control channel for endpoint manager
    let (ep_control_tx, ep_control_rx) = mpsc::unbounded_channel();

    let thread = std::thread::Builder::new()
        .name("ucx-runtime".into())
        .spawn(move || {
            if let Err(err) = run_runtime(
                channels,
                msg_rx,
                resp_rx,
                event_rx,
                worker_addr_rx,
                ep_control_rx,
                cancel_token,
            ) {
                error!("UCX runtime thread exited with error: {:?}", err);
            }
        })?;

    Ok((worker_addr_tx, ep_control_tx, thread))
}

/// Request to get the local worker address
pub struct GetWorkerAddressRequest {
    pub reply: oneshot::Sender<Result<Bytes>>,
}

/// Task sent to a sender task

#[derive(Clone)]
pub struct SendTask {
    pub target_instance: InstanceId,
    pub target_blob: Bytes, // decoded UCX WorkerAddress
    pub header: Bytes,
    pub payload: Bytes,
    pub on_error: std::sync::Arc<dyn crate::TransportErrorHandler>,
}

/// Main runtime loop - runs in dedicated thread with LocalSet
fn run_runtime(
    channels: TransportAdapter,
    msg_rx: flume::Receiver<SendTask>,
    resp_rx: flume::Receiver<SendTask>,
    event_rx: flume::Receiver<SendTask>,
    mut worker_addr_rx: mpsc::UnboundedReceiver<GetWorkerAddressRequest>,
    ep_control_rx: mpsc::UnboundedReceiver<EndpointControlMessage>,
    cancel_token: CancellationToken,
) -> Result<()> {
    // Create endpoint request channel for internal use by sender tasks
    let (ep_tx, ep_rx) = mpsc::unbounded_channel();
    // Create single-threaded tokio runtime
    let rt = Builder::new_current_thread()
        .enable_time()
        .build()
        .context("failed to build UCX runtime")?;

    let local = LocalSet::new();

    // Initialize UCX
    let context = UcxContext::new().context("failed to initialize UCX context")?;

    // Note: we need two UCX workers because we need a mechanism to shutdown receiving active messages
    // in a safe way, i.e. in a way that senders will get an error notification instead of silently
    // messages. In this way, we can drain all remaining incoming messages while still being able to
    // complete our outstanding sends and receives.

    // Create a second UCX worker to send active messages and receive responses/acks/events
    let worker = context
        .create_worker()
        .context("failed to initialize UCX worker")?;

    local.spawn_local(worker.clone().polling());

    // Spawn endpoint manager task
    let ep_worker = worker.clone();
    let ep_cancel = cancel_token.clone();
    local.spawn_local(endpoint_manager_task(
        ep_rx,
        ep_control_rx,
        ep_worker,
        ep_cancel,
    ));

    // Active message recveiver task is tied to the primary cancel token
    local.spawn_local(receiver_task(
        LANE_MESSAGE,
        worker.clone(),
        channels.message_stream,
        cancel_token.clone(),
    ));

    local.spawn_local(receiver_task(
        LANE_RESPONSE,
        worker.clone(),
        channels.response_stream,
        cancel_token.clone(),
    ));
    local.spawn_local(receiver_task(
        LANE_EVENT,
        worker.clone(),
        channels.event_stream,
        cancel_token.clone(),
    ));

    // Spawn 3 sender tasks
    local.spawn_local(sender_task(
        LANE_MESSAGE,
        msg_rx,
        ep_tx.clone(),
        cancel_token.clone(),
    ));
    local.spawn_local(sender_task(
        LANE_RESPONSE,
        resp_rx,
        ep_tx.clone(),
        cancel_token.clone(),
    ));
    local.spawn_local(sender_task(
        LANE_EVENT,
        event_rx,
        ep_tx.clone(),
        cancel_token.clone(),
    ));

    // Main loop handles worker address requests
    local.block_on(&rt, async move {
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    debug!("UCX runtime received cancellation token");
                    break;
                }
                Some(req) = worker_addr_rx.recv() => {
                    let result = worker.address()
                        .map(|addr| addr.as_bytes().clone())
                        .context("Failed to get UCX worker address");
                    let _ = req.reply.send(result);
                }
                else => break,
            }
        }
    });

    debug!("UCX runtime exiting");
    Ok(())
}

/// Receiver task - receives messages on a specific lane and forwards to appropriate channel
async fn receiver_task(
    lane_id: u16,
    worker: Rc<Worker>,
    sender: flume::Sender<(Bytes, Bytes)>,
    cancel_token: CancellationToken,
) {
    debug!("Starting receiver task for lane {}", lane_id);

    // Register AM stream for this lane
    let stream = match worker.am_stream(lane_id) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to register AM stream for lane {}: {}", lane_id, e);
            return;
        }
    };

    debug!("AM stream registered for lane {}", lane_id);

    let mut draining = false;

    loop {
        tokio::select! {
            _ = cancel_token.cancelled(), if !draining => {
                debug!("Receiver task for lane {} cancelled, draining remaining messages", lane_id);
                draining = true;
                // Continue looping to drain UCX stream
            }
            msg = stream.wait_msg() => {
                match msg {
                    Some(mut message) => {
                        // UCX AM provides header bytes directly - no framing needed
                        let header = Bytes::copy_from_slice(message.header());

                        // Get payload data - check if data is present
                        let payload = if message.contains_data() {
                            match message.recv_data().await {
                                Ok(data) => Bytes::from(data),
                                Err(e) => {
                                    error!("Failed to receive UCX payload on lane {}: {}", lane_id, e);
                                    continue;
                                }
                            }
                        } else {
                            Bytes::new()
                        };

                        // Send to appropriate channel
                        if sender.send_async((header, payload)).await.is_err() {
                            warn!("Channel closed for lane {}", lane_id);
                            break;
                        }
                    }
                    None => {
                        // UCX signals stream is drained/unregistered
                        debug!("AM stream drained and closed for lane {}", lane_id);
                        break;
                    }
                }
            }
        }
    }

    debug!("Receiver task for lane {} exiting", lane_id);
}

/// Sender task - receives send requests and sends via UCX
async fn sender_task(
    lane_id: u16,
    rx: flume::Receiver<SendTask>,
    ep_tx: mpsc::UnboundedSender<EndpointRequest>,
    cancel_token: CancellationToken,
) {
    debug!("Starting sender task for lane {}", lane_id);

    // Local endpoint cache for fast path lookups
    let mut endpoint_cache: HashMap<dynamo_identity::InstanceId, Rc<Endpoint>> = HashMap::new();
    let mut draining = false;

    loop {
        tokio::select! {
            _ = cancel_token.cancelled(), if !draining => {
                debug!("Sender task for lane {} cancelled, draining remaining messages", lane_id);
                draining = true;
                // Continue looping to drain pending sends
            }
            result = rx.recv_async() => {
                match result {
                    Ok(task) => {
                        // Fast path: check local cache
                        let endpoint = if let Some(ep) = endpoint_cache.get(&task.target_instance) {
                            ep.clone()
                        } else {
                            // Slow path: request from endpoint manager
                            let (reply_tx, reply_rx) = oneshot::channel();
                            if ep_tx.send(EndpointRequest::GetEndpoint {
                                instance_id: task.target_instance,
                                ucx_blob: task.target_blob.clone(),
                                reply: reply_tx,
                            }).is_err() {
                                error!("Endpoint manager channel closed");
                                break;
                            }

                            // Wait for endpoint creation
                            let ep = match reply_rx.await {
                                Ok(Ok(ep)) => ep,
                                Ok(Err(e)) => {
                                    task.on_error.on_error(task.header, task.payload, e.to_string());
                                    continue;
                                }
                                Err(_) => {
                                    task.on_error.on_error(
                                        task.header,
                                        task.payload,
                                        "endpoint manager died".into()
                                    );
                                    continue;
                                }
                            };

                            // Cache the endpoint for future sends
                            endpoint_cache.insert(task.target_instance, ep.clone());
                            ep
                        };

                        // Send via UCX - header and payload are separate, UCX handles framing
                        if let Err(e) = endpoint.am_send(
                            lane_id,
                            &task.header,
                            &task.payload,
                            false,
                            None
                        ).await {
                            task.on_error.on_error(task.header, task.payload, e.to_string());
                        }
                    }
                    Err(_) => {
                        // All senders have been dropped - proper shutdown signal
                        debug!("Sender task for lane {} channel closed, exiting", lane_id);
                        break;
                    }
                }
            }
        }
    }

    debug!("Sender task for lane {} exiting", lane_id);
}

/// Endpoint manager task - handles lazy endpoint creation with deduplication
async fn endpoint_manager_task(
    mut rx: mpsc::UnboundedReceiver<EndpointRequest>,
    mut control_rx: mpsc::UnboundedReceiver<EndpointControlMessage>,
    worker: Rc<Worker>,
    cancel_token: CancellationToken,
) {
    debug!("Starting endpoint manager task");

    // Endpoint cache
    let mut cache: HashMap<InstanceId, Rc<Endpoint>> = HashMap::new();

    // Pending requests for endpoints being created
    let mut pending: HashMap<InstanceId, Vec<oneshot::Sender<Result<Rc<Endpoint>>>>> =
        HashMap::new();

    // Channel for receiving endpoint creation results
    let (result_tx, mut result_rx) = mpsc::unbounded_channel();

    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                debug!("Endpoint manager cancelled");
                break;
            }
            Some(req) = rx.recv() => {
                match req {
                    EndpointRequest::GetEndpoint { instance_id, ucx_blob, reply } => {
                        // Fast path: already in cache
                        if let Some(ep) = cache.get(&instance_id) {
                            let _ = reply.send(Ok(ep.clone()));
                            continue;
                        }

                        // Check if creation is already in progress
                        if let Some(waiters) = pending.get_mut(&instance_id) {
                            waiters.push(reply);
                            continue;
                        }

                        // Start new creation
                        pending.insert(instance_id, vec![reply]);

                        // Spawn creation task
                        let worker_clone = worker.clone();
                        let blob = ucx_blob.clone();
                        let id = instance_id;
                        let tx = result_tx.clone();

                        tokio::task::spawn_local(async move {
                            let result = create_endpoint(worker_clone, id, blob).await;
                            let _ = tx.send(result);
                        });
                    }
                }
            }
            Some(msg) = control_rx.recv() => {
                match msg {
                    EndpointControlMessage::CheckHealth { instance_id, reply } => {
                        // Check if endpoint exists in cache and its status
                        let status = if let Some(ep) = cache.get(&instance_id) {
                            // Check endpoint status - if get_status() returns error, it's unhealthy
                            if ep.get_status().is_ok() {
                                EndpointHealthStatus::Healthy
                            } else {
                                EndpointHealthStatus::Unhealthy
                            }
                        } else {
                            EndpointHealthStatus::NotConnected
                        };
                        let _ = reply.send(status);
                    }
                }
            }
            Some((instance_id, result)) = result_rx.recv() => {
                // Endpoint creation completed (successfully or not)
                if let Some(waiters) = pending.remove(&instance_id) {
                    match result {
                        Ok(ref ep) => {
                            // Cache successful endpoint
                            cache.insert(instance_id, ep.clone());

                            // Notify all waiters
                            for waiter in waiters {
                                let _ = waiter.send(Ok(ep.clone()));
                            }
                        }
                        Err(ref e) => {
                            // Notify all waiters of failure
                            let error_msg = e.to_string();
                            for waiter in waiters {
                                let _ = waiter.send(Err(anyhow!("{}", error_msg)));
                            }
                        }
                    }
                }
            }
            else => break,
        }
    }

    debug!(
        "Endpoint manager exiting with {} cached endpoints",
        cache.len()
    );
}

/// Create an endpoint (called from spawned task)
async fn create_endpoint(
    worker: Rc<Worker>,
    instance_id: InstanceId,
    ucx_blob: Bytes,
) -> (InstanceId, Result<Rc<Endpoint>>) {
    debug!("Creating endpoint for instance {}", instance_id);

    let start = Instant::now();

    // Parse UCX WorkerAddress directly from decoded bytes
    let ucx_addr = UcpWorkerAddress::from_bytes(ucx_blob);

    // Create endpoint by connecting to the worker address
    let result = match worker.connect_addr(&ucx_addr) {
        Ok(ep) => {
            debug!(
                "Created endpoint for instance {} in {:?}",
                instance_id,
                start.elapsed()
            );
            Ok(Rc::new(ep))
        }
        Err(e) => Err(anyhow!("Failed to create UCX endpoint: {}", e)),
    };

    (instance_id, result)
}
