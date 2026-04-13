// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPU Memory Service client library.
//!
//! Async client for communicating with the GMS server over Unix domain sockets.
//! Handles connection, handshake, request/response correlation, and FD receiving.

use std::os::fd::RawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::net::UnixStream;

use crate::error::{GmsError, GmsResult};
use crate::protocol::*;

/// Maximum allowed frame size (64 MiB). Prevents OOM from malformed/malicious peers.
const MAX_FRAME_SIZE: usize = 64 * 1024 * 1024;

/// Default RPC timeout (30 seconds).
const DEFAULT_RPC_TIMEOUT: Duration = Duration::from_secs(30);

/// Client for the GPU Memory Service.
///
/// Provides async methods for all GMS operations. Maintains a single
/// connection to the server with a granted lock type.
pub struct GmsClient {
    stream: UnixStream,
    session_id: u64,
    lock_type: GrantedLockType,
    request_counter: AtomicU64,
    committed: bool,
    rpc_timeout: Duration,
}

impl GmsClient {
    /// Connect to a GMS server and perform handshake.
    ///
    /// # Arguments
    /// * `socket_path` - Path to the server's Unix domain socket
    /// * `lock_type` - Requested lock type
    /// * `timeout` - Optional timeout for lock acquisition
    pub async fn connect(
        socket_path: &Path,
        lock_type: RequestedLockType,
        timeout: Option<Duration>,
    ) -> GmsResult<Self> {
        let stream = UnixStream::connect(socket_path)
            .await
            .map_err(GmsError::Io)?;

        let mut client = Self {
            stream,
            session_id: 0,
            lock_type: GrantedLockType::Ro, // Will be set after handshake
            request_counter: AtomicU64::new(1),
            committed: false,
            rpc_timeout: DEFAULT_RPC_TIMEOUT,
        };

        // Perform handshake
        let req = HandshakeRequest {
            lock_type,
            timeout_ms: timeout.map(|t| t.as_millis() as u64),
        };

        let header = GmsHeader {
            op: GmsOp::HandshakeRequest,
            session_id: 0,
            request_id: 0,
        };

        client.send_message(&header, &req).await?;
        let (_resp_header, resp_bytes) = client.recv_message().await?;
        let resp: HandshakeResponse = decode(&resp_bytes)?;

        if !resp.success {
            return Err(GmsError::LockTimeout);
        }

        client.session_id = _resp_header.session_id;
        client.lock_type = resp.granted_lock_type.unwrap_or(GrantedLockType::Ro);
        client.committed = resp.committed;

        tracing::info!(
            "Connected to GMS: session={}, lock={:?}",
            client.session_id,
            client.lock_type
        );

        Ok(client)
    }

    /// Allocate GPU memory.
    pub async fn allocate(&mut self, size: u64, tag: &str) -> GmsResult<AllocateResponse> {
        let req = AllocateRequest {
            size,
            tag: tag.to_string(),
        };
        let (_header, payload) = self.rpc(GmsOp::Allocate, &req).await?;
        let resp: AllocateResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Export an allocation as a POSIX FD.
    ///
    /// Returns the response metadata and the received file descriptor.
    /// The caller must close the FD when done.
    pub async fn export(&mut self, allocation_id: &str) -> GmsResult<(ExportResponse, RawFd)> {
        let req = ExportRequest {
            allocation_id: allocation_id.to_string(),
        };

        let request_id = self.next_request_id();
        let header = GmsHeader {
            op: GmsOp::Export,
            session_id: self.session_id,
            request_id,
        };
        self.send_message(&header, &req).await?;

        // Receive response with FD via SCM_RIGHTS
        let (data, fds) = self.recv_message_with_fds().await?;
        let resp: ExportResponse = decode(&data)?;

        let fd = fds
            .into_iter()
            .next()
            .ok_or_else(|| GmsError::Protocol("export response missing FD".into()))?;

        Ok((resp, fd))
    }

    /// Get allocation info.
    pub async fn get_allocation(&mut self, allocation_id: &str) -> GmsResult<GetAllocationResponse> {
        let req = GetAllocationRequest {
            allocation_id: allocation_id.to_string(),
        };
        let (_header, payload) = self.rpc(GmsOp::GetAllocation, &req).await?;
        let resp: GetAllocationResponse = decode(&payload)?;
        Ok(resp)
    }

    /// List allocations, optionally filtered by tag.
    pub async fn list_allocations(
        &mut self,
        tag: Option<&str>,
    ) -> GmsResult<ListAllocationsResponse> {
        let req = ListAllocationsRequest {
            tag: tag.map(|s| s.to_string()),
        };
        let (_header, payload) = self.rpc(GmsOp::ListAllocations, &req).await?;
        let resp: ListAllocationsResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Free a single allocation.
    pub async fn free(&mut self, allocation_id: &str) -> GmsResult<FreeResponse> {
        let req = FreeRequest {
            allocation_id: allocation_id.to_string(),
        };
        let (_header, payload) = self.rpc(GmsOp::Free, &req).await?;
        let resp: FreeResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Clear all allocations.
    pub async fn clear_all(&mut self) -> GmsResult<ClearAllResponse> {
        let req = ClearAllRequest;
        let (_header, payload) = self.rpc(GmsOp::ClearAll, &req).await?;
        let resp: ClearAllResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Store a metadata entry.
    pub async fn metadata_put(
        &mut self,
        key: &str,
        allocation_id: &str,
        offset: u64,
        value: &[u8],
    ) -> GmsResult<()> {
        let req = MetadataPutRequest {
            key: key.to_string(),
            allocation_id: allocation_id.to_string(),
            offset_bytes: offset,
            value: value.to_vec(),
        };
        let (_header, _payload) = self.rpc(GmsOp::MetadataPut, &req).await?;
        Ok(())
    }

    /// Retrieve a metadata entry.
    pub async fn metadata_get(&mut self, key: &str) -> GmsResult<MetadataGetResponse> {
        let req = MetadataGetRequest {
            key: key.to_string(),
        };
        let (_header, payload) = self.rpc(GmsOp::MetadataGet, &req).await?;
        let resp: MetadataGetResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Delete a metadata entry.
    pub async fn metadata_delete(&mut self, key: &str) -> GmsResult<MetadataDeleteResponse> {
        let req = MetadataDeleteRequest {
            key: key.to_string(),
        };
        let (_header, payload) = self.rpc(GmsOp::MetadataDelete, &req).await?;
        let resp: MetadataDeleteResponse = decode(&payload)?;
        Ok(resp)
    }

    /// List metadata keys.
    pub async fn metadata_list(&mut self, prefix: &str) -> GmsResult<MetadataListResponse> {
        let req = MetadataListRequest {
            prefix: prefix.to_string(),
        };
        let (_header, payload) = self.rpc(GmsOp::MetadataList, &req).await?;
        let resp: MetadataListResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Commit all allocations and metadata.
    ///
    /// After commit, the connection transitions to Committed state and
    /// the connection is closed by the server.
    pub async fn commit(&mut self) -> GmsResult<CommitResponse> {
        let req = CommitRequest;
        let (_header, payload) = self.rpc(GmsOp::Commit, &req).await?;
        let resp: CommitResponse = decode(&payload)?;
        self.committed = true;
        Ok(resp)
    }

    /// Get current lock state.
    pub async fn get_lock_state(&mut self) -> GmsResult<GetLockStateResponse> {
        let req = GetLockStateRequest;
        let (_header, payload) = self.rpc(GmsOp::GetLockState, &req).await?;
        let resp: GetLockStateResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Get allocation state.
    pub async fn get_allocation_state(&mut self) -> GmsResult<GetAllocationStateResponse> {
        let req = GetAllocationStateRequest;
        let (_header, payload) = self.rpc(GmsOp::GetAllocationState, &req).await?;
        let resp: GetAllocationStateResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Get memory layout hash.
    pub async fn get_state_hash(&mut self) -> GmsResult<GetStateHashResponse> {
        let req = GetStateHashRequest;
        let (_header, payload) = self.rpc(GmsOp::GetStateHash, &req).await?;
        let resp: GetStateHashResponse = decode(&payload)?;
        Ok(resp)
    }

    /// Whether this session has committed.
    pub fn committed(&self) -> bool {
        self.committed
    }

    /// The granted lock type.
    pub fn lock_type(&self) -> GrantedLockType {
        self.lock_type
    }

    /// The session ID.
    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    // ==================== Internal ====================

    fn next_request_id(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Set the RPC timeout duration.
    pub fn set_rpc_timeout(&mut self, timeout: Duration) {
        self.rpc_timeout = timeout;
    }

    /// Send a request and receive the response.
    async fn rpc<T: serde::Serialize>(
        &self,
        op: GmsOp,
        body: &T,
    ) -> GmsResult<(GmsHeader, Vec<u8>)> {
        let request_id = self.next_request_id();
        let header = GmsHeader {
            op,
            session_id: self.session_id,
            request_id,
        };
        self.send_message(&header, body).await?;
        tokio::time::timeout(self.rpc_timeout, self.recv_message())
            .await
            .map_err(|_| GmsError::RpcTimeout)?
    }

    async fn send_message<T: serde::Serialize>(
        &self,
        header: &GmsHeader,
        body: &T,
    ) -> GmsResult<()> {
        let header_bytes = crate::protocol::encode_header(header)
            .map_err(|e| GmsError::Protocol(format!("encode header: {e}")))?;
        let payload_bytes = crate::protocol::encode(body)
            .map_err(|e| GmsError::Protocol(format!("encode payload: {e}")))?;

        let mut frame =
            Vec::with_capacity(8 + header_bytes.len() + payload_bytes.len());
        frame.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
        frame.extend_from_slice(&(payload_bytes.len() as u32).to_be_bytes());
        frame.extend_from_slice(&header_bytes);
        frame.extend_from_slice(&payload_bytes);

        let mut written = 0;
        while written < frame.len() {
            self.stream.writable().await.map_err(GmsError::Io)?;
            match self.stream.try_write(&frame[written..]) {
                Ok(n) => written += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(e) => return Err(GmsError::Io(e)),
            }
        }

        Ok(())
    }

    async fn recv_message(&self) -> GmsResult<(GmsHeader, Vec<u8>)> {
        let mut frame_header = [0u8; 8];

        self.stream
            .readable()
            .await
            .map_err(GmsError::Io)?;

        let mut total_read = 0;
        while total_read < 8 {
            match self.stream.try_read(&mut frame_header[total_read..]) {
                Ok(0) => return Err(GmsError::Protocol("connection closed".into())),
                Ok(n) => total_read += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    self.stream.readable().await.map_err(GmsError::Io)?;
                }
                Err(e) => return Err(GmsError::Io(e)),
            }
        }

        let header_len = u32::from_be_bytes(frame_header[0..4].try_into().expect("frame header is 8 bytes")) as usize;
        let payload_len = u32::from_be_bytes(frame_header[4..8].try_into().expect("frame header is 8 bytes")) as usize;

        let total_len = header_len.checked_add(payload_len).ok_or_else(|| {
            GmsError::Protocol("frame size overflow".into())
        })?;
        if total_len > MAX_FRAME_SIZE {
            return Err(GmsError::Protocol(format!(
                "frame too large: {total_len} bytes (max {MAX_FRAME_SIZE})"
            )));
        }

        let mut data = vec![0u8; total_len];
        let mut total_read = 0;
        while total_read < data.len() {
            match self.stream.try_read(&mut data[total_read..]) {
                Ok(0) => return Err(GmsError::Protocol("connection closed mid-frame".into())),
                Ok(n) => total_read += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    self.stream.readable().await.map_err(GmsError::Io)?;
                }
                Err(e) => return Err(GmsError::Io(e)),
            }
        }

        let header: GmsHeader = crate::protocol::decode_header(&data[..header_len])
            .map_err(|e| GmsError::Protocol(format!("decode header: {e}")))?;
        let payload = data[header_len..].to_vec();

        // Check for error response
        if header.op == GmsOp::Error {
            let err: ErrorResponse = crate::protocol::decode(&payload)
                .map_err(|e| GmsError::Protocol(format!("decode error: {e}")))?;
            return Err(GmsError::Protocol(err.error));
        }

        Ok((header, payload))
    }

    async fn recv_message_with_fds(&self) -> GmsResult<(Vec<u8>, Vec<RawFd>)> {
        // Use ancillary data to receive FDs
        let (frame, fds) =
            crate::ancillary::recv_frame_with_fds(&self.stream, 64 * 1024)
                .await
                .map_err(|e| GmsError::Transport(e))?;

        if frame.len() < 8 {
            return Err(GmsError::Protocol("frame too short".into()));
        }

        let header_len = u32::from_be_bytes(frame[0..4].try_into().expect("frame header is 8 bytes")) as usize;
        let _payload_len = u32::from_be_bytes(frame[4..8].try_into().expect("frame header is 8 bytes")) as usize;

        if 8 + header_len > frame.len() {
            return Err(GmsError::Protocol("header_len exceeds frame".into()));
        }
        let payload = frame[8 + header_len..].to_vec();
        Ok((payload, fds))
    }
}

fn decode<'a, T: serde::Deserialize<'a>>(data: &'a [u8]) -> GmsResult<T> {
    crate::protocol::decode(data).map_err(|e| GmsError::Protocol(format!("decode: {e}")))
}
