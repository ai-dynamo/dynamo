// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire protocol messages for the GPU Memory Service.
//!
//! All messages are serialized with MessagePack (rmp-serde) and ride on
//! Velo's `(header, payload)` framing. The header carries routing info,
//! the payload carries the request/response body.

use serde::{Deserialize, Serialize};

// ==================== Lock Types ====================

/// Lock type requested by the client during handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestedLockType {
    /// Exclusive read-write access (for loaders/writers).
    Rw,
    /// Shared read-only access (for inference engines).
    Ro,
    /// Prefer RW if available, otherwise RO.
    RwOrRo,
}

/// Lock type actually granted by the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrantedLockType {
    /// Exclusive read-write access.
    Rw,
    /// Shared read-only access.
    Ro,
}

// ==================== GMS Header ====================

/// Header for GMS messages, carried in Velo's header field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmsHeader {
    /// Operation code identifying the message type.
    pub op: GmsOp,
    /// Session ID (assigned during handshake).
    pub session_id: u64,
    /// Request ID for request/response correlation.
    pub request_id: u64,
}

/// Operation codes for GMS messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GmsOp {
    // Handshake
    HandshakeRequest,
    HandshakeResponse,
    // Allocation operations
    Allocate,
    AllocateResponse,
    Export,
    ExportResponse,
    GetAllocation,
    GetAllocationResponse,
    ListAllocations,
    ListAllocationsResponse,
    Free,
    FreeResponse,
    ClearAll,
    ClearAllResponse,
    // Metadata operations
    MetadataPut,
    MetadataPutResponse,
    MetadataGet,
    MetadataGetResponse,
    MetadataDelete,
    MetadataDeleteResponse,
    MetadataList,
    MetadataListResponse,
    // Commit
    Commit,
    CommitResponse,
    // State queries
    GetLockState,
    GetLockStateResponse,
    GetAllocationState,
    GetAllocationStateResponse,
    GetStateHash,
    GetStateHashResponse,
    // Error
    Error,
}

// ==================== Handshake ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub lock_type: RequestedLockType,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub success: bool,
    pub committed: bool,
    pub granted_lock_type: Option<GrantedLockType>,
}

// ==================== Allocations ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocateRequest {
    pub size: u64,
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocateResponse {
    pub allocation_id: String,
    pub size: u64,
    pub aligned_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    pub allocation_id: String,
}

/// Export response. The actual FD is sent via SCM_RIGHTS ancillary data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResponse {
    pub allocation_id: String,
    pub size: u64,
    pub aligned_size: u64,
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetAllocationRequest {
    pub allocation_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetAllocationResponse {
    pub allocation_id: String,
    pub size: u64,
    pub aligned_size: u64,
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListAllocationsRequest {
    pub tag: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEntry {
    pub allocation_id: String,
    pub size: u64,
    pub aligned_size: u64,
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListAllocationsResponse {
    pub allocations: Vec<AllocationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeRequest {
    pub allocation_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeResponse {
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearAllRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearAllResponse {
    pub cleared_count: u32,
}

// ==================== Metadata ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataPutRequest {
    pub key: String,
    pub allocation_id: String,
    pub offset_bytes: u64,
    #[serde(with = "serde_bytes")]
    pub value: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataPutResponse {
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataGetRequest {
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataGetResponse {
    pub found: bool,
    pub allocation_id: Option<String>,
    pub offset_bytes: Option<u64>,
    #[serde(with = "serde_bytes")]
    pub value: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataDeleteRequest {
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataDeleteResponse {
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataListRequest {
    pub prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataListResponse {
    pub keys: Vec<String>,
}

// ==================== Commit ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    pub success: bool,
    pub state_hash: String,
}

// ==================== State Queries ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetLockStateRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetLockStateResponse {
    pub state: String,
    pub has_rw_session: bool,
    pub ro_session_count: u32,
    pub waiting_writers: u32,
    pub committed: bool,
    pub is_ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetAllocationStateRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetAllocationStateResponse {
    pub allocation_count: u32,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetStateHashRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetStateHashResponse {
    pub memory_layout_hash: String,
}

// ==================== Error ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: i32,
}

// ==================== Encoding ====================

/// Encode a message to MessagePack bytes.
pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec(msg)
}

/// Decode a message from MessagePack bytes.
pub fn decode<'a, T: Deserialize<'a>>(data: &'a [u8]) -> Result<T, rmp_serde::decode::Error> {
    rmp_serde::from_slice(data)
}

/// Encode a GMS header.
pub fn encode_header(header: &GmsHeader) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec(header)
}

/// Decode a GMS header.
pub fn decode_header(data: &[u8]) -> Result<GmsHeader, rmp_serde::decode::Error> {
    rmp_serde::from_slice(data)
}
