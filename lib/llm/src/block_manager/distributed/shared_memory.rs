// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    fs::OpenOptions,
    io::{Read, Write},
    os::unix::{
        fs::{FileExt, MetadataExt, OpenOptionsExt},
        net::UnixStream as StdUnixStream,
    },
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering, fence},
    },
    time::Duration,
};

use dynamo_memory::{ExternalFileMappingDescriptor, ExternalPinnedStorage};
use dynamo_runtime::config::environment_names::kvbm::shared_memory::{
    DYN_KVBM_G2_BACKING, DYN_KVBM_RCOMMU_ATTACH_TIMEOUT_MS, DYN_KVBM_RCOMMU_ENDPOINT,
    DYN_KVBM_RCOMMU_NUMA_NODE, DYN_KVBM_RCOMMU_REGION_KEY,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
};
use tokio_util::sync::CancellationToken;

use crate::block_manager::storage::PinnedStorage;

const PROTOCOL_VERSION: u16 = 1;
const HEADER_LEN: usize = 4096;
const MAX_CONTROL_FRAME_LEN: usize = 1024 * 1024;
const HEADER_MAGIC: &[u8; 8] = b"RCKVSHM1";
const STATE_OFFSET: usize = 12;
const OWNER_DOMAIN_OFFSET: usize = 16;
const OWNER_EPOCH_OFFSET: usize = 24;
const REGION_ID_OFFSET: usize = 32;
const REGION_GENERATION_OFFSET: usize = 48;
const DATA_OFFSET_OFFSET: usize = 56;
const DATA_LEN_OFFSET: usize = 64;
const DATA_ALIGNMENT_OFFSET: usize = 72;
const LAYOUT_FINGERPRINT_OFFSET: usize = 88;
const HEADER_DIGEST_OFFSET: usize = 120;

#[derive(Clone, Debug, Default)]
pub enum HostMemoryBacking {
    #[default]
    CudaAllocated,
    Rcommu(RcommuShmConfig),
}

#[derive(Clone, Debug)]
pub struct RcommuShmConfig {
    pub endpoint: PathBuf,
    pub region_key: String,
    pub numa_node: Option<u32>,
    pub attach_timeout: Duration,
    pub worker_epoch: u64,
}

impl RcommuShmConfig {
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        let Some(backing) = std::env::var_os(DYN_KVBM_G2_BACKING) else {
            return Ok(None);
        };
        let backing = backing.to_string_lossy();
        match backing.as_ref() {
            "cuda" => return Ok(None),
            "rcommu" => {}
            other => {
                anyhow::bail!("{DYN_KVBM_G2_BACKING} must be 'cuda' or 'rcommu', got {other:?}")
            }
        }
        let endpoint = required_env(DYN_KVBM_RCOMMU_ENDPOINT)?;
        let region_key = required_env(DYN_KVBM_RCOMMU_REGION_KEY)?;
        let numa_node = optional_parse_env(DYN_KVBM_RCOMMU_NUMA_NODE)?;
        let timeout_ms = optional_parse_env(DYN_KVBM_RCOMMU_ATTACH_TIMEOUT_MS)?.unwrap_or(30_000);
        if timeout_ms == 0 {
            anyhow::bail!("{DYN_KVBM_RCOMMU_ATTACH_TIMEOUT_MS} must be positive");
        }
        Ok(Some(Self {
            endpoint: PathBuf::from(endpoint),
            region_key,
            numa_node,
            attach_timeout: Duration::from_millis(timeout_ms),
            worker_epoch: random_nonzero(),
        }))
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.endpoint.as_os_str().is_empty() {
            anyhow::bail!("rcommu owner endpoint must not be empty");
        }
        if self.region_key.is_empty() || self.region_key.len() > 512 {
            anyhow::bail!("rcommu region key must contain between 1 and 512 bytes");
        }
        if self.attach_timeout.is_zero() {
            anyhow::bail!("rcommu attach timeout must be positive");
        }
        if self.worker_epoch == 0 {
            anyhow::bail!("rcommu worker epoch must be non-zero");
        }
        Ok(())
    }

    pub fn for_worker(&self, device_id: usize, rank: Option<i32>) -> Self {
        let rank = rank
            .map(|rank| rank.to_string())
            .unwrap_or_else(|| device_id.to_string());
        let region_key = self
            .region_key
            .replace("{device_id}", &device_id.to_string())
            .replace("{rank}", &rank)
            .replace("{pid}", &std::process::id().to_string());
        Self {
            region_key,
            ..self.clone()
        }
    }
}

fn required_env(name: &str) -> anyhow::Result<String> {
    let value = std::env::var(name).map_err(|_| anyhow::anyhow!("{name} is required"))?;
    if value.is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    Ok(value)
}

fn optional_parse_env<T: std::str::FromStr>(name: &str) -> anyhow::Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .map_err(|error| anyhow::anyhow!("invalid {name}={value:?}: {error}"))
        })
        .transpose()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct OwnerIdentity {
    domain_fingerprint: u64,
    owner_epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RegionIdentity {
    region_id: [u8; 16],
    region_generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ClientIdentity {
    worker_id: String,
    worker_epoch: u64,
    pid: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RegionAccess {
    ExclusiveReadWrite,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SharedRegionSpec {
    region_key: String,
    data_len: u64,
    data_alignment: u64,
    numa_node: Option<u32>,
    layout_fingerprint: [u8; 32],
    access: RegionAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SharedRegionDescriptor {
    protocol_version: u16,
    owner: OwnerIdentity,
    region: RegionIdentity,
    path: PathBuf,
    file_identity: FileIdentity,
    header_len: u64,
    data_offset: u64,
    data_len: u64,
    data_alignment: u64,
    layout_fingerprint: [u8; 32],
    header_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct AttachmentLease {
    attachment_id: [u8; 16],
    attachment_generation: u64,
    owner: OwnerIdentity,
    region: RegionIdentity,
    client: ClientIdentity,
    capability: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct OpenRegionReply {
    descriptor: SharedRegionDescriptor,
    lease: AttachmentLease,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ClientAttachObservation {
    observed_owner_epoch: u64,
    observed_region_generation: u64,
    observed_file_identity: FileIdentity,
    observed_header_digest: [u8; 32],
    mapped_data_len: u64,
    layout_fingerprint: [u8; 32],
    cuda_device: u32,
    cuda_registration_ok: bool,
    nixl_registration_ok: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AttachmentStatus {
    Opening,
    Active,
    Aborted,
    Closed,
    Quarantined,
    Stale,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct AttachmentState {
    attachment_id: [u8; 16],
    region: RegionIdentity,
    status: AttachmentStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum RegionOperation {
    Open {
        client: ClientIdentity,
        spec: SharedRegionSpec,
    },
    Activate {
        lease: AttachmentLease,
        observation: ClientAttachObservation,
    },
    Abort {
        lease: AttachmentLease,
    },
    Detach {
        lease: AttachmentLease,
    },
    GetStatus {
        lease: AttachmentLease,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RegionRequest {
    protocol_version: u16,
    request_id: String,
    operation: RegionOperation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
enum RegionReply {
    Open(Box<OpenRegionReply>),
    Attachment(AttachmentState),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RegionErrorCode {
    InvalidRequest,
    ProtocolMismatch,
    SpecConflict,
    AttachmentBusy,
    OwnerBusy,
    AttachmentNotFound,
    InvalidAttachmentState,
    Unauthorized,
    StaleOwner,
    StaleRegion,
    OperationConflict,
    CapacityExceeded,
    Io,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RegionError {
    code: RegionErrorCode,
    message: String,
}

impl std::fmt::Display for RegionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RegionResponse {
    protocol_version: u16,
    request_id: String,
    result: Result<RegionReply, RegionError>,
}

#[derive(Clone, Debug)]
struct OwnerClient {
    endpoint: PathBuf,
    timeout: Duration,
}

impl OwnerClient {
    fn new(config: &RcommuShmConfig) -> Self {
        Self {
            endpoint: config.endpoint.clone(),
            timeout: config.attach_timeout,
        }
    }

    async fn call(&self, request: &RegionRequest) -> anyhow::Result<RegionReply> {
        let mut last_error = None;
        for _ in 0..2 {
            match tokio::time::timeout(self.timeout, call_once(&self.endpoint, request)).await {
                Ok(Ok(reply)) => return Ok(reply),
                Ok(Err(error)) => last_error = Some(error),
                Err(error) => last_error = Some(error.into()),
            }
        }
        Err(anyhow::anyhow!(
            "rcommu owner request {} failed twice with {:?} per-attempt timeout: {:#}",
            request.request_id,
            self.timeout,
            last_error.expect("two attempts set an error")
        ))
    }

    fn call_blocking(&self, request: &RegionRequest) -> anyhow::Result<RegionReply> {
        let mut last_error = None;
        for _ in 0..2 {
            match self.call_blocking_once(request) {
                Ok(reply) => return Ok(reply),
                Err(error) => last_error = Some(error),
            }
        }
        Err(anyhow::anyhow!(
            "rcommu owner request {} failed twice: {:#}",
            request.request_id,
            last_error.expect("two attempts set an error")
        ))
    }

    fn call_blocking_once(&self, request: &RegionRequest) -> anyhow::Result<RegionReply> {
        let mut stream = StdUnixStream::connect(&self.endpoint)?;
        stream.set_read_timeout(Some(self.timeout))?;
        stream.set_write_timeout(Some(self.timeout))?;
        let payload = serde_json::to_vec(request)?;
        if payload.is_empty() || payload.len() > MAX_CONTROL_FRAME_LEN {
            anyhow::bail!("rcommu owner request frame has invalid length");
        }
        stream.write_all(&(payload.len() as u32).to_be_bytes())?;
        stream.write_all(&payload)?;
        stream.flush()?;
        let mut length = [0_u8; 4];
        stream.read_exact(&mut length)?;
        let length = u32::from_be_bytes(length) as usize;
        if length == 0 || length > MAX_CONTROL_FRAME_LEN {
            anyhow::bail!("rcommu owner response frame has invalid length {length}");
        }
        let mut response = vec![0_u8; length];
        stream.read_exact(&mut response)?;
        decode_response(request, serde_json::from_slice(&response)?)
    }
}

async fn call_once(endpoint: &Path, request: &RegionRequest) -> anyhow::Result<RegionReply> {
    let mut stream = UnixStream::connect(endpoint).await?;
    write_frame(&mut stream, request).await?;
    let response: RegionResponse = read_frame(&mut stream).await?;
    decode_response(request, response)
}

fn decode_response(
    request: &RegionRequest,
    response: RegionResponse,
) -> anyhow::Result<RegionReply> {
    if response.protocol_version != PROTOCOL_VERSION {
        anyhow::bail!(
            "rcommu owner returned protocol version {}, expected {PROTOCOL_VERSION}",
            response.protocol_version
        );
    }
    if response.request_id != request.request_id {
        anyhow::bail!("rcommu owner response request_id mismatch");
    }
    response
        .result
        .map_err(|error| anyhow::anyhow!("rcommu owner rejected request: {error}"))
}

async fn write_frame<T: Serialize>(stream: &mut UnixStream, value: &T) -> anyhow::Result<()> {
    let payload = serde_json::to_vec(value)?;
    if payload.is_empty() || payload.len() > MAX_CONTROL_FRAME_LEN {
        anyhow::bail!("rcommu owner request frame has invalid length");
    }
    stream.write_u32(payload.len() as u32).await?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
    Ok(())
}

async fn read_frame<T: DeserializeOwned>(stream: &mut UnixStream) -> anyhow::Result<T> {
    let length = stream.read_u32().await? as usize;
    if length == 0 || length > MAX_CONTROL_FRAME_LEN {
        anyhow::bail!("rcommu owner response frame has invalid length {length}");
    }
    let mut payload = vec![0_u8; length];
    stream.read_exact(&mut payload).await?;
    Ok(serde_json::from_slice(&payload)?)
}

pub(super) struct OpeningAttachment {
    client: OwnerClient,
    descriptor: SharedRegionDescriptor,
    lease: AttachmentLease,
    armed: bool,
}

impl OpeningAttachment {
    pub(super) async fn open(
        config: &RcommuShmConfig,
        worker_id: String,
        data_len: u64,
        layout_fingerprint: [u8; 32],
        cuda_device: u32,
    ) -> anyhow::Result<(PinnedStorage, Self)> {
        let client = OwnerClient::new(config);
        let request = RegionRequest {
            protocol_version: PROTOCOL_VERSION,
            request_id: request_id(),
            operation: RegionOperation::Open {
                client: ClientIdentity {
                    worker_id,
                    worker_epoch: config.worker_epoch,
                    pid: std::process::id(),
                },
                spec: SharedRegionSpec {
                    region_key: config.region_key.clone(),
                    data_len,
                    data_alignment: 4096,
                    numa_node: config.numa_node,
                    layout_fingerprint,
                    access: RegionAccess::ExclusiveReadWrite,
                },
            },
        };
        let open = match client.call(&request).await? {
            RegionReply::Open(open) => *open,
            _ => anyhow::bail!("rcommu owner returned a non-open reply for open request"),
        };
        if let Err(error) = validate_header(&open.descriptor) {
            abort_best_effort(&client, &open.lease).await;
            return Err(error.context("validate rcommu shared-memory header"));
        }
        let mapping = ExternalFileMappingDescriptor {
            path: open.descriptor.path.clone(),
            device: open.descriptor.file_identity.device,
            inode: open.descriptor.file_identity.inode,
            offset: open.descriptor.data_offset,
            len: open.descriptor.data_len,
            alignment: open.descriptor.data_alignment,
        };
        let storage = match ExternalPinnedStorage::new(mapping, cuda_device) {
            Ok(storage) => PinnedStorage::from_external(storage),
            Err(error) => {
                abort_best_effort(&client, &open.lease).await;
                return Err(error.into());
            }
        };
        Ok((
            storage,
            Self {
                client,
                descriptor: open.descriptor,
                lease: open.lease,
                armed: true,
            },
        ))
    }

    pub(super) async fn activate(
        &mut self,
        cuda_device: u32,
        failure_token: CancellationToken,
    ) -> anyhow::Result<Arc<ActiveAttachmentSession>> {
        let request = RegionRequest {
            protocol_version: PROTOCOL_VERSION,
            request_id: request_id(),
            operation: RegionOperation::Activate {
                lease: self.lease.clone(),
                observation: ClientAttachObservation {
                    observed_owner_epoch: self.descriptor.owner.owner_epoch,
                    observed_region_generation: self.descriptor.region.region_generation,
                    observed_file_identity: self.descriptor.file_identity,
                    observed_header_digest: self.descriptor.header_digest,
                    mapped_data_len: self.descriptor.data_len,
                    layout_fingerprint: self.descriptor.layout_fingerprint,
                    cuda_device,
                    cuda_registration_ok: true,
                    nixl_registration_ok: true,
                },
            },
        };
        match self.client.call(&request).await {
            Ok(RegionReply::Attachment(state)) if state.status == AttachmentStatus::Active => {
                self.armed = false;
                Ok(ActiveAttachmentSession::new(
                    self.client.clone(),
                    self.lease.clone(),
                    failure_token,
                ))
            }
            Ok(_) => anyhow::bail!("rcommu owner did not activate the attachment"),
            Err(error) => {
                // The activate request is idempotent, but two timeouts can still leave its
                // result ambiguous. Query the lease before deciding whether to abort.
                let status = status_best_effort(&self.client, &self.lease).await;
                if status == Some(AttachmentStatus::Active) {
                    self.armed = false;
                    return Ok(ActiveAttachmentSession::new(
                        self.client.clone(),
                        self.lease.clone(),
                        failure_token,
                    ));
                }
                Err(error)
            }
        }
    }

    pub(super) async fn abort(mut self) {
        if self.armed {
            abort_best_effort(&self.client, &self.lease).await;
            self.armed = false;
        }
    }
}

impl Drop for OpeningAttachment {
    fn drop(&mut self) {
        if self.armed {
            spawn_terminal_request(
                self.client.clone(),
                RegionOperation::Abort {
                    lease: self.lease.clone(),
                },
            );
        }
    }
}

pub(super) struct ActiveAttachmentSession {
    client: OwnerClient,
    lease: AttachmentLease,
    closed: AtomicBool,
    healthy: Arc<AtomicBool>,
    health_cancellation: CancellationToken,
    health_task: Option<tokio::task::JoinHandle<()>>,
}

impl std::fmt::Debug for ActiveAttachmentSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ActiveAttachmentSession")
            .field("attachment_id", &self.lease.attachment_id)
            .field("closed", &self.closed.load(Ordering::Acquire))
            .finish()
    }
}

impl ActiveAttachmentSession {
    fn new(
        client: OwnerClient,
        lease: AttachmentLease,
        failure_token: CancellationToken,
    ) -> Arc<Self> {
        let health_cancellation = CancellationToken::new();
        let healthy = Arc::new(AtomicBool::new(true));
        let health_task = tokio::spawn(monitor_owner_health(
            client.clone(),
            lease.clone(),
            health_cancellation.clone(),
            failure_token,
            Arc::clone(&healthy),
        ));
        Arc::new(Self {
            client,
            lease,
            closed: AtomicBool::new(false),
            healthy,
            health_cancellation,
            health_task: Some(health_task),
        })
    }

    pub(super) fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
}

impl Drop for ActiveAttachmentSession {
    fn drop(&mut self) {
        self.health_cancellation.cancel();
        if let Some(task) = self.health_task.take() {
            task.abort();
        }
        if !self.closed.swap(true, Ordering::AcqRel) {
            spawn_terminal_request(
                self.client.clone(),
                RegionOperation::Detach {
                    lease: self.lease.clone(),
                },
            );
        }
    }
}

async fn monitor_owner_health(
    client: OwnerClient,
    lease: AttachmentLease,
    cancellation: CancellationToken,
    failure_token: CancellationToken,
    healthy: Arc<AtomicBool>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = cancellation.cancelled() => return,
            _ = interval.tick() => {
                let status = tokio::select! {
                    _ = cancellation.cancelled() => return,
                    status = status_best_effort(&client, &lease) => status,
                };
                if status != Some(AttachmentStatus::Active) {
                    healthy.store(false, Ordering::Release);
                    tracing::error!(
                        ?status,
                        "rcommu shared-memory owner or active attachment became unavailable; cancelling KVBM worker"
                    );
                    failure_token.cancel();
                    return;
                }
            }
        }
    }
}

fn spawn_terminal_request(client: OwnerClient, operation: RegionOperation) {
    let request = RegionRequest {
        protocol_version: PROTOCOL_VERSION,
        request_id: request_id(),
        operation,
    };
    let _ = std::thread::Builder::new()
        .name("kvbm-rcommu-detach".into())
        .spawn(move || {
            if let Err(error) = client.call_blocking(&request) {
                tracing::error!(%error, "failed to close rcommu shared-memory attachment");
            }
        });
}

async fn abort_best_effort(client: &OwnerClient, lease: &AttachmentLease) {
    let request = RegionRequest {
        protocol_version: PROTOCOL_VERSION,
        request_id: request_id(),
        operation: RegionOperation::Abort {
            lease: lease.clone(),
        },
    };
    if let Err(error) = client.call(&request).await {
        tracing::error!(%error, "failed to abort rcommu shared-memory attachment");
    }
}

async fn status_best_effort(
    client: &OwnerClient,
    lease: &AttachmentLease,
) -> Option<AttachmentStatus> {
    let request = RegionRequest {
        protocol_version: PROTOCOL_VERSION,
        request_id: request_id(),
        operation: RegionOperation::GetStatus {
            lease: lease.clone(),
        },
    };
    match client.call(&request).await {
        Ok(RegionReply::Attachment(state)) => Some(state.status),
        _ => None,
    }
}

fn validate_header(descriptor: &SharedRegionDescriptor) -> anyhow::Result<()> {
    if descriptor.protocol_version != PROTOCOL_VERSION
        || descriptor.header_len != HEADER_LEN as u64
        || descriptor.data_offset < HEADER_LEN as u64
    {
        anyhow::bail!("rcommu shared-memory descriptor version or offsets are unsupported");
    }
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 || !descriptor.data_offset.is_multiple_of(page_size as u64) {
        anyhow::bail!("rcommu shared-memory data offset is not page aligned");
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&descriptor.path)?;
    let metadata = file.metadata()?;
    if metadata.dev() != descriptor.file_identity.device
        || metadata.ino() != descriptor.file_identity.inode
    {
        anyhow::bail!("rcommu shared-memory file identity changed before attach");
    }
    let required_len = descriptor
        .data_offset
        .checked_add(descriptor.data_len)
        .ok_or_else(|| anyhow::anyhow!("rcommu shared-memory data range overflows"))?;
    if metadata.len() < required_len {
        anyhow::bail!("rcommu shared-memory file is shorter than its descriptor");
    }
    let mut header = [0_u8; HEADER_LEN];
    file.read_exact_at(&mut header, 0)?;
    if &header[..8] != HEADER_MAGIC
        || get_u16(&header, 8) != PROTOCOL_VERSION
        || get_u16(&header, 10) as usize != HEADER_LEN
    {
        anyhow::bail!("rcommu shared-memory header magic or version mismatch");
    }
    if get_u32(&header, STATE_OFFSET) != 1 {
        anyhow::bail!("rcommu shared-memory region is not ready");
    }
    fence(Ordering::Acquire);
    let mut expected_digest = [0_u8; 32];
    expected_digest.copy_from_slice(&header[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + 32]);
    let mut hasher = blake3::Hasher::new();
    hasher.update(&header[..STATE_OFFSET]);
    hasher.update(&header[STATE_OFFSET + 4..HEADER_DIGEST_OFFSET]);
    let computed_digest = *hasher.finalize().as_bytes();
    if !constant_time_eq(&expected_digest, &computed_digest)
        || !constant_time_eq(&expected_digest, &descriptor.header_digest)
    {
        anyhow::bail!("rcommu shared-memory header digest mismatch");
    }

    let mut region_id = [0_u8; 16];
    region_id.copy_from_slice(&header[REGION_ID_OFFSET..REGION_ID_OFFSET + 16]);
    let mut layout_fingerprint = [0_u8; 32];
    layout_fingerprint
        .copy_from_slice(&header[LAYOUT_FINGERPRINT_OFFSET..LAYOUT_FINGERPRINT_OFFSET + 32]);
    let observed_owner = OwnerIdentity {
        domain_fingerprint: get_u64(&header, OWNER_DOMAIN_OFFSET),
        owner_epoch: get_u64(&header, OWNER_EPOCH_OFFSET),
    };
    let observed_region = RegionIdentity {
        region_id,
        region_generation: get_u64(&header, REGION_GENERATION_OFFSET),
    };
    if observed_owner != descriptor.owner {
        anyhow::bail!("rcommu shared-memory owner epoch is stale");
    }
    if observed_region != descriptor.region {
        anyhow::bail!("rcommu shared-memory region generation is stale");
    }
    if get_u64(&header, DATA_OFFSET_OFFSET) != descriptor.data_offset
        || get_u64(&header, DATA_LEN_OFFSET) != descriptor.data_len
        || get_u64(&header, DATA_ALIGNMENT_OFFSET) != descriptor.data_alignment
        || layout_fingerprint != descriptor.layout_fingerprint
    {
        anyhow::bail!("rcommu shared-memory header does not match its descriptor");
    }
    Ok(())
}

fn get_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("fixed header"))
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed header"))
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed header"))
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .fold(0_u8, |difference, (left, right)| {
                difference | (left ^ right)
            })
            == 0
}

fn request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn random_nonzero() -> u64 {
    loop {
        let value = rand::random();
        if value != 0 {
            return value;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::{FileExt, MetadataExt};

    use super::*;

    #[test]
    fn owner_wire_request_uses_versioned_tagged_contract() {
        let request = RegionRequest {
            protocol_version: PROTOCOL_VERSION,
            request_id: "request-1".into(),
            operation: RegionOperation::Open {
                client: ClientIdentity {
                    worker_id: "worker-0".into(),
                    worker_epoch: 9,
                    pid: 10,
                },
                spec: SharedRegionSpec {
                    region_key: "deployment/instance/rank-0/g2".into(),
                    data_len: 4096,
                    data_alignment: 4096,
                    numa_node: None,
                    layout_fingerprint: [7; 32],
                    access: RegionAccess::ExclusiveReadWrite,
                },
            },
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["protocol_version"], PROTOCOL_VERSION);
        assert_eq!(value["request_id"], "request-1");
        assert_eq!(value["operation"]["operation"], "open");
        assert_eq!(value["operation"]["spec"]["access"], "exclusive_read_write");
    }

    #[test]
    fn region_key_placeholders_are_resolved_per_worker() {
        let config = RcommuShmConfig {
            endpoint: "/tmp/owner.sock".into(),
            region_key: "deployment/device-{device_id}/rank-{rank}/pid-{pid}".into(),
            numa_node: None,
            attach_timeout: Duration::from_secs(1),
            worker_epoch: 1,
        };
        let resolved = config.for_worker(3, Some(7));
        assert_eq!(
            resolved.region_key,
            format!("deployment/device-3/rank-7/pid-{}", std::process::id())
        );
    }

    #[test]
    fn validates_owner_header_identity_digest_and_bounds() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(8192).unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let owner = OwnerIdentity {
            domain_fingerprint: 11,
            owner_epoch: 12,
        };
        let region = RegionIdentity {
            region_id: [13; 16],
            region_generation: 14,
        };
        let layout_fingerprint = [15; 32];
        let mut header = [0_u8; HEADER_LEN];
        header[..8].copy_from_slice(HEADER_MAGIC);
        header[8..10].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        header[10..12].copy_from_slice(&(HEADER_LEN as u16).to_le_bytes());
        header[STATE_OFFSET..STATE_OFFSET + 4].copy_from_slice(&1_u32.to_le_bytes());
        header[OWNER_DOMAIN_OFFSET..OWNER_DOMAIN_OFFSET + 8]
            .copy_from_slice(&owner.domain_fingerprint.to_le_bytes());
        header[OWNER_EPOCH_OFFSET..OWNER_EPOCH_OFFSET + 8]
            .copy_from_slice(&owner.owner_epoch.to_le_bytes());
        header[REGION_ID_OFFSET..REGION_ID_OFFSET + 16].copy_from_slice(&region.region_id);
        header[REGION_GENERATION_OFFSET..REGION_GENERATION_OFFSET + 8]
            .copy_from_slice(&region.region_generation.to_le_bytes());
        header[DATA_OFFSET_OFFSET..DATA_OFFSET_OFFSET + 8]
            .copy_from_slice(&(HEADER_LEN as u64).to_le_bytes());
        header[DATA_LEN_OFFSET..DATA_LEN_OFFSET + 8].copy_from_slice(&4096_u64.to_le_bytes());
        header[DATA_ALIGNMENT_OFFSET..DATA_ALIGNMENT_OFFSET + 8]
            .copy_from_slice(&4096_u64.to_le_bytes());
        header[LAYOUT_FINGERPRINT_OFFSET..LAYOUT_FINGERPRINT_OFFSET + 32]
            .copy_from_slice(&layout_fingerprint);
        let mut hasher = blake3::Hasher::new();
        hasher.update(&header[..STATE_OFFSET]);
        hasher.update(&header[STATE_OFFSET + 4..HEADER_DIGEST_OFFSET]);
        let digest = *hasher.finalize().as_bytes();
        header[HEADER_DIGEST_OFFSET..HEADER_DIGEST_OFFSET + 32].copy_from_slice(&digest);
        file.as_file().write_all_at(&header, 0).unwrap();

        let descriptor = SharedRegionDescriptor {
            protocol_version: PROTOCOL_VERSION,
            owner,
            region,
            path: file.path().to_path_buf(),
            file_identity: FileIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            },
            header_len: HEADER_LEN as u64,
            data_offset: HEADER_LEN as u64,
            data_len: 4096,
            data_alignment: 4096,
            layout_fingerprint,
            header_digest: digest,
        };
        validate_header(&descriptor).unwrap();

        let mut forged = descriptor;
        forged.header_digest[0] ^= 1;
        assert!(validate_header(&forged).is_err());
    }

    #[tokio::test]
    async fn unhealthy_owner_status_cancels_worker() {
        let temporary = tempfile::TempDir::new().unwrap();
        let endpoint = temporary.path().join("owner.sock");
        let listener = tokio::net::UnixListener::bind(&endpoint).unwrap();
        let lease = AttachmentLease {
            attachment_id: [1; 16],
            attachment_generation: 2,
            owner: OwnerIdentity {
                domain_fingerprint: 3,
                owner_epoch: 4,
            },
            region: RegionIdentity {
                region_id: [5; 16],
                region_generation: 6,
            },
            client: ClientIdentity {
                worker_id: "worker".into(),
                worker_epoch: 7,
                pid: std::process::id(),
            },
            capability: [8; 32],
        };
        let expected_lease = lease.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request: RegionRequest = read_frame(&mut stream).await.unwrap();
            assert!(matches!(
                request.operation,
                RegionOperation::GetStatus { .. }
            ));
            write_frame(
                &mut stream,
                &RegionResponse {
                    protocol_version: PROTOCOL_VERSION,
                    request_id: request.request_id,
                    result: Ok(RegionReply::Attachment(AttachmentState {
                        attachment_id: expected_lease.attachment_id,
                        region: expected_lease.region,
                        status: AttachmentStatus::Closed,
                    })),
                },
            )
            .await
            .unwrap();
        });
        let client = OwnerClient {
            endpoint,
            timeout: Duration::from_millis(100),
        };
        let health_cancellation = CancellationToken::new();
        let failure = CancellationToken::new();
        let healthy = Arc::new(AtomicBool::new(true));
        let monitor = tokio::spawn(monitor_owner_health(
            client,
            lease,
            health_cancellation.clone(),
            failure.clone(),
            Arc::clone(&healthy),
        ));
        tokio::time::timeout(Duration::from_secs(2), failure.cancelled())
            .await
            .expect("health monitor should cancel the worker");
        assert!(!healthy.load(Ordering::Acquire));
        health_cancellation.cancel();
        monitor.await.unwrap();
        server.await.unwrap();
    }
}
