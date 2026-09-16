// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C-compatible ABI definitions for dynamically loaded AISimulate placement
//! policies.
//!
//! This crate contains the contract only: no loader, simulator, router, or
//! allocator crosses this boundary. Inputs are borrowed for a call. A plugin
//! owns result slices and diagnostic bytes until the host releases them with
//! the callbacks from the same [`PluginVTableV1`].

use std::ffi::{c_char, c_void};

/// The first compatible major version of the placement plugin ABI.
pub const PLACEMENT_PLUGIN_ABI_MAJOR_V1: u32 = 1;
/// Latest compatible minor version of the placement plugin ABI major V1.
///
/// Minor one adds routing-safe prompt identity and tagged admission metadata.
pub const PLACEMENT_PLUGIN_ABI_MINOR_V1: u32 = 1;
/// The first version of the placement creation payload.
pub const PLACEMENT_CREATE_PAYLOAD_VERSION_V1: u32 = 1;
/// Maximum workers accepted in a V1 creation payload.
pub const MAX_CREATE_WORKERS_V1: u64 = 65_536;
/// Maximum option bytes accepted in a V1 creation payload.
pub const MAX_CREATE_OPTION_BYTES_V1: u64 = 1_048_576;
/// Maximum mutations accepted in one V1 batch.
pub const MAX_BATCH_MUTATIONS_V1: u64 = 65_536;
/// Maximum prompt token identities accepted on one V1 admission.
pub const MAX_ADMISSION_PROMPT_TOKEN_IDS_V1: u64 = 1_048_576;
/// Maximum canonical block hashes accepted per identity form on one admission.
pub const MAX_ADMISSION_PROMPT_BLOCK_HASHES_V1: u64 = 1_048_576;
/// Maximum tagged metadata bytes accepted on one V1 admission.
pub const MAX_ADMISSION_METADATA_BYTES_V1: u64 = 65_536;

/// A versioned C-ABI operation status.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusV1(pub i32);

impl StatusV1 {
    /// The operation completed successfully.
    pub const OK: Self = Self(0);
    /// The caller supplied an invalid record, pointer, or bound.
    pub const INVALID_ARGUMENT: Self = Self(1);
    /// The provider does not implement the requested operation.
    pub const UNSUPPORTED: Self = Self(2);
    /// The provider rejected a validly encoded placement operation.
    pub const REJECTED: Self = Self(3);
    /// The provider failed internally without unwinding across the ABI.
    pub const INTERNAL: Self = Self(4);
}

/// A byte slice borrowed for an input call or owned by an output producer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ByteSliceV1 {
    /// Start of bytes, or null only when `len` is zero.
    pub data: *const u8,
    /// Byte count represented with a fixed-width scalar.
    pub len: u64,
}

impl ByteSliceV1 {
    /// Empty bytes.
    pub const EMPTY: Self = Self {
        data: std::ptr::null(),
        len: 0,
    };
}

impl Default for ByteSliceV1 {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// A borrowed sequence of materialized prompt token identifiers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TokenIdSliceV1 {
    /// Start of token identifiers, or null only when `len` is zero.
    pub data: *const u32,
    /// Number of token identifiers.
    pub len: u64,
}

impl Default for TokenIdSliceV1 {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            len: 0,
        }
    }
}

/// A request identifier independent of a Rust UUID implementation.
pub type RequestIdV1 = [u8; 16];

/// A borrowed sequence of scheduler identifiers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SchedulerIdSliceV1 {
    /// Start of identifiers, or null only when `len` is zero.
    pub data: *const u64,
    /// Number of identifiers.
    pub len: u64,
}

impl Default for SchedulerIdSliceV1 {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            len: 0,
        }
    }
}

/// A borrowed sequence of KV block hashes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockHashSliceV1 {
    /// Start of hashes, or null only when `len` is zero.
    pub data: *const u64,
    /// Number of hashes.
    pub len: u64,
}

impl Default for BlockHashSliceV1 {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            len: 0,
        }
    }
}

/// Resolved topology for one worker.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WorkerTopologyV1 {
    /// Stable worker identity.
    pub worker_id: u64,
    /// Scheduler identities served by this worker.
    pub scheduler_ids: SchedulerIdSliceV1,
}

/// A borrowed sequence of resolved workers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WorkerTopologySliceV1 {
    /// Start of workers, or null only when `len` is zero.
    pub data: *const WorkerTopologyV1,
    /// Number of workers.
    pub len: u64,
}

impl Default for WorkerTopologySliceV1 {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            len: 0,
        }
    }
}

impl From<&WorkerTopologyV1> for WorkerTopologySliceV1 {
    fn from(value: &WorkerTopologyV1) -> Self {
        Self {
            data: value,
            len: 1,
        }
    }
}

/// Capacity facts for one worker at creation or during a worker transition.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WorkerCapacityV1 {
    /// Worker to which these facts apply.
    pub worker_id: u64,
    /// Total KV-cache blocks managed by the worker.
    pub total_kv_blocks: u64,
    /// KV-cache blocks currently available to new work.
    pub available_kv_blocks: u64,
    /// Maximum simultaneously running requests.
    pub max_running_requests: u64,
    /// Reserved for compatible-minor capacity flags.
    pub flags: u32,
    /// Reserved for compatible-minor extension.
    pub reserved: u32,
}

/// A borrowed sequence of worker capacity facts.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WorkerCapacitySliceV1 {
    /// Start of facts, or null only when `len` is zero.
    pub data: *const WorkerCapacityV1,
    /// Number of facts.
    pub len: u64,
}

impl Default for WorkerCapacitySliceV1 {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            len: 0,
        }
    }
}

/// Provider-independent bounds negotiated during creation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementLimitsV1 {
    /// Maximum mutations permitted in one batch.
    pub max_mutations: u64,
    /// Maximum admission decision records returned in one result.
    pub max_admission_results: u64,
    /// Maximum released-placement records returned in one result.
    pub max_released: u64,
    /// Maximum UTF-8 diagnostic bytes returned in one result.
    pub max_diagnostic_bytes: u64,
}

/// Bounded, versioned input used to create one configured placement policy.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementCreateRequestV1 {
    /// Size of this structure as compiled by the caller.
    pub struct_size: u32,
    /// Version of the creation payload, starting at one.
    pub payload_version: u32,
    /// Reserved for compatible-minor flags.
    pub flags: u32,
    /// Reserved for compatible-minor extension.
    pub reserved: u32,
    /// Deterministic selector seed supplied by AISimulate.
    pub selector_seed: [u8; 32],
    /// Fully resolved worker and scheduler topology.
    pub workers: WorkerTopologySliceV1,
    /// Capacity facts keyed by worker identity.
    pub capacities: WorkerCapacitySliceV1,
    /// Namespace identifying the provider-specific options format.
    pub options_namespace: ByteSliceV1,
    /// Bounded provider-specific options in the declared namespace.
    pub provider_options: ByteSliceV1,
    /// Data-plane limits negotiated before the instance exists.
    pub limits: PlacementLimitsV1,
}

/// A backward-compatible short name for the placement creation payload.
pub type CreateRequestV1 = PlacementCreateRequestV1;

/// Routing-safe prompt identity supplied with one admission.
///
/// The identity forms are deliberately separate from provider creation options:
/// a provider can route with these stable identities without interpreting an
/// opaque provider-specific configuration payload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PromptIdentityV1 {
    /// Presence bits selecting the identity slices below.
    pub flags: u32,
    /// Reserved for compatible-minor identity flags.
    pub reserved: u32,
    /// Materialized token IDs in prompt order when present.
    pub materialized_token_ids: TokenIdSliceV1,
    /// Canonical local hashes for complete prompt blocks when present.
    pub local_block_hashes: BlockHashSliceV1,
    /// Canonical rolling sequence hashes for complete prompt blocks when present.
    pub sequence_block_hashes: BlockHashSliceV1,
}

impl PromptIdentityV1 {
    /// `materialized_token_ids` is present. An empty slice is a known-empty prompt.
    pub const MATERIALIZED_TOKEN_IDS_PRESENT: u32 = 1 << 0;
    /// `local_block_hashes` is present. An empty slice has no complete blocks.
    pub const LOCAL_BLOCK_HASHES_PRESENT: u32 = 1 << 1;
    /// `sequence_block_hashes` is present. An empty slice has no complete blocks.
    pub const SEQUENCE_BLOCK_HASHES_PRESENT: u32 = 1 << 2;

    const KNOWN_FLAGS: u32 = Self::MATERIALIZED_TOKEN_IDS_PRESENT
        | Self::LOCAL_BLOCK_HASHES_PRESENT
        | Self::SEQUENCE_BLOCK_HASHES_PRESENT;

    /// No prompt identity was materialized by the caller.
    pub const OMITTED: Self = Self {
        flags: 0,
        reserved: 0,
        materialized_token_ids: TokenIdSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        local_block_hashes: BlockHashSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        sequence_block_hashes: BlockHashSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
    };
}

impl Default for PromptIdentityV1 {
    fn default() -> Self {
        Self::OMITTED
    }
}

/// Discriminant for [`PlacementMetadataV1`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionMetadataFormatV1(pub u32);

impl AdmissionMetadataFormatV1 {
    /// No metadata is carried; the byte slice must be canonical empty.
    pub const NONE: Self = Self(0);
    /// UTF-8 encoded JSON metadata. Its meaning is application-neutral.
    pub const JSON_UTF8: Self = Self(1);
}

/// Bounded, tagged metadata associated with one admission.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementMetadataV1 {
    /// Self-describing encoding of `bytes`.
    pub format: AdmissionMetadataFormatV1,
    /// Reserved for compatible-minor metadata flags.
    pub flags: u32,
    /// Metadata bytes in the declared format.
    pub bytes: ByteSliceV1,
}

impl PlacementMetadataV1 {
    /// Canonical absent metadata.
    pub const EMPTY: Self = Self {
        format: AdmissionMetadataFormatV1::NONE,
        flags: 0,
        bytes: ByteSliceV1::EMPTY,
    };
}

impl Default for PlacementMetadataV1 {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Input facts for a request admission mutation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementAdmissionV1 {
    /// Request being admitted.
    pub request_id: RequestIdV1,
    /// Optional request/session fields present in `flags`.
    pub flags: u32,
    /// Router priority.
    pub priority: i32,
    /// Number of prompt tokens in the request.
    pub prompt_tokens: u64,
    /// Requested maximum generated tokens.
    pub max_output_tokens: u64,
    /// Routing-safe prompt identity, with explicit omitted versus known-empty forms.
    pub prompt_identity: PromptIdentityV1,
    /// Tagged, bounded application-neutral request metadata.
    pub metadata: PlacementMetadataV1,
    /// Optional session identifier encoded as UTF-8.
    pub session_id: ByteSliceV1,
}

/// An engine observation that may affect routing.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EngineObservationV1 {
    /// Worker that emitted the observation.
    pub worker_id: u64,
    /// Scheduler that emitted the observation.
    pub scheduler_id: u64,
    /// Request associated with the observation, or all zeros when absent.
    pub request_id: RequestIdV1,
    /// Engine event discriminant.
    pub event_kind: u32,
    /// Event-specific flags.
    pub flags: u32,
    /// Event-specific count or token value.
    pub value: u64,
}

/// A KV-cache observation that may affect cache-aware placement.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvObservationV1 {
    /// Worker that owns the cache event.
    pub worker_id: u64,
    /// Scheduler associated with the event.
    pub scheduler_id: u64,
    /// Monotonic producer event identity.
    pub event_id: u64,
    /// KV event discriminant.
    pub event_kind: u32,
    /// Event-specific flags.
    pub flags: u32,
    /// Event block hashes in their producer order.
    pub block_hashes: BlockHashSliceV1,
}

/// A scheduler command observation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SchedulerCommandObservationV1 {
    /// Worker receiving the command.
    pub worker_id: u64,
    /// Scheduler receiving the command.
    pub scheduler_id: u64,
    /// Request associated with the command, or all zeros when absent.
    pub request_id: RequestIdV1,
    /// Scheduler command discriminant.
    pub command_kind: u32,
    /// Command-specific flags.
    pub flags: u32,
    /// Command-specific count or token value.
    pub value: u64,
}

/// A scheduler-pass boundary observation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PassBoundaryObservationV1 {
    /// Worker whose pass changed state.
    pub worker_id: u64,
    /// Scheduler whose pass changed state.
    pub scheduler_id: u64,
    /// Pass discriminant.
    pub pass_kind: u32,
    /// Boundary discriminant, such as start or end.
    pub boundary_kind: u32,
    /// Work observed in the pass.
    pub request_count: u64,
    /// Token count observed in the pass.
    pub token_count: u64,
}

/// An offload or restore observation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OffloadObservationV1 {
    /// Worker whose memory tier changed.
    pub worker_id: u64,
    /// Scheduler associated with the operation.
    pub scheduler_id: u64,
    /// Request associated with the operation, or all zeros when absent.
    pub request_id: RequestIdV1,
    /// Offload operation discriminant.
    pub operation_kind: u32,
    /// Operation-specific flags.
    pub flags: u32,
    /// Number of affected blocks.
    pub block_count: u64,
}

/// Request lifecycle data shared by cancel, terminal, and prefill mutations.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RequestLifecycleV1 {
    /// Request whose lifecycle advanced.
    pub request_id: RequestIdV1,
    /// Terminal status or lifecycle-specific flags.
    pub flags: u32,
    /// Reserved for compatible-minor extension.
    pub reserved: u32,
}

/// Payload of one ordered placement mutation.
#[repr(C)]
#[derive(Clone, Copy)]
pub union PlacementMutationPayloadV1 {
    /// Admission input.
    pub admission: PlacementAdmissionV1,
    /// Engine observation input.
    pub engine: EngineObservationV1,
    /// KV-cache observation input.
    pub kv: KvObservationV1,
    /// Scheduler command input.
    pub scheduler_command: SchedulerCommandObservationV1,
    /// Scheduler pass-boundary input.
    pub pass_boundary: PassBoundaryObservationV1,
    /// Offload input.
    pub offload: OffloadObservationV1,
    /// Pending cancellation, terminal notification, or prefill completion.
    pub request_lifecycle: RequestLifecycleV1,
    /// Worker transition input.
    pub worker: WorkerTopologyV1,
}

/// Discriminants for [`PlacementMutationV1::kind`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementMutationKindV1(pub u32);

impl PlacementMutationKindV1 {
    /// Admit and immediately place or queue a request.
    pub const ADMIT: Self = Self(1);
    /// Observe an engine event.
    pub const OBSERVE_ENGINE: Self = Self(2);
    /// Observe a KV-cache event.
    pub const OBSERVE_KV: Self = Self(3);
    /// Observe a scheduler command.
    pub const OBSERVE_SCHEDULER_COMMAND: Self = Self(4);
    /// Observe a scheduler-pass boundary.
    pub const OBSERVE_PASS_BOUNDARY: Self = Self(5);
    /// Observe an offload or restore event.
    pub const OBSERVE_OFFLOAD: Self = Self(6);
    /// Cancel a pending request.
    pub const CANCEL_PENDING: Self = Self(7);
    /// Notify that a request became terminal.
    pub const REQUEST_TERMINAL: Self = Self(8);
    /// Notify that a request completed prefill.
    pub const PREFILL_COMPLETED: Self = Self(9);
    /// Notify that a worker became ready.
    pub const WORKER_READY: Self = Self(10);
    /// Notify that a worker is draining.
    pub const WORKER_DRAINING: Self = Self(11);
    /// Notify that a worker was removed.
    pub const WORKER_REMOVED: Self = Self(12);
    /// Notify that topology mutations are settled.
    pub const TOPOLOGY_SETTLED: Self = Self(13);
}

/// One ordered lifecycle mutation. The batch order is normative.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlacementMutationV1 {
    /// Size of this record as compiled by the caller.
    pub struct_size: u32,
    /// Operation discriminant selecting `payload`.
    pub kind: PlacementMutationKindV1,
    /// Operation-specific flags.
    pub flags: u32,
    /// Monotonic batch-local sequence for diagnostics.
    pub sequence: u64,
    /// Simulated time at which this mutation occurs.
    pub now_ms: f64,
    /// Payload selected by `kind`.
    pub payload: PlacementMutationPayloadV1,
}

impl PlacementMutationV1 {
    /// Creates a topology-settled lifecycle mutation.
    #[must_use]
    pub const fn topology_settled(now_ms: f64) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            kind: PlacementMutationKindV1::TOPOLOGY_SETTLED,
            flags: 0,
            sequence: 0,
            now_ms,
            payload: PlacementMutationPayloadV1 {
                request_lifecycle: RequestLifecycleV1 {
                    request_id: [0; 16],
                    flags: 0,
                    reserved: 0,
                },
            },
        }
    }
}

/// A borrowed ordered mutation batch.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlacementMutationSliceV1 {
    /// Start of mutations, or null only when `len` is zero.
    pub data: *const PlacementMutationV1,
    /// Number of mutations in normative application order.
    pub len: u64,
}

/// Cache-overlap facts reported for a placement.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementCacheSampleV1 {
    /// Cache facts are present when this bit is set.
    pub flags: u32,
    /// Prefix blocks available on the selected worker.
    pub overlap_blocks: u32,
    /// Largest prefix overlap available on an eligible worker.
    pub best_available_overlap_blocks: u32,
    /// Input-sequence blocks considered by the policy.
    pub isl_blocks: u32,
}

impl PlacementCacheSampleV1 {
    /// Cache facts are present.
    pub const PRESENT: u32 = 1;
}

/// A selected placement returned for an admission or release.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementV1 {
    /// Request selected for this worker.
    pub request_id: RequestIdV1,
    /// Stable selected worker identity.
    pub worker_id: u64,
    /// Scheduler selected within that worker.
    pub scheduler_id: u64,
    /// Cache tokens reported to the caller.
    pub reported_overlap_tokens: u64,
    /// Cache-overlap facts when present.
    pub cache_sample: PlacementCacheSampleV1,
    /// Policy replica that made the decision, or zero when not replicated.
    pub placement_replica_id: u64,
    /// Reserved for compatible-minor extension.
    pub reserved: u64,
}

/// Admission-decision discriminants.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionDecisionV1(pub u32);

impl AdmissionDecisionV1 {
    /// The policy immediately selected the included placement.
    pub const IMMEDIATE: Self = Self(1);
    /// The policy accepted the request but queued it.
    pub const QUEUED: Self = Self(2);
}

/// The decision associated with one admission mutation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementResultV1 {
    /// Request admitted by the corresponding mutation.
    pub request_id: RequestIdV1,
    /// Immediate or queued admission decision.
    pub decision: AdmissionDecisionV1,
    /// Reserved for C alignment and compatible-minor extension.
    pub reserved: [u8; 4],
    /// Selected placement when `decision` is immediate.
    pub placement: PlacementV1,
}

/// A plugin-owned sequence of admission results.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlacementResultSliceV1 {
    /// Start of records, or null only when `len` is zero.
    pub data: *const PlacementResultV1,
    /// Number of records.
    pub len: u64,
}

impl From<&PlacementResultV1> for PlacementResultSliceV1 {
    fn from(value: &PlacementResultV1) -> Self {
        Self {
            data: value,
            len: 1,
        }
    }
}

/// A plugin-owned sequence of released placements.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlacementSliceV1 {
    /// Start of records, or null only when `len` is zero.
    pub data: *const PlacementV1,
    /// Number of records.
    pub len: u64,
}

impl From<&PlacementV1> for PlacementSliceV1 {
    fn from(value: &PlacementV1) -> Self {
        Self {
            data: value,
            len: 1,
        }
    }
}

/// A structured diagnostic attached to a placement batch result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementDiagnosticV1 {
    /// Provider-defined diagnostic class.
    pub code: u32,
    /// Mutation index associated with this diagnostic, or `u64::MAX`.
    pub mutation_index: u64,
    /// Bounded UTF-8 diagnostic bytes.
    pub message: ByteSliceV1,
}

impl PlacementDiagnosticV1 {
    /// Empty diagnostic used to initialize an output buffer.
    pub const EMPTY: Self = Self {
        code: 0,
        mutation_index: u64::MAX,
        message: ByteSliceV1::EMPTY,
    };
}

/// A plugin-owned sequence of batch diagnostics.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlacementDiagnosticSliceV1 {
    /// Start of diagnostics, or null only when `len` is zero.
    pub data: *const PlacementDiagnosticV1,
    /// Number of diagnostics.
    pub len: u64,
}

impl From<&PlacementDiagnosticV1> for PlacementDiagnosticSliceV1 {
    fn from(value: &PlacementDiagnosticV1) -> Self {
        Self {
            data: value,
            len: 1,
        }
    }
}

/// Result of applying one ordered mutation batch.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlacementBatchResultV1 {
    /// Size of this record as compiled by the plugin.
    pub struct_size: u32,
    /// Reserved for compatible-minor result flags.
    pub flags: u32,
    /// Number of mutations committed before completion or failure.
    pub applied_mutations: u64,
    /// Final pending-request count after the committed prefix.
    pub pending_count: u64,
    /// One decision for each admitted request in the committed prefix.
    pub admission_results: PlacementResultSliceV1,
    /// Requests released by any mutation in the committed prefix.
    pub released: PlacementSliceV1,
    /// Structured diagnostics for the batch.
    pub diagnostics: PlacementDiagnosticSliceV1,
}

/// An opaque plugin-owned placement instance.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct PlacementHandleV1(pub *mut c_void);

/// Creates one fully configured placement instance.
pub type CreateFnV1 = unsafe extern "C" fn(
    PlacementCreateRequestV1,
    *mut PlacementHandleV1,
    *mut ByteSliceV1,
) -> StatusV1;

/// Plugin identity and ABI compatibility metadata.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginDescriptorV1 {
    /// ABI major implemented by this plugin.
    pub abi_major: u32,
    /// ABI minor implemented by this plugin.
    pub abi_minor: u32,
    /// Size of this descriptor as compiled by the plugin.
    pub struct_size: u32,
    /// Reserved for compatible-minor descriptor flags.
    pub flags: u32,
    /// Placement capabilities supported by this plugin.
    pub capabilities: u64,
    /// NUL-terminated static UTF-8 provider identifier.
    pub provider_id: *const c_char,
    /// Immutable table used for placement operations.
    pub vtable: *const PluginVTableV1,
}

// Safety: descriptors are process-lifetime immutable plugin metadata. Hosts
// validate them while retaining the shared library that owns them.
unsafe impl Sync for PluginDescriptorV1 {}

impl PluginDescriptorV1 {
    /// ABI major required by this V1 descriptor.
    pub const ABI_MAJOR: u32 = PLACEMENT_PLUGIN_ABI_MAJOR_V1;
    /// Minimum ABI minor required by this V1 descriptor.
    pub const ABI_MINOR: u32 = PLACEMENT_PLUGIN_ABI_MINOR_V1;
}

/// All required callable operations supplied by a placement plugin.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PluginVTableV1 {
    /// Size of this table as compiled by the plugin.
    pub struct_size: u32,
    /// Reserved for compatible-minor table flags.
    pub flags: u32,
    /// Creates one configured placement instance.
    pub create: Option<CreateFnV1>,
    /// Applies an ordered lifecycle batch atomically up to its reported prefix.
    pub apply_batch: Option<
        unsafe extern "C" fn(
            PlacementHandleV1,
            PlacementMutationSliceV1,
            *mut PlacementBatchResultV1,
        ) -> StatusV1,
    >,
    /// Releases all output slices returned in one batch result.
    pub release_results: Option<unsafe extern "C" fn(PlacementBatchResultV1)>,
    /// Releases a diagnostic or create-error byte slice returned by this table.
    pub release_bytes: Option<unsafe extern "C" fn(ByteSliceV1)>,
    /// Returns the last provider error as plugin-owned UTF-8 bytes.
    pub last_error: Option<unsafe extern "C" fn(PlacementHandleV1, *mut ByteSliceV1) -> StatusV1>,
    /// Destroys one placement instance.
    pub destroy: Option<unsafe extern "C" fn(PlacementHandleV1)>,
}

impl PluginVTableV1 {
    /// Bytes a consumer must be able to read for every V1 operation.
    pub const REQUIRED_SIZE: usize = std::mem::size_of::<Self>();
}

/// Why a loaded V1 plugin descriptor cannot be used by a host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorValidationError {
    /// The loader received a null descriptor pointer.
    NullDescriptor,
    /// The plugin implements an incompatible ABI major.
    IncompatibleMajor,
    /// The plugin predates the minimum compatible ABI minor.
    IncompatibleMinor,
    /// The descriptor was compiled with fewer fields than V1 requires.
    DescriptorTooSmall,
    /// The plugin did not identify its provider.
    MissingProviderId,
    /// The plugin did not supply an operation table.
    MissingVTable,
    /// The table was compiled with fewer fields than V1 requires.
    VTableTooSmall,
    /// A required V1 operation was absent.
    MissingOperation,
}

/// Why a V1 placement creation payload cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateRequestValidationError {
    /// The request was compiled with fewer fields than V1 requires.
    RequestTooSmall,
    /// The request uses an unsupported payload version.
    UnsupportedPayloadVersion,
    /// A non-empty slice has a null pointer.
    NullSlice,
    /// The topology or capacity slice exceeds the V1 bound.
    TooManyWorkers,
    /// Provider-specific options exceed the V1 bound.
    OptionsTooLarge,
    /// A negotiated result or mutation bound is invalid.
    InvalidLimits,
}

/// Validates that a plugin descriptor has the complete V1 shape.
///
/// # Safety
///
/// `descriptor` must be null or point to readable [`PluginDescriptorV1`]
/// memory. A non-null table pointer must point to readable [`PluginVTableV1`]
/// memory. The loader retains the library that owns both records.
pub unsafe fn validate_descriptor_v1(
    descriptor: *const PluginDescriptorV1,
) -> Result<*const PluginVTableV1, DescriptorValidationError> {
    if descriptor.is_null() {
        return Err(DescriptorValidationError::NullDescriptor);
    }
    // Safety: required by this function's contract.
    let descriptor = unsafe { &*descriptor };
    if descriptor.abi_major != PLACEMENT_PLUGIN_ABI_MAJOR_V1 {
        return Err(DescriptorValidationError::IncompatibleMajor);
    }
    if descriptor.abi_minor < PLACEMENT_PLUGIN_ABI_MINOR_V1 {
        return Err(DescriptorValidationError::IncompatibleMinor);
    }
    if (descriptor.struct_size as usize) < std::mem::size_of::<PluginDescriptorV1>() {
        return Err(DescriptorValidationError::DescriptorTooSmall);
    }
    if descriptor.provider_id.is_null() {
        return Err(DescriptorValidationError::MissingProviderId);
    }
    if descriptor.vtable.is_null() {
        return Err(DescriptorValidationError::MissingVTable);
    }
    // Safety: required by this function's contract.
    let vtable = unsafe { &*descriptor.vtable };
    if (vtable.struct_size as usize) < PluginVTableV1::REQUIRED_SIZE {
        return Err(DescriptorValidationError::VTableTooSmall);
    }
    if vtable.create.is_none()
        || vtable.apply_batch.is_none()
        || vtable.release_results.is_none()
        || vtable.release_bytes.is_none()
        || vtable.last_error.is_none()
        || vtable.destroy.is_none()
    {
        return Err(DescriptorValidationError::MissingOperation);
    }
    Ok(descriptor.vtable)
}

/// Validates the bounded creation payload without creating an instance.
pub fn validate_create_request_v1(
    request: &PlacementCreateRequestV1,
) -> Result<(), CreateRequestValidationError> {
    if (request.struct_size as usize) < std::mem::size_of::<PlacementCreateRequestV1>() {
        return Err(CreateRequestValidationError::RequestTooSmall);
    }
    if request.payload_version != PLACEMENT_CREATE_PAYLOAD_VERSION_V1 {
        return Err(CreateRequestValidationError::UnsupportedPayloadVersion);
    }
    if !valid_slice(request.workers.data, request.workers.len)
        || !valid_slice(request.capacities.data, request.capacities.len)
        || !valid_slice(
            request.options_namespace.data,
            request.options_namespace.len,
        )
        || !valid_slice(request.provider_options.data, request.provider_options.len)
    {
        return Err(CreateRequestValidationError::NullSlice);
    }
    if request.workers.len > MAX_CREATE_WORKERS_V1 || request.capacities.len > MAX_CREATE_WORKERS_V1
    {
        return Err(CreateRequestValidationError::TooManyWorkers);
    }
    if request.options_namespace.len > MAX_CREATE_OPTION_BYTES_V1
        || request.provider_options.len > MAX_CREATE_OPTION_BYTES_V1
    {
        return Err(CreateRequestValidationError::OptionsTooLarge);
    }
    if request.limits.max_mutations == 0
        || request.limits.max_mutations > MAX_BATCH_MUTATIONS_V1
        || request.limits.max_admission_results > MAX_BATCH_MUTATIONS_V1
        || request.limits.max_released > MAX_BATCH_MUTATIONS_V1
        || request.limits.max_diagnostic_bytes > MAX_CREATE_OPTION_BYTES_V1
    {
        return Err(CreateRequestValidationError::InvalidLimits);
    }
    Ok(())
}

/// Validates a borrowed mutation batch's pointer, bound, and finite times.
///
/// # Safety
///
/// A non-empty `batch` must reference readable [`PlacementMutationV1`] records
/// for the duration of this call.
pub unsafe fn validate_mutation_batch_v1(batch: PlacementMutationSliceV1) -> Result<(), StatusV1> {
    if !valid_slice(batch.data, batch.len) || batch.len > MAX_BATCH_MUTATIONS_V1 {
        return Err(StatusV1::INVALID_ARGUMENT);
    }
    if batch.len == 0 {
        return Ok(());
    }
    // Safety: required by this function's contract and guarded by the null
    // check above.
    let mutations = unsafe { std::slice::from_raw_parts(batch.data, batch.len as usize) };
    for mutation in mutations {
        if (mutation.struct_size as usize) < std::mem::size_of::<PlacementMutationV1>()
            || !mutation.now_ms.is_finite()
        {
            return Err(StatusV1::INVALID_ARGUMENT);
        }
        if mutation.kind == PlacementMutationKindV1::ADMIT {
            // Safety: `admission` is the active union member for an ADMIT mutation.
            let admission = unsafe { mutation.payload.admission };
            if !valid_admission(&admission) {
                return Err(StatusV1::INVALID_ARGUMENT);
            }
        }
    }
    Ok(())
}

fn valid_admission(admission: &PlacementAdmissionV1) -> bool {
    let identity = admission.prompt_identity;
    if identity.reserved != 0 || identity.flags & !PromptIdentityV1::KNOWN_FLAGS != 0 {
        return false;
    }
    if !valid_optional_identity_slice(
        identity.flags,
        PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT,
        identity.materialized_token_ids.data,
        identity.materialized_token_ids.len,
        MAX_ADMISSION_PROMPT_TOKEN_IDS_V1,
    ) || !valid_optional_identity_slice(
        identity.flags,
        PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT,
        identity.local_block_hashes.data,
        identity.local_block_hashes.len,
        MAX_ADMISSION_PROMPT_BLOCK_HASHES_V1,
    ) || !valid_optional_identity_slice(
        identity.flags,
        PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT,
        identity.sequence_block_hashes.data,
        identity.sequence_block_hashes.len,
        MAX_ADMISSION_PROMPT_BLOCK_HASHES_V1,
    ) {
        return false;
    }
    if identity.flags & PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT != 0
        && identity.materialized_token_ids.len != admission.prompt_tokens
    {
        return false;
    }
    if identity.flags & PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT != 0
        && identity.flags & PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT != 0
        && identity.local_block_hashes.len != identity.sequence_block_hashes.len
    {
        return false;
    }

    let metadata = admission.metadata;
    if metadata.flags != 0
        || !valid_slice(metadata.bytes.data, metadata.bytes.len)
        || metadata.bytes.len > MAX_ADMISSION_METADATA_BYTES_V1
    {
        return false;
    }
    match metadata.format {
        AdmissionMetadataFormatV1::NONE => metadata.bytes.data.is_null() && metadata.bytes.len == 0,
        AdmissionMetadataFormatV1::JSON_UTF8 => {
            if metadata.bytes.len == 0 {
                return false;
            }
            // Safety: a non-empty metadata slice has a non-null pointer, checked above.
            let bytes = unsafe {
                std::slice::from_raw_parts(metadata.bytes.data, metadata.bytes.len as usize)
            };
            serde_json::from_slice::<serde_json::Value>(bytes).is_ok()
        }
        _ => false,
    }
}

fn valid_optional_identity_slice<T>(
    flags: u32,
    present: u32,
    data: *const T,
    len: u64,
    max: u64,
) -> bool {
    if flags & present == 0 {
        return data.is_null() && len == 0;
    }
    valid_slice(data, len) && len <= max
}

fn valid_slice<T>(data: *const T, len: u64) -> bool {
    len == 0 || !data.is_null()
}

/// Fixed entry point exported by every V1 placement plugin.
pub type PluginEntryV1 = unsafe extern "C" fn() -> *const PluginDescriptorV1;
