/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYN_SNAPSHOT_CUDA_VMM_PROTOCOL_H
#define DYN_SNAPSHOT_CUDA_VMM_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>
#include <sys/un.h>

#define DYN_VMM_MAGIC 0x44564d4dU
#define DYN_VMM_VERSION 5U
#define DYN_VMM_CAPABILITY_MAGIC 0x44564d43U
#define DYN_VMM_CAPABILITY_VERSION 2U
#define DYN_VMM_FABRIC_TOKEN_MAGIC 0x44564d46U
#define DYN_VMM_FABRIC_TOKEN_VERSION 2U
#define DYN_VMM_FABRIC_HANDLE_SIZE 64U
#define DYN_VMM_SOCKET_PREFIX "cuda-vmm-"
#define DYN_VMM_PARTICIPANT_ID_SIZE 33U
#define DYN_VMM_FABRIC_PARTICIPANT_ID_SIZE 32U
#define DYN_VMM_ALLOCATION_UUID_SIZE 16U
#define DYN_VMM_GPU_UUID_SIZE 16U
#define DYN_VMM_MAXIMUM_ALLOCATION_SIZE (UINT64_C(512) << 20)

enum dyn_vmm_operation {
  DYN_VMM_INSPECT = 1,
  DYN_VMM_READ_OWNER = 2,
  DYN_VMM_DETACH_IMPORTS = 3,
  DYN_VMM_DETACH_OWNERS = 4,
  DYN_VMM_RESTORE_OWNER = 5,
  DYN_VMM_RESTORE_IMPORT = 6,
  DYN_VMM_IDENTIFY = 7,
  DYN_VMM_SET_PLACEMENT = 8,
  DYN_VMM_EXPORT_OWNER = 9,
  DYN_VMM_RESTORE_MULTICAST = 10,
  DYN_VMM_FINALIZE_RESTORE = 11,
  DYN_VMM_ABORT_RESTORE = 12,
};

enum dyn_vmm_role {
  DYN_VMM_OWNER = 1,
  DYN_VMM_IMPORTER = 2,
};

enum dyn_vmm_phase {
  DYN_VMM_ACTIVE = 0,
  DYN_VMM_DETACHED = 1,
  DYN_VMM_RESTORED = 2,
  DYN_VMM_FAILED = 3,
};

enum dyn_vmm_object_kind {
  DYN_VMM_ALLOCATION = 1,
  DYN_VMM_MULTICAST = 2,
};

enum dyn_vmm_record_flags {
  DYN_VMM_APPLICATION_HANDLE_LIVE = 1U << 0,
  DYN_VMM_RETAIN_RESTORE_HANDLE = 1U << 1,
};

enum dyn_vmm_multicast_bind_api {
  DYN_VMM_MULTICAST_BIND_MEM = 1,
  DYN_VMM_MULTICAST_BIND_MEM_V2 = 2,
};

struct dyn_vmm_header {
  uint32_t magic;
  uint16_t version;
  uint16_t operation;
  int32_t status;
  uint32_t count;
  uint32_t handle_type;
  uint32_t object_kind;
  uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  uint64_t object_id;
  uint64_t payload_size;
  char message[96];
  char participant_id[DYN_VMM_PARTICIPANT_ID_SIZE];
  uint8_t reserved_identity[71];
};

/*
 * CUDA structures are copied as opaque trailing bytes. Checkpoint and restore
 * must use the same architecture, CUDA ABI, and shim build.
 */
struct dyn_vmm_record {
  uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  uint64_t object_id;
  uint64_t address;
  uint64_t size;
  uint64_t offset;
  uint32_t role;
  uint32_t object_kind;
  uint32_t requested_handle_type;
  uint32_t flags;
  int32_t device_ordinal;
  uint32_t properties_size;
  uint32_t access_count;
  uint32_t access_size;
  uint8_t gpu_uuid[DYN_VMM_GPU_UUID_SIZE];
};

struct dyn_vmm_capability {
  uint32_t magic;
  uint16_t version;
  uint16_t object_kind;
  uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  char owner_socket_path[sizeof(((struct sockaddr_un*)0)->sun_path)];
  char owner_participant_id[DYN_VMM_PARTICIPANT_ID_SIZE];
  uint8_t reserved_identity[91];
};

struct dyn_vmm_fabric_token {
  uint32_t magic;
  uint16_t version;
  uint16_t handle_type;
  uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  char owner_participant_id[DYN_VMM_FABRIC_PARTICIPANT_ID_SIZE];
  uint32_t owner_namespace_pid;
  uint32_t object_kind;
};

/*
 * One fixed multicast extension follows each multicast record's opaque
 * properties and access bytes. Capture uses backing_allocation_uuid; restore
 * uses backing_object_id and requires the UUID bytes to be otherwise zero.
 */
struct dyn_vmm_multicast_record {
  uint8_t backing_allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  uint64_t backing_object_id;
  uint64_t multicast_offset;
  uint64_t memory_offset;
  uint64_t bind_size;
  uint64_t bind_flags;
  uint64_t object_flags;
  uint64_t object_handle_types;
  uint64_t object_size;
  uint32_t num_devices;
  uint32_t backing_role;
  uint32_t bind_api;
  uint32_t reserved;
};

struct dyn_vmm_placement {
  int32_t device_ordinal;
  uint32_t reserved;
  uint8_t source_gpu_uuid[DYN_VMM_GPU_UUID_SIZE];
  uint8_t target_gpu_uuid[DYN_VMM_GPU_UUID_SIZE];
};

_Static_assert(sizeof(struct dyn_vmm_header) == 256, "VMM header layout changed");
_Static_assert(offsetof(struct dyn_vmm_header, handle_type) == 16, "VMM header handle type offset changed");
_Static_assert(offsetof(struct dyn_vmm_header, object_kind) == 20, "VMM header object kind offset changed");
_Static_assert(offsetof(struct dyn_vmm_header, allocation_uuid) == 24, "VMM header UUID offset changed");
_Static_assert(offsetof(struct dyn_vmm_header, object_id) == 40, "VMM header object ID offset changed");
_Static_assert(offsetof(struct dyn_vmm_header, payload_size) == 48, "VMM header payload size offset changed");
_Static_assert(offsetof(struct dyn_vmm_header, message) == 56, "VMM header message offset changed");
_Static_assert(offsetof(struct dyn_vmm_header, participant_id) == 152, "VMM header participant offset changed");
_Static_assert(offsetof(struct dyn_vmm_header, reserved_identity) == 185, "VMM header reserved identity offset changed");
_Static_assert(
    offsetof(struct dyn_vmm_header, reserved_identity) +
            sizeof(((struct dyn_vmm_header*)0)->reserved_identity) ==
        sizeof(struct dyn_vmm_header),
    "VMM header reserved identity must end at the header boundary");
_Static_assert(sizeof(struct dyn_vmm_record) == 96, "VMM record layout changed");
_Static_assert(sizeof(struct dyn_vmm_multicast_record) == 96, "VMM multicast record layout changed");
_Static_assert(sizeof(struct dyn_vmm_capability) == 256, "VMM capability layout changed");
_Static_assert(sizeof(struct dyn_vmm_fabric_token) == DYN_VMM_FABRIC_HANDLE_SIZE, "VMM FABRIC token layout changed");
_Static_assert(offsetof(struct dyn_vmm_fabric_token, magic) == 0, "VMM FABRIC token magic offset changed");
_Static_assert(offsetof(struct dyn_vmm_fabric_token, version) == 4, "VMM FABRIC token version offset changed");
_Static_assert(offsetof(struct dyn_vmm_fabric_token, handle_type) == 6, "VMM FABRIC token handle type offset changed");
_Static_assert(offsetof(struct dyn_vmm_fabric_token, allocation_uuid) == 8, "VMM FABRIC token UUID offset changed");
_Static_assert(
    offsetof(struct dyn_vmm_fabric_token, owner_participant_id) == 24, "VMM FABRIC token participant offset changed");
_Static_assert(offsetof(struct dyn_vmm_fabric_token, owner_namespace_pid) == 56, "VMM FABRIC token PID offset changed");
_Static_assert(offsetof(struct dyn_vmm_fabric_token, object_kind) == 60, "VMM FABRIC token kind offset changed");
_Static_assert(sizeof(struct dyn_vmm_placement) == 40, "VMM placement layout changed");

#endif
