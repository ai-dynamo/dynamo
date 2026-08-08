/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYN_SNAPSHOT_CUDA_VMM_PROTOCOL_H
#define DYN_SNAPSHOT_CUDA_VMM_PROTOCOL_H

#include <stdint.h>
#include <sys/un.h>

#define DYN_VMM_MAGIC 0x44564d4dU
#define DYN_VMM_VERSION 2U
#define DYN_VMM_CAPABILITY_MAGIC 0x44564d43U
#define DYN_VMM_CAPABILITY_VERSION 1U
#define DYN_VMM_SOCKET_PREFIX "cuda-vmm-"
#define DYN_VMM_PARTICIPANT_ID_SIZE 33U
#define DYN_VMM_ALLOCATION_UUID_SIZE 16U
#define DYN_VMM_GPU_UUID_SIZE 16U

enum dyn_vmm_operation {
  DYN_VMM_INSPECT = 1,
  DYN_VMM_READ_OWNER = 2,
  DYN_VMM_DETACH_IMPORTS = 3,
  DYN_VMM_DETACH_OWNERS = 4,
  DYN_VMM_RESTORE_OWNER = 5,
  DYN_VMM_RESTORE_IMPORT = 6,
  DYN_VMM_IDENTIFY = 7,
  DYN_VMM_QUERY_PLACEMENT = 8,
  DYN_VMM_EXPORT_OWNER = 9,
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
};

enum dyn_vmm_record_flags {
  DYN_VMM_APPLICATION_HANDLE_LIVE = 1U << 0,
};

struct dyn_vmm_header {
  uint32_t magic;
  uint16_t version;
  uint16_t operation;
  int32_t status;
  uint32_t count;
  uint32_t reserved[2];
  uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  uint64_t object_id;
  uint64_t payload_size;
  char message[96];
  char participant_id[DYN_VMM_PARTICIPANT_ID_SIZE];
  uint8_t reserved_identity[64];
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
  uint16_t reserved;
  uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  char owner_socket_path[sizeof(((struct sockaddr_un*)0)->sun_path)];
  char owner_participant_id[DYN_VMM_PARTICIPANT_ID_SIZE];
  uint8_t reserved_identity[91];
};

struct dyn_vmm_placement {
  int32_t device_ordinal;
  uint32_t reserved;
  uint8_t source_gpu_uuid[DYN_VMM_GPU_UUID_SIZE];
  uint8_t current_gpu_uuid[DYN_VMM_GPU_UUID_SIZE];
};

_Static_assert(sizeof(struct dyn_vmm_header) == 256, "VMM header layout changed");
_Static_assert(sizeof(struct dyn_vmm_record) == 96, "VMM record layout changed");
_Static_assert(sizeof(struct dyn_vmm_capability) == 256, "VMM capability layout changed");
_Static_assert(sizeof(struct dyn_vmm_placement) == 40, "VMM placement layout changed");

#endif
