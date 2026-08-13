/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SNAPSHOT_CUDA_VMM_PROTOCOL_H
#define SNAPSHOT_CUDA_VMM_PROTOCOL_H

#include <stdint.h>

#define SNAPSHOT_VMM_MAGIC 0x44564d4dU
#define SNAPSHOT_VMM_VERSION 1U
#define SNAPSHOT_VMM_SOCKET_PREFIX "cuda-vmm-"
#define SNAPSHOT_VMM_ID_SIZE 33U
#define SNAPSHOT_VMM_ALLOCATION_ID_SIZE 16U
#define SNAPSHOT_VMM_TOKEN_SIZE 16U
#define SNAPSHOT_VMM_MAX_ACCESS 8U
#define SNAPSHOT_VMM_MAX_RECORDS 4096U
#define SNAPSHOT_VMM_POSIX_HANDLE_TYPE 1U

enum snapshot_vmm_operation {
  SNAPSHOT_VMM_IDENTIFY = 1,
  SNAPSHOT_VMM_INSPECT = 2,
  SNAPSHOT_VMM_CREATE_CARRIERS = 3,
  SNAPSHOT_VMM_PREPARE = 4,
  SNAPSHOT_VMM_RESTORE_CREATORS = 5,
  SNAPSHOT_VMM_RESTORE_IMPORTERS = 6,
  SNAPSHOT_VMM_EXPORT = 7,
};

enum snapshot_vmm_record_kind {
  SNAPSHOT_VMM_ALLOCATION = 1,
  SNAPSHOT_VMM_MAPPING = 2,
};

enum snapshot_vmm_record_flags {
  SNAPSHOT_VMM_CREATOR = 1U << 0,
  SNAPSHOT_VMM_APPLICATION_HANDLE_LIVE = 1U << 1,
};

struct snapshot_vmm_header {
  uint32_t magic;
  uint16_t version;
  uint16_t operation;
  int32_t status;
  uint32_t count;
  uint64_t payload_size;
  char job_id[SNAPSHOT_VMM_ID_SIZE];
  char participant_id[SNAPSHOT_VMM_ID_SIZE];
  char message[96];
  uint8_t authorization[SNAPSHOT_VMM_TOKEN_SIZE];
  uint8_t allocation_id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  uint8_t reserved[38];
};

struct snapshot_vmm_access {
  int32_t location_type;
  int32_t location_id;
  uint64_t flags;
};

struct snapshot_vmm_record {
  uint32_t kind;
  uint32_t flags;
  uint8_t allocation_id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  uint64_t address;
  uint64_t size;
  uint64_t offset;
  uint64_t allocation_size;
  int32_t allocation_type;
  uint32_t requested_handle_types;
  int32_t allocation_location_type;
  int32_t allocation_location_id;
  uint32_t access_count;
  uint32_t application_handle_count;
  struct snapshot_vmm_access access[SNAPSHOT_VMM_MAX_ACCESS];
};

_Static_assert(sizeof(struct snapshot_vmm_header) == 256, "VMM header layout changed");
_Static_assert(sizeof(struct snapshot_vmm_access) == 16, "VMM access layout changed");
_Static_assert(sizeof(struct snapshot_vmm_record) == 208, "VMM record layout changed");

#endif
