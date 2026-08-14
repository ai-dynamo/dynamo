/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SNAPSHOT_CUDA_VMM_MULTICAST_H
#define SNAPSHOT_CUDA_VMM_MULTICAST_H

#include <cuda.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "posix.h"
#include "protocol.h"

struct snapshot_multicast_member {
  uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  CUmemGenericAllocationHandle handle;
  CUdeviceptr address;
  size_t allocation_offset;
  CUdevice device;
  bool temporary_handle;
};

struct snapshot_multicast_callbacks {
  void* (*real_symbol)(const char* name);
  int (*allocate_logical_handle)(CUmemGenericAllocationHandle* output);
  int (*random_bytes)(void* output, size_t size);
  int (*member_from_handle)(CUmemGenericAllocationHandle logical, struct snapshot_multicast_member* member);
  int (*member_from_address)(CUdeviceptr address, size_t size, struct snapshot_multicast_member* member);
  int (*member_from_id)(const uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE], struct snapshot_multicast_member* member);
  void (*release_state_lock)(void);
  void (*acquire_state_lock)(void);
};

void snapshot_multicast_initialize(
    const struct snapshot_multicast_callbacks* callbacks, const char* job_id, const char* participant_id,
    const char* endpoint);
void snapshot_multicast_reset(void);

bool snapshot_multicast_is_handle(CUmemGenericAllocationHandle logical);
bool snapshot_multicast_has_mapping(CUdeviceptr address, size_t size);
CUresult snapshot_multicast_release(CUmemGenericAllocationHandle logical);
CUresult snapshot_multicast_retain(CUmemGenericAllocationHandle* output, void* address);
CUresult snapshot_multicast_map(
    CUdeviceptr address, size_t size, size_t offset, CUmemGenericAllocationHandle logical, unsigned long long flags);
CUresult snapshot_multicast_unmap(CUdeviceptr address, size_t size);
CUresult snapshot_multicast_set_access(
    CUdeviceptr address, size_t size, const CUmemAccessDesc* descriptors, size_t count);
CUresult snapshot_multicast_export(
    void* shareable, CUmemGenericAllocationHandle logical, CUmemAllocationHandleType type, unsigned long long flags);
CUresult snapshot_multicast_import(
    CUmemGenericAllocationHandle* output, const struct snapshot_vmm_posix_capability* capability, int raw_fd);
CUresult snapshot_multicast_get_properties(CUmemAllocationProp* properties, CUmemGenericAllocationHandle logical);

CUresult snapshot_multicast_create(CUmemGenericAllocationHandle* output, const CUmulticastObjectProp* properties);
CUresult snapshot_multicast_add_device(CUmemGenericAllocationHandle logical, CUdevice device);
CUresult snapshot_multicast_bind_mem(
    CUmemGenericAllocationHandle multicast, CUdevice device, bool device_explicit, size_t multicast_offset,
    CUmemGenericAllocationHandle memory, size_t memory_offset, size_t size, unsigned long long flags);
CUresult snapshot_multicast_bind_address(
    CUmemGenericAllocationHandle multicast, CUdevice device, bool device_explicit, size_t multicast_offset,
    CUdeviceptr memory, size_t size, unsigned long long flags);
CUresult snapshot_multicast_unbind(CUmemGenericAllocationHandle multicast, CUdevice device, size_t offset, size_t size);

size_t snapshot_multicast_record_count(void);
int snapshot_multicast_write_records(struct snapshot_vmm_record* records, size_t count);
int snapshot_multicast_prepare(void);
int snapshot_multicast_restore_creators(void);
int snapshot_multicast_restore_importers(void);
int snapshot_multicast_restore_devices(void);
int snapshot_multicast_restore_topology(void);
int snapshot_multicast_validate_restored(void);
CUresult snapshot_multicast_export_raw(
    const uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE], const uint8_t authorization[SNAPSHOT_VMM_TOKEN_SIZE],
    int* output);
const char* snapshot_multicast_error(void);

#endif
