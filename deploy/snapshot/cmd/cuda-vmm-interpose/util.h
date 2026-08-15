/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SNAPSHOT_CUDA_VMM_UTIL_H
#define SNAPSHOT_CUDA_VMM_UTIL_H

#include <stdbool.h>
#include <stddef.h>

#include "protocol.h"

int snapshot_vmm_write_all(int fd, const void* value, size_t size);
int snapshot_vmm_read_all(int fd, void* value, size_t size);
int snapshot_vmm_pread_all(int fd, void* value, size_t size);
bool snapshot_vmm_is_lower_hex_id(const char value[SNAPSHOT_VMM_ID_SIZE]);
bool snapshot_vmm_header_strings_terminated(const struct snapshot_vmm_header* header);
int snapshot_vmm_set_socket_timeouts(int fd, int seconds);
int snapshot_vmm_send_header(int fd, const struct snapshot_vmm_header* header, int passed_fd);
int snapshot_vmm_receive_header(int fd, struct snapshot_vmm_header* header, int* passed_fd);

#endif
