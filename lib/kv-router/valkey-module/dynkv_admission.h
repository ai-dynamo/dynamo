/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_ADMISSION_H
#define DYNKV_ADMISSION_H

#include "dynkv_types.h"

bool admission_identity_read(Reader *, AdmissionIdentity *);
bool admission_identity_append(Buffer *, const AdmissionIdentity *);
bool admission_identity_parse_payload(const uint8_t *, size_t, AdmissionIdentity *, Reader *);
bool admission_apply_prefix_append(Buffer *, uint8_t, uint64_t);
bool admission_reservation_matches_raw(const Reservation *, const uint8_t *, uint32_t);
bool admission_candidates_valid(RouterIndex *, const AdmissionRequest *, uint64_t);
int router_index_admission_apply_reserve(RouterIndex *, uint64_t, const AdmissionIdentity *, uint64_t, uint32_t, uint64_t, uint32_t, uint64_t, uint32_t, uint32_t, const uint8_t *, uint32_t, Reservation **, bool *);
int router_index_admission_apply_release(RouterIndex *, uint64_t, const AdmissionIdentity *, uint64_t, bool *, bool *);
int router_index_admission_apply_renew(RouterIndex *, uint64_t, const AdmissionIdentity *, uint64_t, uint64_t, Reservation **, bool *);
bool admission_replicate_payload(ValkeyModuleCtx *, ValkeyModuleString *, const Buffer *);
bool admission_commit_payload(ValkeyModuleCtx *, ValkeyModuleString *, RouterIndex *, const Buffer *);
bool admission_cleanup_payload(Buffer *, uint64_t);
bool admission_reserve_apply_payload(Buffer *, uint64_t, const AdmissionIdentity *, const AdmissionSelection *, const AdmissionRequest *);
bool admission_release_apply_payload(Buffer *, uint64_t, const AdmissionIdentity *, uint64_t);
bool admission_renew_apply_payload(Buffer *, uint64_t, const AdmissionIdentity *, uint64_t, uint64_t);
bool admission_commit_cleanup(ValkeyModuleCtx *, ValkeyModuleString *, RouterIndex *, uint64_t);
int dynkv_admit_apply_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_select_reserve_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_release_command(ValkeyModuleCtx *, ValkeyModuleString **, int);
int dynkv_renew_command(ValkeyModuleCtx *, ValkeyModuleString **, int);

#endif /* DYNKV_ADMISSION_H */
