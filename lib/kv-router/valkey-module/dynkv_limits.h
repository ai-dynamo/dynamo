/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYNKV_LIMITS_H
#define DYNKV_LIMITS_H

/* Canonical wire limits. lib/llm/build.rs generates the Rust constants here. */
#define DYNKV_MAX_MATCH_HASHES 65536
#define DYNKV_MAX_SELECT_CANDIDATES 16384
#define DYNKV_MAX_ADMISSION_DOMAIN_LENGTH 128
#define DYNKV_MAX_ADMISSION_CANDIDATES 16384
#define DYNKV_MAX_ADMISSION_LEASE_MS 600000
#define DYNKV_MAX_ADMISSION_REQUEST_BYTES 16777216
#define DYNKV_MAX_REGISTRATION_RANKS 65536

#endif /* DYNKV_LIMITS_H */
