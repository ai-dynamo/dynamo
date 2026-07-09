/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#include "dynkv_gc.h"
#include "dynkv_lease.h"
#include "dynkv_persistence.h"
#include "dynkv_state.h"

bool gc_read_cleanup_epoch_identity(
    Reader *reader,
    uint64_t *worker_id,
    uint64_t *owner_nonce,
    uint64_t *expires_at_ms,
    uint64_t *cleanup_generation) {
    return reader_u64(reader, worker_id) && reader_u64(reader, owner_nonce) &&
           reader_u64(reader, expires_at_ms) &&
           reader_u64(reader, cleanup_generation);
}

/* Reject conflicting/duplicate identities before any exact-plan mutation. */
bool gc_payload_identities_unique(const uint8_t *data, size_t length) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint32_t count = 0;
    if (!reader_u8(&reader, &version) ||
        (version != DYNKV_GC_WIRE_VERSION_LEGACY &&
         version != DYNKV_GC_WIRE_VERSION) ||
        !reader_u32(&reader, &count) || count == 0 || count > DYNKV_MAX_GC_ITEMS) {
        return false;
    }
    ValkeyModuleDict *seen = ValkeyModule_CreateDict(NULL);
    bool valid = true;
    for (uint32_t i = 0; i < count && valid; ++i) {
        uint8_t kind = 0;
        uint8_t identity[1 + 4 + DYNKV_MAX_ADMISSION_DOMAIN_LENGTH + 16];
        size_t identity_length = 1;
        uint64_t first = 0;
        uint64_t worker_id = 0;
        uint32_t dp_rank = 0;
        uint64_t ignored_u64 = 0;
        if (!reader_u8(&reader, &kind)) {
            valid = false;
            break;
        }
        identity[0] =
            kind == DYNKV_GC_EXPIRE_WORKER_LEASE ||
                    kind == DYNKV_GC_BEGIN_LEASE_CLEANUP ||
                    kind == DYNKV_GC_FINALIZE_LEASE_CLEANUP
                ? DYNKV_GC_REMOVE_WORKER_EPOCH
                : kind;
        if (kind == DYNKV_GC_REMOVE_OWNER) {
            valid = reader_u64(&reader, &first) &&
                    reader_u64(&reader, &worker_id) &&
                    reader_u32(&reader, &dp_rank) &&
                    reader_u64(&reader, &ignored_u64);
            encode_u64_be(identity + 1, first);
            worker_key(identity + 9, worker_id, dp_rank);
            identity_length = 21;
        } else if (kind == DYNKV_GC_REMOVE_ADMISSION_RANK) {
            uint32_t domain_length = 0;
            valid = reader_u32(&reader, &domain_length) && domain_length != 0 &&
                    domain_length <= DYNKV_MAX_ADMISSION_DOMAIN_LENGTH &&
                    reader.length - reader.offset >= domain_length;
            if (valid) {
                encode_u32_be(identity + 1, domain_length);
                memcpy(identity + 5, reader.data + reader.offset, domain_length);
                reader.offset += domain_length;
                valid = reader_u64(&reader, &worker_id) &&
                        reader_u32(&reader, &dp_rank) &&
                        reader_u64(&reader, &ignored_u64) &&
                        reader_u64(&reader, &ignored_u64);
                worker_key(identity + 5 + domain_length, worker_id, dp_rank);
                identity_length = 17 + domain_length;
            }
        } else if (kind == DYNKV_GC_REMOVE_NODE) {
            valid = reader_u64(&reader, &first) &&
                    reader_u64(&reader, &ignored_u64) &&
                    reader_u64(&reader, &ignored_u64);
            encode_u64_be(identity + 1, first);
            identity_length = 9;
        } else if (kind == DYNKV_GC_REMOVE_WORKER) {
            valid = reader_u64(&reader, &worker_id) &&
                    reader_u32(&reader, &dp_rank) &&
                    reader_u64(&reader, &ignored_u64);
            worker_key(identity + 1, worker_id, dp_rank);
            identity_length = 13;
        } else if (kind == DYNKV_GC_REMOVE_WORKER_EPOCH) {
            valid = reader_u64(&reader, &worker_id) &&
                    reader_u64(&reader, &ignored_u64);
            encode_u64_be(identity + 1, worker_id);
            identity_length = 9;
        } else if (kind == DYNKV_GC_EXPIRE_WORKER_LEASE) {
            valid = reader_u64(&reader, &ignored_u64) &&
                    reader_u64(&reader, &worker_id) &&
                    reader_u64(&reader, &ignored_u64) &&
                    reader_u64(&reader, &ignored_u64);
            encode_u64_be(identity + 1, worker_id);
            identity_length = 9;
        } else if (version == DYNKV_GC_WIRE_VERSION &&
                   kind == DYNKV_GC_BEGIN_LEASE_CLEANUP) {
            valid = reader_u64(&reader, &ignored_u64) &&
                    reader_u64(&reader, &worker_id) &&
                    reader_u64(&reader, &ignored_u64) &&
                    reader_u64(&reader, &ignored_u64) &&
                    reader_u64(&reader, &ignored_u64) &&
                    reader_u64(&reader, &ignored_u64);
            encode_u64_be(identity + 1, worker_id);
            identity_length = 9;
        } else if (version == DYNKV_GC_WIRE_VERSION &&
                   (kind == DYNKV_GC_CLEANUP_RESERVATION ||
                    kind == DYNKV_GC_CLEANUP_ADMISSION_RANK ||
                    kind == DYNKV_GC_CLEANUP_OWNER ||
                    kind == DYNKV_GC_CLEANUP_WORKER ||
                    kind == DYNKV_GC_FINALIZE_LEASE_CLEANUP)) {
            uint64_t owner_nonce = 0;
            uint64_t expires_at_ms = 0;
            uint64_t cleanup_generation = 0;
            valid = gc_read_cleanup_epoch_identity(
                &reader,
                &worker_id,
                &owner_nonce,
                &expires_at_ms,
                &cleanup_generation);
            (void)owner_nonce;
            (void)expires_at_ms;
            (void)cleanup_generation;
            if (valid && kind == DYNKV_GC_CLEANUP_RESERVATION) {
                uint32_t domain_length = 0;
                uint64_t client_nonce = 0;
                uint64_t request_nonce = 0;
                valid = reader_u32(&reader, &domain_length) && domain_length != 0 &&
                        domain_length <= DYNKV_MAX_ADMISSION_DOMAIN_LENGTH &&
                        reader.length - reader.offset >= domain_length;
                if (valid) {
                    encode_u32_be(identity + 1, domain_length);
                    memcpy(identity + 5, reader.data + reader.offset, domain_length);
                    reader.offset += domain_length;
                    valid = reader_u64(&reader, &client_nonce) &&
                            reader_u64(&reader, &request_nonce) &&
                            reader_u32(&reader, &dp_rank) &&
                            reader_u64(&reader, &ignored_u64);
                    encode_u64_be(identity + 5 + domain_length, client_nonce);
                    encode_u64_be(identity + 13 + domain_length, request_nonce);
                    identity_length = 21 + domain_length;
                }
            } else if (valid && kind == DYNKV_GC_CLEANUP_ADMISSION_RANK) {
                uint32_t domain_length = 0;
                valid = reader_u32(&reader, &domain_length) && domain_length != 0 &&
                        domain_length <= DYNKV_MAX_ADMISSION_DOMAIN_LENGTH &&
                        reader.length - reader.offset >= domain_length;
                if (valid) {
                    encode_u32_be(identity + 1, domain_length);
                    memcpy(identity + 5, reader.data + reader.offset, domain_length);
                    reader.offset += domain_length;
                    valid = reader_u32(&reader, &dp_rank) &&
                            reader_u64(&reader, &ignored_u64);
                    worker_key(identity + 5 + domain_length, worker_id, dp_rank);
                    identity_length = 17 + domain_length;
                }
            } else if (valid && kind == DYNKV_GC_CLEANUP_OWNER) {
                uint8_t active = 0;
                valid = reader_u32(&reader, &dp_rank) &&
                        reader_u64(&reader, &first) &&
                        reader_u64(&reader, &ignored_u64) &&
                        reader_u8(&reader, &active) && active <= 1;
                encode_u64_be(identity + 1, first);
                worker_key(identity + 9, worker_id, dp_rank);
                identity_length = 21;
            } else if (valid && kind == DYNKV_GC_CLEANUP_WORKER) {
                valid = reader_u32(&reader, &dp_rank);
                worker_key(identity + 1, worker_id, dp_rank);
                identity_length = 13;
            } else if (valid) {
                encode_u64_be(identity + 1, worker_id);
                identity_length = 9;
            }
        } else {
            valid = false;
        }
        if (valid &&
            ValkeyModule_DictGetC(
                seen, identity, identity_length, NULL) != NULL) {
            valid = false;
        }
        if (valid) {
            ValkeyModule_DictSetC(
                seen, identity, identity_length, (void *)(uintptr_t)1);
        }
    }
    valid = valid && reader.offset == reader.length;
    ValkeyModule_FreeDict(NULL, seen);
    return valid;
}

bool gc_process_payload(
    RouterIndex *index,
    const uint8_t *data,
    size_t length,
    bool apply) {
    Reader reader = {.data = data, .length = length, .offset = 0};
    uint8_t version = 0;
    uint32_t count = 0;
    uint64_t generation_advances = 0;
    uint64_t lifecycle_advances = 0;
    bool saw_begin = false;
    if (!reader_u8(&reader, &version) ||
        (version != DYNKV_GC_WIRE_VERSION_LEGACY &&
         version != DYNKV_GC_WIRE_VERSION) ||
        !reader_u32(&reader, &count) || count == 0 || count > DYNKV_MAX_GC_ITEMS) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint8_t kind = 0;
        uint64_t worker_id = 0;
        uint32_t dp_rank = 0;
        uint64_t generation = 0;
        if (!reader_u8(&reader, &kind)) {
            return false;
        }
        if (kind == DYNKV_GC_REMOVE_OWNER) {
            uint64_t external_hash = 0;
            if (!reader_u64(&reader, &external_hash) ||
                !reader_u64(&reader, &worker_id) ||
                !reader_u32(&reader, &dp_rank) ||
                !reader_u64(&reader, &generation) ||
                !(apply ? gc_apply_remove_owner(
                              index,
                              external_hash,
                              worker_id,
                              dp_rank,
                              generation)
                        : gc_can_remove_owner(
                              index,
                              external_hash,
                              worker_id,
                              dp_rank,
                              generation))) {
                return false;
            }
        } else if (kind == DYNKV_GC_REMOVE_ADMISSION_RANK) {
            uint32_t domain_length = 0;
            uint64_t incarnation = 0;
            if (!reader_u32(&reader, &domain_length) || domain_length == 0 ||
                domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH ||
                reader.length - reader.offset < domain_length) {
                return false;
            }
            const uint8_t *domain = reader.data + reader.offset;
            reader.offset += domain_length;
            if (!reader_u64(&reader, &worker_id) ||
                !reader_u32(&reader, &dp_rank) ||
                !reader_u64(&reader, &incarnation) ||
                !reader_u64(&reader, &generation) ||
                !(apply ? gc_apply_remove_admission_rank(
                              index,
                              domain,
                              domain_length,
                              worker_id,
                              dp_rank,
                              incarnation,
                              generation)
                        : gc_can_remove_admission_rank(
                              index,
                              domain,
                              domain_length,
                              worker_id,
                              dp_rank,
                              incarnation,
                              generation))) {
                return false;
            }
        } else if (kind == DYNKV_GC_REMOVE_NODE) {
            uint64_t external_hash = 0;
            uint64_t parent_hash = 0;
            uint64_t local_hash = 0;
            if (!reader_u64(&reader, &external_hash) ||
                !reader_u64(&reader, &parent_hash) ||
                !reader_u64(&reader, &local_hash) ||
                !(apply ? gc_apply_remove_node(
                              index, external_hash, parent_hash, local_hash)
                        : gc_can_remove_node(
                              index, external_hash, parent_hash, local_hash))) {
                return false;
            }
        } else if (kind == DYNKV_GC_REMOVE_WORKER) {
            if (!reader_u64(&reader, &worker_id) ||
                !reader_u32(&reader, &dp_rank) ||
                !reader_u64(&reader, &generation) ||
                !(apply ? gc_apply_remove_worker(
                              index, worker_id, dp_rank, generation)
                        : gc_can_remove_worker(
                              index, worker_id, dp_rank, generation))) {
                return false;
            }
        } else if (kind == DYNKV_GC_REMOVE_WORKER_EPOCH) {
            if (!reader_u64(&reader, &worker_id) ||
                !reader_u64(&reader, &generation) ||
                !(apply ? gc_apply_remove_worker_epoch(
                              index, worker_id, generation)
                        : gc_can_remove_worker_epoch(
                              index, worker_id, generation))) {
                return false;
            }
            if (!apply &&
                ++generation_advances > UINT64_MAX - index->generation_counter) {
                return false;
            }
        } else if (kind == DYNKV_GC_EXPIRE_WORKER_LEASE) {
            uint64_t now_ms = 0;
            uint64_t owner_nonce = 0;
            uint64_t expires_at_ms = 0;
            if (!reader_u64(&reader, &now_ms) ||
                !reader_u64(&reader, &worker_id) ||
                !reader_u64(&reader, &owner_nonce) ||
                !reader_u64(&reader, &expires_at_ms) ||
                !(apply ? gc_apply_expire_worker_lease(
                              index,
                              now_ms,
                              worker_id,
                              owner_nonce,
                              expires_at_ms)
                        : gc_can_expire_worker_lease(
                              index,
                              now_ms,
                              worker_id,
                              owner_nonce,
                              expires_at_ms))) {
                return false;
            }
            if (!apply &&
                (++generation_advances >
                     UINT64_MAX - index->generation_counter ||
                 ++lifecycle_advances >
                     UINT64_MAX - index->lifecycle_generation_counter)) {
                return false;
            }
        } else if (version == DYNKV_GC_WIRE_VERSION &&
                   kind == DYNKV_GC_BEGIN_LEASE_CLEANUP) {
            saw_begin = true;
            uint64_t now_ms = 0;
            uint64_t owner_nonce = 0;
            uint64_t expires_at_ms = 0;
            uint64_t cleanup_generation = 0;
            uint64_t lifecycle_generation = 0;
            if (!reader_u64(&reader, &now_ms) ||
                !reader_u64(&reader, &worker_id) ||
                !reader_u64(&reader, &owner_nonce) ||
                !reader_u64(&reader, &expires_at_ms) ||
                !reader_u64(&reader, &cleanup_generation) ||
                !reader_u64(&reader, &lifecycle_generation) ||
                !(apply ? gc_apply_begin_lease_cleanup(
                              index,
                              now_ms,
                              worker_id,
                              owner_nonce,
                              expires_at_ms,
                              cleanup_generation,
                              lifecycle_generation)
                        : gc_can_begin_lease_cleanup(
                              index,
                              now_ms,
                              worker_id,
                              owner_nonce,
                              expires_at_ms,
                              cleanup_generation,
                              lifecycle_generation))) {
                return false;
            }
            if (!apply &&
                (++generation_advances >
                     UINT64_MAX - index->generation_counter ||
                 ++lifecycle_advances >
                     UINT64_MAX - index->lifecycle_generation_counter)) {
                return false;
            }
        } else if (version == DYNKV_GC_WIRE_VERSION &&
                   (kind == DYNKV_GC_CLEANUP_RESERVATION ||
                    kind == DYNKV_GC_CLEANUP_ADMISSION_RANK ||
                    kind == DYNKV_GC_CLEANUP_OWNER ||
                    kind == DYNKV_GC_CLEANUP_WORKER ||
                    kind == DYNKV_GC_FINALIZE_LEASE_CLEANUP)) {
            uint64_t owner_nonce = 0;
            uint64_t expires_at_ms = 0;
            uint64_t cleanup_generation = 0;
            if (!gc_read_cleanup_epoch_identity(
                    &reader,
                    &worker_id,
                    &owner_nonce,
                    &expires_at_ms,
                    &cleanup_generation)) {
                return false;
            }
            if (kind == DYNKV_GC_CLEANUP_RESERVATION) {
                uint32_t domain_length = 0;
                uint64_t client_nonce = 0;
                uint64_t request_nonce = 0;
                uint64_t reservation_expires_at_ms = 0;
                if (!reader_u32(&reader, &domain_length) || domain_length == 0 ||
                    domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH ||
                    reader.length - reader.offset < domain_length) {
                    return false;
                }
                const uint8_t *domain = reader.data + reader.offset;
                reader.offset += domain_length;
                if (!reader_u64(&reader, &client_nonce) ||
                    !reader_u64(&reader, &request_nonce) ||
                    !reader_u32(&reader, &dp_rank) ||
                    !reader_u64(&reader, &reservation_expires_at_ms) ||
                    !(apply ? gc_apply_cleanup_reservation(
                                  index,
                                  worker_id,
                                  owner_nonce,
                                  expires_at_ms,
                                  cleanup_generation,
                                  domain,
                                  domain_length,
                                  client_nonce,
                                  request_nonce,
                                  dp_rank,
                                  reservation_expires_at_ms)
                            : gc_can_cleanup_reservation(
                                  index,
                                  worker_id,
                                  owner_nonce,
                                  expires_at_ms,
                                  cleanup_generation,
                                  domain,
                                  domain_length,
                                  client_nonce,
                                  request_nonce,
                                  dp_rank,
                                  reservation_expires_at_ms))) {
                    return false;
                }
            } else if (kind == DYNKV_GC_CLEANUP_ADMISSION_RANK) {
                uint32_t domain_length = 0;
                uint64_t incarnation = 0;
                if (!reader_u32(&reader, &domain_length) || domain_length == 0 ||
                    domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH ||
                    reader.length - reader.offset < domain_length) {
                    return false;
                }
                const uint8_t *domain = reader.data + reader.offset;
                reader.offset += domain_length;
                if (!reader_u32(&reader, &dp_rank) ||
                    !reader_u64(&reader, &incarnation) ||
                    !(apply ? gc_apply_cleanup_admission_rank(
                                  index,
                                  worker_id,
                                  owner_nonce,
                                  expires_at_ms,
                                  cleanup_generation,
                                  domain,
                                  domain_length,
                                  dp_rank,
                                  incarnation)
                            : gc_can_cleanup_admission_rank(
                                  index,
                                  worker_id,
                                  owner_nonce,
                                  expires_at_ms,
                                  cleanup_generation,
                                  domain,
                                  domain_length,
                                  dp_rank,
                                  incarnation))) {
                    return false;
                }
            } else if (kind == DYNKV_GC_CLEANUP_OWNER) {
                uint64_t external_hash = 0;
                uint64_t event_id = 0;
                uint8_t active = 0;
                if (!reader_u32(&reader, &dp_rank) ||
                    !reader_u64(&reader, &external_hash) ||
                    !reader_u64(&reader, &event_id) ||
                    !reader_u8(&reader, &active) || active > 1 ||
                    !(apply ? gc_apply_cleanup_owner(
                                  index,
                                  worker_id,
                                  owner_nonce,
                                  expires_at_ms,
                                  cleanup_generation,
                                  dp_rank,
                                  external_hash,
                                  event_id,
                                  active == 1)
                            : gc_can_cleanup_owner(
                                  index,
                                  worker_id,
                                  owner_nonce,
                                  expires_at_ms,
                                  cleanup_generation,
                                  dp_rank,
                                  external_hash,
                                  event_id,
                                  active == 1))) {
                    return false;
                }
            } else if (kind == DYNKV_GC_CLEANUP_WORKER) {
                if (!reader_u32(&reader, &dp_rank) ||
                    !(apply ? gc_apply_cleanup_worker(
                                  index,
                                  worker_id,
                                  owner_nonce,
                                  expires_at_ms,
                                  cleanup_generation,
                                  dp_rank)
                            : gc_can_cleanup_worker(
                                  index,
                                  worker_id,
                                  owner_nonce,
                                  expires_at_ms,
                                  cleanup_generation,
                                  dp_rank))) {
                    return false;
                }
            } else if (!(apply ? gc_apply_finalize_lease_cleanup(
                                      index,
                                      worker_id,
                                      owner_nonce,
                                      expires_at_ms,
                                      cleanup_generation)
                                : gc_can_finalize_lease_cleanup(
                                      index,
                                      worker_id,
                                      owner_nonce,
                                      expires_at_ms,
                                      cleanup_generation))) {
                return false;
            }
        } else {
            return false;
        }
    }
    return reader.offset == reader.length &&
           (apply || !saw_begin ||
            (generation_advances == 1 && lifecycle_advances == 1));
}

bool gc_apply_payload(RouterIndex *index, const uint8_t *data, size_t length) {
    return gc_payload_identities_unique(data, length) &&
           gc_process_payload(index, data, length, false) &&
           gc_process_payload(index, data, length, true);
}

int dynkv_gc_apply_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 3) {
        return ValkeyModule_WrongArity(ctx);
    }
    if (!ValkeyModule_MustObeyClient(ctx)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GC_APPLY_INTERNAL_ONLY");
    }
    size_t payload_length = 0;
    const char *payload = ValkeyModule_StringPtrLen(argv[2], &payload_length);
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_write(ctx, argv[1], &key);
    if (index == NULL ||
        !gc_apply_payload(index, (const uint8_t *)payload, payload_length)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_GC_PLAN");
    }
    ++index->mutation_count;
    ValkeyModule_SignalModifiedKey(ctx, argv[1]);
    return ValkeyModule_ReplyWithSimpleString(ctx, "OK");
}

int dynkv_gc_command(ValkeyModuleCtx *ctx, ValkeyModuleString **argv, int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 4) {
        return ValkeyModule_WrongArity(ctx);
    }
    size_t watermark_length = 0;
    size_t budget_length = 0;
    const char *watermark_data =
        ValkeyModule_StringPtrLen(argv[2], &watermark_length);
    const char *budget_data = ValkeyModule_StringPtrLen(argv[3], &budget_length);
    Reader watermark_reader = {
        .data = (const uint8_t *)watermark_data, .length = watermark_length};
    Reader budget_reader = {
        .data = (const uint8_t *)budget_data, .length = budget_length};
    uint64_t watermark = 0;
    uint32_t budget = 0;
    bool use_current = watermark_length == 7 &&
                       memcmp(watermark_data, "CURRENT", 7) == 0;
    if ((!use_current &&
         (!reader_u64(&watermark_reader, &watermark) ||
          watermark_reader.offset != watermark_reader.length)) ||
        !reader_u32(&budget_reader, &budget) ||
        budget_reader.offset != budget_reader.length || budget == 0 ||
        budget > DYNKV_MAX_GC_ITEMS) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_GC_REQUEST");
    }
    ValkeyModuleKey *read_key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &read_key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    if (use_current) {
        watermark = index == NULL ? 0 : index->generation_counter;
    }
    if (watermark > (index == NULL ? 0 : index->generation_counter)) {
        return ValkeyModule_ReplyWithError(ctx, "DYNKV_GC_FUTURE_WATERMARK");
    }
    GcResult result = {0};
    if (index != NULL) {
        ValkeyModuleKey *write_key = NULL;
        index = router_index_for_write(ctx, argv[1], &write_key);
        Buffer payload = {0};
        if (!buffer_u8(&payload, DYNKV_GC_WIRE_VERSION) ||
            !buffer_u32(&payload, 0)) {
            buffer_free(&payload);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_GC_OOM");
        }
        uint64_t now_ms = 0;
        if (!admission_now_ms(&now_ms)) {
            buffer_free(&payload);
            return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_GC_REQUEST");
        }
        bool first_record_set = false;
        uint8_t first_phase = 0;
        uint8_t first_key[DYNKV_MAX_GC_CURSOR_BYTES];
        size_t first_key_length = 0;
        while (result.examined < budget) {
            void *data = NULL;
            uint8_t phase = 0;
            if (!gc_next_record(index, &data, &phase)) {
                break;
            }
            if (first_record_set && phase == first_phase &&
                index->gc_cursor_length == first_key_length &&
                memcmp(index->gc_cursor, first_key, first_key_length) == 0) {
                break;
            }
            if (!first_record_set) {
                first_record_set = true;
                first_phase = phase;
                first_key_length = index->gc_cursor_length;
                memcpy(first_key, index->gc_cursor, first_key_length);
            }
            ++result.examined;
            if (!gc_plan_record(
                    index,
                    phase,
                    data,
                    watermark,
                    now_ms,
                    budget,
                    &payload,
                    &result)) {
                buffer_free(&payload);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_GC_STATE");
            }
            if (result.stop_planning) {
                break;
            }
        }
        if (result.reclaimed != 0) {
            encode_u32_be(payload.data + 1, (uint32_t)result.reclaimed);
            if (!gc_payload_identities_unique(payload.data, payload.length) ||
                !gc_process_payload(
                    index, payload.data, payload.length, false)) {
                buffer_free(&payload);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_INVALID_GC_STATE");
            }
            if (ValkeyModule_Replicate(
                    ctx,
                    "DYNKV.GC_APPLY",
                    "sb",
                    argv[1],
                    (char *)payload.data,
                    payload.length) != VALKEYMODULE_OK) {
                buffer_free(&payload);
                return ValkeyModule_ReplyWithError(ctx, "DYNKV_REPLICATION_FAILED");
            }
            if (!gc_process_payload(index, payload.data, payload.length, true)) {
                buffer_free(&payload);
                return ValkeyModule_ReplyWithError(
                    ctx, "DYNKV_GC_COMMIT_INVARIANT_FAILED");
            }
            ++index->mutation_count;
            ValkeyModule_SignalModifiedKey(ctx, argv[1]);
        }
        buffer_free(&payload);
    }
    ValkeyModule_ReplyWithArray(ctx, 8);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)result.examined);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)result.reclaimed);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)result.owners);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)result.admission_ranks);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)result.nodes);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)result.workers);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)result.worker_epochs);
    return ValkeyModule_ReplyWithLongLong(
        ctx, index == NULL ? 0 : (long long)index->gc_phase);
}

int dynkv_gc_stats_command(
    ValkeyModuleCtx *ctx,
    ValkeyModuleString **argv,
    int argc) {
    ValkeyModule_AutoMemory(ctx);
    if (argc != 2) {
        return ValkeyModule_WrongArity(ctx);
    }
    ValkeyModuleKey *key = NULL;
    RouterIndex *index = router_index_for_read(ctx, argv[1], &key);
    if (index == (RouterIndex *)(uintptr_t)1) {
        return VALKEYMODULE_OK;
    }
    uint64_t direct_tombstones = 0;
    uint64_t retained_legacy_ranks = 0;
    uint64_t direct_epochs = 0;
    uint64_t retained_legacy_epochs = 0;
    uint64_t inactive_owners = 0;
    uint64_t ownerless_nodes = 0;
    if (index != NULL) {
        uint64_t now_ms = 0;
        if (!admission_now_ms(&now_ms)) {
            return ValkeyModule_ReplyWithError(
                ctx, "DYNKV_INVALID_GC_STATE");
        }
        ValkeyModuleDictIter *workers =
            ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
        void *data = NULL;
        while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
            WorkerState *worker = data;
            if (worker->lifecycle_tombstone && worker->lifecycle_managed &&
                !worker->legacy_tainted) {
                ++direct_tombstones;
            } else if (worker->lifecycle_tombstone || worker->legacy_tainted) {
                ++retained_legacy_ranks;
            }
        }
        ValkeyModule_DictIteratorStop(workers);
        ValkeyModuleDictIter *epochs =
            ValkeyModule_DictIteratorStartC(index->worker_epochs, "^", NULL, 0);
        while (ValkeyModule_DictNextC(epochs, NULL, &data) != NULL) {
            WorkerEpoch *epoch = data;
            bool owner_is_inactive = !epoch->registration_owner_set ||
                                     epoch->lease_cleanup_pending ||
                                     epoch->registration_expires_at_ms <= now_ms;
            if (owner_is_inactive && epoch->lifecycle_managed &&
                !epoch->legacy_tainted) {
                ++direct_epochs;
            } else if (owner_is_inactive && epoch->legacy_tainted) {
                ++retained_legacy_epochs;
            }
        }
        ValkeyModule_DictIteratorStop(epochs);
        ValkeyModuleDictIter *nodes =
            ValkeyModule_DictIteratorStartC(index->nodes_by_external, "^", NULL, 0);
        while (ValkeyModule_DictNextC(nodes, NULL, &data) != NULL) {
            IndexNode *node = data;
            ownerless_nodes += node->owner_count == 0 ? 1 : 0;
            for (size_t i = 0; i < node->owner_count; ++i) {
                inactive_owners += node->owners[i].active ? 0 : 1;
            }
        }
        ValkeyModule_DictIteratorStop(nodes);
    }
    ValkeyModule_ReplyWithArray(ctx, 8);
    ValkeyModule_ReplyWithLongLong(
        ctx, index == NULL ? 0 : (long long)index->generation_counter);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)direct_tombstones);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)retained_legacy_ranks);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)direct_epochs);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)retained_legacy_epochs);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)inactive_owners);
    ValkeyModule_ReplyWithLongLong(ctx, (long long)ownerless_nodes);
    return ValkeyModule_ReplyWithLongLong(
        ctx, index == NULL ? 0 : (long long)index->gc_phase);
}
