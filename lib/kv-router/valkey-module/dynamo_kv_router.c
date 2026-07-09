/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dynkv_state.h"

ValkeyModuleType *RouterIndexType;

void encode_u32_be(uint8_t out[4], uint32_t value) {
    out[0] = (uint8_t)(value >> 24);
    out[1] = (uint8_t)(value >> 16);
    out[2] = (uint8_t)(value >> 8);
    out[3] = (uint8_t)value;
}
void encode_u64_be(uint8_t out[8], uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out[i] = (uint8_t)(value >> (56 - (i * 8)));
    }
}

void worker_key(uint8_t out[12], uint64_t worker_id, uint32_t dp_rank) {
    encode_u64_be(out, worker_id);
    encode_u32_be(out + 8, dp_rank);
}

bool reservation_key(
    uint8_t *out,
    size_t out_capacity,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t client_nonce,
    uint64_t request_nonce,
    size_t *length_out) {
    size_t required = 4 + (size_t)domain_length + 16;
    if (domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH || required > out_capacity) {
        return false;
    }
    encode_u32_be(out, domain_length);
    memcpy(out + 4, domain, domain_length);
    encode_u64_be(out + 4 + domain_length, client_nonce);
    encode_u64_be(out + 12 + domain_length, request_nonce);
    *length_out = required;
    return true;
}

/* The admission domain is part of the capacity and lease authority. */
bool admission_rank_key(
    uint8_t *out,
    size_t out_capacity,
    const uint8_t *domain,
    uint32_t domain_length,
    uint64_t worker_id,
    uint32_t dp_rank,
    size_t *length_out) {
    size_t required = 4 + (size_t)domain_length + 12;
    if (domain_length == 0 || domain_length > DYNKV_MAX_ADMISSION_DOMAIN_LENGTH ||
        required > out_capacity) {
        return false;
    }
    encode_u32_be(out, domain_length);
    memcpy(out + 4, domain, domain_length);
    worker_key(out + 4 + domain_length, worker_id, dp_rank);
    *length_out = required;
    return true;
}

void child_key(uint8_t out[16], uint64_t parent_hash, uint64_t local_hash) {
    encode_u64_be(out, parent_hash);
    encode_u64_be(out + 8, local_hash);
}

bool reader_u8(Reader *reader, uint8_t *value) {
    if (reader->offset >= reader->length) {
        return false;
    }
    *value = reader->data[reader->offset++];
    return true;
}

bool reader_u32(Reader *reader, uint32_t *value) {
    if (reader->length - reader->offset < 4) {
        return false;
    }
    const uint8_t *data = reader->data + reader->offset;
    *value = ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
             ((uint32_t)data[2] << 8) | (uint32_t)data[3];
    reader->offset += 4;
    return true;
}

bool reader_u64(Reader *reader, uint64_t *value) {
    if (reader->length - reader->offset < 8) {
        return false;
    }
    uint64_t result = 0;
    for (size_t i = 0; i < 8; ++i) {
        result = (result << 8) | reader->data[reader->offset + i];
    }
    reader->offset += 8;
    *value = result;
    return true;
}

bool buffer_reserve(Buffer *buffer, size_t extra) {
    if (extra > SIZE_MAX - buffer->length) {
        return false;
    }
    size_t required = buffer->length + extra;
    if (required <= buffer->capacity) {
        return true;
    }
    size_t capacity = buffer->capacity == 0 ? 128 : buffer->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    buffer->data = ValkeyModule_Realloc(buffer->data, capacity);
    buffer->capacity = capacity;
    return true;
}

bool buffer_u8(Buffer *buffer, uint8_t value) {
    if (!buffer_reserve(buffer, 1)) {
        return false;
    }
    buffer->data[buffer->length++] = value;
    return true;
}

bool buffer_u32(Buffer *buffer, uint32_t value) {
    if (!buffer_reserve(buffer, 4)) {
        return false;
    }
    encode_u32_be(buffer->data + buffer->length, value);
    buffer->length += 4;
    return true;
}

bool buffer_u64(Buffer *buffer, uint64_t value) {
    if (!buffer_reserve(buffer, 8)) {
        return false;
    }
    encode_u64_be(buffer->data + buffer->length, value);
    buffer->length += 8;
    return true;
}

bool buffer_bytes(Buffer *buffer, const uint8_t *data, size_t length) {
    if (!buffer_reserve(buffer, length)) {
        return false;
    }
    memcpy(buffer->data + buffer->length, data, length);
    buffer->length += length;
    return true;
}

void buffer_free(Buffer *buffer) {
    ValkeyModule_Free(buffer->data);
    buffer->data = NULL;
    buffer->length = 0;
    buffer->capacity = 0;
}

RouterIndex *router_index_create(void) {
    RouterIndex *index = ValkeyModule_Calloc(1, sizeof(*index));
    /* Use a NULL context: command auto-memory must not own persistent maps. */
    index->nodes_by_external = ValkeyModule_CreateDict(NULL);
    index->children_by_parent_and_local = ValkeyModule_CreateDict(NULL);
    index->workers = ValkeyModule_CreateDict(NULL);
    index->worker_epochs = ValkeyModule_CreateDict(NULL);
    index->admission_ranks = ValkeyModule_CreateDict(NULL);
    index->reservations = ValkeyModule_CreateDict(NULL);
    return index;
}

WorkerEpoch *router_index_worker_epoch_lookup(
    RouterIndex *index,
    uint64_t worker_id) {
    uint8_t key[8];
    encode_u64_be(key, worker_id);
    return ValkeyModule_DictGetC(index->worker_epochs, key, sizeof(key), NULL);
}

void index_node_free(IndexNode *node) {
    if (node == NULL) {
        return;
    }
    ValkeyModule_Free(node->owners);
    ValkeyModule_Free(node);
}

void worker_state_free(WorkerState *worker) {
    if (worker == NULL) {
        return;
    }
    ValkeyModule_FreeDict(NULL, worker->node_members);
    ValkeyModule_Free(worker->nodes);
    ValkeyModule_Free(worker->node_owner_indices);
    ValkeyModule_Free(worker);
}

void worker_epoch_free(WorkerEpoch *epoch) {
    if (epoch == NULL) {
        return;
    }
    ValkeyModule_Free(epoch->worker_states);
    ValkeyModule_Free(epoch->admission_rank_states);
    ValkeyModule_Free(epoch->reservation_states);
    ValkeyModule_Free(epoch);
}

void admission_rank_state_free(AdmissionRankState *rank) {
    if (rank == NULL) {
        return;
    }
    ValkeyModule_Free(rank->domain);
    ValkeyModule_Free(rank);
}

void reservation_free(Reservation *reservation) {
    if (reservation == NULL) {
        return;
    }
    ValkeyModule_Free(reservation->domain);
    ValkeyModule_Free(reservation->request_bytes);
    ValkeyModule_Free(reservation);
}

bool reservation_expires_before(const Reservation *left, const Reservation *right) {
    return left->expires_at_ms < right->expires_at_ms;
}

void router_index_reservation_expiry_heap_swap(
    RouterIndex *index,
    size_t left,
    size_t right) {
    Reservation *temporary = index->reservation_expiry_heap[left];
    index->reservation_expiry_heap[left] = index->reservation_expiry_heap[right];
    index->reservation_expiry_heap[right] = temporary;
    index->reservation_expiry_heap[left]->expiry_heap_index = left;
    index->reservation_expiry_heap[right]->expiry_heap_index = right;
}

void router_index_reservation_expiry_heap_sift_up(RouterIndex *index, size_t position) {
    while (position != 0) {
        size_t parent = (position - 1) / 2;
        if (!reservation_expires_before(
                index->reservation_expiry_heap[position],
                index->reservation_expiry_heap[parent])) {
            return;
        }
        router_index_reservation_expiry_heap_swap(index, position, parent);
        position = parent;
    }
}

void router_index_reservation_expiry_heap_sift_down(
    RouterIndex *index,
    size_t position) {
    while (position < index->reservation_expiry_heap_length / 2) {
        size_t left = position * 2 + 1;
        size_t smallest = left;
        size_t right = left + 1;
        if (right < index->reservation_expiry_heap_length &&
            reservation_expires_before(
                index->reservation_expiry_heap[right],
                index->reservation_expiry_heap[left])) {
            smallest = right;
        }
        if (!reservation_expires_before(
                index->reservation_expiry_heap[smallest],
                index->reservation_expiry_heap[position])) {
            return;
        }
        router_index_reservation_expiry_heap_swap(index, position, smallest);
        position = smallest;
    }
}

bool router_index_reservation_expiry_heap_reserve(
    RouterIndex *index,
    size_t required) {
    if (required <= index->reservation_expiry_heap_capacity) {
        return true;
    }
    size_t capacity = index->reservation_expiry_heap_capacity == 0
                          ? 8
                          : index->reservation_expiry_heap_capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*index->reservation_expiry_heap)) {
        return false;
    }
    Reservation **heap = ValkeyModule_Realloc(
        index->reservation_expiry_heap,
        capacity * sizeof(*index->reservation_expiry_heap));
    if (heap == NULL) {
        return false;
    }
    index->reservation_expiry_heap = heap;
    index->reservation_expiry_heap_capacity = capacity;
    return true;
}

bool router_index_reservation_expiry_heap_insert(
    RouterIndex *index,
    Reservation *reservation) {
    if (index->reservation_expiry_heap_length == SIZE_MAX ||
        !router_index_reservation_expiry_heap_reserve(
            index, index->reservation_expiry_heap_length + 1)) {
        return false;
    }
    size_t position = index->reservation_expiry_heap_length++;
    index->reservation_expiry_heap[position] = reservation;
    reservation->expiry_heap_index = position;
    router_index_reservation_expiry_heap_sift_up(index, position);
    return true;
}

bool router_index_reservation_expiry_heap_remove(
    RouterIndex *index,
    Reservation *reservation) {
    size_t position = reservation->expiry_heap_index;
    if (position >= index->reservation_expiry_heap_length ||
        index->reservation_expiry_heap[position] != reservation) {
        return false;
    }
    size_t last = --index->reservation_expiry_heap_length;
    if (position != last) {
        Reservation *moved = index->reservation_expiry_heap[last];
        index->reservation_expiry_heap[position] = moved;
        moved->expiry_heap_index = position;
        if (position != 0 &&
            reservation_expires_before(
                moved, index->reservation_expiry_heap[(position - 1) / 2])) {
            router_index_reservation_expiry_heap_sift_up(index, position);
        } else {
            router_index_reservation_expiry_heap_sift_down(index, position);
        }
    }
    reservation->expiry_heap_index = SIZE_MAX;
    return true;
}

bool router_index_reservation_expiry_heap_reposition(
    RouterIndex *index,
    Reservation *reservation) {
    size_t position = reservation->expiry_heap_index;
    if (position >= index->reservation_expiry_heap_length ||
        index->reservation_expiry_heap[position] != reservation) {
        return false;
    }
    if (position != 0 &&
        reservation_expires_before(
            reservation, index->reservation_expiry_heap[(position - 1) / 2])) {
        router_index_reservation_expiry_heap_sift_up(index, position);
    } else {
        router_index_reservation_expiry_heap_sift_down(index, position);
    }
    return true;
}

bool worker_lease_expires_before(const WorkerEpoch *left, const WorkerEpoch *right) {
    return left->registration_expires_at_ms < right->registration_expires_at_ms ||
           (left->registration_expires_at_ms == right->registration_expires_at_ms &&
            left->worker_id < right->worker_id);
}

void router_index_worker_lease_expiry_heap_swap(
    RouterIndex *index,
    size_t left,
    size_t right) {
    WorkerEpoch *temporary = index->worker_lease_expiry_heap[left];
    index->worker_lease_expiry_heap[left] = index->worker_lease_expiry_heap[right];
    index->worker_lease_expiry_heap[right] = temporary;
    index->worker_lease_expiry_heap[left]->registration_expiry_heap_index = left;
    index->worker_lease_expiry_heap[right]->registration_expiry_heap_index = right;
}

void router_index_worker_lease_expiry_heap_sift_up(
    RouterIndex *index,
    size_t position) {
    while (position != 0) {
        size_t parent = (position - 1) / 2;
        if (!worker_lease_expires_before(
                index->worker_lease_expiry_heap[position],
                index->worker_lease_expiry_heap[parent])) {
            return;
        }
        router_index_worker_lease_expiry_heap_swap(index, position, parent);
        position = parent;
    }
}

void router_index_worker_lease_expiry_heap_sift_down(
    RouterIndex *index,
    size_t position) {
    while (position < index->worker_lease_expiry_heap_length / 2) {
        size_t left = position * 2 + 1;
        size_t smallest = left;
        size_t right = left + 1;
        if (right < index->worker_lease_expiry_heap_length &&
            worker_lease_expires_before(
                index->worker_lease_expiry_heap[right],
                index->worker_lease_expiry_heap[left])) {
            smallest = right;
        }
        if (!worker_lease_expires_before(
                index->worker_lease_expiry_heap[smallest],
                index->worker_lease_expiry_heap[position])) {
            return;
        }
        router_index_worker_lease_expiry_heap_swap(index, position, smallest);
        position = smallest;
    }
}

bool router_index_worker_lease_expiry_heap_reserve(
    RouterIndex *index,
    size_t required) {
    if (required <= index->worker_lease_expiry_heap_capacity) {
        return true;
    }
    size_t capacity = index->worker_lease_expiry_heap_capacity == 0
                          ? 8
                          : index->worker_lease_expiry_heap_capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*index->worker_lease_expiry_heap)) {
        return false;
    }
    WorkerEpoch **heap = ValkeyModule_Realloc(
        index->worker_lease_expiry_heap,
        capacity * sizeof(*index->worker_lease_expiry_heap));
    if (heap == NULL) {
        return false;
    }
    index->worker_lease_expiry_heap = heap;
    index->worker_lease_expiry_heap_capacity = capacity;
    return true;
}

bool router_index_worker_lease_expiry_heap_insert(
    RouterIndex *index,
    WorkerEpoch *epoch) {
    if (epoch->registration_expiry_heap_index != SIZE_MAX ||
        index->worker_lease_expiry_heap_length == SIZE_MAX ||
        !router_index_worker_lease_expiry_heap_reserve(
            index, index->worker_lease_expiry_heap_length + 1)) {
        return false;
    }
    size_t position = index->worker_lease_expiry_heap_length++;
    index->worker_lease_expiry_heap[position] = epoch;
    epoch->registration_expiry_heap_index = position;
    router_index_worker_lease_expiry_heap_sift_up(index, position);
    return true;
}

bool router_index_worker_lease_expiry_heap_remove(
    RouterIndex *index,
    WorkerEpoch *epoch) {
    size_t position = epoch->registration_expiry_heap_index;
    if (position >= index->worker_lease_expiry_heap_length ||
        index->worker_lease_expiry_heap[position] != epoch) {
        return false;
    }
    size_t last = --index->worker_lease_expiry_heap_length;
    if (position != last) {
        WorkerEpoch *moved = index->worker_lease_expiry_heap[last];
        index->worker_lease_expiry_heap[position] = moved;
        moved->registration_expiry_heap_index = position;
        if (position != 0 &&
            worker_lease_expires_before(
                moved, index->worker_lease_expiry_heap[(position - 1) / 2])) {
            router_index_worker_lease_expiry_heap_sift_up(index, position);
        } else {
            router_index_worker_lease_expiry_heap_sift_down(index, position);
        }
    }
    epoch->registration_expiry_heap_index = SIZE_MAX;
    return true;
}

bool router_index_worker_lease_expiry_heap_reposition(
    RouterIndex *index,
    WorkerEpoch *epoch) {
    size_t position = epoch->registration_expiry_heap_index;
    if (position >= index->worker_lease_expiry_heap_length ||
        index->worker_lease_expiry_heap[position] != epoch) {
        return false;
    }
    if (position != 0 &&
        worker_lease_expires_before(
            epoch, index->worker_lease_expiry_heap[(position - 1) / 2])) {
        router_index_worker_lease_expiry_heap_sift_up(index, position);
    } else {
        router_index_worker_lease_expiry_heap_sift_down(index, position);
    }
    return true;
}

void router_index_free(void *value) {
    RouterIndex *index = value;
    if (index == NULL) {
        return;
    }

    ValkeyModuleDictIter *nodes =
        ValkeyModule_DictIteratorStartC(index->nodes_by_external, "^", NULL, 0);
    void *data = NULL;
    while (ValkeyModule_DictNextC(nodes, NULL, &data) != NULL) {
        index_node_free(data);
    }
    ValkeyModule_DictIteratorStop(nodes);

    ValkeyModuleDictIter *workers =
        ValkeyModule_DictIteratorStartC(index->workers, "^", NULL, 0);
    while (ValkeyModule_DictNextC(workers, NULL, &data) != NULL) {
        worker_state_free(data);
    }
    ValkeyModule_DictIteratorStop(workers);

    ValkeyModuleDictIter *epochs =
        ValkeyModule_DictIteratorStartC(index->worker_epochs, "^", NULL, 0);
    while (ValkeyModule_DictNextC(epochs, NULL, &data) != NULL) {
        worker_epoch_free(data);
    }
    ValkeyModule_DictIteratorStop(epochs);

    ValkeyModuleDictIter *admission_ranks =
        ValkeyModule_DictIteratorStartC(index->admission_ranks, "^", NULL, 0);
    while (ValkeyModule_DictNextC(admission_ranks, NULL, &data) != NULL) {
        admission_rank_state_free(data);
    }
    ValkeyModule_DictIteratorStop(admission_ranks);

    ValkeyModuleDictIter *reservations =
        ValkeyModule_DictIteratorStartC(index->reservations, "^", NULL, 0);
    while (ValkeyModule_DictNextC(reservations, NULL, &data) != NULL) {
        reservation_free(data);
    }
    ValkeyModule_DictIteratorStop(reservations);
    ValkeyModule_Free(index->reservation_expiry_heap);
    ValkeyModule_Free(index->worker_lease_expiry_heap);

    ValkeyModule_FreeDict(NULL, index->nodes_by_external);
    ValkeyModule_FreeDict(NULL, index->children_by_parent_and_local);
    ValkeyModule_FreeDict(NULL, index->workers);
    ValkeyModule_FreeDict(NULL, index->worker_epochs);
    ValkeyModule_FreeDict(NULL, index->admission_ranks);
    ValkeyModule_FreeDict(NULL, index->reservations);
    ValkeyModule_Free(index);
}
