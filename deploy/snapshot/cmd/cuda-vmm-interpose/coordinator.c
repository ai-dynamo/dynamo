/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "protocol.h"
#include "util.h"

#define TIMEOUT_SECONDS 30
#define STATE_FILENAME "cuda-vmm.state"

struct participant {
  char* endpoint;
  char id[SNAPSHOT_VMM_ID_SIZE];
  char job[SNAPSHOT_VMM_ID_SIZE];
  struct snapshot_vmm_record* records;
  uint32_t count;
};

struct multicast {
  uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  char creator[SNAPSHOT_VMM_ID_SIZE];
  uint64_t size;
  uint64_t handle_types;
  uint64_t flags;
  uint32_t num_devices;
  uint32_t participants;
  uint32_t creators;
  uint32_t devices;
  uint32_t bindings;
  struct multicast_device* device_list;
  struct multicast* next;
};

struct multicast_device {
  int32_t device;
  bool bound;
  struct multicast_device* next;
};

struct allocation {
  uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  char creator[SNAPSHOT_VMM_ID_SIZE];
  uint64_t size;
  bool creator_handle;
  bool creator_mapping;
  struct allocation* next;
};

static int
connect_endpoint(const char* endpoint)
{
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  int fd;

  if (strlen(endpoint) >= sizeof(address.sun_path))
    return -1;
  snprintf(address.sun_path, sizeof(address.sun_path), "%s", endpoint);
  fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0 || snapshot_vmm_set_socket_timeouts(fd, TIMEOUT_SECONDS) != 0 ||
      connect(fd, (const struct sockaddr*)&address, sizeof(address)) != 0) {
    if (fd >= 0)
      close(fd);
    return -1;
  }
  return fd;
}

static struct multicast*
find_multicast(struct multicast* multicasts, const uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE])
{
  struct multicast* multicast;

  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    if (memcmp(multicast->id, id, SNAPSHOT_VMM_ALLOCATION_ID_SIZE) == 0)
      return multicast;
  }
  return NULL;
}

static struct multicast_device*
find_multicast_device(struct multicast* multicast, int32_t device)
{
  struct multicast_device* current;

  for (current = multicast->device_list; current != NULL; current = current->next) {
    if (current->device == device)
      return current;
  }
  return NULL;
}

static void
free_multicasts(struct multicast* multicasts)
{
  while (multicasts != NULL) {
    struct multicast* next = multicasts->next;
    while (multicasts->device_list != NULL) {
      struct multicast_device* device_next = multicasts->device_list->next;
      free(multicasts->device_list);
      multicasts->device_list = device_next;
    }
    free(multicasts);
    multicasts = next;
  }
}

static int
exchange(struct participant* participant, uint16_t operation, struct snapshot_vmm_record** records, uint32_t* count)
{
  struct snapshot_vmm_header request;
  struct snapshot_vmm_header response;
  uint64_t payload_size;
  int fd = -1;
  int result = -1;

  memset(&request, 0, sizeof(request));
  request.magic = SNAPSHOT_VMM_MAGIC;
  request.version = SNAPSHOT_VMM_VERSION;
  request.operation = operation;
  if (operation != SNAPSHOT_VMM_IDENTIFY)
    snprintf(request.job_id, sizeof(request.job_id), "%s", participant->job);
  fd = connect_endpoint(participant->endpoint);
  if (fd < 0 || snapshot_vmm_write_all(fd, &request, sizeof(request)) != 0 ||
      snapshot_vmm_read_all(fd, &response, sizeof(response)) != 0)
    goto done;
  if (response.magic != SNAPSHOT_VMM_MAGIC || response.version != SNAPSHOT_VMM_VERSION ||
      response.operation != operation || response.status != 0 || response.count > SNAPSHOT_VMM_MAX_RECORDS ||
      response.payload_size != (uint64_t)response.count * sizeof(struct snapshot_vmm_record)) {
    if (response.message[0] != '\0')
      fprintf(stderr, "%s: %s\n", participant->endpoint, response.message);
    goto done;
  }
  if (operation == SNAPSHOT_VMM_IDENTIFY) {
    snprintf(participant->job, sizeof(participant->job), "%s", response.job_id);
    snprintf(participant->id, sizeof(participant->id), "%s", response.participant_id);
  } else if (strcmp(response.job_id, participant->job) != 0 || strcmp(response.participant_id, participant->id) != 0) {
    goto done;
  }
  payload_size = response.payload_size;
  if (payload_size != 0) {
    *records = calloc(response.count, sizeof(**records));
    if (*records == NULL || snapshot_vmm_read_all(fd, *records, (size_t)payload_size) != 0)
      goto done;
  }
  *count = response.count;
  result = 0;
done:
  if (fd >= 0)
    close(fd);
  if (result != 0) {
    free(*records);
    *records = NULL;
    *count = 0;
  }
  return result;
}

static struct allocation*
find_allocation(struct allocation* allocations, const uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE])
{
  struct allocation* allocation;

  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (memcmp(allocation->id, id, SNAPSHOT_VMM_ALLOCATION_ID_SIZE) == 0)
      return allocation;
  }
  return NULL;
}

static int
validate_topology(struct participant* participants, size_t participant_count, struct allocation** output)
{
  struct allocation* allocations = NULL;
  struct multicast* multicasts = NULL;
  const char* reason = NULL;
  uint64_t value = UINT64_MAX;
  const char* job;
  size_t participant_index;
  uint32_t record_index = UINT32_MAX;

  if (participant_count == 0) {
    fprintf(stderr, "topology validate failed: no participants\n");
    return -1;
  }
  job = participants[0].job;
  if (!snapshot_vmm_is_lower_hex_id(job)) {
    fprintf(stderr, "topology validate failed: invalid first participant job identity\n");
    return -1;
  }
  for (participant_index = 0; participant_index < participant_count; participant_index++) {
    struct participant* participant = &participants[participant_index];
    size_t previous;

    record_index = UINT32_MAX;
    if (!snapshot_vmm_is_lower_hex_id(participant->id)) {
      reason = "invalid participant identity";
      goto failed;
    }
    if (strcmp(participant->job, job) != 0) {
      reason = "mismatched participant job identity";
      goto failed;
    }
    for (previous = 0; previous < participant_index; previous++) {
      if (strcmp(participants[previous].id, participant->id) == 0) {
        reason = "duplicate participant identity";
        goto failed;
      }
    }
    for (record_index = 0; record_index < participant->count; record_index++) {
      const struct snapshot_vmm_record* record = &participant->records[record_index];
      struct allocation* allocation = find_allocation(allocations, record->allocation_id);
      struct multicast* multicast = find_multicast(multicasts, record->allocation_id);

      if (record->kind == SNAPSHOT_VMM_ALLOCATION) {
        if (allocation == NULL) {
          allocation = calloc(1, sizeof(*allocation));
          if (allocation == NULL) {
            reason = "allocation metadata allocation failed";
            goto failed;
          }
          memcpy(allocation->id, record->allocation_id, sizeof(allocation->id));
          allocation->next = allocations;
          allocations = allocation;
        }
        if ((record->flags & SNAPSHOT_VMM_CREATOR) != 0) {
          if (record->requested_handle_types != SNAPSHOT_VMM_POSIX_HANDLE_TYPE) {
            reason = "non-POSIX requested handle type";
            value = record->requested_handle_types;
            goto failed;
          }
          if (record->allocation_size == 0) {
            reason = "zero creator allocation size";
            goto failed;
          }
          if (allocation->creator[0] != '\0' && strcmp(allocation->creator, participant->id) != 0) {
            reason = "conflicting creators";
            goto failed;
          }
          snprintf(allocation->creator, sizeof(allocation->creator), "%s", participant->id);
          allocation->size = record->allocation_size;
          allocation->creator_handle = (record->flags & SNAPSHOT_VMM_APPLICATION_HANDLE_LIVE) != 0;
        }
      } else if (record->kind == SNAPSHOT_VMM_MAPPING) {
        if (record->address == 0) {
          reason = "zero mapping address";
          goto failed;
        }
        if (record->size == 0) {
          reason = "zero mapping size";
          goto failed;
        }
        if (record->access_count > SNAPSHOT_VMM_MAX_ACCESS) {
          reason = "mapping access count exceeds limit";
          value = record->access_count;
          goto failed;
        }
        if (allocation == NULL) {
          allocation = calloc(1, sizeof(*allocation));
          if (allocation == NULL) {
            reason = "allocation metadata allocation failed";
            goto failed;
          }
          memcpy(allocation->id, record->allocation_id, sizeof(allocation->id));
          allocation->next = allocations;
          allocations = allocation;
        }
        if ((record->flags & SNAPSHOT_VMM_CREATOR) != 0)
          allocation->creator_mapping = true;
      } else if (record->kind == SNAPSHOT_VMM_MULTICAST) {
        if (record->handle_types != SNAPSHOT_VMM_POSIX_HANDLE_TYPE || record->num_devices == 0 ||
            record->allocation_size == 0 || !snapshot_vmm_is_lower_hex_id(record->creator_participant)) {
          reason = "invalid multicast properties";
          goto failed;
        }
        if (multicast == NULL) {
          multicast = calloc(1, sizeof(*multicast));
          if (multicast == NULL) {
            reason = "multicast metadata allocation failed";
            goto failed;
          }
          memcpy(multicast->id, record->allocation_id, sizeof(multicast->id));
          multicast->size = record->allocation_size;
          multicast->handle_types = record->handle_types;
          multicast->flags = record->object_flags;
          multicast->num_devices = record->num_devices;
          snprintf(multicast->creator, sizeof(multicast->creator), "%s", record->creator_participant);
          multicast->next = multicasts;
          multicasts = multicast;
        } else if (
            multicast->size != record->allocation_size || multicast->handle_types != record->handle_types ||
            multicast->flags != record->object_flags || multicast->num_devices != record->num_devices ||
            strcmp(multicast->creator, record->creator_participant) != 0) {
          reason = "inconsistent multicast properties";
          goto failed;
        }
        multicast->participants++;
        if ((record->flags & SNAPSHOT_VMM_CREATOR) != 0) {
          if (strcmp(participant->id, multicast->creator) != 0) {
            reason = "invalid multicast creator";
            goto failed;
          }
          multicast->creators++;
        }
      } else if (record->kind == SNAPSHOT_VMM_MULTICAST_DEVICE) {
        struct multicast_device* device;
        if (multicast == NULL) {
          reason = "multicast device precedes object";
          goto failed;
        }
        if (find_multicast_device(multicast, record->device) != NULL) {
          reason = "duplicate multicast device";
          goto failed;
        }
        device = calloc(1, sizeof(*device));
        if (device == NULL) {
          reason = "multicast device metadata allocation failed";
          goto failed;
        }
        device->device = record->device;
        device->next = multicast->device_list;
        multicast->device_list = device;
        multicast->devices++;
      } else if (record->kind == SNAPSHOT_VMM_MULTICAST_BINDING) {
        struct allocation* member = find_allocation(allocations, record->member_id);
        struct multicast_device* device;
        if (multicast == NULL || record->size == 0 || record->offset > multicast->size ||
            record->size > multicast->size - record->offset ||
            (record->binding_kind != SNAPSHOT_VMM_MULTICAST_BIND_MEM &&
             record->binding_kind != SNAPSHOT_VMM_MULTICAST_BIND_ADDR) ||
            (record->api_version != 1 && record->api_version != 2)) {
          reason = "invalid multicast binding";
          goto failed;
        }
        if ((record->binding_kind == SNAPSHOT_VMM_MULTICAST_BIND_MEM && (member == NULL || record->address != 0)) ||
            (record->binding_kind == SNAPSHOT_VMM_MULTICAST_BIND_ADDR && record->address == 0)) {
          reason = "invalid multicast member";
          goto failed;
        }
        device = find_multicast_device(multicast, record->device);
        if (device == NULL) {
          reason = "multicast binding device is absent";
          goto failed;
        }
        device->bound = true;
        multicast->bindings++;
      } else if (record->kind == SNAPSHOT_VMM_MULTICAST_MAPPING) {
        if (multicast == NULL || record->address == 0 || record->size == 0 || record->offset > multicast->size ||
            record->size > multicast->size - record->offset || record->access_count > SNAPSHOT_VMM_MAX_ACCESS) {
          reason = "invalid multicast mapping";
          goto failed;
        }
      } else {
        reason = "unknown record kind";
        value = record->kind;
        goto failed;
      }
    }
  }
  {
    struct allocation* allocation;
    for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
      struct participant* participant;

      if (allocation->creator[0] == '\0') {
        reason = "missing creator";
        goto failed;
      }
      if (allocation->size == 0) {
        reason = "missing allocation size";
        goto failed;
      }
      if (!allocation->creator_handle && !allocation->creator_mapping) {
        reason = "missing creator anchor";
        goto failed;
      }
      for (participant_index = 0; participant_index < participant_count; participant_index++) {
        participant = &participants[participant_index];
        for (record_index = 0; record_index < participant->count; record_index++) {
          const struct snapshot_vmm_record* record = &participant->records[record_index];
          if (record->kind == SNAPSHOT_VMM_MAPPING &&
              memcmp(record->allocation_id, allocation->id, sizeof(allocation->id)) == 0 &&
              (record->offset > allocation->size || record->size > allocation->size - record->offset)) {
            reason = "mapping out of bounds";
            goto failed;
          }
        }
      }
    }
  }
  {
    struct multicast* multicast;
    for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
      if (multicast->participants != participant_count) {
        reason = "incomplete multicast participant group";
        value = multicast->participants;
        goto failed;
      }
      if (multicast->creators != 1) {
        reason = "multicast group must have exactly one creator";
        value = multicast->creators;
        goto failed;
      }
      if (multicast->devices != multicast->num_devices) {
        reason = "incomplete multicast device group";
        value = multicast->devices;
        goto failed;
      }
      if (multicast->bindings < multicast->num_devices) {
        reason = "incomplete multicast binding group";
        value = multicast->bindings;
        goto failed;
      }
      {
        struct multicast_device* device;
        for (device = multicast->device_list; device != NULL; device = device->next) {
          if (!device->bound) {
            reason = "multicast device has no binding";
            value = (uint64_t)(uint32_t)device->device;
            goto failed;
          }
        }
      }
    }
  }
  *output = allocations;
  free_multicasts(multicasts);
  return 0;
failed:
  if (value != UINT64_MAX)
    fprintf(
        stderr, "topology validate failed: %s (value %llu, participant index %zu, record index %u)\n", reason,
        (unsigned long long)value, participant_index, (unsigned)record_index);
  else if (participant_index < participant_count && record_index != UINT32_MAX)
    fprintf(
        stderr, "topology validate failed: %s (participant index %zu, record index %u)\n", reason, participant_index,
        (unsigned)record_index);
  else if (participant_index < participant_count)
    fprintf(stderr, "topology validate failed: %s (participant index %zu)\n", reason, participant_index);
  else
    fprintf(stderr, "topology validate failed: %s\n", reason);
  while (allocations != NULL) {
    struct allocation* next = allocations->next;
    free(allocations);
    allocations = next;
  }
  free_multicasts(multicasts);
  return -1;
}

static int write_state(struct participant* participants, size_t count, FILE* output);

static int
write_state_atomic(const char* path, struct participant* participants, size_t count)
{
  char temporary[PATH_MAX];
  FILE* output = NULL;
  int fd;
  int length;
  int result = -1;

  length = snprintf(temporary, sizeof(temporary), "%s.tmp.XXXXXX", path);
  if (length < 0 || (size_t)length >= sizeof(temporary))
    return -1;
  fd = mkstemp(temporary);
  if (fd < 0)
    return -1;
  output = fdopen(fd, "w");
  if (output == NULL) {
    close(fd);
    unlink(temporary);
    return -1;
  }
  if (write_state(participants, count, output) != 0 || fsync(fileno(output)) != 0)
    goto done;
  if (fclose(output) != 0) {
    output = NULL;
    goto done;
  }
  output = NULL;
  if (rename(temporary, path) != 0)
    goto done;
  result = 0;
done:
  if (output != NULL)
    fclose(output);
  if (result != 0)
    unlink(temporary);
  return result;
}

static int
record_compare(const void* left, const void* right)
{
  return memcmp(left, right, sizeof(struct snapshot_vmm_record));
}

static int
participant_compare(const void* left, const void* right)
{
  const struct participant* a = left;
  const struct participant* b = right;
  return strcmp(a->id, b->id);
}

static int
write_state(struct participant* participants, size_t count, FILE* output)
{
  size_t index;

  qsort(participants, count, sizeof(*participants), participant_compare);
  if (fprintf(output, "snapshot-cuda-posix-v2\n") < 0 || fprintf(output, "job %s\n", participants[0].job) < 0)
    return -1;
  for (index = 0; index < count; index++) {
    struct participant* participant = &participants[index];
    uint32_t record_index;

    qsort(participant->records, participant->count, sizeof(*participant->records), record_compare);
    if (fprintf(output, "participant %s %u\n", participant->id, participant->count) < 0)
      return -1;
    for (record_index = 0; record_index < participant->count; record_index++) {
      const uint8_t* bytes = (const uint8_t*)&participant->records[record_index];
      size_t byte_index;
      for (byte_index = 0; byte_index < sizeof(struct snapshot_vmm_record); byte_index++) {
        if (fprintf(output, "%02x", bytes[byte_index]) < 0)
          return -1;
      }
      if (fputc('\n', output) == EOF)
        return -1;
    }
  }
  return fflush(output);
}

static int
hex_digit(int value)
{
  if (value >= '0' && value <= '9')
    return value - '0';
  if (value >= 'a' && value <= 'f')
    return value - 'a' + 10;
  return -1;
}

static int
read_record(FILE* input, struct snapshot_vmm_record* record)
{
  uint8_t* bytes = (uint8_t*)record;
  size_t index;

  for (index = 0; index < sizeof(*record); index++) {
    int high = hex_digit(fgetc(input));
    int low = hex_digit(fgetc(input));
    if (high < 0 || low < 0)
      return -1;
    bytes[index] = (uint8_t)((high << 4) | low);
  }
  return fgetc(input) == '\n' ? 0 : -1;
}

static int
read_state(FILE* input, struct participant** output, size_t* output_count)
{
  char line[128];
  char job[SNAPSHOT_VMM_ID_SIZE];
  struct participant* participants = NULL;
  size_t count = 0;

  if (fgets(line, sizeof(line), input) == NULL || strcmp(line, "snapshot-cuda-posix-v2\n") != 0 ||
      fgets(line, sizeof(line), input) == NULL || sscanf(line, "job %32s", job) != 1 ||
      !snapshot_vmm_is_lower_hex_id(job))
    return -1;
  while (fgets(line, sizeof(line), input) != NULL) {
    struct participant* participant;
    char id[SNAPSHOT_VMM_ID_SIZE];
    unsigned int record_count;
    unsigned int index;

    if (sscanf(line, "participant %32s %u", id, &record_count) != 2 || !snapshot_vmm_is_lower_hex_id(id) ||
        record_count > SNAPSHOT_VMM_MAX_RECORDS)
      goto failed;
    participants = realloc(participants, (count + 1) * sizeof(*participants));
    if (participants == NULL)
      goto failed;
    participant = &participants[count++];
    memset(participant, 0, sizeof(*participant));
    snprintf(participant->id, sizeof(participant->id), "%s", id);
    snprintf(participant->job, sizeof(participant->job), "%s", job);
    participant->records = calloc(record_count == 0 ? 1 : record_count, sizeof(*participant->records));
    if (participant->records == NULL)
      goto failed;
    participant->count = record_count;
    for (index = 0; index < record_count; index++) {
      if (read_record(input, &participant->records[index]) != 0)
        goto failed;
    }
  }
  *output = participants;
  *output_count = count;
  return count == 0 ? -1 : 0;
failed:
  if (participants != NULL) {
    size_t index;
    for (index = 0; index < count; index++) free(participants[index].records);
  }
  free(participants);
  return -1;
}

static int
identify(struct participant* participants, size_t count)
{
  size_t index;

  for (index = 0; index < count; index++) {
    struct snapshot_vmm_record* records = NULL;
    uint32_t record_count = 0;
    if (exchange(&participants[index], SNAPSHOT_VMM_IDENTIFY, &records, &record_count) != 0)
      return -1;
    free(records);
  }
  return 0;
}

static int
inspect(struct participant* participants, size_t count)
{
  size_t index;

  if (identify(participants, count) != 0)
    return -1;
  for (index = 0; index < count; index++) {
    if (exchange(
            &participants[index], SNAPSHOT_VMM_INSPECT, &participants[index].records, &participants[index].count) != 0)
      return -1;
  }
  return 0;
}

static void
free_participants(struct participant* participants, size_t count)
{
  size_t index;

  if (participants == NULL)
    return;
  for (index = 0; index < count; index++) {
    free(participants[index].endpoint);
    free(participants[index].records);
  }
  free(participants);
}

static void
free_allocations(struct allocation* allocations)
{
  while (allocations != NULL) {
    struct allocation* next = allocations->next;
    free(allocations);
    allocations = next;
  }
}

static int
command_all(struct participant* participants, size_t count, uint16_t operation)
{
  size_t index;

  for (index = 0; index < count; index++) {
    struct snapshot_vmm_record* records = NULL;
    uint32_t record_count = 0;
    if (exchange(&participants[index], operation, &records, &record_count) != 0)
      return -1;
    free(records);
  }
  return 0;
}

static int
same_topology(struct participant* expected, size_t expected_count, struct participant* actual, size_t actual_count)
{
  size_t index;

  if (expected_count != actual_count)
    return -1;
  qsort(expected, expected_count, sizeof(*expected), participant_compare);
  qsort(actual, actual_count, sizeof(*actual), participant_compare);
  for (index = 0; index < expected_count; index++) {
    if (strcmp(expected[index].job, actual[index].job) != 0 || strcmp(expected[index].id, actual[index].id) != 0 ||
        expected[index].count != actual[index].count)
      return -1;
    qsort(expected[index].records, expected[index].count, sizeof(*expected[index].records), record_compare);
    qsort(actual[index].records, actual[index].count, sizeof(*actual[index].records), record_compare);
    if (memcmp(
            expected[index].records, actual[index].records,
            (size_t)expected[index].count * sizeof(*expected[index].records)) != 0)
      return -1;
  }
  return 0;
}

int
main(int argc, char** argv)
{
  struct participant* participants = NULL;
  struct participant* expected = NULL;
  struct allocation* allocations = NULL;
  size_t participant_count = 0;
  size_t expected_count = 0;
  size_t index;
  bool prepare;
  int result = EXIT_FAILURE;
  FILE* state = NULL;

  char state_path[PATH_MAX];
  const char* proc_root;
  const char* checkpoint_dir;
  const char* control_dir = "/snapshot-control";
  int length;

  if (argc < 9 || (argc - 6) % 3 != 0 || (strcmp(argv[1], "--prepare") != 0 && strcmp(argv[1], "--restore") != 0)) {
    fprintf(
        stderr,
        "usage: %s (--prepare|--restore) --proc-root PATH "
        "--checkpoint-dir PATH --process OBSERVED_PID NAMESPACE_PID...\n",
        argv[0]);
    return EXIT_FAILURE;
  }
  prepare = strcmp(argv[1], "--prepare") == 0;
  if (strcmp(argv[2], "--proc-root") != 0 || strcmp(argv[4], "--checkpoint-dir") != 0)
    return EXIT_FAILURE;
  proc_root = argv[3];
  checkpoint_dir = argv[5];
  length = snprintf(state_path, sizeof(state_path), "%s/%s", checkpoint_dir, STATE_FILENAME);
  if (length < 0 || (size_t)length >= sizeof(state_path))
    return EXIT_FAILURE;
  if (!prepare) {
    state = fopen(state_path, "r");
    if (state == NULL)
      return errno == ENOENT ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (proc_root[0] == '\0') {
    const char* control = getenv("DYN_SNAPSHOT_CONTROL_DIR");
    if (control != NULL && control[0] != '\0')
      control_dir = control;
  }
  participant_count = (size_t)(argc - 6) / 3;
  participants = calloc(participant_count, sizeof(*participants));
  if (participants == NULL)
    goto done;
  for (index = 0; index < participant_count; index++) {
    char endpoint[sizeof(((struct sockaddr_un*)0)->sun_path)];
    char* end;
    long observed;
    long namespace;
    int length;

    if (strcmp(argv[6 + index * 3], "--process") != 0)
      goto done;
    errno = 0;
    observed = strtol(argv[7 + index * 3], &end, 10);
    if (errno != 0 || *end != '\0' || observed <= 0 || observed > INT_MAX)
      goto done;
    errno = 0;
    namespace = strtol(argv[8 + index * 3], &end, 10);
    if (errno != 0 || *end != '\0' || namespace <= 0 || namespace > INT_MAX)
      goto done;
    if (proc_root[0] == '\0')
      length =
          snprintf(endpoint, sizeof(endpoint), "%s/%s%ld.sock", control_dir, SNAPSHOT_VMM_SOCKET_PREFIX, namespace);
    else
      length = snprintf(
          endpoint, sizeof(endpoint), "%s/%ld/root%s/%s%ld.sock", proc_root, observed, control_dir,
          SNAPSHOT_VMM_SOCKET_PREFIX, namespace);
    if (length < 0 || (size_t)length >= sizeof(endpoint))
      goto done;
    participants[index].endpoint = strdup(endpoint);
    if (participants[index].endpoint == NULL)
      goto done;
  }
  if (prepare) {
    if (inspect(participants, participant_count) != 0) {
      fprintf(stderr, "prepare failed: participant inspect\n");
      goto done;
    }
    if (validate_topology(participants, participant_count, &allocations) != 0) {
      fprintf(stderr, "prepare failed: topology validate\n");
      goto done;
    }
    if (command_all(participants, participant_count, SNAPSHOT_VMM_CREATE_CARRIERS) != 0) {
      fprintf(stderr, "prepare failed: checkpoint carrier creation\n");
      goto done;
    }
    if (command_all(participants, participant_count, SNAPSHOT_VMM_PREPARE_MULTICAST) != 0) {
      fprintf(stderr, "prepare failed: multicast teardown\n");
      goto done;
    }
    if (command_all(participants, participant_count, SNAPSHOT_VMM_PREPARE) != 0) {
      fprintf(stderr, "prepare failed: participant prepare\n");
      goto done;
    }
    if (write_state_atomic(state_path, participants, participant_count) != 0) {
      fprintf(stderr, "prepare failed: atomic state write\n");
      goto done;
    }
  } else {
    if (read_state(state, &expected, &expected_count) != 0)
      goto done;
    result = fclose(state);
    state = NULL;
    if (result != 0 || identify(participants, participant_count) != 0 || expected_count != participant_count)
      goto done;
    result = EXIT_FAILURE;
    for (index = 0; index < participant_count; index++) {
      if (strcmp(participants[index].job, expected[0].job) != 0)
        goto done;
    }
    if (command_all(participants, participant_count, SNAPSHOT_VMM_RESTORE_CREATORS) != 0 ||
        command_all(participants, participant_count, SNAPSHOT_VMM_RESTORE_IMPORTERS) != 0 ||
        command_all(participants, participant_count, SNAPSHOT_VMM_RESTORE_MULTICAST_CREATORS) != 0 ||
        command_all(participants, participant_count, SNAPSHOT_VMM_RESTORE_MULTICAST_IMPORTERS) != 0 ||
        command_all(participants, participant_count, SNAPSHOT_VMM_RESTORE_MULTICAST_DEVICES) != 0 ||
        command_all(participants, participant_count, SNAPSHOT_VMM_RESTORE_MULTICAST) != 0) {
      goto done;
    }
    for (index = 0; index < participant_count; index++) {
      free(participants[index].records);
      participants[index].records = NULL;
      participants[index].count = 0;
    }
    if (inspect(participants, participant_count) != 0 ||
        validate_topology(participants, participant_count, &allocations) != 0 ||
        same_topology(expected, expected_count, participants, participant_count) != 0)
      goto done;
  }
  result = EXIT_SUCCESS;
done:
  if (state != NULL)
    fclose(state);
  free_allocations(allocations);
  free_participants(expected, expected_count);
  free_participants(participants, participant_count);
  return result;
}
