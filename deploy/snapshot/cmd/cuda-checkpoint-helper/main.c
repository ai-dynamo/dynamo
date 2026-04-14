#include <ctype.h>
#include <cuda.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
print_usage(FILE* stream)
{
  return fprintf(
             stream,
             "Usage:\n"
             "  cuda-checkpoint-helper --get-state --pid <pid>\n"
             "  cuda-checkpoint-helper --get-restore-tid --pid <pid>\n"
             "  cuda-checkpoint-helper --action lock|checkpoint|restore|unlock --pid <pid> [--timeout <ms>] "
             "[--device-map <uuids>]\n") < 0
             ? 1
             : 0;
}

static void
print_cuda_error(CUresult status)
{
  const char* name = NULL;
  const char* msg = NULL;

  (void)cuGetErrorName(status, &name);
  (void)cuGetErrorString(status, &msg);

  if (name == NULL) {
    name = "CUDA_ERROR_UNKNOWN";
  }
  if (msg == NULL) {
    msg = "unknown CUDA error";
  }

  fprintf(stderr, "%s: %s\n", name, msg);
}

static int
parse_pid(const char* pid_str, int* pid_out)
{
  char* end = NULL;
  long pid = strtol(pid_str, &end, 10);

  if (pid_str[0] == '\0' || end == NULL || *end != '\0' || pid <= 0 || pid > INT_MAX) {
    return -1;
  }

  *pid_out = (int)pid;
  return 0;
}

static int
parse_timeout_ms(const char* timeout_str, unsigned int* timeout_ms_out)
{
  char* end = NULL;
  unsigned long timeout_ms = strtoul(timeout_str, &end, 10);

  if (timeout_str[0] == '\0' || end == NULL || *end != '\0' || timeout_ms > UINT_MAX) {
    return -1;
  }

  *timeout_ms_out = (unsigned int)timeout_ms;
  return 0;
}

static int
parse_hex_byte(const char* src, unsigned char* byte_out)
{
  char tmp[3];
  char* end = NULL;
  long value;

  tmp[0] = src[0];
  tmp[1] = src[1];
  tmp[2] = '\0';

  value = strtol(tmp, &end, 16);
  if (end == NULL || *end != '\0' || value < 0 || value > 255) {
    return -1;
  }

  *byte_out = (unsigned char)value;
  return 0;
}

static int
parse_uuid(const char* uuid_str, CUuuid* uuid_out)
{
  size_t len;
  int i;

  if (uuid_str == NULL || uuid_out == NULL) {
    return -1;
  }

  len = strlen(uuid_str);
  if (len == 40) {
    if (strncmp(uuid_str, "GPU-", 4) != 0) {
      return -1;
    }
    uuid_str += 4;
    len -= 4;
  }

  if (len != 36) {
    return -1;
  }

  for (i = 0; i < 16; ++i) {
    if (*uuid_str == '-') {
      ++uuid_str;
    }
    if (!isxdigit((unsigned char)uuid_str[0]) || !isxdigit((unsigned char)uuid_str[1])) {
      return -1;
    }
    if (parse_hex_byte(uuid_str, (unsigned char*)&uuid_out->bytes[i]) != 0) {
      return -1;
    }
    uuid_str += 2;
  }

  return *uuid_str == '\0' ? 0 : -1;
}

static int
parse_device_map(const char* device_map, CUcheckpointGpuPair** pairs_out, unsigned int* count_out)
{
  char* copy = NULL;
  char* pair = NULL;
  char* pair_save = NULL;
  unsigned int count = 0;
  CUcheckpointGpuPair* pairs = NULL;

  *pairs_out = NULL;
  *count_out = 0;

  if (device_map == NULL || device_map[0] == '\0') {
    return 0;
  }

  copy = strdup(device_map);
  if (copy == NULL) {
    return -1;
  }

  for (pair = copy; *pair != '\0'; ++pair) {
    if (*pair == ',') {
      ++count;
    }
  }
  ++count;

  pairs = calloc(count, sizeof(*pairs));
  if (pairs == NULL) {
    free(copy);
    return -1;
  }

  count = 0;
  pair = strtok_r(copy, ",", &pair_save);
  while (pair != NULL) {
    char* uuid_save = NULL;
    char* old_uuid = strtok_r(pair, "=", &uuid_save);
    char* new_uuid = strtok_r(NULL, "=", &uuid_save);

    if (old_uuid == NULL || new_uuid == NULL || strtok_r(NULL, "=", &uuid_save) != NULL) {
      free(copy);
      free(pairs);
      return -1;
    }
    if (parse_uuid(old_uuid, &pairs[count].oldUuid) != 0 || parse_uuid(new_uuid, &pairs[count].newUuid) != 0) {
      free(copy);
      free(pairs);
      return -1;
    }

    ++count;
    pair = strtok_r(NULL, ",", &pair_save);
  }

  free(copy);
  *pairs_out = pairs;
  *count_out = count;
  return 0;
}

static const char*
process_state_string(CUprocessState state)
{
  switch (state) {
    case CU_PROCESS_STATE_RUNNING:
      return "running";
    case CU_PROCESS_STATE_LOCKED:
      return "locked";
    case CU_PROCESS_STATE_CHECKPOINTED:
      return "checkpointed";
    case CU_PROCESS_STATE_FAILED:
      return "failed";
    default:
      return "unknown";
  }
}

static CUresult
do_lock(int pid, unsigned int timeout_ms)
{
  CUcheckpointLockArgs args;

  memset(&args, 0, sizeof(args));
  args.timeoutMs = timeout_ms;
  return cuCheckpointProcessLock(pid, &args);
}

static CUresult
do_checkpoint(int pid)
{
  CUcheckpointCheckpointArgs args;

  memset(&args, 0, sizeof(args));
  return cuCheckpointProcessCheckpoint(pid, &args);
}

static CUresult
do_restore(int pid, const char* device_map)
{
  CUcheckpointRestoreArgs args;
  CUcheckpointGpuPair* pairs = NULL;
  unsigned int pair_count = 0;
  CUresult status;

  memset(&args, 0, sizeof(args));
  if (parse_device_map(device_map, &pairs, &pair_count) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  args.gpuPairs = pairs;
  args.gpuPairsCount = pair_count;
  status = cuCheckpointProcessRestore(pid, &args);
  free(pairs);
  return status;
}

static CUresult
do_unlock(int pid)
{
  CUcheckpointUnlockArgs args;

  memset(&args, 0, sizeof(args));
  return cuCheckpointProcessUnlock(pid, &args);
}

static CUresult
do_get_state(int pid, CUprocessState* state_out)
{
  return cuCheckpointProcessGetState(pid, state_out);
}

static CUresult
do_get_restore_tid(int pid, int* tid_out)
{
  return cuCheckpointProcessGetRestoreThreadId(pid, tid_out);
}

int
main(int argc, char** argv)
{
  const char* action = NULL;
  const char* device_map = "";
  int pid = 0;
  int have_pid = 0;
  int do_get_state_flag = 0;
  int do_get_restore_tid_flag = 0;
  unsigned int timeout_ms = 0;
  int i;
  CUresult status;

  if (argc == 1) {
    return print_usage(stderr);
  }

  for (i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--get-state") == 0) {
      do_get_state_flag = 1;
      continue;
    }
    if (strcmp(argv[i], "--get-restore-tid") == 0) {
      do_get_restore_tid_flag = 1;
      continue;
    }
    if (strcmp(argv[i], "--action") == 0) {
      if (++i >= argc) {
        return print_usage(stderr);
      }
      action = argv[i];
      continue;
    }
    if (strcmp(argv[i], "--pid") == 0 || strcmp(argv[i], "-p") == 0) {
      if (++i >= argc || parse_pid(argv[i], &pid) != 0) {
        return print_usage(stderr);
      }
      have_pid = 1;
      continue;
    }
    if (strcmp(argv[i], "--timeout") == 0 || strcmp(argv[i], "-t") == 0) {
      if (++i >= argc || parse_timeout_ms(argv[i], &timeout_ms) != 0) {
        return print_usage(stderr);
      }
      continue;
    }
    if (strcmp(argv[i], "--device-map") == 0 || strcmp(argv[i], "-d") == 0) {
      if (++i >= argc) {
        return print_usage(stderr);
      }
      device_map = argv[i];
      continue;
    }
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      return print_usage(stdout);
    }
    return print_usage(stderr);
  }

  if ((do_get_state_flag + do_get_restore_tid_flag + (action != NULL ? 1 : 0)) != 1) {
    return print_usage(stderr);
  }
  if (!have_pid) {
    return print_usage(stderr);
  }

  if (do_get_state_flag) {
    CUprocessState state;

    if (timeout_ms != 0 || device_map[0] != '\0') {
      return print_usage(stderr);
    }
    status = do_get_state(pid, &state);
    if (status != CUDA_SUCCESS) {
      print_cuda_error(status);
      return 1;
    }
    return fprintf(stdout, "%s\n", process_state_string(state)) < 0 ? 1 : 0;
  }

  if (do_get_restore_tid_flag) {
    int tid = 0;

    if (timeout_ms != 0 || device_map[0] != '\0') {
      return print_usage(stderr);
    }
    status = do_get_restore_tid(pid, &tid);
    if (status != CUDA_SUCCESS) {
      print_cuda_error(status);
      return 1;
    }
    return fprintf(stdout, "%d\n", tid) < 0 ? 1 : 0;
  }

  if (strcmp(action, "lock") == 0) {
    status = do_lock(pid, timeout_ms);
  } else if (strcmp(action, "checkpoint") == 0) {
    if (timeout_ms != 0 || device_map[0] != '\0') {
      return print_usage(stderr);
    }
    status = do_checkpoint(pid);
  } else if (strcmp(action, "restore") == 0) {
    if (timeout_ms != 0) {
      return print_usage(stderr);
    }
    status = do_restore(pid, device_map);
  } else if (strcmp(action, "unlock") == 0) {
    if (timeout_ms != 0 || device_map[0] != '\0') {
      return print_usage(stderr);
    }
    status = do_unlock(pid);
  } else {
    return print_usage(stderr);
  }

  if (status != CUDA_SUCCESS) {
    print_cuda_error(status);
    return 1;
  }
  return 0;
}
