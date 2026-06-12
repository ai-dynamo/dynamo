#define _GNU_SOURCE
#include <ctype.h>
#include <cuda.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
get_cuda_error(CUresult status, const char** name_out, const char** msg_out)
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

  *name_out = name;
  *msg_out = msg;
}

static void
print_cuda_error(const char* api_name, CUresult status)
{
  const char* name = NULL;
  const char* msg = NULL;

  get_cuda_error(status, &name, &msg);
  fprintf(stderr, "%s failed: %s: %s\n", api_name, name, msg);
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

static void
print_env_value(const char* name)
{
  const char* value = getenv(name);

  fprintf(stderr, "env.%s=%s\n", name, value != NULL ? value : "<unset>");
}

static void
print_cuda_versions(void)
{
  int driver_version = 0;
  CUresult status;

  fprintf(stderr, "cuda_header_version=%d\n", CUDA_VERSION);
  status = cuDriverGetVersion(&driver_version);
  if (status == CUDA_SUCCESS) {
    fprintf(stderr, "cuda_driver_version=%d\n", driver_version);
  } else {
    print_cuda_error("cuDriverGetVersion", status);
  }
}

static void
print_libcuda_path(void)
{
  Dl_info info;

  if (dladdr((void*)cuInit, &info) != 0 && info.dli_fname != NULL) {
    fprintf(stderr, "libcuda_path=%s\n", info.dli_fname);
    return;
  }
  fprintf(stderr, "libcuda_path=<unknown>\n");
}

static void
print_symbol_status(const char* symbol)
{
  dlerror();
  fprintf(stderr, "symbol.%s=%s\n", symbol, dlsym(RTLD_DEFAULT, symbol) != NULL ? "present" : "missing");
}

static void
print_cuda_symbol_status(void)
{
  print_symbol_status("cuInit");
  print_symbol_status("cuDriverGetVersion");
  print_symbol_status("cuCheckpointProcessLock");
  print_symbol_status("cuCheckpointProcessCheckpoint");
  print_symbol_status("cuCheckpointProcessRestore");
  print_symbol_status("cuCheckpointProcessUnlock");
  print_symbol_status("cuCheckpointProcessGetState");
  print_symbol_status("cuCheckpointProcessGetRestoreThreadId");
}

static void
print_file_excerpt(const char* path)
{
  FILE* file = fopen(path, "r");
  char buffer[4096];
  size_t n;

  if (file == NULL) {
    fprintf(stderr, "%s=<unreadable: %s>\n", path, strerror(errno));
    return;
  }

  n = fread(buffer, 1, sizeof(buffer) - 1, file);
  buffer[n] = '\0';
  fclose(file);
  fprintf(stderr, "%s=%s%s\n", path, buffer, n == sizeof(buffer) - 1 ? "<truncated>" : "");
}

static void
print_proc_status(int pid)
{
  char path[64];
  FILE* file;
  char line[512];
  static const char* const prefixes[] = {
    "Name:", "State:", "Tgid:", "Pid:", "PPid:", "TracerPid:", "NSpid:", "Threads:", NULL,
  };

  snprintf(path, sizeof(path), "/proc/%d/status", pid);
  file = fopen(path, "r");
  if (file == NULL) {
    fprintf(stderr, "%s=<unreadable: %s>\n", path, strerror(errno));
    return;
  }

  while (fgets(line, sizeof(line), file) != NULL) {
    int i;

    for (i = 0; prefixes[i] != NULL; ++i) {
      if (strncmp(line, prefixes[i], strlen(prefixes[i])) == 0) {
        line[strcspn(line, "\n")] = '\0';
        fprintf(stderr, "target.%s\n", line);
        break;
      }
    }
  }
  fclose(file);
}

static void
print_proc_cmdline(int pid)
{
  char path[64];
  FILE* file;
  char buffer[1024];
  size_t n;
  size_t i;

  snprintf(path, sizeof(path), "/proc/%d/cmdline", pid);
  file = fopen(path, "r");
  if (file == NULL) {
    fprintf(stderr, "%s=<unreadable: %s>\n", path, strerror(errno));
    return;
  }

  n = fread(buffer, 1, sizeof(buffer) - 1, file);
  fclose(file);
  buffer[n] = '\0';
  for (i = 0; i < n; ++i) {
    if (buffer[i] == '\0') {
      buffer[i] = ' ';
    }
  }
  fprintf(stderr, "target.cmdline=%s%s\n", n > 0 ? buffer : "<empty>", n == sizeof(buffer) - 1 ? "<truncated>" : "");
}

static void
print_proc_namespace(int pid, const char* namespace_name)
{
  char path[64];
  char target[256];
  ssize_t n;

  snprintf(path, sizeof(path), "/proc/%d/ns/%s", pid, namespace_name);
  n = readlink(path, target, sizeof(target) - 1);
  if (n < 0) {
    fprintf(stderr, "%s=<unreadable: %s>\n", path, strerror(errno));
    return;
  }
  target[n] = '\0';
  fprintf(stderr, "target.ns.%s=%s\n", namespace_name, target);
}

static const char*
file_type_string(mode_t mode)
{
  if (S_ISCHR(mode)) {
    return "char";
  }
  if (S_ISDIR(mode)) {
    return "dir";
  }
  if (S_ISREG(mode)) {
    return "regular";
  }
  if (S_ISLNK(mode)) {
    return "symlink";
  }
  return "other";
}

static void
print_path_status(const char* path)
{
  struct stat st;

  if (stat(path, &st) != 0) {
    fprintf(stderr, "path.%s=missing:%s\n", path, strerror(errno));
    return;
  }
  fprintf(stderr, "path.%s=present,type=%s,mode=%o\n", path, file_type_string(st.st_mode), st.st_mode & 07777);
}

static void
print_directory_sample(const char* path)
{
  DIR* dir = opendir(path);
  struct dirent* entry;
  int printed = 0;

  if (dir == NULL) {
    fprintf(stderr, "dir.%s=<unreadable: %s>\n", path, strerror(errno));
    return;
  }
  fprintf(stderr, "dir.%s=", path);
  while ((entry = readdir(dir)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    if (printed >= 32) {
      fprintf(stderr, "<truncated>");
      break;
    }
    fprintf(stderr, "%s%s", printed > 0 ? " " : "", entry->d_name);
    ++printed;
  }
  if (printed == 0) {
    fprintf(stderr, "<empty>");
  }
  fprintf(stderr, "\n");
  closedir(dir);
}

static void
print_device_visibility(void)
{
  static const char* const paths[] = {
    "/dev/nvidiactl",
    "/dev/nvidia0",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-caps",
    "/dev/nvidia-caps-imex-channels",
    "/dev/nvidia-nvswitchctl",
    "/dev/nvidia-nvswitch0",
    "/dev/nvidia-nvlink",
    "/dev/nvidia-fs",
    NULL,
  };
  int i;

  for (i = 0; paths[i] != NULL; ++i) {
    print_path_status(paths[i]);
  }
  print_directory_sample("/dev/nvidia-caps");
  print_directory_sample("/dev/nvidia-caps-imex-channels");
}

static void
print_checkpoint_state(int pid, const char* failed_api_name)
{
  CUprocessState state;
  CUresult status;
  int tid = 0;

  if (strcmp(failed_api_name, "cuCheckpointProcessGetState") != 0) {
    status = cuCheckpointProcessGetState(pid, &state);
    if (status == CUDA_SUCCESS) {
      fprintf(stderr, "target.cuda_state=%s\n", process_state_string(state));
    } else {
      print_cuda_error("cuCheckpointProcessGetState", status);
    }
  }

  if (strcmp(failed_api_name, "cuCheckpointProcessGetRestoreThreadId") != 0) {
    status = cuCheckpointProcessGetRestoreThreadId(pid, &tid);
    if (status == CUDA_SUCCESS) {
      fprintf(stderr, "target.cuda_restore_tid=%d\n", tid);
    } else {
      print_cuda_error("cuCheckpointProcessGetRestoreThreadId", status);
    }
  }
}

static void
print_failure_diagnostics(const char* api_name, CUresult status, int pid, const char* action, const char* device_map)
{
  const char* name = NULL;
  const char* msg = NULL;

  get_cuda_error(status, &name, &msg);
  fprintf(stderr, "%s failed: %s: %s\n", api_name, name, msg);
  fprintf(stderr, "cuda-checkpoint-helper diagnostics begin\n");
  fprintf(stderr, "helper.action=%s\n", action != NULL ? action : "<none>");
  fprintf(stderr, "helper.pid=%d\n", pid);
  fprintf(stderr, "helper.device_map=%s\n", device_map != NULL && device_map[0] != '\0' ? device_map : "<empty>");
  print_cuda_versions();
  print_libcuda_path();
  print_cuda_symbol_status();
  print_env_value("LD_LIBRARY_PATH");
  print_env_value("CUDA_VISIBLE_DEVICES");
  print_env_value("NVIDIA_VISIBLE_DEVICES");
  print_env_value("NVIDIA_DRIVER_CAPABILITIES");
  print_file_excerpt("/proc/driver/nvidia/version");
  print_proc_status(pid);
  print_proc_cmdline(pid);
  print_proc_namespace(pid, "pid");
  print_proc_namespace(pid, "mnt");
  print_proc_namespace(pid, "ipc");
  print_proc_namespace(pid, "net");
  print_checkpoint_state(pid, api_name);
  print_device_visibility();
  fprintf(stderr, "cuda-checkpoint-helper diagnostics end\n");
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
  const char* api_name = NULL;
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
    api_name = "cuCheckpointProcessGetState";
    status = do_get_state(pid, &state);
    if (status != CUDA_SUCCESS) {
      print_cuda_error(api_name, status);
      return 1;
    }
    return fprintf(stdout, "%s\n", process_state_string(state)) < 0 ? 1 : 0;
  }

  if (do_get_restore_tid_flag) {
    int tid = 0;

    if (timeout_ms != 0 || device_map[0] != '\0') {
      return print_usage(stderr);
    }
    api_name = "cuCheckpointProcessGetRestoreThreadId";
    status = do_get_restore_tid(pid, &tid);
    if (status != CUDA_SUCCESS) {
      print_cuda_error(api_name, status);
      return 1;
    }
    return fprintf(stdout, "%d\n", tid) < 0 ? 1 : 0;
  }

  if (strcmp(action, "lock") == 0) {
    api_name = "cuCheckpointProcessLock";
    status = do_lock(pid, timeout_ms);
  } else if (strcmp(action, "checkpoint") == 0) {
    if (timeout_ms != 0 || device_map[0] != '\0') {
      return print_usage(stderr);
    }
    api_name = "cuCheckpointProcessCheckpoint";
    status = do_checkpoint(pid);
  } else if (strcmp(action, "restore") == 0) {
    if (timeout_ms != 0) {
      return print_usage(stderr);
    }
    /* cuCheckpointProcessRestore requires persistence mode or prior cuInit. */
    api_name = "cuInit";
    status = cuInit(0);
    if (status == CUDA_SUCCESS) {
      api_name = "cuCheckpointProcessRestore";
      status = do_restore(pid, device_map);
    }
  } else if (strcmp(action, "unlock") == 0) {
    if (timeout_ms != 0 || device_map[0] != '\0') {
      return print_usage(stderr);
    }
    api_name = "cuCheckpointProcessUnlock";
    status = do_unlock(pid);
  } else {
    return print_usage(stderr);
  }

  if (status != CUDA_SUCCESS) {
    print_failure_diagnostics(api_name, status, pid, action, device_map);
    return 1;
  }
  return 0;
}
