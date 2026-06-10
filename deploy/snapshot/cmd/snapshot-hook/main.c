#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

typedef int (*nccl_checkpoint_fn_t)(void);
typedef pid_t (*fork_fn_t)(void);

static pid_t g_started_pid = 0;
static pthread_t g_thread;

static void log_msg(const char* fmt, ...) {
  const char* enabled = getenv("DYN_SNAPSHOT_HOOK_LOG");
  if (enabled == NULL || enabled[0] == '\0' || strcmp(enabled, "0") == 0) {
    return;
  }

  flockfile(stderr);
  fprintf(stderr, "dynamo-snapshot-hook[%d]: ", getpid());
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  funlockfile(stderr);
}

static void replyf(int fd, const char* fmt, ...) {
  char buf[512];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  if (n < 0) {
    return;
  }
  if ((size_t)n >= sizeof(buf)) {
    n = (int)sizeof(buf) - 1;
  }
  (void)write(fd, buf, (size_t)n);
}

static int read_cmd(int fd, char* buf, size_t len) {
  size_t pos = 0;
  while (pos + 1 < len) {
    char c;
    ssize_t n = read(fd, &c, 1);
    if (n == 0) {
      break;
    }
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (c == '\n' || c == '\r') {
      break;
    }
    buf[pos++] = c;
  }
  buf[pos] = '\0';
  return 0;
}

static void call_nccl_symbol(int fd, const char* symbol) {
  dlerror();
  void* sym = dlsym(RTLD_DEFAULT, symbol);
  if (sym == NULL) {
    const char* err = dlerror();
    replyf(fd, "missing %s %s\n", symbol, err != NULL ? err : "unknown");
    return;
  }

  nccl_checkpoint_fn_t fn = (nccl_checkpoint_fn_t)sym;
  int ret = fn();
  replyf(fd, "ok %d\n", ret);
}

static void handle_client(int fd) {
  char cmd[128];
  if (read_cmd(fd, cmd, sizeof(cmd)) != 0) {
    replyf(fd, "error read failed\n");
    return;
  }

  if (strcmp(cmd, "ping") == 0) {
    replyf(fd, "ok 0\n");
  } else if (strcmp(cmd, "nccl_prepare") == 0) {
    call_nccl_symbol(fd, "ncclCheckpointPrepare");
  } else if (strcmp(cmd, "nccl_restore") == 0) {
    call_nccl_symbol(fd, "ncclCheckpointRestore");
  } else {
    replyf(fd, "error unknown command %s\n", cmd);
  }
}

static int make_socket_dir(const char* control_dir, char* out, size_t out_len) {
  int n = snprintf(out, out_len, "%s/snapshot-hook", control_dir);
  if (n < 0 || (size_t)n >= out_len) {
    return -1;
  }
  if (mkdir(out, 0777) != 0 && errno != EEXIST) {
    return -1;
  }
  (void)chmod(out, 0777);
  return 0;
}

static int bind_socket(const char* dir, char* path, size_t path_len) {
  int n = snprintf(path, path_len, "%s/%d.sock", dir, getpid());
  if (n < 0 || (size_t)n >= path_len) {
    errno = ENAMETOOLONG;
    return -1;
  }

  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    return -1;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  if (strlen(path) >= sizeof(addr.sun_path)) {
    close(fd);
    errno = ENAMETOOLONG;
    return -1;
  }
  strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
  (void)unlink(path);

  if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }
  (void)chmod(path, 0666);
  if (listen(fd, 16) != 0) {
    close(fd);
    (void)unlink(path);
    return -1;
  }
  return fd;
}

static void* hook_thread_main(void* arg) {
  (void)arg;
  const char* control_dir = getenv("DYN_SNAPSHOT_CONTROL_DIR");
  if (control_dir == NULL || control_dir[0] == '\0') {
    return NULL;
  }

  char dir[PATH_MAX];
  if (make_socket_dir(control_dir, dir, sizeof(dir)) != 0) {
    log_msg("failed to create socket dir under %s: %s", control_dir, strerror(errno));
    return NULL;
  }

  char path[PATH_MAX];
  int listen_fd = bind_socket(dir, path, sizeof(path));
  if (listen_fd < 0) {
    log_msg("failed to bind hook socket: %s", strerror(errno));
    return NULL;
  }
  log_msg("listening on %s", path);

  for (;;) {
    struct pollfd pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.fd = listen_fd;
    pfd.events = POLLIN;
    int ready = poll(&pfd, 1, 1000);
    if (ready == 0) {
      if (access(path, F_OK) != 0 && errno == ENOENT) {
        close(listen_fd);
        listen_fd = bind_socket(dir, path, sizeof(path));
        if (listen_fd < 0) {
          log_msg("failed to rebind missing hook socket: %s", strerror(errno));
          return NULL;
        }
        log_msg("rebound missing hook socket on %s", path);
      }
      continue;
    }
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      log_msg("poll failed: %s", strerror(errno));
      continue;
    }
    if ((pfd.revents & POLLIN) == 0) {
      continue;
    }
    int fd = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
    if (fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      log_msg("accept failed: %s", strerror(errno));
      continue;
    }
    handle_client(fd);
    close(fd);
  }

  return NULL;
}

static void start_hook_thread(void) {
  if (getenv("DYN_SNAPSHOT_CONTROL_DIR") == NULL) {
    return;
  }

  pid_t pid = getpid();
  if (g_started_pid == pid) {
    return;
  }
  g_started_pid = pid;

  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0) {
    g_started_pid = 0;
    return;
  }
  (void)pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  int ret = pthread_create(&g_thread, &attr, hook_thread_main, NULL);
  (void)pthread_attr_destroy(&attr);
  if (ret != 0) {
    g_started_pid = 0;
    log_msg("pthread_create failed: %s", strerror(ret));
  }
}

__attribute__((constructor)) static void dynamo_snapshot_hook_init(void) {
  start_hook_thread();
}

pid_t fork(void) {
  static fork_fn_t real_fork = NULL;
  if (real_fork == NULL) {
    real_fork = (fork_fn_t)dlsym(RTLD_NEXT, "fork");
  }
  if (real_fork == NULL) {
    errno = ENOSYS;
    return -1;
  }

  pid_t pid = real_fork();
  if (pid == 0) {
    g_started_pid = 0;
    start_hook_thread();
  }
  return pid;
}
