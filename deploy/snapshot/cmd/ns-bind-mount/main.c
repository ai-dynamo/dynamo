/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ns-bind-mount: bind-mount or unmount a directory in another process's mount namespace.
 *
 * Mount:      ns-bind-mount <pid> <src> <dst> [ro]
 * Mount-fd:   ns-bind-mount mount-fd <ns_fd> <src> <dst> [ro]
 * Unmount:    ns-bind-mount umount <pid> <dst> [created]
 * Unmount-fd: ns-bind-mount umount-fd <ns_fd> <dst> [created]
 *
 * mount-fd is the preferred form: the caller (Go) opens /proc/<pid>/ns/mnt
 * before launching the helper and passes the fd through ExtraFiles, so the
 * namespace is pinned at open time rather than re-resolved from the PID inside
 * the helper.  Both mount paths apply mount_setattr(MOUNT_ATTR_RDONLY) to the
 * cloned tree *before* attaching so the mount is never visible as writable
 * inside the target namespace.  Unmount enters the namespace the same way and
 * calls umount2(MNT_DETACH).  Both subcommands run as single-threaded C
 * processes so setns(CLONE_NEWNS) is allowed (prohibited in multithreaded Go
 * programs).
 *
 * Requires Linux 5.12+ (mount_setattr; open_tree/move_mount need only 5.2).
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef __NR_open_tree
#define __NR_open_tree 428
#endif
#ifndef __NR_move_mount
#define __NR_move_mount 429
#endif
#ifndef __NR_mount_setattr
#define __NR_mount_setattr 442
#endif

#define OPEN_TREE_CLONE 1
#define MOVE_MOUNT_F_EMPTY_PATH 0x00000004

#ifndef MOUNT_ATTR_RDONLY
#define MOUNT_ATTR_RDONLY 0x00000001
#define MOUNT_ATTR_NOSUID 0x00000002
#define MOUNT_ATTR_NODEV 0x00000004
struct mount_attr {
  uint64_t attr_set;
  uint64_t attr_clr;
  uint64_t propagation;
  uint64_t userns_fd;
};
#endif

#define SNAPSHOT_BINARIES_SRC "/snapshot-binaries"
#define SNAPSHOT_BINARIES_DST "/tmp/snapshot-binaries"
#define PAGEBROKER_RESTORE_SRC "/pagebroker/staging/restore"
#define PAGEBROKER_DST "/tmp/pagebroker"

static int
has_path_prefix(const char* path, const char* prefix)
{
  size_t length = strlen(prefix);
  return strncmp(path, prefix, length) == 0 && (path[length] == '\0' || path[length] == '/');
}

static int
is_descendant_path(const char* path, const char* directory)
{
  size_t length = strlen(directory);
  if (!has_path_prefix(path, directory))
    return 0;
  return path[length] == '/' && path[length + 1] != '\0' && path[length + 1] != '/';
}

static int has_dot_component(const char* path);

static int
check_mount_paths(const char* src, const char* dst)
{
  if (has_dot_component(src) || has_dot_component(dst)) {
    fprintf(stderr, "mount paths must not contain '.' or '..' components\n");
    return -1;
  }
  if (src[0] != '/' || dst[0] != '/') {
    fprintf(stderr, "mount paths must be absolute\n");
    return -1;
  }
  if (has_path_prefix(src, SNAPSHOT_BINARIES_SRC) && has_path_prefix(dst, SNAPSHOT_BINARIES_DST))
    return 0;
  if (is_descendant_path(src, PAGEBROKER_RESTORE_SRC) && strcmp(dst, PAGEBROKER_DST) == 0)
    return 0;
  fprintf(stderr, "invalid mount paths: %s -> %s\n", src, dst);
  return -1;
}

static int
check_umount_path(const char* dst)
{
  if (has_dot_component(dst)) {
    fprintf(stderr, "dst must not contain '.' or '..' components: %s\n", dst);
    return -1;
  }
  if (dst[0] != '/') {
    fprintf(stderr, "dst must be an absolute path: %s\n", dst);
    return -1;
  }
  if (has_path_prefix(dst, SNAPSHOT_BINARIES_DST) || strcmp(dst, PAGEBROKER_DST) == 0)
    return 0;
  fprintf(stderr, "invalid unmount path: %s\n", dst);
  return -1;
}

/* Returns 1 if path contains a dot or dot-dot path component, 0 otherwise. */
static int
has_dot_component(const char* path)
{
  for (const char* p = path; *p;) {
    const char* seg = p;
    while (*p && *p != '/') p++;
    size_t len = (size_t)(p - seg);
    if (len == 1 && seg[0] == '.')
      return 1;
    if (len == 2 && seg[0] == '.' && seg[1] == '.')
      return 1;
    while (*p == '/') p++;
  }
  return 0;
}

static int
sys_open_tree(int dfd, const char* path, unsigned flags)
{
  return (int)syscall(__NR_open_tree, dfd, path, flags);
}

static int
sys_move_mount(int from_dfd, const char* from_path, int to_dfd, const char* to_path, unsigned flags)
{
  return (int)syscall(__NR_move_mount, from_dfd, from_path, to_dfd, to_path, flags);
}

/* Open the mount namespace of the given pid. */
static int
open_mnt_ns(int pid)
{
  char ns_path[256];
  snprintf(ns_path, sizeof(ns_path), "/proc/%d/ns/mnt", pid);
  int ns_fd = open(ns_path, O_RDONLY | O_CLOEXEC);
  if (ns_fd < 0) {
    fprintf(stderr, "open %s: %s\n", ns_path, strerror(errno));
    return -1;
  }
  return ns_fd;
}

/* Enter the mount namespace of the given pid.  Returns 0 on success. */
static int
enter_mnt_ns(int pid)
{
  int ns_fd = open_mnt_ns(pid);
  if (ns_fd < 0)
    return -1;
  if (setns(ns_fd, CLONE_NEWNS) < 0) {
    fprintf(stderr, "setns fd %d: %s\n", ns_fd, strerror(errno));
    close(ns_fd);
    return -1;
  }
  close(ns_fd);
  return 0;
}

/* Parse a positive pid from str.  Returns the pid on success, -1 on error. */
static int
parse_pid(const char* str)
{
  char* end;
  long val = strtol(str, &end, 10);
  if (*end != '\0' || val <= 0 || val > INT_MAX) {
    fprintf(stderr, "invalid pid: %s\n", str);
    return -1;
  }
  return (int)val;
}

/* Apply read-only attributes to tree_fd before attaching it so the mount is
 * never visible as writable inside the target namespace. */
static int
apply_rdonly_attrs(int tree_fd)
{
  struct mount_attr attr = {
      .attr_set = MOUNT_ATTR_RDONLY | MOUNT_ATTR_NOSUID | MOUNT_ATTR_NODEV,
  };
  if (syscall(__NR_mount_setattr, tree_fd, "", AT_EMPTY_PATH, &attr, sizeof attr) < 0) {
    fprintf(stderr, "mount_setattr ro: %s\n", strerror(errno));
    return -1;
  }
  return 0;
}

/* Create or verify the target directory.  Returns 1 if this call created it,
 * 0 if it already existed as a plain directory, -1 on error. */
static int
ensure_dst_dir(const char* dst)
{
  if (mkdir(dst, 0700) == 0)
    return 1;
  if (errno != EEXIST) {
    fprintf(stderr, "mkdir %s: %s\n", dst, strerror(errno));
    return -1;
  }
  /* dst already existed — verify it is a plain directory, not a symlink,
   * so a process inside the namespace cannot redirect the mount. */
  struct stat st;
  if (lstat(dst, &st) < 0) {
    fprintf(stderr, "lstat %s: %s\n", dst, strerror(errno));
    return -1;
  }
  if (!S_ISDIR(st.st_mode)) {
    fprintf(stderr, "dst %s exists but is not a plain directory\n", dst);
    return -1;
  }
  return 0;
}

static int
do_umount(int argc, char* argv[])
{
  if (argc < 4) {
    fprintf(stderr, "usage: ns-bind-mount umount <pid> <dst> [created]\n");
    return 1;
  }
  int pid = parse_pid(argv[2]);
  if (pid < 0)
    return 1;
  const char* dst = argv[3];
  int created_dst = (argc >= 5 && strcmp(argv[4], "created") == 0);

  if (check_umount_path(dst) < 0)
    return 1;

  if (enter_mnt_ns(pid) < 0)
    return 1;

  /* MNT_DETACH: lazy unmount — succeeds even if the path is busy. */
  if (umount2(dst, MNT_DETACH) < 0) {
    if (errno != ENOENT && errno != EINVAL) {
      fprintf(stderr, "umount2 %s: %s\n", dst, strerror(errno));
      return 1;
    }
    /* Already gone (CRIU removed it during namespace restore). */
  }

  if (created_dst)
    rmdir(dst);
  return 0;
}

/* Unmount via an open namespace fd rather than a pid.  The caller (Go) passes
 * an already-open /proc/<pid>/ns/mnt fd inherited through ExtraFiles; using
 * the fd avoids the PID-reuse window between mount time and cleanup.
 * The optional "created" argument instructs the helper to remove dst — only
 * set when the mount subcommand reported that it created the directory. */
static int
do_umount_fd(int argc, char* argv[])
{
  if (argc < 4) {
    fprintf(stderr, "usage: ns-bind-mount umount-fd <ns_fd> <dst> [created]\n");
    return 1;
  }
  char* end;
  long fd_val = strtol(argv[2], &end, 10);
  if (*end != '\0' || fd_val < 0 || fd_val > INT_MAX) {
    fprintf(stderr, "invalid fd: %s\n", argv[2]);
    return 1;
  }
  int ns_fd = (int)fd_val;
  const char* dst = argv[3];
  int created_dst = (argc >= 5 && strcmp(argv[4], "created") == 0);

  if (check_umount_path(dst) < 0)
    return 1;

  if (setns(ns_fd, CLONE_NEWNS) < 0) {
    fprintf(stderr, "setns fd %d: %s\n", ns_fd, strerror(errno));
    return 1;
  }

  if (umount2(dst, MNT_DETACH) < 0) {
    if (errno != ENOENT && errno != EINVAL) {
      fprintf(stderr, "umount2 %s: %s\n", dst, strerror(errno));
      return 1;
    }
    /* Already gone (CRIU removed it during namespace restore).
     * Fall through so we clean up the directory if we created it. */
  }

  /* Only remove the directory if the mount subcommand created it. */
  if (created_dst)
    rmdir(dst);
  return 0;
}

static int
mount_directory(int ns_fd, const char* src, const char* dst, int readonly)
{
  if (check_mount_paths(src, dst) < 0)
    return 1;

  int tree_fd = sys_open_tree(AT_FDCWD, src, OPEN_TREE_CLONE | O_CLOEXEC);
  if (tree_fd < 0) {
    fprintf(stderr, "open_tree %s: %s\n", src, strerror(errno));
    return 1;
  }
  if (readonly && apply_rdonly_attrs(tree_fd) < 0) {
    close(tree_fd);
    return 1;
  }
  if (setns(ns_fd, CLONE_NEWNS) < 0) {
    fprintf(stderr, "setns fd %d: %s\n", ns_fd, strerror(errno));
    close(tree_fd);
    return 1;
  }

  int created_dst = ensure_dst_dir(dst);
  if (created_dst < 0) {
    close(tree_fd);
    return 1;
  }
  if (sys_move_mount(tree_fd, "", AT_FDCWD, dst, MOVE_MOUNT_F_EMPTY_PATH) < 0) {
    fprintf(stderr, "move_mount -> %s: %s\n", dst, strerror(errno));
    close(tree_fd);
    if (created_dst)
      rmdir(dst);
    return 1;
  }
  close(tree_fd);
  printf("created_dst=%d\n", created_dst);
  return 0;
}

/* Mount via an already-open namespace fd.  The caller (Go) opens
 * /proc/<pid>/ns/mnt before launching the helper and passes the fd through
 * ExtraFiles, so the namespace is pinned at Go-side open time rather than
 * re-resolved from the PID — eliminating the PID-reuse window. */
static int
do_mount_fd(int argc, char* argv[])
{
  if (argc < 5) {
    fprintf(stderr, "usage: ns-bind-mount mount-fd <ns_fd> <src> <dst> [ro]\n");
    return 1;
  }
  char* end;
  long fd_val = strtol(argv[2], &end, 10);
  if (*end != '\0' || fd_val < 0 || fd_val > INT_MAX) {
    fprintf(stderr, "invalid fd: %s\n", argv[2]);
    return 1;
  }
  int ns_fd = (int)fd_val;
  const char* src = argv[3];
  const char* dst = argv[4];
  int readonly = (argc >= 6 && strcmp(argv[5], "ro") == 0);

  return mount_directory(ns_fd, src, dst, readonly);
}

int
main(int argc, char* argv[])
{
  if (argc >= 2 && strcmp(argv[1], "mount-fd") == 0)
    return do_mount_fd(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "umount-fd") == 0)
    return do_umount_fd(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "umount") == 0)
    return do_umount(argc, argv);

  if (argc < 4) {
    fprintf(
        stderr,
        "usage: ns-bind-mount <pid> <src> <dst> [ro]\n"
        "       ns-bind-mount mount-fd <ns_fd> <src> <dst> [ro]\n"
        "       ns-bind-mount umount <pid> <dst> [created]\n"
        "       ns-bind-mount umount-fd <ns_fd> <dst> [created]\n");
    return 1;
  }

  int pid = parse_pid(argv[1]);
  if (pid < 0)
    return 1;
  const char* src = argv[2];
  const char* dst = argv[3];
  int readonly = (argc >= 5 && strcmp(argv[4], "ro") == 0);

  int ns_fd = open_mnt_ns(pid);
  if (ns_fd < 0)
    return 1;
  int result = mount_directory(ns_fd, src, dst, readonly);
  close(ns_fd);
  return result;
}
