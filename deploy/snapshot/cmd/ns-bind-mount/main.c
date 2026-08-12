/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Single-threaded mount-namespace helper. Go cannot call setns(CLONE_NEWNS)
 * from its multithreaded runtime, so this exists purely to perform a mount on
 * its behalf. It carries no policy: which paths and which attributes are the
 * caller's to choose.
 *
 *   mount-fd  <ns_fd> <src> <dst> [ro] [nosuid] [nodev] [noexec]
 *   umount-fd <ns_fd> <dst> [created] [strict]
 *
 * The destination is still validated, because the target namespace belongs to
 * an untrusted container that could otherwise swap it between the check and the
 * mount. That is a property of mounting safely, not a restriction on the caller.
 */
#define _GNU_SOURCE
#include <dirent.h>
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
#ifndef __NR_openat2
#define __NR_openat2 437
#endif
#ifndef AT_EMPTY_PATH
#define AT_EMPTY_PATH 0x1000
#endif
#ifndef UMOUNT_NOFOLLOW
#define UMOUNT_NOFOLLOW 0x00000008
#endif
#define OPEN_TREE_CLONE 1
#define MOVE_MOUNT_F_EMPTY_PATH 0x00000004
#define MOVE_MOUNT_T_EMPTY_PATH 0x00000040
/* openat2 resolve flags, named locally because <linux/openat2.h> is not present
 * on every builder image. */
#define NS_RESOLVE_NO_XDEV 0x01
#define NS_RESOLVE_NO_MAGICLINKS 0x02
#define NS_RESOLVE_NO_SYMLINKS 0x04
#define NS_RESOLVE_BENEATH 0x08
struct ns_open_how {
  uint64_t flags, mode, resolve;
};
#ifndef MOUNT_ATTR_RDONLY
#define MOUNT_ATTR_RDONLY 0x00000001
#define MOUNT_ATTR_NOSUID 0x00000002
#define MOUNT_ATTR_NODEV 0x00000004
#define MOUNT_ATTR_NOEXEC 0x00000008
struct mount_attr {
  uint64_t attr_set, attr_clr, propagation, userns_fd;
};
#endif

/* The four attributes this helper understands. Anything outside this set is a
 * caller error rather than a silently dropped restriction. */
#define MANAGED_MOUNT_ATTRS (MOUNT_ATTR_RDONLY | MOUNT_ATTR_NOSUID | MOUNT_ATTR_NODEV | MOUNT_ATTR_NOEXEC)

static int
parse_fd(const char* value)
{
  char* end;
  errno = 0;
  long fd = strtol(value, &end, 10);
  if (errno || end == value || *end || fd < 0 || fd > INT_MAX) {
    fprintf(stderr, "invalid fd: %s\n", value);
    return -1;
  }
  return (int)fd;
}

static int
sys_open_tree(int dfd, const char* path, unsigned flags)
{
  return (int)syscall(__NR_open_tree, dfd, path, flags);
}

static int
sys_openat2(int dfd, const char* path, uint64_t flags, uint64_t resolve)
{
  struct ns_open_how how = {.flags = flags, .mode = 0, .resolve = resolve};
  return (int)syscall(__NR_openat2, dfd, path, &how, sizeof(how));
}

/* Applies exactly the requested attributes. attr_clr is the complement rather
 * than zero: a cloned tree inherits its source mount's attributes, so an
 * attribute the caller did not ask for has to be cleared explicitly or it
 * survives. Without this, a bundle mounted from a noexec filesystem would stay
 * noexec however it was requested. */
static int
set_mount_attrs(int tree_fd, uint64_t attrs)
{
  struct mount_attr attr = {
      .attr_set = attrs,
      .attr_clr = MANAGED_MOUNT_ATTRS & ~attrs,
  };
  if (syscall(__NR_mount_setattr, tree_fd, "", AT_EMPTY_PATH, &attr, sizeof(attr)) < 0) {
    fprintf(stderr, "mount_setattr: %s\n", strerror(errno));
    return -1;
  }
  return 0;
}

/* Resolves dst exactly once and returns its descriptor, or -1. Sets *created
 * when this helper made the directory.
 *
 * The destination lives in a namespace the workload controls, so it is resolved
 * once and every later step works from the returned descriptor: re-resolving
 * the path per step would let a container swap it in between. The single
 * openat2 also subsumes the old lstat/S_ISDIR/mountinfo checks, and errno says
 * which one failed: EXDEV for an existing mount (including a same-filesystem
 * bind), ELOOP for a symlink anywhere in the path, ENOTDIR for a non-directory. */
static int
ensure_empty_destination(const char* dst, int* created)
{
  *created = 0;

  /* The parent and the final component are resolved separately because
   * RESOLVE_NO_XDEV has to apply to the final component alone: /tmp is itself a
   * mount in most containers, so asking for it across the whole path would
   * reject every destination. Against the parent, RESOLVE_NO_SYMLINKS is what
   * matters, so a target image pointing /tmp elsewhere is refused. */
  const char* sep = strrchr(dst, '/');
  if (!sep || !sep[1]) {
    fprintf(stderr, "invalid destination %s\n", dst);
    return -1;
  }
  char parent[PATH_MAX];
  size_t parent_len = sep == dst ? 1 : (size_t)(sep - dst);
  if (parent_len >= sizeof(parent)) {
    fprintf(stderr, "destination %s is too long\n", dst);
    return -1;
  }
  memcpy(parent, dst, parent_len);
  parent[parent_len] = '\0';
  const char* leaf = sep + 1;

  int parent_fd = sys_openat2(
      AT_FDCWD, parent, O_PATH | O_DIRECTORY | O_CLOEXEC, NS_RESOLVE_NO_SYMLINKS | NS_RESOLVE_NO_MAGICLINKS);
  if (parent_fd < 0) {
    fprintf(
        stderr,
        "open %s in target namespace: %s"
        " (%s must be a real directory, not a symlink)\n",
        parent, strerror(errno), parent);
    return -1;
  }

  if (mkdirat(parent_fd, leaf, 0700) == 0) {
    *created = 1;
  } else if (errno != EEXIST) {
    fprintf(stderr, "mkdir %s: %s\n", dst, strerror(errno));
    close(parent_fd);
    return -1;
  }

  int fd = sys_openat2(
      parent_fd, leaf, O_RDONLY | O_DIRECTORY | O_CLOEXEC,
      NS_RESOLVE_BENEATH | NS_RESOLVE_NO_SYMLINKS | NS_RESOLVE_NO_MAGICLINKS | NS_RESOLVE_NO_XDEV);
  close(parent_fd);
  if (fd < 0) {
    const char* why = errno == EXDEV     ? "is already a mountpoint"
                      : errno == ELOOP   ? "is a symlink"
                      : errno == ENOTDIR ? "is not a directory"
                                         : strerror(errno);
    fprintf(stderr, "destination %s %s\n", dst, why);
    if (*created)
      rmdir(dst);
    return -1;
  }

  if (!*created) {
    int scan_fd = dup(fd);
    DIR* dir = scan_fd < 0 ? NULL : fdopendir(scan_fd);
    if (!dir) {
      fprintf(stderr, "opendir %s: %s\n", dst, strerror(errno));
      if (scan_fd >= 0)
        close(scan_fd);
      close(fd);
      return -1;
    }
    struct dirent* entry;
    while ((entry = readdir(dir))) {
      if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
        fprintf(
            stderr,
            "destination %s is not empty (contains %s);"
            " the restore target image must leave %s absent or empty\n",
            dst, entry->d_name, dst);
        closedir(dir);
        close(fd);
        return -1;
      }
    }
    closedir(dir);
  }
  return fd;
}

static int
mount_tree(int ns_fd, const char* src, const char* dst, uint64_t attrs)
{
  /* Resolve the source without following symlinks, then clone from that
   * descriptor. open_tree takes no resolve flags of its own, so the openat2
   * gives the source the same treatment the destination gets. */
  int src_fd =
      sys_openat2(AT_FDCWD, src, O_PATH | O_DIRECTORY | O_CLOEXEC, NS_RESOLVE_NO_SYMLINKS | NS_RESOLVE_NO_MAGICLINKS);
  if (src_fd < 0) {
    fprintf(stderr, "open source %s: %s\n", src, strerror(errno));
    return 1;
  }
  int tree_fd = sys_open_tree(src_fd, "", OPEN_TREE_CLONE | AT_EMPTY_PATH | O_CLOEXEC);
  close(src_fd);
  if (tree_fd < 0) {
    fprintf(stderr, "open_tree %s: %s\n", src, strerror(errno));
    return 1;
  }
  if (set_mount_attrs(tree_fd, attrs) < 0) {
    close(tree_fd);
    return 1;
  }
  if (setns(ns_fd, CLONE_NEWNS) < 0) {
    fprintf(stderr, "setns: %s\n", strerror(errno));
    close(tree_fd);
    return 1;
  }
  int created = 0;
  int dst_fd = ensure_empty_destination(dst, &created);
  if (dst_fd < 0) {
    close(tree_fd);
    return 1;
  }
  /* Publish destination ownership before move_mount so a caller that loses
   * the helper after the mount becomes visible can still remove a directory
   * this helper created during conservative rollback. */
  printf("created_dst=%d\n", created);
  fflush(stdout);
  int rc = 0;
  if (syscall(__NR_move_mount, tree_fd, "", dst_fd, "", MOVE_MOUNT_F_EMPTY_PATH | MOVE_MOUNT_T_EMPTY_PATH) < 0) {
    fprintf(stderr, "move_mount -> %s: %s\n", dst, strerror(errno));
    if (created)
      rmdir(dst);
    rc = 1;
  }
  close(dst_fd);
  close(tree_fd);
  return rc;
}

static int
unmount_tree(int ns_fd, const char* dst, int strict, int created)
{
  if (setns(ns_fd, CLONE_NEWNS) < 0) {
    fprintf(stderr, "setns: %s\n", strerror(errno));
    return 1;
  }
  /* UMOUNT_NOFOLLOW reports a symlinked destination as EINVAL, which is
   * indistinguishable from "nothing is mounted here" and would be swallowed
   * below. Name it explicitly: a destination that turned into a symlink means
   * the target moved the mount, so the artifact may still be attached somewhere
   * and the unmount must not be reported as clean. */
  struct stat st;
  if (fstatat(AT_FDCWD, dst, &st, AT_SYMLINK_NOFOLLOW) == 0 && S_ISLNK(st.st_mode)) {
    fprintf(stderr, "destination %s is a symlink; refusing to unmount through it\n", dst);
    return 1;
  }
  if (umount2(dst, (strict ? 0 : MNT_DETACH) | UMOUNT_NOFOLLOW) < 0 && errno != ENOENT && errno != EINVAL) {
    fprintf(stderr, "umount2 %s: %s\n", dst, strerror(errno));
    return 1;
  }
  if (created)
    rmdir(dst);
  return 0;
}

/* Maps an attribute token to its MOUNT_ATTR_* bit, or 0 if unknown. */
static uint64_t
parse_attr(const char* token)
{
  if (!strcmp(token, "ro"))
    return MOUNT_ATTR_RDONLY;
  if (!strcmp(token, "nosuid"))
    return MOUNT_ATTR_NOSUID;
  if (!strcmp(token, "nodev"))
    return MOUNT_ATTR_NODEV;
  if (!strcmp(token, "noexec"))
    return MOUNT_ATTR_NOEXEC;
  return 0;
}

static void
usage(void)
{
  fprintf(
      stderr,
      "usage: ns-bind-mount mount-fd  <ns-fd> <src> <dst> [ro] [nosuid] [nodev] [noexec]\n"
      "       ns-bind-mount umount-fd <ns-fd> <dst> [created] [strict]\n");
}

int
main(int argc, char** argv)
{
  if (argc < 3) {
    usage();
    return 1;
  }
  int ns_fd = parse_fd(argv[2]);
  if (ns_fd < 0)
    return 1;

  if (!strcmp(argv[1], "mount-fd")) {
    if (argc < 5) {
      usage();
      return 1;
    }
    uint64_t attrs = 0;
    for (int i = 5; i < argc; i++) {
      uint64_t bit = parse_attr(argv[i]);
      if (!bit) {
        fprintf(stderr, "unknown mount attribute: %s\n", argv[i]);
        return 1;
      }
      attrs |= bit;
    }
    return mount_tree(ns_fd, argv[3], argv[4], attrs);
  }

  if (!strcmp(argv[1], "umount-fd")) {
    if (argc < 4) {
      usage();
      return 1;
    }
    /* Matched by name in any order: two optional positional flags would make
     * "umount-fd 3 /tmp/x created" ambiguous. */
    int created = 0, strict = 0;
    for (int i = 4; i < argc; i++) {
      if (!strcmp(argv[i], "created"))
        created = 1;
      else if (!strcmp(argv[i], "strict"))
        strict = 1;
      else {
        fprintf(stderr, "unknown unmount flag: %s\n", argv[i]);
        return 1;
      }
    }
    return unmount_tree(ns_fd, argv[3], strict, created);
  }

  fprintf(stderr, "invalid operation: %s\n", argv[1]);
  usage();
  return 1;
}
