/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include "util.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

int
snapshot_vmm_write_all(int fd, const void* value, size_t size)
{
  const uint8_t* current = value;

  while (size != 0) {
    ssize_t count = write(fd, current, size);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return -1;
    current += count;
    size -= (size_t)count;
  }
  return 0;
}

int
snapshot_vmm_read_all(int fd, void* value, size_t size)
{
  uint8_t* current = value;

  while (size != 0) {
    ssize_t count = read(fd, current, size);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return -1;
    current += count;
    size -= (size_t)count;
  }
  return 0;
}

int
snapshot_vmm_pread_all(int fd, void* value, size_t size)
{
  uint8_t* current = value;
  off_t offset = 0;

  while (size != 0) {
    ssize_t count = pread(fd, current, size, offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return -1;
    current += count;
    offset += count;
    size -= (size_t)count;
  }
  return 0;
}

bool
snapshot_vmm_is_lower_hex_id(const char value[SNAPSHOT_VMM_ID_SIZE])
{
  size_t index;

  if (value == NULL)
    return false;
  for (index = 0; index < SNAPSHOT_VMM_ID_SIZE - 1; index++) {
    if (!((value[index] >= '0' && value[index] <= '9') || (value[index] >= 'a' && value[index] <= 'f')))
      return false;
  }
  return value[SNAPSHOT_VMM_ID_SIZE - 1] == '\0';
}

int
snapshot_vmm_set_socket_timeouts(int fd, int seconds)
{
  struct timeval timeout = {.tv_sec = seconds};

  return setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0 &&
                 setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0
             ? 0
             : -1;
}

int
snapshot_vmm_send_header(int fd, const struct snapshot_vmm_header* header, int passed_fd)
{
  char control[CMSG_SPACE(sizeof(int))] = {0};
  struct iovec vector = {.iov_base = (void*)header, .iov_len = sizeof(*header)};
  struct msghdr message = {.msg_iov = &vector, .msg_iovlen = 1};
  ssize_t count;

  if (passed_fd >= 0) {
    struct cmsghdr* item;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    item = CMSG_FIRSTHDR(&message);
    item->cmsg_level = SOL_SOCKET;
    item->cmsg_type = SCM_RIGHTS;
    item->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(item), &passed_fd, sizeof(passed_fd));
  }
  do {
    count = sendmsg(fd, &message, MSG_NOSIGNAL);
  } while (count < 0 && errno == EINTR);
  return count == (ssize_t)sizeof(*header) ? 0 : -1;
}

int
snapshot_vmm_receive_header(int fd, struct snapshot_vmm_header* header, int* passed_fd)
{
  char control[CMSG_SPACE(sizeof(int))] = {0};
  struct iovec vector = {.iov_base = header, .iov_len = sizeof(*header)};
  struct msghdr message = {
      .msg_iov = &vector,
      .msg_iovlen = 1,
      .msg_control = control,
      .msg_controllen = sizeof(control),
  };
  struct cmsghdr* item;
  ssize_t count;

  *passed_fd = -1;
  do {
    count = recvmsg(fd, &message, MSG_WAITALL | MSG_CMSG_CLOEXEC);
  } while (count < 0 && errno == EINTR);
  if (count != (ssize_t)sizeof(*header) || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0)
    return -1;
  item = CMSG_FIRSTHDR(&message);
  if (item != NULL && item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_RIGHTS &&
      item->cmsg_len == CMSG_LEN(sizeof(int)))
    memcpy(passed_fd, CMSG_DATA(item), sizeof(*passed_fd));
  return item == NULL || CMSG_NXTHDR(&message, item) == NULL ? 0 : -1;
}
