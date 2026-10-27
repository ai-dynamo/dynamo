// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* LD_PRELOAD shim: turns Nagle off on every TCP connection a process accepts,
 * as the frontend's own accept loop does since #16728. Without it, an HTTP
 * server that writes the response headers and then a small SSE chunk holds
 * the chunk until the client acknowledges the headers: one extra round trip.
 *
 *   cc -shared -fPIC -Os -s -o libnodelay.so nodelay.c
 */
#define _GNU_SOURCE
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

int
accept4(int fd, struct sockaddr* addr, socklen_t* len, int flags)
{
  int conn = syscall(SYS_accept4, fd, addr, len, flags);
  if (conn >= 0) {
    int one = 1;
    /* Fails harmlessly on non-TCP sockets. */
    setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  }
  return conn;
}

int
accept(int fd, struct sockaddr* addr, socklen_t* len)
{
  return accept4(fd, addr, len, 0);
}
