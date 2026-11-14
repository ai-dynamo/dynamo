// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Thin C shim over the vendored NVTX v3 headers (third_party/nvtx3), compiled by
// build.rs when the `nvtx` feature is on. The v3 API is header-only: every call below
// compiles to a stub that lazily binds to the profiler's injection library
// (NVTX_INJECTION64_PATH) and is inert when none is attached.
//
// The declarations in `ffi` (src/nvtx.rs) must match these definitions exactly.

#include <stdint.h>

#include "nvtx3/nvToolsExt.h"

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#define DYNAMO_NVTX_THREAD_ID ((uint32_t)syscall(SYS_gettid))
#elif defined(__APPLE__)
#include <sys/syscall.h>
#include <unistd.h>
#define DYNAMO_NVTX_THREAD_ID ((uint32_t)syscall(SYS_thread_selfid))
#else
#error "dynamo nvtx shim: unsupported platform"
#endif

static void
dynamo_nvtx_fill_payload(nvtxEventAttributes_t* attr, const char* message, uint64_t payload)
{
  attr->version = NVTX_VERSION;
  attr->size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr->messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr->message.ascii = message;
  attr->payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  attr->payload.ullValue = payload;
}

uint64_t
dynamo_nvtx_range_start(const char* message)
{
  return (uint64_t)nvtxRangeStartA(message);
}

uint64_t
dynamo_nvtx_range_start_payload(const char* message, uint64_t payload)
{
  nvtxEventAttributes_t attr = {0};
  dynamo_nvtx_fill_payload(&attr, message, payload);
  return (uint64_t)nvtxRangeStartEx(&attr);
}

void
dynamo_nvtx_range_end(uint64_t id)
{
  nvtxRangeEnd((nvtxRangeId_t)id);
}

void
dynamo_nvtx_mark(const char* message)
{
  nvtxMarkA(message);
}

void
dynamo_nvtx_mark_payload(const char* message, uint64_t payload)
{
  nvtxEventAttributes_t attr = {0};
  dynamo_nvtx_fill_payload(&attr, message, payload);
  nvtxMarkEx(&attr);
}

void
dynamo_nvtx_range_push(const char* message)
{
  nvtxRangePushA(message);
}

void
dynamo_nvtx_range_pop(void)
{
  nvtxRangePop();
}

void
dynamo_nvtx_name_thread(const char* name)
{
  nvtxNameOsThreadA(DYNAMO_NVTX_THREAD_ID, name);
}
