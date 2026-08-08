/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>

CUresult CUDAAPI
cuGetProcAddress_v2(
    const char* symbol, void** output, int version, cuuint64_t flags,
    CUdriverProcAddressQueryResult* status)
{
  (void)symbol;
  (void)version;
  (void)flags;
  if (output != NULL)
    *output = NULL;
  if (status != NULL)
    *status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  return CUDA_SUCCESS;
}

void*
fake_cuda_prefix_resolver_address(void)
{
  return (void*)&cuGetProcAddress_v2;
}
