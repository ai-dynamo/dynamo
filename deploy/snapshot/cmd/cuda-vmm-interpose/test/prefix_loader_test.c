/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void
require(int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "%s\n", message);
    exit(1);
  }
}

int
main(void)
{
  void* library = dlopen("libcuda.software", RTLD_NOW | RTLD_LOCAL);
  void* resolver;
  void* (*real_resolver_address)(void);

  require(library != NULL, dlerror());
  resolver = dlsym(library, "cuGetProcAddress_v2");
  real_resolver_address =
      (void* (*)(void))dlsym(library, "fake_cuda_prefix_resolver_address");
  require(
      resolver != NULL && real_resolver_address != NULL,
      "unrelated CUDA-prefix DSO lookups failed");
  require(
      resolver == real_resolver_address(),
      "unrelated CUDA-prefix DSO resolver was replaced by the shim");
  require(dlclose(library) == 0, "cannot close unrelated CUDA-prefix DSO");
  return 0;
}
