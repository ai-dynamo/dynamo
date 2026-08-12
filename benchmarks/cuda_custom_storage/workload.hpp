/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "roundtrip_common.hpp"

namespace cuda_custom_storage_benchmark {

int RunWorkload(int ready_fd, int command_fd, const Options& options);

}  // namespace cuda_custom_storage_benchmark
