# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Preserve legacy Agent names from Dynamo's runtime partition mapping.
export PARTITION_ID="${LOGICAL_PARTITION_INDEX}"
export RANK_IN_PARTITION="${PARTITION_RANK}"
