#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export EPD_MODE=e_p_d
exec bash "$(dirname "$(readlink -f "$0")")/disagg_multimodal_e_pd.sh" "$@"
