# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Put the example root on sys.path so `import riva_nim` resolves to the local
# package. The package is deliberately NOT named `riva`, so it does not shadow
# the nvidia-riva-client package (imported as top-level `riva`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
