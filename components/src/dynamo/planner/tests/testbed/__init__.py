# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power Planner stress testbed — pure in-memory simulation.

NEVER imports pynvml. NEVER instantiates kubernetes.client.CoreV1Api in a way
that can issue real requests. Asserted at import time + pytest session.

See docs/design-docs/powerplanner-testbed-design.md for the full design.
"""

import sys

_BANNED = ("pynvml",)


class _ImportGuard:
    def find_spec(self, name, path, target=None):
        if name in _BANNED:
            raise ImportError(
                f"Testbed forbids import of {name!r}: this would risk "
                f"touching real hardware. See docs/design-docs/"
                f"powerplanner-testbed-design.md §1.3."
            )
        return None


sys.meta_path.insert(0, _ImportGuard())
