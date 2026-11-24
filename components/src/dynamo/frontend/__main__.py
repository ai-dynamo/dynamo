#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import sys

# Capability inspection needs no serving configuration, event loop, or backend packages.
if "--list-parser-capabilities" in sys.argv[1:]:
    from dynamo.frontend.parser_capabilities import main
else:
    from dynamo.frontend.main import main

if __name__ == "__main__":
    main()
