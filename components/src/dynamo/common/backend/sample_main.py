# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim -> `dynamo.sample_engine.main`. Removed in the follow-up migration commit."""
from dynamo.sample_engine.main import main

if __name__ == "__main__":
    main()
