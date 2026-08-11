<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom Worker Selection Example

- Keep each policy in its own crate and register it through `catalog`.
- Keep scorer and picker algorithms small enough to explain in `README.md`.
- Declare every signal read by a scorer or picker with `required_worker_inputs`.
- Parse and validate YAML parameters in the provider before creating the factory.
- Keep blocking I/O, panics, and shared mutable state out of `score` and `pick`.
- Add new crates to the root Cargo workspace and add focused registration tests.
- Update `README.md` and the custom worker-selection docs page when the example workflow changes.
