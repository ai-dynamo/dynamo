# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NIXL 1.3.0 LIBFABRIC backport

Dynamo release 1.3 replaces only the NIXL 1.3.0 LIBFABRIC plugin with a build
carrying the corrected FI_MORE batching and endpoint-selection changes from
[ai-dynamo/nixl PR #1966](https://github.com/ai-dynamo/nixl/pull/1966). The
rest of the NIXL wheel remains unchanged.

The release cannot safely bump the complete NIXL package: its SGLang 0.5.14
runtime pins the `nixl` and `nixl-cu13` packages to 1.3.0, and replacing the
whole wheel would change a release-pinned integration and ABI surface. This
temporary source backport is removed when the SGLang base image carries a NIXL
release containing the finalized upstream fix.

Upstream tracking and reproducibility:

- NIXL base commit: `f852e14226f0fd67378469d638bf45b12d1926e5`
- NIXL base tree: `4e7e85eaad3947371e68c5bdace02340aea772ba`
- Upstream PR head: `d1e60ec95140ff018a1db6057e3f51dfa4c0f562`
- Upstream PR tree: `76620c0937796f87187f335c0616367174f38ff1`
- Patched NIXL tree: `659b5b21f4d0d7ad0d79c54a8bd47be642ca21a2`
- Patch SHA-256: `c24dc80783ba949153e41efeb4c14e63147ca2cf2dfd4174bc727bdec28e7f26`

The image build pins those identities, compiles the plugin, checks its ELF
architecture and loader ABI, and validates the corrected source shape. The
source/model validator is a regression gate, not an end-to-end transport test;
release qualification still requires Dynamo plus SGLang validation on both
H100 and GB200 EFA nodes.
