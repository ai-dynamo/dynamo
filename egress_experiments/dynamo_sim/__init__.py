# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dynamo half of the diagram: Rust ingress, the one asyncio loop, the
worker handler, and the two egress paths (pull and push).
"""
