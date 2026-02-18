# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MyEngine: a sample backend implementation for Dynamo.

This backend echoes the request's input tokens repeated 5 times,
streamed token by token. It exists as a minimal reference for writing new backends.
"""
