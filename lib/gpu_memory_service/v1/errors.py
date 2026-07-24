# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class GMSError(RuntimeError):
    pass


class FatalGMSError(GMSError):
    """The process no longer has a safe V1 ownership transition."""
