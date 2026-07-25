# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class GMSError(RuntimeError):
    """Recoverable GMS protocol or lifecycle error."""


class FatalGMSError(GMSError):
    """Irrecoverable GMS ownership or cleanup error."""


class AllocationNotFoundError(GMSError):
    """Raised when an allocation ID is not present in the active epoch."""
