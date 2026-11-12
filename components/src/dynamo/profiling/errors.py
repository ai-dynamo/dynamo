# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral control errors with stable status codes."""


class CoordinatorError(ValueError):
    status = 500


class BadRequest(CoordinatorError):
    status = 400


class NotFound(CoordinatorError):
    status = 404


class Conflict(CoordinatorError):
    status = 409


class Unsupported(CoordinatorError):
    status = 422


class Unavailable(CoordinatorError):
    status = 503
