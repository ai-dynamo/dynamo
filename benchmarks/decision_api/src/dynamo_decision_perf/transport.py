# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use server-generated IDs without changing AIPerf measurement correlation."""

from aiperf.transports.aiohttp_transport import AioHttpTransport


class DecisionHttpTransport(AioHttpTransport):
    def build_headers(self, request_info):
        # The optional client ID currently conflicts with Dynamo's server ID pair.
        return {
            key: value
            for key, value in super().build_headers(request_info).items()
            if key.lower() != "x-request-id"
        }
