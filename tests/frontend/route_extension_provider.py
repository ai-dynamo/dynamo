# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A minimal frontend route provider for the route-extension e2e test.

The e2e test registers this provider under the ``dynamo.frontend.routes``
entry-point group (via a throwaway ``.dist-info`` on ``PYTHONPATH``, so no real
package install is required) and launches::

    python -m dynamo.frontend --frontend-route-extension <name>

then calls ``/hello_world`` and validates the response. ``FrontendRoute`` is
imported lazily inside ``routes()`` so this module stays importable (for the
shared constants below) without the ai-dynamo wheel present.
"""

from __future__ import annotations

from typing import Any

HELLO_PATH = "/hello_world"
HELLO_BODY = {"message": "hello world!"}


def _hello(ctx: Any) -> dict:
    """Static handler — independent of any model or worker."""
    return dict(HELLO_BODY)


def routes():
    """Provider callable: return the custom routes to register on the frontend."""
    from dynamo.llm import FrontendRoute

    return [FrontendRoute("GET", HELLO_PATH, _hello)]
