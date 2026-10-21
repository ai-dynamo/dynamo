# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference check and check runner for worker validation gates.

* ``python -m dynamo.validation.completion``: one chat completion through the
  worker's loopback OpenAI endpoint must return content.
* ``python -m dynamo.validation.runner``: a check runner that serves checks
  like this one to ``DYN_PUBLISH_GATE`` and ``DYN_READY_GATE``, as a sidecar
  or a helper process in the worker container.
"""
