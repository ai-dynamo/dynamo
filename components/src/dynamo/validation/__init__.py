# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference check and check runner for worker validation gates.

* ``python -m dynamo.validation.completion``: one chat completion through the
  worker's loopback OpenAI endpoint must return content.
* ``python -m dynamo.validation.runner``: a check runner for ``url`` gates, run
  as a sidecar with its own image.
"""
