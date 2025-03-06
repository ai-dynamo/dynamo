#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# wrapper over bento images to handle Dynemo base image

import os

import bentoml

# TODO: "dynemo:latest-vllm" image will not be available to image builder
# so we need to use the dynemo-base image. We'd consider publishing the image to a public registry.
image_name = os.getenv(
    "NOVA_IMAGE", "nvcr.io/nvidian/nim-llm-dev/yatai-bentos:dynemo-base-0886e19"
)
NOVA_IMAGE = bentoml.images.PythonImage(base_image=image_name)
