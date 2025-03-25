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

# wrapper over bento images to handle Dynamo base image

import os

import bentoml

# TODO: "dynamo:latest-vllm-dev" image will not be available to image builder in k8s
# so We'd consider publishing the base image for releases to public nvcr.io registry.
image_name = os.getenv("DYNAMO_IMAGE", "dynamo:latest-vllm-dev")
DYNAMO_IMAGE = bentoml.images.PythonImage(base_image=image_name)
