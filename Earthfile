# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

VERSION 0.8

############### ARTIFACTS TARGETS ##############################
# These targets are invoked in child Earthfiles to pass top-level files that are out of their build context
# https://docs.earthly.dev/earthly-0.6/best-practices#copying-files-from-outside-the-build-context

############### SHARED LIBRARY TARGETS ##############################
golang-base:
    FROM golang:1.23
    RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/* && curl -sSfL https://github.com/golangci/golangci-lint/releases/download/v1.61.0/golangci-lint-1.61.0-linux-amd64.tar.gz | tar -xzv && mv golangci-lint-1.61.0-linux-amd64/golangci-lint /usr/local/bin/

operator-src:
    FROM +golang-base
    COPY ./deploy/dynamo/operator /artifacts/operator
    SAVE ARTIFACT /artifacts/operator

############### ALL TARGETS ##############################
all-test:
    BUILD ./deploy/dynamo/operator+test
    BUILD ./deploy/dynamo/api-server+test

all-docker:
    ARG CI_REGISTRY_IMAGE=my-registry
    ARG CI_COMMIT_SHA=latest
    BUILD ./deploy/dynamo/operator+docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
    BUILD ./deploy/dynamo/api-server+docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA

all-lint:
    BUILD ./deploy/dynamo/operator+lint

all:
    BUILD +all-test
    BUILD +all-docker
    BUILD +all-lint

# For testing
custom:
    ARG CI_REGISTRY_IMAGE=my-registry
    ARG CI_COMMIT_SHA=latest
    BUILD +all-test
