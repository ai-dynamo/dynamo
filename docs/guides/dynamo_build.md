<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.s
-->

# Building Dynamo (`dynamo build`)

This guide explains how to use the `dynamo build` command to containerize Dynamo inference graphs (pipelines) for deployment.

`dynamo build` is a command-line tool that helps containerize inference graphs created with Dynamo SDK. Run `dynamo build --containerize` to build a stand-alone Docker container that encapsulates your entire inference graph. This image can then be shared and run standalone.

```{note}
This experimental feature is tested on the examples in the `examples/` directory. You need to make some modifications. Pay particular attention if your inference graph introduces custom dependencies.
```

## Building a containerized inference graph

The basic workflow for using `dynamo build` includes:

#. Defining your inference graph and testing locally with `dynamo serve`
#. Specifying a base image for your inference graph. More on this below.
#. Running `dynamo build` to build a containerized inference graph

### Basic Usage

```bash
dynamo build <graph_definition> --containerize
```