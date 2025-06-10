<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# About the Dynamo Command Line Interface

The Dynamo CLI serves, containerizes, and deploys Dynamo applications efficiently. It provides intuitive commands to manage your Dynamo services.

## CLI Capabilities

With the Dynamo CLI, you can:

* Chat with models quickly using `run`
* Serve multiple services locally using `serve`
* Package your services into archive (called `dynamo artifact`) using `build`
* Deploy pipelines to Dynamo Cloud using `deploy`

## Commands

### `run`

Use `run` to start an interactive chat session with a model. This command executes the `dynamo-run` Rust binary under the hood. For more details, see [Running Dynamo](dynamo_run.md).

#### Example
```bash
dynamo run deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

### `serve`

Use `serve` to run your defined inference graph locally. You'll need to specify your file and intended class using the file:Class syntax. For more details, see [Serving Inference Graphs](dynamo_serve.md).

#### Usage
```bash
dynamo serve [SERVICE]
```

#### Arguments
* `SERVICE`: Specify the service to start using file:Class syntax

#### Flags
* `--file`/`-f`: Path to optional YAML configuration file. For configuration examples, see the [SDK docs](../API/sdk.md)
* `--dry-run`: Print the dependency graph and values without starting services
* `--service-name`: Start only the specified service name
* `--working-dir`: Set the directory for finding the Service instance
* Additional flags following Class.key=value pattern are passed to the service constructor. For details, see the configuration section of the [SDK docs](../API/sdk.md)

#### Example
```bash
cd examples
# Start the Frontend, Middle, and Backend components
dynamo serve hello_world:Frontend

# Start only the Middle component in the graph that is discoverable from the Frontend service
dynamo serve --service-name Middle hello_world:Frontend
```

### `build`

Use `build` to package your inference graph and its dependencies into an archive. Combine this with the `--containerize` flag to create a single Docker container for your inference graph. As with `serve`, you point toward the first service in your dependency graph. For more details, see [Serving Inference Graphs](dynamo_serve.md).

#### Usage
```bash
dynamo build [SERVICE]
```

#### Arguments
* `SERVICE`: Specify the service to build using file:Class syntax

#### Flags
* `--working-dir`: Specify the directory for finding the Service instance
* `--containerize`: Choose whether to create a container from the dynamo artifact after building

#### Example
```bash
cd examples/hello_world
dynamo build hello_world:Frontend
```

### `deploy`

Use `deploy` to create a pipeline on Dynamo Cloud using either interactive prompts or a YAML configuration file. For more details, see [Deploying Inference Graphs to Kubernetes](dynamo_deploy/README.md).

#### Usage
```bash
dynamo deploy [PIPELINE]
```

#### Arguments
* `PIPELINE`: The pipeline to deploy; defaults to *None*; required

#### Flags
* `--name`/`-n`: Set the deployment name. Defaults to *None*; required
* `--config-file`/`-f`: Specify the configuration file path. Defaults to *None*; required
* `--wait`/`--no-wait`: Choose whether to wait for deployment readiness. Defaults to wait
* `--timeout`: Set maximum deployment time in seconds. Defaults to 3600
* `--endpoint`/`-e`: Specify the Dynamo Cloud deployment endpoint. Defaults to *None*; required
* `--help`/`-h`: Display command help

For a detailed deployment example, see [Operator Deployment](dynamo_deploy/operator_deployment.md).
