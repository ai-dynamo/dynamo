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
limitations under the License.
-->

# Guide to Dynamo CLI

After installing Dynamo with the following command, Dynamo can be used primarily through its CLI.
```
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0
python3 -m venv venv
source venv/bin/activate

pip install ai-dynamo[all]
```

## Dynamo workflow
Dynamo CLI has the following 4 sub-commands.

- :runner: dynamo run: quickly spin up a server to experiment with a specified model, input and output target.
- :palm_up_hand: dynamo serve: compose a graph of workers locally and serve.
- :hammer: (Experimental) dynamo build: containerize either the entire graph or parts of graph to multiple containers
- :rocket: (Experimental) dynamo deploy: deploy to K8 with helm charts or custom operators
- :cloud: (Experimental) dynamo cloud: interact with your dynamo cloud server

For more detailed examples on serving LLMs with disaggregated serving, KV aware routing, etc,  please refer to [LLM deployment examples](https://github.com/ai-dynamo/dynamo/blob/main/examples/llm/README.md)

