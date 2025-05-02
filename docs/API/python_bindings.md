<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dynamo Python Bindings

Python bindings for the Dynamo runtime system, enabling distributed computing capabilities for machine learning workloads.

## 🚀 Quick Start

1. Install `uv`: https://docs.astral.sh/uv/#getting-started
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install `protoc` protobuf compiler: https://grpc.io/docs/protoc-installation/.

For example on an Ubuntu/Debian system:
```
apt install protobuf-compiler
```

3. Setup a virtualenv

```
uv venv
source .venv/bin/activate
uv pip install maturin
```

4. Build and install dynamo wheel
```
maturin develop --uv
```

## Run Examples

### Prerequisite

See [README.md](../runtime/README.md#prerequisites).

### Hello World Example

1. Start 3 separate shells, and activate the virtual environment in each
```
source .venv/bin/activate
```

2. In one shell (shell 1), run example server the instance-1
```
python3 ./examples/hello_world/server.py
```

3. (Optional) In another shell (shell 2), run example the server instance-2
```
python3 ./examples/hello_world/server.py
```

4. In the last shell (shell 3), run the example client:
```
python3 ./examples/hello_world/client.py
```

If you run the example client in rapid succession, and you started more than
one server instance above, you should see the requests from the client being
distributed across the server instances in each server's output. If only one
server instance is started, you should see the requests go to that server
each time.

## Performance

The performance impacts of synchronizing the Python and Rust async runtimes
is a critical consideration when optimizing the performance of a highly
concurrent and parallel distributed system.

The Python GIL is a global critical section and is ultimately the death of
parallelism. To compound that, when Rust async futures become ready,
accessing the GIL on those async event loop needs to be considered carefully.
Under high load, accessing the GIL or performing CPU intensive tasks on
on the event loop threads can starve out other async tasks for CPU resources.
However, performing a `tokio::task::spawn_blocking` is not without overheads
as well.

If bouncing many small message back-and-forth between the Python and Rust
event loops where Rust requires GIL access, this is pattern where moving the
code from Python to Rust will give you significant gains.
