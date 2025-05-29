<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Dynamo Runtime

<h4>A Datacenter Scale Distributed Inference Serving Framework</h4>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Rust implementation of the Dynamo runtime system, enabling distributed computing capabilities for machine learning workloads.

## Prerequisites

### Install Rust and Cargo using [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build

```
cargo build
cargo test
```

### Start Dependencies

#### Docker Compose

The simplest way to deploy the pre-requisite services is using
[docker-compose](https://docs.docker.com/compose/install/linux/),
defined in [deploy/metrics/docker-compose.yml](../../deploy/metrics/docker-compose.yml).

```
docker-compose up -d
```

This will deploy a [NATS.io](https://nats.io/) server and an [etcd](https://etcd.io/)
server used to communicate between and discover components at runtime.


#### Local (alternate)

To deploy the pre-requisite services locally instead of using `docker-compose`
above, you can manually launch each:

- [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) server with [Jetstream](https://docs.nats.io/nats-concepts/jetstream)
    - example: `nats-server -js --trace`
- [etcd](https://etcd.io) server
    - follow instructions in [etcd installation](https://etcd.io/docs/v3.5/install/) to start an `etcd-server` locally


### Run Examples

When developing or running examples, any process or user that shared your core-services (`etcd` and `nats.io`) will
be operating within your distributed runtime.

The current examples use a hard-coded `namespace`. We will address the `namespace` collisions later.

All examples require the `etcd` and `nats.io` pre-requisites to be running and available.

#### Rust `hello_world`

With two terminals open, in one window:

```
cd examples/hello_world
cargo run --bin server
```

In the second terminal, execute:

```
cd examples/hello_world
cargo run --bin client
```

which should yield some output similar to:
```
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 6.25s
     Running `target/debug/client`
Annotated { data: Some("h"), id: None, event: None, comment: None }
Annotated { data: Some("e"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("o"), id: None, event: None, comment: None }
Annotated { data: Some(" "), id: None, event: None, comment: None }
Annotated { data: Some("w"), id: None, event: None, comment: None }
Annotated { data: Some("o"), id: None, event: None, comment: None }
Annotated { data: Some("r"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("d"), id: None, event: None, comment: None }
```

#### Python

See the [README.md](../API/python_bindings.md) for details

The Python and Rust `hello_world` client and server examples are interchangeable,
so you can start the Python `server.py` and talk to it from the Rust `client`.
