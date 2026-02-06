---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

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
defined in [deploy/docker-compose.yml](https://github.com/ai-dynamo/dynamo/tree/main/deploy/docker-compose.yml).

```
# At the root of the repository:
docker compose -f deploy/docker-compose.yml up -d
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

See the [README.md](https://github.com/ai-dynamo/dynamo/tree/main/lib/runtime/lib/bindings/python/README.md) for details

The Python and Rust `hello_world` client and server examples are interchangeable,
so you can start the Python `server.py` and talk to it from the Rust `client`.
