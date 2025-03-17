# Documentation for the Dynamo SDK and CLI 

# Introduction 

Nvidia Dynamo is a flexible and performant distributed inferencing solution for large-scale deployments. It is an ecosystems of tools, frameworks, and abstractions that makes the design, customization, and deployment of frontier-level models onto datacenter-scale infrastructure easy to reason about and optimized for your specific inferencing workloads. Dynamo's core is written in Rust and contains a set of well-define Python bindings. Examples for those can be found [here](../../../../README.md). 

Dynamo SDK is a layer on top of the core. It is a Python framework that aims to make it easy to create inference graphs and deploy them locally and onto a target K8s cluster. The SDK was heavily inspired by [BentoML](https://github.com/bentoml/BentoML) and we leverage many of its core primatives throughout the SDK. The Dynamo CLI is a companion tool that allows you to spin up an inference pipeline locally, containerize it, and then also deploy it. You can find a toy hello-world example [here](../README.md)

# Installation 

The SDK can be installed using pip:

```bash
pip install ai-dynamo
```

# Core Concepts 

## Defining a Service

Services in Dynamo are easily defined using the `@service` decorator and `@dynamo_endpoint` decorator. Here, we define a "Hello World" Dynamo service and chain it to an HTTP frontend to interact with it:

```python
from pydantic import BaseModel
from dynamo.sdk import DYNAMO_IMAGE, api, depends, dynamo_endpoint, service

class RequestType(BaseModel):
    text: str

class ResponseType(BaseModel):
    text: str

@service(
    resources={"cpu": "2"},
    dynamo={
        "enabled": True,
        "namespace": "inference",
    },
    workers=3,
    image=DYNAMO_IMAGE,
)
class Backend:
    def __init__(self) -> None:
        print("Init tasks...")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        for token in req.text.split():
            yield f"Echo: {token}"

@service(
    resources={"cpu": "1"},
    traffic={"timeout": 60},
    image=DYNAMO_IMAGE,
)
class Frontend:
    backend = depends(Backend)

    @api
    async def generate(self, text):
        txt = RequestType(text=text)
        async for response in self.backend.generate(txt.model_dump_json()):
            yield response
```

If we save this file as hello_world.py, we can then test our dynamo service locally using the following:

```bash
dynamo serve hello_world:Frontend # name of the module:entrypoint of the graph

# In another terminal, we can test our service using curl
curl -X POST http://localhost:8000/generate -d '{"text": "Hello World"}'

# We should see the following output:
Echo: Hello
Echo: World
```

Let's break down the example:

### Decorators

The `@service` class decorator is used to define a service. It takes in a set of keyword arguments that are used to configure the service. 

- `dynamo`: A dictionary that defines the Dynamo configuration for the service and whether it is enabled or not.
- `resources`: A dictionary that defines the resources for the service when deployed onto a K8s cluster. These do not apply locally.
- `workers`: Parallelism of the service
- `image`: The base image to use when building a container for the service (experimental)

Decorating a method of a service with `@dynamo_endpoint` will expose that method as an endpoint of that Dynamo service and will enable other services to create clients to that endpoint.

> **Note:** Frontend has no dynamo configuration defined and as such is a regular HTTP service. To expose an HTTP endpoint, we can simply decorate a method of that service with `@api`.

### How serve works?

When running with `Dynamo serve`, each service will be spun up in its own process. 

### Dependencies

Dependencies are defined using the `depends` function. This function takes in a service class and returns a client to that service. This client can be used to call the service's endpoints.

```python
backend = depends(Backend)
```

In our simple example, Frontend uses its client to call the `generate` endpoint of the Backend service as if it were a local python function, abstracting away that this is taking place over the Dynamo transport.


servicedecorator, decorator values, dynamo section 
endpoint decorator link this and above to the python bindings docs
class based (similar to bento and ray)
async on start (working on adding support for all other hooks)
passing in arguments + configuring with a YAML (and how to read those args in the service class)

### Composing services into an inference graph
2 ways
- using the depends. this is the most supported way and is the recommended way when you are looking to deploy. 
- depends for us means edges that will spin up BUT ALSO are an astraction over creating a client for a dynemo endpoint (from above)
- deploy is the maximal set of edges. link lets you define that at runtime for experimentation. our examples using this experimental link syntax

### Learning by example. Aggregated KV routing. 

### CLI flows

1. serve is your go to 
2. then you build a `bento`. explain slightly and share the structure 
3. this is what is used to deploy onto a cluster 