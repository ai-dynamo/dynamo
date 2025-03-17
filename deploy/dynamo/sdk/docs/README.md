# Documentation for the Dynamo SDK and CLI 

# Introduction 

Nvidia Dynamo is a flexible and performant distributed inferencing solution for large-scale deployments. It is an ecosystem of tools, frameworks, and abstractions that makes the design, customization, and deployment of frontier-level models onto datacenter-scale infrastructure easy to reason about and optimized for your specific inferencing workloads. Dynamo's core is written in Rust and contains a set of well-defined Python bindings. Docs and examples for those can be found [here](../../../../README.md). 

Dynamo SDK is a layer on top of the core. It is a Python framework that makes it easy to create inference graphs and deploy them locally and onto a target K8s cluster. The SDK was heavily inspired by [BentoML's](https://github.com/bentoml/BentoML) open source deployment patterns and leverages many of its core primitives. The Dynamo CLI is a companion tool that allows you to spin up an inference pipeline locally, containerize it, and deploy it. You can find a toy hello-world example [here](../README.md).

# Installation 

The SDK can be installed using pip:

```bash
pip install ai-dynamo
```

# Core Concepts 
As you read about each concept, it is helpful to have the basic example up as well so you can refer back to it. 

## Defining a Service

A Service is a core building block for a project. You can think of it as a logical unit of work. For example, you might have a service responsible for preprocessing and tokenizing and another service running the model worker itself.

```python
@service(
    dynamo={
        "enabled": True, 
        "namespace": "dynamo", 
    },
    resources={"gpu": 2, "cpu": "10", "memory": "20Gi"}, 
    workers=1, 
)
```

Key configuration options:
1. `dynamo`: Dictionary that defines the Dynamo configuration and enables/disables it. When enabled, a dynamo worker is created under the hood which can register with the Distributed Runtime [TODO:link]
2. `resources`: Dictionary defining resource requirements. Used primarily when deploying to K8s, but gpu is also used for local execution.
3. `workers`: Number of parallel instances of the service to spin up.

## Writing a Service

Lets walk through a dummy service to understand how you write a dynamo service.

```python
import ServiceB

@service(dynamo={"enabled": True, "namespace": "dynamo"}, resources={"gpu": 1})
class ServiceA:
    # Define service dependencies
    service_b = depends(ServiceB)

    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.engine = None

    @async_on_start
    async def async_init(self):
        # Initialize resources that require async operations
        self.engine = await initialize_model_engine(self.model_name)
        print(f"ServiceA initialized with model: {self.model_name}")

    @async_on_shutdown
    async def async_shutdown(self):
        # Clean up resources
        if self.engine:
            await self.engine.shutdown()
            print("ServiceA engine shut down")

    @dynamo_endpoint()
    async def generate(self, request: ChatCompletionRequest):
        # Call dependent service
        processed_request = await self.service_b.preprocess(request)
        
        # Use the engine to generate a response
        response = await self.engine.generate(processed_request)
        return response
```

### Class-Based Architecture 
Dynamo follows a class-based architecture similar to BentoML making it intuitive for users familiar with those frameworks. Each service is defined as a Python class, with the following components:
1. Class attributes for dependencies using depends()
2. An __init__ method for standard initialization
3. Optional lifecycle hooks like @async_on_start and @async_on_shutdown
4. Endpoints defined with @dynamo_endpoint()

This approach provides a clean separation of concerns and makes the service structure easy to understand.

### Service Dependencies with `depends()`
The `depends()` function is a powerful BentoML feature that lets you create a dependency between services. When you use `depends(ServiceB)`, several things happen:
1. It ensures that `ServiceB` is deployed when `ServiceA` is deployed by adding it to an internal service dependency graph
2. It creates a client to the endpoints of `ServiceB` that is being served under the hood. 
3. You are able to access `ServiceB` endpoints as if it were a local function!

This abstraction dramatically simplifies building distributed applications, as you can write code that looks local but actually communicates across service boundaries via the Dynamo's core distributed service primatives. You can findn more docs on depends [here](https://docs.bentoml.com/en/latest/build-with-bentoml/distributed-services.html#interservice-communication)

### Lifecycle Hooks
Dynamo supports key lifecycle hooks to manage service initialization and cleanup. We currently only support a subset of BentoML's lifecycle hooks but are working on adding support for the rest.

#### `@async_on_start`

The `@async_on_start` hook is called when the service is started and is used to run an async process outside of the main `__init__` function.

```python
@async_on_start
async def async_init(self):
    # Perfect for operations that need to be awaited
    self.db = await connect_to_db()
    self.tokenizer = await load_tokenizer()
    self.engine = await initialize_engine(self.model)
```
This is especially useful for:
- Initializing external connections
- Setting up runtime resources that require async operations

#### `@async_on_shutdown`
The `@async_on_shutdown` hook is called when the service is shutdown handles cleanup.

```python
@async_on_shutdown
async def async_shutdown(self):
    if self._engine_context is not None:
        await self._engine_context.__aexit__(None, None, None)
    print("VllmWorkerRouterLess shutting down")
```

This ensures resources are properly released, preventing memory leaks and making sure external connections are properly closed. This is helpful to clean up VLLM engines that have been started outside of the main process.

## Configuring a Service

sccratch
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