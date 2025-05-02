# Dynamo SDK

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Writing a Service](#writing-a-service)
- [Configuring a Service](#configuring-a-service)
- [Deploying a Single Service](#deploying-a-single-service)
- [Composing Services into an Graph](#composing-services-into-an-graph)

## Introduction

Dynamo is a flexible and performant distributed inferencing solution for large-scale deployments. It is an ecosystem of tools, frameworks, and abstractions that makes the design, customization, and deployment of frontier-level models onto datacenter-scale infrastructure easy to reason about and optimized for your specific inferencing workloads. Dynamo's core is written in Rust and contains a set of well-defined Python bindings. Docs and examples for those can be found [here](./python_bindings.md).

Dynamo SDK is a layer on top of the core. It is a Python framework that makes it easy to create inference graphs and deploy them locally and onto a target K8s cluster. The SDK was heavily inspired by [BentoML's](https://github.com/bentoml/BentoML) open source deployment patterns and leverages many of its core primitives. The Dynamo CLI is a companion tool that allows you to spin up an inference pipeline locally, containerize it, and deploy it. You can find a toy hello-world example and instructions for deploying it [here](../examples/hello_world.md).

## Installation

The SDK can be installed using pip:

```bash
pip install ai-dynamo
```

## Core Concepts
As you read about each concept, it is helpful to have the [basic example](../examples/hello_world.md) up as well so you can refer back to it.

### Defining a Service

A Service is a core building block for a project. You can think of it as a logical unit of work. For example, you might have a service responsible for preprocessing and tokenizing and another service running the model worker itself. You define a service using the `@service` decorator on a class.

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
1. `dynamo`: Dictionary that defines the Dynamo configuration and enables/disables it. When enabled, a dynamo worker is created under the hood which can register with the [Distributed Runtime](../architecture/architecture.md)
2. `resources`: Dictionary defining resource requirements. The GPUs field is used for local and remote deployment. The other fields are used to determine resources when deploying to K8s.
3. `workers`: Number of parallel instances of the service to spin up.

### Writing a Service

Let's walk through an example to understand how you write a dynamo service.

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

    @dynamo_endpoint()
    async def generate(self, request: ChatCompletionRequest):
        # Call dependent service
        processed_request = await self.service_b.preprocess(request)

        # Use the engine to generate a response
        response = await self.engine.generate(processed_request)
        return response
```

#### Class-Based Architecture
Dynamo follows a class-based architecture similar to BentoML making it intuitive for users familiar with those frameworks. Each service is defined as a Python class, with the following components:
1. Class attributes for dependencies using `depends()`
2. An `__init__` method for standard initialization
3. Optional lifecycle hooks like `@async_on_start` and `@async_on_shutdown`
4. Endpoints defined with `@dynamo_endpoint()`. Optionally, an endpoint can be given a name
   via `@dynamo_endpoint("my_endpoint_name")`, but otherwise will default to the name of the
   function being decorated if omitted.

This approach provides a clean separation of concerns and makes the service structure easy to understand.

#### Service Dependencies with `depends()`
The `depends()` function is a powerful BentoML feature that lets you create a dependency between services. When you use `depends(ServiceB)`, several things happen:
1. It ensures that `ServiceB` is deployed when `ServiceA` is deployed by adding it to an internal service dependency graph
2. It creates a client to the endpoints of `ServiceB` that is being served under the hood.
3. You are able to access `ServiceB` endpoints as if it were a local function!

```python
# What happens internally when you use depends(ServiceB)
service_b = await runtime.namespace("dynamo").component("ServiceB").endpoint("preprocess").client()

# But with Dynamo SDK, you simply write:
service_b = depends(ServiceB)

# And then call methods directly:
result = await service_b.preprocess(data)
```

**NOTE** - through the SDK, we also provide you with a way to access the underlying bindings if you need. Sometimes you might want to write complicated logic that causes you to directly create a client to another Service without depending on it. You can do this via:

```python
import VllmWorker

# this runtime object gives you access to the underlying python bindings
runtime = dynamo_context["runtime"]
comp_ns, comp_name = VllmWorker.dynamo_address() # dynamo://{namespace}/{name}
print(f"[Processor] comp_ns: {comp_ns}, comp_name: {comp_name}")
self.worker_client = (
    await runtime.namespace(comp_ns)
    .component(comp_name)
    .endpoint("generate")
    .client()
)
```

This is used in some of our prebuilt examples and is a powerful way to leverage the benefits of the SDK while being able to access Dynamo's core primitives.

You can find more docs on depends [here](https://docs.bentoml.com/en/latest/build-with-bentoml/distributed-services.html#interservice-communication)

#### Lifecycle Hooks
Dynamo supports key lifecycle hooks to manage service initialization and cleanup. We currently only support a subset of BentoML's lifecycle hooks but are working on adding support for the rest.

##### `@async_on_start`

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

### Configuring a Service

Dynamo SDK provides a flexible configuration system that allows you to define service parameters through multiple methods:

1. Directly in the `@service` decorator
2. Through YAML configuration files
3. Via command-line arguments
4. Using environment variables

These methods can be used together with clear precedence rules, giving you fine-grained control over service configuration across different environments.

#### Configuration via Service Decorator

The most basic method is to specify parameters directly in the service decorator:

```python
@service(
    dynamo={"enabled": True, "namespace": "prod"},
    resources={"gpu": 2, "cpu": "4", "memory": "16Gi"},
    workers=2,
)
class MyService:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
```

This defines static configuration values in code. Note that the constructor parameters (`model_name` and `temperature`) are also configurable values that can be overridden.

#### Configuration via YAML

For more flexible configuration, especially across environments, you can use YAML files:

```yaml
# config.yaml
MyService:
  # Override service decorator settings
  ServiceArgs:
    workers: 4
    resources:
      gpu: 4

  # Service instance parameters
  model_name: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  temperature: 0.8
```

The YAML file has a hierarchical structure:
- Top level keys are service class names
- `ServiceArgs` contains parameters for the service decorator
- Other keys are passed as arguments to the service constructor
- Additional keys specific to the service can be accessed via the config system

#### Loading YAML Configuration

Use the CLI to load configuration from a YAML file:

```bash
dynamo serve service:MyService -f config.yaml
```

The configuration is parsed and stored in the `DYNAMO_SERVICE_CONFIG` environment variable, which is then passed to the service workers.

#### Configuration Precedence

When multiple configuration sources are used, they follow this precedence order (highest to lowest):

1. Command-line arguments
2. YAML configuration
3. Service decorator defaults
4. Constructor defaults

#### Accessing Configuration in Services

Inside a service, you can access configuration using the `ServiceConfig` class:

```python
from dynamo.sdk.lib.config import ServiceConfig

class MyService:
    def __init__(self):
        config = ServiceConfig.get_instance()

        # Get with default value
        self.model_name = config.get("MyService", {}).get("model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.temperature = config.get("MyService", {}).get("temperature", 0.7)

        # Require a config value (raises error if missing)
        self.api_key = config.require("MyService", "api_key")

        # Get all config for this service
        all_my_config = config.get("MyService", {})
```

#### Parsing Configuration as CLI Arguments

For services that need to extract their configuration as command-line arguments (common when integrating and validating with external libraries), the SDK provides a helper method:

```python
from dynamo.sdk.lib.config import ServiceConfig

def setup_my_lib():
    config = ServiceConfig.get_instance()

    # Get all MyService config with prefix "lib_" as CLI args
    cli_args = config.as_args("MyService", prefix="lib_")
    # Returns: ["--option1", "value1", "--flag2", "--option3", "value3"]

    # Pass to an external library's argument parser
    lib_parser = MyLibArgumentParser()
    lib_args = lib_parser.parse_args(cli_args)
    return lib_args
```

This pattern is used in the example vLLM integration:

```python
def parse_vllm_args(service_name, prefix) -> AsyncEngineArgs:
    config = ServiceConfig.get_instance()
    vllm_args = config.as_args(service_name, prefix=prefix)
    parser = FlexibleArgumentParser()

    # Add custom arguments
    parser.add_argument("--router", type=str, choices=["random", "round-robin", "kv"], default="random")
    parser.add_argument("--remote-prefill", action="store_true")

    # Add VLLM's arguments
    parser = AsyncEngineArgs.add_cli_args(parser)

    # Parse both custom and VLLM arguments
    args = parser.parse_args(vllm_args)

    # Convert to engine arguments
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Add custom args to the engine args
    engine_args.router = args.router
    engine_args.remote_prefill = args.remote_prefill

    return engine_args
```

#### Overriding Service Decorator with ServiceArgs

The `ServiceArgs` section in YAML configuration allows you to override any parameter in the `@service` decorator:

```yaml
MyService:
  ServiceArgs:
    dynamo:
      namespace: "staging"  # Override namespace
    resources:
      gpu: 4  # Use more GPUs
    workers: 8  # Scale up workers
```

This is particularly useful for:
- Changing resource allocations between environments
- Modifying worker counts based on expected load
- Switching between namespaces for different deployments

Under the hood, the `DynamoService` class reads these arguments during initialization:

```python
def _get_service_args(self, service_name: str) -> Optional[dict]:
    """Get ServiceArgs from environment config if specified"""
    config_str = os.environ.get("DYNAMO_SERVICE_CONFIG")
    if config_str:
        config = json.loads(config_str)
        service_config = config.get(service_name, {})
        return service_config.get("ServiceArgs")
    return None
```
#### Complete Configuration Example

Here's a comprehensive example showing how all these pieces fit together:

1. First, define your service with basic defaults:

```python
@service(
    dynamo={"enabled": True, "namespace": "default"},
    resources={"gpu": 1},
    workers=1,
)
class LLMService:
    def __init__(self, model_name="gpt-2", temperature=0.7, max_tokens=1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get additional configuration
        config = ServiceConfig.get_instance()
        service_config = config.get("LLMService", {})

        # Extract service-specific parameters
        self.cache_size = service_config.get("cache_size", 1000)
        self.use_kv_cache = service_config.get("use_kv_cache", True)
```

2. Create a YAML configuration for production:

```yaml
# prod_config.yaml
LLMService:
  ServiceArgs:
    dynamo:
      namespace: "prod"
    resources:
      gpu: 4
      memory: "64Gi"
    workers: 8

  # Constructor parameters
  model_name: "llama-3-70b-instruct"
  temperature: 0.8
  max_tokens: 2048

  # Service-specific parameters
  cache_size: 10000
  use_kv_cache: true
```

3. Deploy with mixed configuration:

```bash
dynamo serve service:LLMService -f prod_config.yaml --LLMService.temperature=0.9
```

The service will receive the combined configuration with the command-line value taking precedence, resulting in effective configuration of:
- `dynamo.namespace = "prod"`
- `resources.gpu = 4`
- `workers = 8`
- `model_name = "llama-3-70b-instruct"`
- `temperature = 0.9` (from CLI override)
- `max_tokens = 2048`
- `cache_size = 10000`
- `use_kv_cache = true`

#### Service Configuration Best Practices

1. **Use the Service Decorator for Defaults**: Put reasonable defaults in the service decorator
2. **Use Constructor Parameters for Runtime Options**: Parameters that might change between deployments
3. **Use YAML for Environment Configuration**: Separate configuration by environment (dev/staging/prod)
4. **Use CLI for Quick Testing**: Override specific values for experimentation
5. **Document Configuration Keys**: Make sure to document all available configuration options

Following these practices will help you create flexible and maintainable Dynamo services that can be easily configured for different environments and use cases.

### Deploying a Single Service
You can deploy a single service for local development even if you have a dependancy graph defined using `depends()` using `dynamo serve --service-name <ClassName> <entrypoint> <configuration arguments>`. You can see an example of this in our multinode documentation [here](../examples/multinode.md).

### Composing Services into an Graph
There are two main ways to compose services in Dynamo:
1. Use `depends()` (Recommended)
The depends() approach is the recommended way for production deployments:
- Automatically deploys all dependencies
- Creates a static inference graph at deployment time
- Provides type hints and better IDE support

2. Use `.link()` (Experimental)
Our `.link()` syntax is an flexible and experimental way to compose various services. Linking allows you to compose checks at runtime and view behavior. Under the hood - we are editing the dependency graph between various services. This is useful for experimentation and development but we suggest writing a static graph for your final production deployment.

#### Understanding the `.link()` syntax
Lets take the example of a `Processor` component. This component can currently do 2 things:
1. Process a request and send it to a `Router` to decide what worker to send it to.
2. Process a request and send it to a `Worker` directly.

A snippet of the Processor is shown below:

```python
class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    worker = depends(VllmWorker)
    router = depends(Router)

    # logic for processing a request based on router or worker
```

You can think of all the depends statements as the maximal set of edges for the processor. At runtime, you may want to follow only a single path. By default, our processor will spin up both the VllmWorker and Router as separate services (because `depends()` is defined for both). However, if you want to only spin up the Router, you can do this by linking the Router to the Processor which will remove the `worker` dependency from the Processor.

```python
Processor.link(Router)
```

This will remove the `worker` dependency from the Processor and only spin up the Router.