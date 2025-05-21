# Dynamo Service Dependencies and Linking

Dynamo provides a flexible way to define and compose service dependencies through two key mechanisms: `depends` and `link`. This system allows for both strong type safety and runtime flexibility in service composition.

## The Relationship Between `depends` and `link`

`depends` and `link` work together to create a flexible dependency injection system:

- `depends` declares what a service needs (the "what")
- `link` provides how those needs are satisfied (the "how")

Think of it like this:
- `depends` is like writing a job description - it specifies the requirements
- `link` is like hiring someone for the job - it provides the actual implementation

Here's a simple example:

```python
# Define what we need (the interface)
class StorageInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    async def save(self, data: str):
        pass

# Declare that we need storage (using depends)
@service
class MyService:
    storage = depends(StorageInterface)  # "I need something that can store data"

    @dynamo_endpoint()
    async def process(self, data: str):
        await self.storage.save(data)

# Provide different implementations (the "how")
@service
class FileStorage(StorageInterface):
    @dynamo_endpoint()
    async def save(self, data: str):
        with open("data.txt", "w") as f:
            f.write(data)

@service
class DatabaseStorage(StorageInterface):
    @dynamo_endpoint()
    async def save(self, data: str):
        await db.execute("INSERT INTO data VALUES (?)", data)

# Link different implementations (hiring for the job)
file_service = MyService.link(FileStorage)      # "Use file storage"
db_service = MyService.link(DatabaseStorage)    # "Use database storage"
```

The key aspects of this relationship are:

1. **Declaration vs. Implementation**:
   - `depends` declares what capabilities are needed
   - `link` provides the actual implementation of those capabilities

2. **Flexibility**:
   - A service using `depends` doesn't know or care about the implementation details
   - The same service can be linked to different implementations without changing its code

3. **Type Safety**:
   - `depends` ensures that only compatible implementations can be linked
   - The compiler/type checker can verify that linked implementations satisfy the interface

4. **Runtime Composition**:
   - `link` allows you to compose services at runtime
   - You can create different configurations by linking different implementations

This separation of concerns is what makes the system so powerful - services can be written to depend on interfaces without knowing the implementation details, and implementations can be swapped in and out as needed.

## Defining Interfaces

Interfaces in Dynamo are the foundation of your service architecture. They define the contracts that your services must follow, ensuring type safety and clear communication between components. Here's how to define an interface:

```python
from dynamo.sdk.lib.decorators import abstract_dynamo_endpoint
from dynamo.sdk.lib.service import AbstractDynamoService

class WorkerInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    async def generate(self, request: GenerateRequest):
        """Generate text based on the request.

        Args:
            request: The generation request containing input text

        Yields:
            Generated text tokens
        """
        pass
```

By inheriting from `AbstractDynamoService`, you're telling Dynamo that this is an interface that other services can implement. The `@abstract_dynamo_endpoint` decorator marks methods that must be implemented by any concrete service that uses this interface.

You can also include non-abstract methods in your interfaces to provide default implementations:

```python
class WorkerInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    async def generate(self, request: GenerateRequest):
        pass

    async def preprocess(self, text: str) -> str:
        """Default implementation of text preprocessing."""
        return text.strip()
```

## Declaring Dependencies

Once you have your interfaces defined, you can declare dependencies in your services. The `depends` decorator is flexible - it can accept either an interface or a concrete service class:

```python
@service(dynamo={"enabled": True})
class MyService:
    # Declare a dependency on an interface - any implementation will work
    worker = depends(WorkerInterface)

    # Declare a dependency on a specific implementation
    specific_worker = depends(VllmWorker)

    @dynamo_endpoint()
    async def process(self, request: Request):
        # Use the worker dependency
        async for result in self.worker.generate(request):
            yield result
```

When you use an interface as a dependency, you're saying "I need something that can do these things" without specifying exactly which implementation to use. This gives you flexibility to swap implementations at runtime. On the other hand, when you use a concrete service class, you're locking in that specific implementation.

## How Linking Works

The `link` method automatically detects which interfaces need to be satisfied when you inject an implementation. It works by:

1. Analyzing the implementation's inheritance chain to find all interfaces it implements
2. Matching these interfaces against the dependencies declared in the service
3. Satisfying any matching dependencies with the implementation

For example:

```python
# Define interfaces
class WorkerInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    async def generate(self, request: GenerateRequest):
        pass

class RouterInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    async def route(self, request: RouteRequest):
        pass

# Implement a service that satisfies both interfaces
@service
class CombinedService(WorkerInterface, RouterInterface):
    @dynamo_endpoint()
    async def generate(self, request: GenerateRequest):
        yield "generated text"

    @dynamo_endpoint()
    async def route(self, request: RouteRequest):
        yield "routed text"

# When linking, Dynamo automatically detects which interfaces are satisfied
@service
class MyService:
    worker = depends(WorkerInterface)
    router = depends(RouterInterface)

# Both dependencies are satisfied by the same implementation
service = MyService.link(CombinedService)
```

This automatic detection means you don't need to explicitly specify which interfaces are being satisfied - Dynamo figures it out based on the implementation's inheritance. This makes it easy to:

1. Use a single implementation to satisfy multiple interface dependencies
2. Add new interfaces to implementations without changing the linking code
3. Ensure type safety by checking that all required interfaces are implemented

## Complete Example: LLM Pipeline

Let's put it all together in a complete example. We'll create a simple LLM pipeline with different worker implementations and routing strategies:

```python
from dynamo.sdk.lib.decorators import abstract_dynamo_endpoint, dynamo_endpoint, service
from dynamo.sdk.lib.service import AbstractDynamoService
from dynamo.sdk.lib.dependency import depends
from typing import AsyncGenerator

# Define our interfaces
class WorkerInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    async def generate(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """Generate text based on the request."""
        pass

class RouterInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    async def route(self, request: RouteRequest) -> AsyncGenerator[str, None]:
        """Route requests to appropriate workers."""
        pass

# Implement different worker strategies
@service
class VllmWorker(WorkerInterface):
    @dynamo_endpoint()
    async def generate(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        for token in request.text.split():
            yield token.upper()

@service
class TRTLLMWorker(WorkerInterface):
    @dynamo_endpoint()
    async def generate(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        for token in request.text.split():
            yield token.lower()

# Implement different routing strategies
@service
class FastRouter(RouterInterface):
    worker = depends(WorkerInterface)  # Can use any WorkerInterface implementation

    @dynamo_endpoint()
    async def route(self, request: RouteRequest) -> AsyncGenerator[str, None]:
        async for response in self.worker.generate(request):
            await asyncio.sleep(0.1)
            yield response

@service
class SlowRouter(RouterInterface):
    worker = depends(WorkerInterface)  # Can use any WorkerInterface implementation

    @dynamo_endpoint()
    async def route(self, request: RouteRequest) -> AsyncGenerator[str, None]:
        async for response in self.worker.generate(request):
            await asyncio.sleep(1)
            yield response

# Create our frontend service
@service
class Frontend:
    router = depends(RouterInterface)  # Can use any RouterInterface implementation

    @dynamo_endpoint()
    async def chat(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        route_request = RouteRequest(text=request.message)
        async for response in self.router.route(route_request):
            yield response

# Compose different pipelines
fast_pipeline = Frontend.link(FastRouter).link(TRTLLMWorker)
slow_pipeline = Frontend.link(SlowRouter).link(VllmWorker)
mixed_pipeline = Frontend.link(FastRouter).link(VllmWorker)
```

This example demonstrates several important concepts:

1. **Interface Definition**: Both `WorkerInterface` and `RouterInterface` define clear contracts that their implementations must follow.

2. **Flexible Dependencies**: The `FastRouter` and `SlowRouter` both depend on `WorkerInterface`, allowing them to work with any worker implementation.

3. **Multiple Implementations**: We have two different worker implementations (`VllmWorker` and `TRTLLMWorker`) and two different router implementations (`FastRouter` and `SlowRouter`).

4. **Pipeline Composition**: We can create different pipeline configurations by linking different implementations together.

## Advanced Patterns

### Multiple Dependencies

Sometimes you need multiple instances of the same type of service. For example, you might want to route requests to different workers based on some criteria:

```python
@service
class ConditionalRouter(RouterInterface):
    worker1 = depends(WorkerInterface)
    worker2 = depends(WorkerInterface)

    @dynamo_endpoint()
    async def route(self, request: RouteRequest) -> AsyncGenerator[str, None]:
        if request.should_use_worker1:
            async for response in self.worker1.generate(request):
                yield response
        else:
            async for response in self.worker2.generate(request):
                yield response
```

### Testing with Mocks

The interface-based approach makes testing easy. You can create mock implementations for testing:

```python
class MockWorker(WorkerInterface):
    @dynamo_endpoint()
    async def generate(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        yield "mock response"

test_pipeline = Frontend.link(MockWorker)
```

## Best Practices

1. **Define Clear Interfaces**: Make your interfaces as specific as possible while maintaining flexibility. Include docstrings that clearly describe the expected behavior.

2. **Use Interface Dependencies**: Prefer using interface dependencies over concrete service dependencies. This gives you more flexibility to swap implementations.

3. **Include Type Hints**: Always include type hints for better IDE support and type checking. This makes your code more maintainable and helps catch errors early.

4. **Document Dependencies**: Clearly document what each service depends on and why. This helps other developers understand your service architecture.

5. **Test Different Compositions**: Test your services with different pipeline configurations to ensure they work correctly in all scenarios.

## See Also

- [LLM Hello World Example](../examples/llm_hello_world/llm_hello_world.py) - A complete working example of service composition
- [Interface Example](../examples/interface_example/README.md) - Detailed explanation of interface usage