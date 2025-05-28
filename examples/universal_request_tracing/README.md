# Universal X-Request-Id Support Guide

This document demonstrates how to use Dynamo SDK's built-in X-Request-Id support without manual implementation in each component.

## Overview

Dynamo SDK now provides built-in X-Request-Id support through decorators and Mixin classes:

- **Automatic Extraction**: Extract X-Request-Id from HTTP headers or generate new UUID
- **Automatic Propagation**: Automatically propagate request ID across all components
- **Automatic Response**: Automatically add X-Request-Id header to HTTP responses
- **Thread-Safe**: Use thread-local storage to ensure concurrency safety

## Usage

### 1. Frontend Component - Using Class Decorator

```python
from dynamo.sdk import auto_trace_endpoints, service, endpoint
from fastapi import FastAPI, Request

@auto_trace_endpoints  # Automatically add X-Request-Id support to all endpoints
@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    app=FastAPI(title="My Frontend"),
)
class Frontend:
    processor = depends(Processor)

    @endpoint(is_api=True, path="/v1/chat/completions", methods=["POST"])
    async def chat_completions(self, request: Request, chat_request: ChatCompletionRequest):
        # request_id is automatically extracted from X-Request-Id header and passed to method
        # Response will automatically include X-Request-Id header
        async for response in self.processor.chat_completions(chat_request):
            yield response
```

### 2. Processor Component - Using Mixin Class

```python
from dynamo.sdk import RequestTracingMixin, service, endpoint
from typing import Optional

@service(dynamo={"enabled": True, "namespace": "dynamo"})
class Processor(RequestTracingMixin):  # Inherit Mixin to get request ID utilities
    worker = depends(VllmWorker)

    @endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest, request_id: Optional[str] = None):
        # Use Mixin method to ensure we have request_id
        request_id = self.ensure_request_id(request_id)

        # Log with request_id
        self.log_with_request_id("info", f"Processing chat completion: {raw_request.model}")

        # Pass request_id to downstream components
        async for response in self._generate(raw_request, request_id):
            yield response
```

### 3. Manual Decorator Usage

```python
from dynamo.sdk import trace_frontend_endpoint, trace_processor_method

class MyComponent:
    @trace_frontend_endpoint  # Specifically for frontend endpoints
    @endpoint(is_api=True)
    async def my_endpoint(self, request: Request, data: MyData):
        # X-Request-Id automatically handled
        pass

    @trace_processor_method  # Specifically for processor methods
    async def process_data(self, data, request_id: Optional[str] = None):
        # request_id automatically obtained from context or generated
        pass
```

### 4. Getting Current Request ID

```python
from dynamo.sdk import get_current_request_id

async def some_internal_method(self):
    # Get current request ID anywhere
    current_request_id = get_current_request_id()
    if current_request_id:
        logger.info(f"Processing with request_id: {current_request_id}")
```

## Complete Examples

### Frontend Component

```python
from dynamo.sdk import auto_trace_endpoints, service, endpoint, depends
from fastapi import FastAPI, Request
from components.processor import Processor

@auto_trace_endpoints
@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    app=FastAPI(title="Universal Tracing Frontend"),
)
class Frontend:
    processor = depends(Processor)

    @endpoint(is_api=True, path="/v1/chat/completions", methods=["POST"])
    async def chat_completions(self, request: Request, chat_request: ChatCompletionRequest):
        """
        X-Request-Id is automatically:
        1. Extracted from request.headers
        2. Generated as new UUID if not present
        3. Passed to processor.chat_completions()
        4. Added to response headers
        """
        async for response in self.processor.chat_completions(chat_request):
            yield response

    @endpoint(is_api=True, path="/v1/completions", methods=["POST"])
    async def completions(self, request: Request, completion_request: CompletionRequest):
        async for response in self.processor.completions(completion_request):
            yield response
```

### Processor Component

```python
from dynamo.sdk import RequestTracingMixin, service, endpoint, depends
from typing import Optional

@service(dynamo={"enabled": True, "namespace": "dynamo"})
class Processor(RequestTracingMixin):
    worker = depends(VllmWorker)
    router = depends(Router)

    @endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest, request_id: Optional[str] = None):
        # Ensure we have request_id (from parameter, context, or generate new)
        request_id = self.ensure_request_id(request_id)

        # Log with request_id
        self.log_with_request_id("info", f"Processing chat completion for model: {raw_request.model}")

        async for response in self._generate(raw_request, RequestType.CHAT, request_id):
            yield response

    @endpoint(name="completions")
    async def completions(self, raw_request: CompletionRequest, request_id: Optional[str] = None):
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("info", f"Processing completion for model: {raw_request.model}")

        async for response in self._generate(raw_request, RequestType.COMPLETION, request_id):
            yield response

    async def _generate(self, raw_request, request_type: RequestType, request_id: str):
        # request_id automatically propagates to router and worker through Dynamo's Context system
        self.log_with_request_id("debug", f"Starting generation with request_id: {request_id}")

        # Processing logic...
        if self.use_router:
            engine_generator = await self.router_client.generate(request_obj)
        else:
            engine_generator = await self.worker_client.generate(request_obj)

        async for response in engine_generator:
            yield response
```

## 📁 Directory Structure

```
examples/universal_request_tracing/
├── README.md                    # This document
├── components/
│   ├── frontend.py             # Frontend with @auto_trace_endpoints
│   ├── processor.py            # Processor with RequestTracingMixin
│   ├── router.py               # Router with RequestTracingMixin
│   ├── worker.py               # Worker with RequestTracingMixin
│   ├── prefiller.py            # Prefiller with RequestTracingMixin
│   └── decoder.py              # Decoder with RequestTracingMixin
├── test_universal_tracing.py   # Complete test script
```

## All Component Examples

### Router Component

```python
from dynamo.sdk import RequestTracingMixin, endpoint, service

@service(dynamo={"enabled": True, "namespace": "dynamo"})
class Router(RequestTracingMixin):
    @endpoint(name="route")
    async def route(self, request_data: str, request_id: Optional[str] = None):
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("info", "Routing request to optimal worker")

        optimal_worker = await self._find_optimal_worker(request_data)
        return optimal_worker, 0.75  # worker_id, prefix_hit_rate
```

### Worker Component

```python
from dynamo.sdk import RequestTracingMixin, endpoint, service

@service(dynamo={"enabled": True, "namespace": "dynamo"})
class Worker(RequestTracingMixin):
    @endpoint(name="generate")
    async def generate(self, request_data: str, request_id: Optional[str] = None):
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("info", "Starting text generation")

        async for token in self._generate_tokens(request_data):
            yield token
```

### Prefiller Component

```python
from dynamo.sdk import RequestTracingMixin, endpoint, service

@service(dynamo={"enabled": True, "namespace": "dynamo"})
class Prefiller(RequestTracingMixin):
    @endpoint(name="prefill")
    async def prefill(self, request_data: str, request_id: Optional[str] = None):
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("info", "Starting KV cache prefill")

        cache_key = self._generate_cache_key(request_data)
        return await self._perform_prefill(request_data, cache_key)
```

### Decoder Component

```python
from dynamo.sdk import RequestTracingMixin, endpoint, service

@service(dynamo={"enabled": True, "namespace": "dynamo"})
class Decoder(RequestTracingMixin):
    @endpoint(name="decode")
    async def decode(self, hidden_states: List[float], request_id: Optional[str] = None):
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("info", "Starting token decoding")

        async for token_result in self._decode_tokens(hidden_states):
            yield token_result
```

## Testing

```python
import asyncio
import aiohttp

async def test_universal_tracing():
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-universal-123"
    }

    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8080/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            # Verify X-Request-Id is correctly echoed back
            assert response.headers.get("X-Request-Id") == "test-universal-123"
            print("✅ Universal request tracing works!")

if __name__ == "__main__":
    asyncio.run(test_universal_tracing())
```

## Advantages

### 1. Zero Configuration
- Just add decorator or inherit Mixin class
- No manual handling of HTTP headers or responses
- Automatic request ID propagation

### 2. Consistency
- All components use the same request ID format
- Unified logging format and tracing approach
- Compatible with vLLM and OpenAI standards

### 3. Flexibility
- Supports class decorators, method decorators, and Mixin classes
- Can get current request ID anywhere
- Supports manual request ID passing

### 4. Performance
- Uses thread-local storage, no global locks
- Minimal performance overhead
- Automatic context cleanup

## Migration Guide

### Migrating from Manual Implementation

**Before (Manual Implementation):**
```python
def extract_or_generate_request_id(request: Request) -> str:
    request_id = request.headers.get("x-request-id")
    if request_id is None:
        request_id = str(uuid.uuid4())
    return request_id

@endpoint(is_api=True)
async def chat_completions(self, request: Request, chat_request: ChatCompletionRequest):
    request_id = extract_or_generate_request_id(request)

    async def content_generator():
        async for response in self.processor.chat_completions(chat_request, request_id):
            yield response

    response = StreamingResponse(content_generator())
    response.headers["X-Request-Id"] = request_id
    return response
```

**After (Using Universal Support):**
```python
@auto_trace_endpoints
class Frontend:
    @endpoint(is_api=True)
    async def chat_completions(self, request: Request, chat_request: ChatCompletionRequest):
        # Everything is handled automatically!
        async for response in self.processor.chat_completions(chat_request):
            yield response
```

## Important Notes

1. **FastAPI Request Object**: Frontend endpoints need to include `Request` parameter to extract HTTP headers
2. **Optional Parameters**: Processor methods should accept `Optional[str] request_id` parameter
3. **Thread Safety**: Uses thread-local storage, safe in async environments
4. **Context Propagation**: Router and Worker components automatically get request ID through Dynamo's Context system

This universal approach greatly simplifies X-Request-Id implementation, allowing developers to focus on business logic rather than infrastructure code.
