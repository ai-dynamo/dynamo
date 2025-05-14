# Openai v1/completions interface examples

## Step 1: Modify the parameters of subprocess.run in frontend.py

Change "chat-models" in subprocess.run to "completions"
```python
# DYNAMO_DIR/examples/llm/components/frontend.py

# The first modification
def setup_model(self):
    """Configure the model for HTTP service using llmctl."""
    subprocess.run(
        [
            "llmctl",
            "http",
            "remove",
            "completions",        # "chat-models" -> "completions"
            self.frontend_config.served_model_name,
        ],
        check=False,
    )
    # The second modification
    subprocess.run(
        [
            "llmctl",
            "http",
            "add",
            "completions",        # "chat-models" -> "completions"
            self.frontend_config.served_model_name,
            self.frontend_config.endpoint,
        ],
        check=False,
    )

# The third modification
@async_on_shutdown
def cleanup(self):
    """Clean up resources before shutdown."""

    # circusd manages shutdown of http server process, we just need to remove the model using the on_shutdown hook
    subprocess.run(
        [
            "llmctl",
            "http",
            "remove",
            "completions",            # "chat-models" -> "completions"
            self.frontend_config.served_model_name,
        ],
        check=False,
    )

```

## **Step 2**: Annotation processor.py chat_completions method and enable completions

```python
# DYNAMO_DIR/examples/LLM/components/processor.py

# @dynamo_endpoint(name="chat/completions")
# async def chat_completions(self, raw_request: ChatCompletionRequest):
#     async for response in self._generate(raw_request, RequestType.CHAT):
#         yield response

@dynamo_endpoint()
async def completions(self, raw_request: CompletionRequest):
    async for response in self._generate(raw_request, RequestType.COMPLETION):
        yield response

```

## **Step 3**: Modify the endpoint in the Frontend of the yaml file

```yaml
# The yaml files required for dynamo serve

Frontend:
  served_model_name: Meta-Llama-3.1-8B-Instruct
  endpoint: dynamo.Processor.completions        # chat/completions -> completions
  port: 8000
```