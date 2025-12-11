# Tool Calling with Dynamo

You can connect Dynamo to external tools and services using function calling (also known as tool calling). By providing a list of available functions, Dynamo can choose
to output function arguments for the relevant function(s) which you can execute to augment the prompt with relevant external information.

Tool calling (AKA function calling) is controlled using the `tool_choice` and `tools` request parameters.


## Prerequisites

To enable this feature, you should set the following flag while launching the backend worker

- `--dyn-tool-call-parser` : select the parser from the available parsers list using the below command

```bash
# <backend> can be vllm, sglang, trtllm, etc. based on your installation
python -m dynamo.<backend> --help"
```

> [!NOTE]
> If no tool call parser is provided by the user, Dynamo will try to use default tool call parsing based on `<TOOLCALL>` and `<|python_tag|>` tool tags.

> [!TIP]
> If your model's default chat template doesn't support tool calling, but the model itself does, you can specify a custom chat template per worker
> with `python -m dynamo.<backend> --custom-jinja-template </path/to/template.jinja>`.


Parser to Model Mapping

| Parser Name | Supported Models                                                      |
|-------------|-----------------------------------------------------------------------|
| hermes      | Qwen/Qwen2.5-*, Qwen/QwQ-32B, NousResearch/Hermes-2-Pro-*, NousResearch/Hermes-2-Theta-*, NousResearch/Hermes-3-* |
| mistral | mistralai/Mistral-7B-Instruct-v0.3, Additional mistral function-calling models are compatible as well.|
| llama3_json | meta-llama/Llama-3.1-*, meta-llama/Llama-3.2-* |
| harmony | openai/gpt-oss-* |
| nemotron_deci | nvidia/nemotron-* |
| phi4 | Phi-4-* |
| deepseek_v3 | deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-R1-0528 |
| deepseek_v3_1 | deepseek-ai/DeepSeek-V3.1 |
| pythonic |  meta-llama/Llama-4-* |
| jamba |  ai21labs/AI21-Jamba-*-1.5, ai21labs/AI21-Jamba-*-1.6, ai21labs/AI21-Jamba-*-1.7, |


## Examples

### Launch Dynamo Frontend and Backend

```bash
# launch backend worker
python -m dynamo.vllm --model openai/gpt-oss-20b --dyn-tool-call-parser harmony

# launch frontend worker
python -m dynamo.frontend
```

### Tool Calling Request Examples

- Example 1
```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8081/v1", api_key="dummy")

def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."
tool_functions = {"get_weather": get_weather}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "What's the weather like in San Francisco in Celsius?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=10000
)
print(f"{response}")
tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}")
```

- Example 2
```python

# Use tools defined in example 1

time_tool = {
    "type": "function",
    "function": {
        "name": "get_current_time_nyc",
        "description": "Get the current time in NYC.",
        "parameters": {}
    }
}


tools.append(time_tool)

messages = [
    {"role": "user", "content": "What's the current time in New York?"}
]


response = client.chat.completions.create(
    model="openai/gpt-oss-20b", #client.models.list().data[1].id,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=100,
)
print(f"{response}")
tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
```

- Example 3


```python

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_tourist_attractions",
            "description": "Get a list of top tourist attractions for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to find attractions for.",
                    }
                },
                "required": ["city"],
            },
        },
    },
]

def get_messages():
    return [
        {
            "role": "user",
            "content": (
                "I'm planning a trip to Tokyo next week. what are some top tourist attractions in Tokyo? "
            ),
        },
    ]


messages = get_messages()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=100,
)
print(f"{response}")
tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
```
