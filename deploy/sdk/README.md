# Dynamo SDK

Dynamo is a python based SDK for building and deploying distributed inference applications. Dynamo leverages concepts from open source projects like [BentoML](https://github.com/bentoml/bentoml) to provide a developer friendly experience to go from local development to K8s deployment.

## Installation

```bash
pip install ai-dynamo
```

## Quickstart
Lets build a simple distributed pipeline with 3 components: `Frontend`, `Middle` and `Backend`. The structure of the pipeline looks like this:

```
Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │
      ▼
┌─────────────┐
│  Backend    │
└─────────────┘
```

The code for the pipeline looks like this:

```python
# filename: pipeline.py

from dynamo.sdk import service, dynamo_endpoint, depends
from pydantic import BaseModel

class RequestType(BaseModel):
    text: str

@service(resources={"cpu": "1"})
class Frontend:
    middle = depends(Middle)

    @api
    async def generate(self, text: str):
        request = RequestType(text=text)
        async for response in self.middle.generate(request.model_dump_json()):
            yield f"Frontend: {response}"

@service(
    resources={"cpu": "1"},
    dynamo={"enabled": True, "namespace": "inference"}
)
class Middle:
    backend = depends(Backend)

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        text = f"{req.text}-mid"
        for token in text.split():
            yield f"Mid: {token}"

@service(
    resources={"cpu": "1"},
    dynamo={"enabled": True, "namespace": "inference"}
)
class Backend:
    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        text = f"{req.text}-back"
        for token in text.split():
            yield f"Backend: {token}"
```

You can run this pipeline locally by spinning up ETCD and NATS and then running the pipeline:

```bash
# Spin up ETCD and NATS
docker compose -f deploy/docker-compose.yml up -d
```

then

```bash
# Run the pipeline
dynamo serve pipeline:Frontend
```

Once it's up and running, you can make a request to the pipeline using

```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "federer"}'
```

You should see the following output

```bash
federer-mid-back
```

You can find in-depth documentation for the Dynamo SDK [here](./docs/sdk/README.md) and the Dynamo CLI [here](./docs/cli/README.md)

