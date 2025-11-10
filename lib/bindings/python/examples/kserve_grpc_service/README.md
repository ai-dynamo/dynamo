# KServe gRPC Example

This directory contains a minimal Dynamo worker that serves a KServe-compatible
gRPC endpoint (`server.py`) and a Python client (`test_client.py`) that exercises
the endpoint using the Triton `tritonclient.grpc` API.

## Prerequisites

- The Dynamo Python bindings installed
- Client dependencies:
  - `numpy`
  - `tritonclient[grpc]`

You can install the Python dependencies into your active environment with:

```bash
uv pip install numpy tritonclient[grpc]
```

## Running the mock server

1. From the repository root, set `PYTHONPATH` so Python can locate the local
   Dynamo package:

   ```bash
   export PYTHONPATH=$(pwd)
   ```

2. Start the worker:

   ```bash
   python lib/bindings/python/examples/kserve_grpc_service/server.py
   ```

   The server registers a mock completions model named `mock_model` and listens
   on `0.0.0.0:8787`. Leave this process running while you test the endpoint.

## Sending a request with the Triton client

With the server running, invoke the example client from a separate terminal:

```bash
python lib/bindings/python/examples/kserve_grpc_service/test_client.py \
  --model mock_model \
  --prompt "Hello from Dynamo!"
```


You can override the `--host`, `--port`, and `--prompt` options as needed. The script sends an inference request over gRPC using the `InferenceServerClient` and prints the decoded `ModelInferResponse` payload. You should see the prompt `Hello from Dynamo!` successfully received and printed by the server.

## Alternative tooling

For debugging purposes you can still call the endpoint directly with
[`grpcurl`](https://github.com/fullstorydev/grpcurl) by running
`grpcurl.sh` in this directory.
