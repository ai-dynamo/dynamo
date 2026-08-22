# Remote static workflow

This CPU-only proof runs one static fan-out graph through three independently
discovered Dynamo worker processes. It bypasses the HTTP frontend and uses only
inline `text` and `json` values.

From the repository root, with the Dynamo Python bindings installed:

```bash
examples/custom_backend/workflow_remote/launch.sh
```

The client waits for all three endpoint populations, invokes the encoder once,
fans its token list into classifier and generator calls, joins both terminal
responses, and verifies the final result.

Each stage implements unary `StageRunner.run(...)` and returns one mapping. `RemoteStageServer` adapts that method to Dynamo's streaming endpoint contract, so stage implementations do not write `yield` or handle transport streams.
