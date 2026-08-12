# Remote workflow proof

This example compiles one placement-neutral workflow into three discovered
remote stage bindings. The compiler creates no processes; the deployment system
does. For a local proof, start each stage in its own terminal:

```bash
export DYN_DISCOVERY_BACKEND=file
export DYN_EVENT_PLANE=zmq

python -m examples.custom_backend.workflow_remote.worker encoder
python -m examples.custom_backend.workflow_remote.worker classifier
python -m examples.custom_backend.workflow_remote.worker generator
```

Run the orchestrator in a fourth terminal with the same environment:

```bash
python -m examples.custom_backend.workflow_remote.client
```

The encoder result fans out through two independent inline edges. Expected
output resembles:

```text
{'scores': {'workflow': 0.2, 'other': 0.8},
 'text': 'processes across runs workflow dynamo'}
```

Only text, bytes, and JSON can cross these v0 inline bindings. Tensor edges are
rejected at compile time until an explicit tensor carrier is selected.
