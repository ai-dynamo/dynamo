# User ensemble

Experimental: This workflow API is under active development and may evolve based on feedback. You can alternatively compose the same discoverable Dynamo endpoints directly, as `bespoke/` demonstrates.

This example serves the same application in two ways. Both keep the custom encoder, dummy classifier, request adaptation, and response merge inside one orchestrator process. Only a stock aggregated `dynamo.vllm` worker is remote.

```text
                                      stock dynamo.vllm
                                    ┌───────────────────┐
request ──> inline encoder ────────> │ GenerateRequest   │ ──┐
                 │                  │ + encoder_result  │   │
                 └─> classifier     └───────────────────┘   │
                         └──────────────────────────────────> merge ──> chunk
```

The declarative version uses `Workflow` and `WorkflowOrchestrator.bind(...)` to own graph scheduling, endpoint discovery, Generate-stream folding, validation, cancellation, and result propagation. Inline runners and the remote Generate endpoint are bound together in the orchestrator worker. The bespoke version contains the equivalent endpoint client, fan-out, stream collection, sibling cancellation, and merge control flow explicitly. This is a runnable code comparison, not a performance claim.

## Run

From the repository root, launch one implementation:

```bash
examples/experimental/workflow/user_ensemble/workflow/launch.sh
# or
examples/experimental/workflow/user_ensemble/bespoke/launch.sh
```

Then send the shared request:

```bash
python -m examples.experimental.workflow.user_ensemble.common.client
```

Both launchers use the TCP request plane with MsgPack because `encoder_result` contains binary tensor data. The bundled Hitchhiker encoder makes the image represent a known phrase so a successful response contains `42`; replace `DYN_ENCODER_CLASS` with an application encoder implementing `VisionEncoderBackend`.
