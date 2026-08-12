# Encoder, classifier, and decoder workflow

This example authors one encoder fan-out workflow and runs every stage in one
process. Workflow authoring and stage contracts stay at the package root, while
the `local` package owns aggregated resource loading and serving.

Run it from the repository root:

```bash
./examples/custom_backend/user_ensemble/local/launch.sh
```

The worker loads the configured encoder, classifier, and vLLM decoder, compiles
the shared workflow with default local bindings, and exposes it through the
ordinary Dynamo frontend.
