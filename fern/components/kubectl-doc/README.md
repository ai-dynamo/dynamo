# kubectl-doc Fern Component

This directory contains the vendored `kubectl-doc` React/Fern component and the Dynamo-owned lazy loading wrapper.

## Version

`VERSION` pins the upstream `sttts/kubectl-doc` tag used for vendoring the shared component runtime.

## Human-Run Vendoring

Run this only when intentionally bumping the component implementation:

```bash
python3 fern/kubectl_doc.py vendor
```

To test a local checkout before tagging:

```bash
python3 fern/kubectl_doc.py vendor --checkout /path/to/kubectl-doc
```

The vendoring script updates only the shared component files copied from `kubectl-doc`:

- `KubeSchemaDoc.tsx`
- `kubectl-doc-runtime.d.ts`
- `kubectl-doc-runtime.js`
- `kubectl-doc-styles.ts`

`LazyKubeSchemaDoc.tsx`, `schemaSources.generated.ts`, and Dynamo's contract test are maintained in this repo.

## Generated CRD API Reference

CRD-specific API reference artifacts are generated from `deploy/operator/config/crd/bases`:

```bash
make -C fern gen
```

This rewrites:

- `docs/kubernetes/api-reference/*.mdx`
- `fern/kubectl-doc-schemas/*.json`
- `schemaSources.generated.ts`

CI runs:

```bash
make -C fern check-generated
```

That command reruns generation and fails if the checked-in API reference is stale.
