# Golden Manifest Design

`golden` compares controller output stored in a Kubernetes API server with a
small, reviewable YAML contract. It operates on unstructured objects and can be
used with either `operatorenv`, `clusterenv`, or an explicitly selected real
cluster.

## API

```go
golden.MatchManifests(t, env.Client(), env.Namespace(), "testdata/deployment.yaml")
```

Every YAML document contributes one expected object. Documents may have
different kinds. For each GroupVersionKind in the file, the actual namespaced
objects must be exactly the expected set, and every expected document must
match exactly one actual object. Cluster-scoped selection and more elaborate
search strategies are deferred until a test requires them.

## Match Directives

Mappings are strict by default. A strict mapping rejects unspecified actual
fields. `$strict: false` makes only that mapping non-strict; when matching a
specified child field, its mapping is strict again unless it has its own
`$strict: false`. `$$strict` escapes a real field named `$strict`.

Scalar string directives are:

| Value | Meaning |
|---|---|
| `$ignore` | The scalar field must exist; its value is ignored. |
| `$exists` | The scalar field must exist; its value is ignored. |
| `$notexists` | The containing mapping must not have the field. |
| `$glob:<glob>` | The scalar string must match the glob. |
| `$pattern:<regexp>` | The scalar string must match the regular expression. |

For mapping values, the existence directives use a one-key mapping such as
`{$ignore: true}` or `{$notexists: true}`. Directives do not apply to sequence
values. Sequences match exactly and in order.

## Mismatch Output

After the retry timeout, a mismatch writes `<expected>.new`. Existing YAML
comments and unchanged directives are preserved throughout the document tree.
The generated file is changed only as far as needed to match the observed
objects:

- strict mappings gain missing actual fields, drop fields absent from the
  actual object, and update mismatching values;
- non-strict mappings keep their selected fields, and a selected field absent
  from the actual object becomes `$notexists`;
- unmatched actual objects are appended and expected objects with no actual
  counterpart are removed.

The `.new` file is diagnostic output. A reviewer still decides whether its
changes describe the intended controller contract.
