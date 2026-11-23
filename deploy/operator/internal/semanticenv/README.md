<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Semantic Environment Composition

`semanticenv.Compose` accepts ordered Kubernetes environment layers, from general
to specific. It returns their composed environment or an error naming a cycle and
its source field paths. It does not depend on Dynamo APIs, feature gates, or a
Kubernetes client.

```go
environment, err := semanticenv.Compose(
    semanticenv.Layer{Source: "operator.defaults", Env: defaults},
    semanticenv.Layer{Source: "spec.env", Env: graphEnvironment},
    semanticenv.Layer{
        Source: "spec.components[0].podTemplate.spec.containers[0].env",
        Env: containerEnvironment,
    },
)
```

Later layers replace earlier definitions of the same name, including the entire
`valueFrom`. Surviving entries retain their relative order within each layer.
Unescaped `$(NAME)` references to definitions in other layers add dependency edges.
The stable topological order uses original layer and entry positions for ties,
not alphabetical name order.

Within a layer, Kubernetes' sequential semantics remain unchanged: forward
references are not repaired, unresolved and self-references remain runtime
lookups, and repeated names remain ordered entries. Existing API validation is
the caller's responsibility. `$$(NAME)` is escaped; shell variable syntax is not
interpreted. `valueFrom` and external inputs do not require resolving their values.
The composer neither substitutes values nor mutates its inputs, and its output
does not alias the supplied Kubernetes objects.

Keep all original layers until final composition. Composition is **not
associative**: feeding an intermediate result back as one layer invents local
ordering constraints between entries that originally belonged to different
layers. For example:

```text
operator:   SYSTEM=$(B)
graph:      A=$(SYSTEM)
container:  B=b

valid final order: B, SYSTEM, A
```

Composing graph and container first produces `A, B`. Treating that result as one
layer would force `A` before `B`, contradicting the actual dependency chain.
Persisting or passing environments between rendering stages must therefore retain
their source layers, or happen only after final composition.

`ContainerLayers` carries these sources through Pod transformations. Capture them
before mutation, record authoritative values with `Bind`, remove intentionally
absent values with `Omit`, then `Apply` once all bindings and removals are known.
A nil collector leaves legacy lowerings unchanged. Dynamo enables composition
only for resources whose stored operator origin is at least 1.8.0.

Each collector belongs to one logical Pod lineage and identifies containers by
name. After capture or registration, make all environment changes through the
collector: `container.Env` is only a staged legacy view, and `Apply` replaces it
from the retained sources. Register replacement containers explicitly; use
`Clone` for renamed or derived containers so their provenance is retained.
Independent Pod copies may share a collector only while its sources are read-only.

The caller controls precedence for operator-owned bindings by supplying their
layer after overridable defaults and authored values. Runtime providers must
place externally injected definitions before entries that reference them.

`Lookup(name, layers...)` observes one binding from the supplied sources before
final composition. It composes and replays only that binding's effective,
transitive dependencies, so an unrelated unfinished feature binding does not
make a known alias unreadable. Its presence result is separate from `Value.Known`:
runtime inputs and conflicts among the selected dependencies remain unknown.
Lookup does not supply later operator bindings or validate the complete environment;
callers must still run final `Compose` after all sources and owned bindings exist.
`Replay` evaluates an already-composed list in order; neither API reads runtime inputs.

Contract: [DEP #18214](https://github.com/ai-dynamo/dynamo/issues/18214).
