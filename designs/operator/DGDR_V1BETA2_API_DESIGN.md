<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DGDR v1beta2 API design

## Scope

These rules apply to `DynamoGraphDeploymentRequest` v1beta2 and the accepted
`NativeSearchInput` snapshots used by `DynamoGraphDeploymentRun`. They cover typed
fields, admission defaults and validation, AISimulate lowering, and
`spec.aisim.inputOverrides`. They do not define a generic contract for every
Dynamo API or every kind of override.

## Invariants

1. **Typed fields have an independent contract.** Validate typed values against
   their own schema and genuine API consistency constraints. An input override
   cannot repair an invalid typed value or make an otherwise valid typed field
   invalid. Do not inspect the merged backend input to invent typed-field rules.
2. **Unused does not mean invalid.** Accept a valid hardware fact even when the
   selected search does not use it. For example, positive KV-transfer bandwidth
   remains valid in an aggregated-only search. A backend limitation is not
   automatically a permanent restriction on the typed Kubernetes API.
3. **Input overrides are the final overlay.** Lower the defaulted typed input,
   then mechanically apply `inputOverrides` with the existing merge and deletion
   semantics. Overrides may replace lowered values or delete them with `null`.
   Preserve documented operator-owned execution fields; this escape hatch does
   not grant ownership of reserved fields.
4. **The backend owns its execution semantics.** Do not maintain handwritten
   admission logic that reinterprets AISimulate modes, transfer calculations,
   defaults, or scoring quality. The existing validation of the final merged
   input against AISimulate's published schema remains a separate check. Typed
   validity alone does not guarantee that every override or backend run succeeds.
5. **Defaulting resolves the persisted contract.** Preserve explicit values and
   the API's distinctions between absence and an empty object. Request admission
   resolves defaults before the controller snapshots the input into a Run. A Run
   must already be a resolved snapshot; overrides cannot substitute for required
   typed snapshot fields. The exactly empty `hardware: {}` opt-out remains valid;
   a hardware object containing `interconnect` is not that opt-out.
6. **Field documentation states stable semantics.** Keep meaning, units,
   measurement scope, optionality, and omitted-field behavior in API comments.
   Reference backend documentation for simulation algorithms, parallelism,
   contention, and benchmarking procedures. Do not copy backend internals into
   the generated Kubernetes API contract.

## Examples

These are partial v1beta2 `spec` examples; other required fields are omitted.
The typed per-GPU bandwidth is valid with an aggregated-only override:

```yaml
spec:
  hardware:
    interconnect:
      kvTransferBandwidthGBps: 27.7
  aisim:
    inputOverrides:
      engine:
        mode: aggregated
```

This establishes typed validity, not guaranteed backend acceptance. AISimulate's
schema or execution may impose its own restrictions.

An override can replace the lowered value without changing the typed field:

```yaml
spec:
  hardware:
    interconnect:
      kvTransferBandwidthGBps: 27.7
  aisim:
    inputOverrides:
      engine:
        kv_transfer:
          bandwidth_gb_per_second: 10
```

Setting `engine.kv_transfer: null` in that overlay deletes the lowered transfer
model. In contrast, this typed value is invalid even though the override deletes
its output:

```yaml
spec:
  hardware:
    interconnect:
      kvTransferBandwidthGBps: 0
  aisim:
    inputOverrides:
      engine:
        kv_transfer: null
```

The schema requires typed bandwidth to be greater than zero. If it is omitted,
lowering supplies no bandwidth and AISimulate's default applies.

## Review checks

- Find the typed owner of each rule before adding validation. Does the rule
  enforce that type's contract, or reconstruct a backend execution decision?
- Vary overrides while holding typed values fixed. Typed-field validity must
  remain fixed, even if the separate backend-schema result changes.
- Check override replacement and deletion after lowering, and ensure reserved
  execution-field ownership is preserved.
- Check Request defaulting and direct Run admission, including absent hardware,
  the empty-object opt-out, and a nonempty partial hardware object.
- Apply [structural defaulting](STRUCTURAL_DEFAULTING.md) and
  [structural validation](STRUCTURAL_VALIDATION.md) for traversal, ownership,
  signatures, and admission-chain tests. Structural organization is distinct
  from the semantic limits on validation described here.
- Check field comments and generated references for contract facts; link to the
  backend for model details.

## Decision source

Derived from the API review in
[PR #19475](https://github.com/ai-dynamo/dynamo/pull/19475), especially the
[override boundary](https://github.com/ai-dynamo/dynamo/pull/19475#discussion_r4695467072),
[unused bandwidth](https://github.com/ai-dynamo/dynamo/pull/19475#discussion_r4695503257),
and [documentation scope](https://github.com/ai-dynamo/dynamo/pull/19475#discussion_r4695377253).
