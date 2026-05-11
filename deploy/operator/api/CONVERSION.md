# Conversion Rules

These rules apply to API conversion code and tests under `deploy/operator/api`.
Every API type change in any version must update conversion code/tests, or
explicitly document why conversion is unaffected.

## Structure

Top-level API objects implement `sigs.k8s.io/controller-runtime/pkg/conversion.Convertible`:

```go
type Convertible interface {
	runtime.Object
	ConvertTo(dst Hub) error
	ConvertFrom(src Hub) error
}
```

Top-level methods have only the `Convertible` parameters:

```go
func (src *DynamoWidget) ConvertTo(dstRaw conversion.Hub) error
func (dst *DynamoWidget) ConvertFrom(srcRaw conversion.Hub) error
```

Their call stack is:

1. Copy Kubernetes metadata from `src` to `dst`.
2. Decode sparse spec/status payloads into typed `restored` values.
3. Scrub private conversion annotations from `dst`.
4. Call typed conversion functions for spec/status.
5. Restore destination-only fields from `restored`.
6. Collect source-only fields in typed `save` values.
7. Encode non-empty `save` values into sparse spec/status payloads on `dst`.

Example:

```text
(*DynamoWidget).ConvertTo(dstRaw)
  ConvertFromDynamoWidgetSpec(&src.Spec, &dst.Spec, restored.Spec, &save.Spec, ctx)
    ConvertFromDynamoWidgetNestedSpec(&src.Spec.Nested, &dst.Spec.Nested, restoredNested, saveNested, ctx)
  ConvertFromDynamoWidgetStatus(&src.Status, &dst.Status, restored.Status, &save.Status)
  saveDynamoWidgetAnnotations(save, dst)
```

The conversion algorithm follows the API Go types inductively:

- Export conversion functions from `v1alpha1`.
- Name conversion functions after the v1alpha1 type:

```go
func ConvertFromDynamoWidgetSpec(
	src *DynamoWidgetSpec,
	dst *v1beta1.DynamoWidgetSpec,
	restored *v1beta1.DynamoWidgetSpec,
	save *DynamoWidgetSpec,
	ctx DynamoWidgetSpecConversionContext,
) error

func ConvertToDynamoWidgetSpec(
	src *v1beta1.DynamoWidgetSpec,
	dst *DynamoWidgetSpec,
	restored *DynamoWidgetSpec,
	save *v1beta1.DynamoWidgetSpec,
	ctx DynamoWidgetSpecConversionContext,
) error
```

- `ConvertFrom` means v1alpha1 -> hub.
- `ConvertTo` means hub -> v1alpha1.
- Do not include `V1alpha1` in conversion function names.
- Do not add wrapper conversion functions with alternate names.
- Parameter order is fixed: `src`, `dst`, `restored`, `save`, `ctx`.
- Include `restored`, `save`, `ctx`, and `error` only when needed.
- Put `ctx` last.
- Return no value for infallible conversions.
- Return `error` when the converter parses JSON, validates preserved payloads, or
  otherwise can reject input.
- `dst` is caller-owned and non-nil.
- Callers allocate nested `dst` objects explicitly.
- Converters do not accept `**T`.
- Converters do not encode absence by assigning `dst` to nil.
- Converters perform only their own API struct's conversion.
- Every nested API struct gets its own systematic
  `ConvertFrom<SubStruct>` / `ConvertTo<SubStruct>` conversion function pair.
- Parent converters call those nested converters instead of inlining nested
  struct conversion.

Allowed local helpers:

- `restore*` readers for decoded sparse payloads.
- `save*` writers for sparse payloads.
- `ensure*` allocators for sparse save payloads.
- Side-effect-free predicates.
- Field-group projections, such as pod template composition/decomposition.

## Invariants

- `src` live fields are always the source of truth.
- A represented field always comes from `src`, including nil, empty, false, and
  zero values.
- `restored` data is used only for destination fields that `src` cannot
  represent.
- `save` data contains only source fields that `dst` cannot represent.
- Preserve unrepresentable data only through the type's private sparse spec and
  status payloads.
- New conversion style allows at most two private conversion annotations per
  type: one spec payload annotation and one status payload annotation.
- Do not add per-field, per-list, per-subobject, or other conversion
  annotations.
- Do not overlay preserved data onto converted live fields.
- Do not use preserved data as a fallback for represented fields.
- Do not mutate `src`.

## Sparse Payloads

- Payloads are typed API structs from the source version.
- Payloads must be sparse by construction.
- Save only source fields the target version cannot represent.
- Save only the minimal context needed to match preserved fields back to live
  source structure.
- Skip empty save objects.
- Do not allocate nested save structs unless they contain at least one saved
  field.
- Keep payload annotation constants unexported and local to conversion files.
- Only API conversion code and conversion tests may know payload annotations or
  payload shapes.
- Controllers, reconcilers, webhooks, internal helpers, and general API helpers
  must not read, write, delete, filter, decode, encode, branch on, or expose
  payload annotations or payloads.
- Non-conversion code may copy Kubernetes metadata opaquely, but must not
  identify conversion annotations.
- If non-conversion code needs data from a sparse payload, add or call a real
  conversion helper instead.
- Restore code may read represented fields from a sparse payload only as
  matching context.
- Restore code must not restore represented fields from a sparse payload.

## Compound Values

- Restore compound values leaf-by-leaf.
- Do not restore a whole pod template, container, job, resource requirements,
  or similar compound object and then patch represented fields over it.
- For projected fields, compare the live target object with the projected
  object and save only the unrepresentable remainder.
- Keep sparse matching/decomposition context inside the spec/status payload.
- Sparse context is not a source of truth.

## Named Lists

- Match preserved list-map entries by the declared list-map key, not by slice
  index.
- Ignore preserved entries whose key is no longer present in live `src`.
- New live entries get no preserved data unless the sparse payload has the same
  key.
- Saved entries for named lists must include the list-map key, such as
  `ComponentName` for DGD components.

## Legacy DGDR Exception

- Do not add new legacy formats.
- Existing legacy DGDR annotations are temporary downgrade shims only.
- Keep legacy keys named `legacyAnn*`.
- Keep their `TODO(sttts)` removal comments.
- Isolate legacy reads/writes in legacy helpers.
- Decode legacy data into the same typed `restored` model used by structural
  conversion.
- Legacy data is an old-value cache only.
- Legacy data must not override fields representable by live `src`.
- Keep the old converter as a test oracle while downgrade compatibility is
  required.
- Legacy-vs-structural fuzz tests may normalize inputs to the old converter's
  representable behavior.
- Main round-trip fuzz tests must not keep workarounds for fixed structural
  conversion bugs.

## API Changes

For every added, removed, renamed, or semantically changed API field, choose one
explicit conversion policy:

- Native mapping: convert from live `src`.
- Source-only preservation: save into the source-version sparse payload on
  `dst`.
- Target-only restore: restore from `restored` after live fields are converted.
- Derived or lossy mapping: document the mapping and add focused tests.
- Intentional drop: document why it is outside the conversion contract and add
  a focused test when observable.

If the other version later grows a native field for the same concept:

- live `src` wins over stale sparse payload data;
- the sparse payload stops preserving that now-representable value.

`TestV1Beta1ConversionFieldSetIsAcknowledged` is a schema-change tripwire for
listed v1beta1 roots:

- Do not update `knownV1Beta1ConversionFieldSet` first.
- Implement the conversion decision.
- Add or update focused tests.
- Then refresh the snapshot from the failing test diff.

## Mutability

- Conversion must not mutate `src`.
- Sharing backing data between `src` and `dst` is allowed when treated as
  read-only after assignment.
- Do not deep-copy solely because a value came from `src`.
- Deep-copy before editing, appending, sorting, normalizing, clearing, merging,
  or otherwise mutating data that came from `src`.
- Mutating through `dst` after assigning shared data from `src` still violates
  this rule.

## Tests

Required coverage for conversion changes:

- Focused tests for each new or changed conversion policy.
- Found conversion regressions get focused tests in the relevant
  `*_bugs_test.go` files.
- Regular round-trip fuzz:

```text
hub -> spoke -> hub
spoke -> hub -> spoke
```

- Mutability fuzz for stale sparse payload behavior:

```text
fuzz in
convert to other
mutate other without deleting private spec/status annotations
convert other -> in -> other2
compare other and other2, ignoring only private spec/status annotations
```

- Mutation must touch nested existing objects, arrays, and slices.

## Review Checklist

- Every represented field comes from live `src`.
- Every restored field is unrepresentable by live `src`.
- Every saved field is unrepresentable by `dst`.
- Save payloads are sparse.
- Sparse context stays inside spec/status payloads.
- Sparse context is used only for matching.
- Named lists are matched by list-map key.
- Compound values are restored leaf-by-leaf.
- Nil and empty shapes are preserved where tests require them.
- `src` is not mutated.
- No new conversion annotations were added beyond spec/status.
- Non-conversion code does not interpret sparse payloads.

## Verification

Run for conversion changes:

```sh
GOCACHE=/tmp/dynamo-go-cache go test ./api/v1alpha1 -count=1
GOCACHE=/tmp/dynamo-go-cache go test ./api/... -count=1
GOCACHE=/tmp/dynamo-go-cache go test ./api -run TestFuzzRoundTrip -roundtrip-fuzz-iters=3000 -count=1 -v
```
