# Conversion Invariants and Implementation Guide

This document defines the invariants expected from conversions between the
spoke API versions and the hub API version. The fuzz round-trip tests in this
directory should enforce these rules.

## Goals

- Make live source fields visibly authoritative.
- Restore only fields that the source API version cannot represent.
- Save only fields that the target API version cannot represent.
- Keep preservation annotations sparse, so stale snapshots cannot quietly
  override later live edits.
- Keep representability decisions explicit and reviewable in typed conversion
  helpers.
- Use generics only for mechanical plumbing, not for conversion policy.

## Source of Truth

Live object fields are the source of truth.

Preservation annotations should store sparse typed payloads. Some legacy
annotations may still contain coarse snapshots, including full spec/status
payloads, but restore logic must use them only as old-value caches. Annotations
must not overlay, underlay, or otherwise override fields that are representable
by the API version being converted from.

This is about representability in the version's schema, not whether a specific
object currently represents the field with a non-zero value. If a field can be
expressed natively by the source version, the source-version field is
authoritative, including its zero, nil, or empty value.

The conversion shape should be:

```text
semantic = convert(live fields)
preserved = decode annotation, if present

for each known unrepresentable field:
  find the matching live subobject
  copy only that unrepresentable field from preserved into semantic

return semantic
```

This is a semantic description, not a required top-level control flow.
Recursive helpers may express the same invariant directly, for example:

```text
convertFooFrom(&src.Foo, &dst.Foo, &preserved.Foo)
```

Such helpers must still treat `src.Foo` as authoritative for every field
representable by the source version, and use `preserved.Foo` only for fields
that the source version cannot represent.

The conversion shape must not be:

```text
return overlay(preserved, semantic)
return overlay(semantic, preserved)
return preserved if annotation exists
```

## Structural Helpers

Conversion policy belongs in API conversion code. Controllers, reconcilers, and
internal packages must not implement or wrap conversion. Temporary callers that
need converted data must call the same exported structural converters used by
the real conversion path. Read-only getters may live elsewhere; converters may
not.

Converters for API structs are exported from the `v1alpha1` package and are
named after the v1alpha1 type:

```go
func ConvertFrom<TypeName>(src *TypeName, dst *v1beta1.HubType, ...)
func ConvertTo<TypeName>(src *v1beta1.HubType, dst *TypeName, ...)
```

Do not include `V1alpha1` in the name, and do not add wrapper functions with
alternate names.

The parameter order is fixed:

- `src`: live source object; authoritative for every representable field,
  including nil, empty, and zero values.
- `dst`: caller-owned, non-nil destination object.
- `restored`: optional target-version data decoded from preservation
  annotations; use it only for target-only fields.
- `save`: optional source-version data to encode into preservation annotations;
  write only source-only fields and keys needed to find them later.
- `ctx`: optional typed context, always last.

`restored`, `save`, and `ctx` are included only when the converter needs them.
Return `error` only when conversion can fail.

Exported conversion-only helper types, such as typed contexts needed by
exported converter signatures, must opt out of Kubernetes object generation and
API reference docs.

Callers perform nil, disabled, zero, or absence checks before calling a
converter and allocate nested `dst` objects explicitly. Converters do not accept
`**T` and do not encode absence by setting `dst` to nil.

Converters must mirror API type structure. A converter handles the fields of
its own type and delegates each converted nested API struct to that nested
struct's `ConvertFrom<SubStruct>` or `ConvertTo<SubStruct>` function. Do not
convert multiple nested API layers at once in a parent converter.

Local helpers are allowed only for conversion-private implementation details
that are not API-struct converters: `restore*` readers, `save*` writers,
`ensure*` allocators for sparse save payloads, side-effect-free predicates, and
field-group projections such as podTemplate composition/decomposition. Use
`restore` when reading `restored` and `save` when writing `save`.

Converters may share data between `src` and `dst`, but they must never mutate
`src`. Deep-copy only when the converter will mutate the copy or when separate
ownership is required.

## Preservation Annotations

Preservation annotations exist only to make unrepresentable data survive a
round trip through a version that cannot express it natively.

Preservation annotations are private to conversion code. Only API conversion
code and its conversion tests may know their keys or payload shape. Controllers,
reconcilers, webhooks, internal helper packages, and general API helpers must
not read, write, delete, filter, decode, encode, branch on, or expose them.
Keep annotation key constants unexported and local to conversion files. Do not
define shared constants, prefixes, string literals, getters, predicates, or
wrapper helpers for conversion annotations outside conversion code.

Non-conversion code may copy Kubernetes metadata opaquely as part of normal
object handling, but it must not identify or interpret individual conversion
annotations. If non-conversion code needs data that currently appears only in a
preservation annotation, add or call a real conversion helper instead of
decoding the annotation.

It is acceptable for the annotation payload to include representable fields as
context. Restore code must explicitly ignore those fields unless they are needed
only to locate the unrepresentable data.

For compound objects with mixed representability, such as pod templates or job
specs, restore code must copy individual unrepresentable leaves. It must not
restore the whole compound object and then patch represented fields over it.

Save payloads should be sparse by construction. A helper should write only the
source-version fields that the target version cannot represent:

```go
// Save source-only fields that dst cannot represent.
save.FrontendSidecar = src.FrontendSidecar
save.PodTemplate = sparseHubOnlyPodTemplateRemainder(src.PodTemplate, projected)
if experimentalIsHubOnlyShape(src.Experimental) {
	save.Experimental = src.Experimental
}
```

After helper execution, callers should skip empty save objects:

```go
if !dcdHubSpecSaveIsZero(&save) {
	encodeDCDSaveAnnotation(dst, &save)
}
```

Typed zero checks are preferred over broad reflection when they keep the
preserved shape clearer. `apiequality.Semantic.DeepEqual` is appropriate for
Kubernetes API structs when nil/empty semantic equality is intended.

## Named Lists

For list-map fields, preserved data must be matched by the declared list-map
key, not by slice index.

For example, `v1beta1.spec.components[]` data is matched by `name`. If the live
object no longer contains that name, the preserved subobject is stale and must
be ignored. If a live object introduces a new name, it gets no preserved data
unless the annotation has a matching key.

Saved entries for named lists must include the list-map key. For example, a
saved DGD component needs `ComponentName` so the preserved fields can be
matched back to the live component later.

## Origin Hints

Some annotations record that a field was generated by conversion from another
version. These annotations are hints for lossless no-op round trips, not sources
of truth.

If a later edit changes source-version-representable semantics, the converted
source-version object must change visibly.

If a later edit changes only target-version-only semantics, the converted
source-version object may look unchanged, but its preservation annotation must
change so converting back restores the edited target-version-only data.

If a later edit changes both, the converted source-version object must change
visibly for the representable part, and the annotation must preserve the
target-version-only remainder.

The bug class to avoid is letting a stale origin annotation restore the old
generated value after a live edit changed source-version-representable
semantics.

Example: v1alpha1 can represent the frontend sidecar image, but cannot
represent every field of the generated v1beta1 sidecar container.

```yaml
# v1alpha1 input
spec:
  services:
    epp:
      frontendSidecar:
        image: frontend:v1
```

Converting to v1beta1 generates a sidecar container and records an origin hint:

```yaml
metadata:
  annotations:
    nvidia.com/dgd-comp-epp-frontend-sidecar-origin: '{"image":"frontend:v1"}'
spec:
  components:
  - name: epp
    frontendSidecar: sidecar-frontend
    podTemplate:
      spec:
        containers:
        - name: main
        - name: sidecar-frontend
          image: frontend:v1
```

If v1beta1 edits the image, v1alpha1 must change visibly:

```yaml
# edited v1beta1
containers:
- name: sidecar-frontend
  image: frontend:v2

# converted v1alpha1
frontendSidecar:
  image: frontend:v2
```

If v1beta1 edits only a container field that v1alpha1 cannot represent, the
visible v1alpha1 field may stay the same, but preservation must carry the
v1beta1-only data:

```yaml
# edited v1beta1
containers:
- name: sidecar-frontend
  image: frontend:v1
  securityContext:
    runAsNonRoot: true

# converted v1alpha1
frontendSidecar:
  image: frontend:v1
metadata:
  annotations:
    nvidia.com/dgd-spec: '{... "securityContext":{"runAsNonRoot":true} ...}'
```

## Generics

Use generics for boring mechanics only.

Good candidates:

- Decode/encode typed annotation payloads.
- Test whether a typed save payload is empty.
- Convert a list-map into a keyed map.
- Convert a keyed map into a deterministic sorted list.
- Match restored/save child objects by key.

Bad candidates:

- Deciding which fields are representable.
- PodTemplate/main-container semantic origin logic.

Generic helpers should reduce repeated mechanics without obscuring conversion
policy.

## Review Checklist

For each helper:

- Does every represented field come from `src`?
- Does every restored field come only from `restored` and only when the source
  version cannot represent it?
- Does every saved field represent data that `dst` cannot express?
- Are named-list fields matched by their list-map key, never by index?
- Is the save payload sparse?
- Are origin annotations used only as hints, not as shortcuts?
- Are conversion annotations private to conversion code and absent from
  controllers/internal helpers?
- Are nil and empty shapes preserved where round-trip tests require them?

## Mutability

Conversion functions must not mutate their input object.

`src` and `dst` may share backing data when the converted value can be assigned
as-is. Do not add `DeepCopy` calls only because a field came from `src`.
Aliasing is acceptable when the conversion code treats the shared value as
read-only after assignment.

Use `DeepCopy` only when the conversion needs an independently mutable value:
for example, before editing, appending to, sorting, normalizing, clearing,
merging into, or otherwise mutating a value that came from `src`. This applies
equally to direct mutation through `src` and indirect mutation through a `dst`
field that aliases `src`; both violate the "do not mutate input" invariant.

The round-trip fuzz tests snapshot inputs through YAML before conversion because
marshalling observes the actual in-memory shape, including aliasing bugs that a
plain structural comparison may miss.

## Fuzz Test Expectations

The regular round-trip tests verify unchanged objects:

```text
hub -> spoke -> hub
spoke -> hub -> spoke
```

The mutability round-trip test verifies stale annotation behavior:

```text
fuzz in
convert to other
mutate other without deleting preservation annotations
convert other -> in -> other2
compare other and other2, ignoring only preservation annotations
```

The mutation step must update nested existing objects, including elements inside
arrays and slices. This is what exposes stale annotation overlays on deep
fields.

## Adding v1beta1 Fields

In Kubernetes conversion, `v1beta1` is the hub version and `v1alpha1` is the
spoke version. Objects may be converted in either direction:

```text
v1beta1 hub -> v1alpha1 spoke
v1alpha1 spoke -> v1beta1 hub
```

The conversion helpers use these names:

- `src`: the live object we are converting from now. It is the source of truth.
- `dst`: the object we are building.
- `restored`: older `dst`-version data decoded from preservation annotations.
- `save`: `src`-version data that `dst` cannot represent directly and that
  will be written into `dst.metadata.annotations` for a future conversion.
- `ctx`: extra typed context passed to nested helpers.

`TestV1Beta1ConversionFieldSetIsAcknowledged` is a schema-change tripwire. It
reflects over the v1beta1 spec/status structs covered by this conversion
cleanup and compares their JSON field paths to a checked-in field-set snapshot.
It currently covers DGD, DCD, and DGDSA. DGDR already shipped with conversion
annotations, so changes to its annotation/storage contract should be handled in
a separate compatibility-focused PR.

When a v1beta1 field is added, this test should fail before the field can be
silently dropped by conversion. Treat that failure as a prompt to make an
explicit conversion decision:

- Native mapping: add the field to the relevant `convert*ToHub` /
  `convert*FromHub` helper, sourced from the live `src` value.
- Source-only preservation: if the target version cannot represent the field,
  add it to the sparse `save` payload. The payload is serialized into an
  annotation on `dst`. On a later conversion back, restore that field from
  `restored` only if the then-current `src` version still cannot represent it.
- Derived or lossy mapping: document the semantic mapping and add a targeted
  regression test for the lossy shape.
- Intentional drop: document why the field is not part of the conversion
  contract and add a targeted test if the omission is observable.

For example, if a new v1beta1 field has no v1alpha1 field, then on
`v1beta1 -> v1alpha1` the converter should copy it into `save` and encode that
save payload into a v1alpha1 annotation. On `v1alpha1 -> v1beta1`, the
converter may restore that field from `restored`. If v1alpha1 later grows a
native field for the same concept, the live `src` field must win over any stale
annotation.

Then add or update a focused conversion test for the new field. Fuzz is a broad
backstop; the field-set test tells reviewers that the schema changed, while the
focused test proves the chosen conversion policy.

After the conversion code and tests are updated, refresh the
`knownV1Beta1ConversionFieldSet` snapshot in
`conversion_field_coverage_test.go` by applying the diff from the failing test
output. Do not update the snapshot before the conversion decision is
implemented.

The field-set snapshot walks v1beta1 API structs and v1beta1-owned nested
types. Kubernetes/library structs such as `corev1.PodTemplateSpec` are treated
as leaves; additions inside those upstream types are covered by the existing
pod-template/job conversion tests rather than by this schema tripwire.

## Verification

Each conversion change should run:

```sh
GOCACHE=/tmp/dynamo-go-cache go test ./api/v1alpha1 -count=1
GOCACHE=/tmp/dynamo-go-cache go test ./api/... -count=1
GOCACHE=/tmp/dynamo-go-cache go test ./api -run TestFuzzRoundTrip -roundtrip-fuzz-iters=3000 -count=1 -v
git diff --check
docker buildx build --platform linux/arm64 --target linter --progress=plain --build-context snapshot=../snapshot .
```
