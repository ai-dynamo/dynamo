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

Preservation annotations are allowed to store coarse snapshots, including full
spec/status payloads, but restore logic must use them only as old-value caches.
Annotations must not overlay, underlay, or otherwise override fields that are
representable by the API version being converted from.

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

Conversion helpers should generally follow a `src`, `dst`, `restored`, `save`,
`ctx` shape:

```go
func convert_v1beta1_Foo_To_v1alpha1_Foo(
	src *v1beta1.Foo,
	dst *Foo,
	restored *Foo,
	save *v1beta1.Foo,
	ctx fooConversionContext,
) error {
	// Convert representable fields from src to dst.

	// Restore target-only fields that src cannot represent.

	// Save source-only fields that dst cannot represent.

	return nil
}
```

The parameters have fixed meaning:

- `src`: live source object. It is authoritative for every field representable
  by the source version, including nil, empty, and zero values.
- `dst`: converted target object.
- `restored`: typed target-version data decoded from preservation annotations.
  It may restore only target fields that `src` cannot represent.
- `save`: typed source-version data that will be encoded into preservation
  annotations. It may contain only source fields that `dst` cannot represent,
  plus matching keys needed to locate those fields later.
- `ctx`: typed high-level context needed by lower-level helpers.

This mirrors the conversion-gen style, but because these conversions are
handwritten, context should be typed instead of `any`. Avoid one global context
type; prefer small family-specific contexts such as `dgdConversionContext`,
`dcdConversionContext`, `dgdrConversionContext`, and
`sharedSpecConversionContext`. Context should carry only cross-cutting
information that leaves cannot derive from their local `src/restored/save`
arguments.

## Preservation Annotations

Preservation annotations exist only to make unrepresentable data survive a
round trip through a version that cannot express it natively.

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
save.PodTemplate = src.PodTemplate
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
    nvidia.com/dgd-hub-spec: '{... "securityContext":{"runAsNonRoot":true} ...}'
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
- DGDR profiling blob merge/prune logic.
- DGDR status phase/fingerprint behavior.
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
- Are nil and empty shapes preserved where round-trip tests require them?

## Mutability

Conversion functions must not mutate their input object.

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

## Verification

Each conversion change should run:

```sh
GOCACHE=/tmp/dynamo-go-cache go test ./api/v1alpha1 -count=1
GOCACHE=/tmp/dynamo-go-cache go test ./api/... -count=1
GOCACHE=/tmp/dynamo-go-cache go test ./api -run TestFuzzRoundTrip -roundtrip-fuzz-iters=3000 -count=1 -v
git diff --check
docker buildx build --platform linux/arm64 --target linter --progress=plain --build-context snapshot=../snapshot .
```
