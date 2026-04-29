# Conversion Invariants

This document defines the invariants expected from conversions between the
spoke API versions and the hub API version. The fuzz round-trip tests in this
directory should enforce these rules.

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

The semantic conversion shape is:

```text
dst = convert(live fields)
restore = decode target-version annotation, if present

for each known unrepresentable field:
  find the matching live subobject
  copy only that unrepresentable field from restore into dst

save source-version fields that the target version cannot represent

return dst and save
```

This is a semantic description, not a required top-level control flow.
Recursive helpers may express the same invariant directly, for example:

```text
convertFoo(&src.Foo, &dst.Foo, &restore.Foo, &save.Foo)
```

Such helpers must still treat `src.Foo` as authoritative for every field
representable by the source version, use `restore.Foo` only for target fields
that the source version cannot represent, and write `save.Foo` only for source
fields that the target version cannot represent.

The conversion shape must not be:

```text
return overlay(preserved, semantic)
return overlay(semantic, preserved)
return preserved if annotation exists
```

## Conversion Helper Pattern

Conversion helper names should follow the converted type names and direction.
The shape mirrors `conversion-gen` naming, but uses private helper names:

```go
func convert_v1beta1_DynamoGraphDeployment_To_v1alpha1_DynamoGraphDeployment(
    src *v1beta1.DynamoGraphDeployment,
    dst *v1alpha1.DynamoGraphDeployment,
    restore *v1alpha1.DynamoGraphDeployment,
    save *v1beta1.DynamoGraphDeployment,
) error
```

The parameters have fixed meaning:

- `src` is the live source-version object and is authoritative for every field
  that the source version can represent.
- `dst` is the converted target-version object.
- `restore` is typed target-version data decoded from preservation annotations.
  It may restore only target fields that `src` cannot represent.
- `save` is typed source-version data that will be encoded into preservation
  annotations. It may save only source fields that `dst` cannot represent.

Root conversion functions should own annotation mechanics:

- Decode existing preservation annotations into typed `restore` objects.
- Allocate typed `save` objects.
- Call typed conversion helpers.
- Scrub stale internal annotations.
- Encode `save` back into annotations only when it contains meaningful
  unrepresentable data.

Nested conversion helpers should not know annotation names. They should receive
the corresponding `restore` and `save` subobjects from their caller.

Each helper should keep the representability decisions visible in three
sections:

```go
// Convert representable fields from src to dst.
dst.Spec.Foo = src.Spec.Foo

// Restore target-only fields that src cannot represent.
if restore != nil {
    dst.Spec.LegacyOnly = restore.Spec.LegacyOnly
}

// Save source-only fields that dst cannot represent.
save.Spec.HubOnly = src.Spec.HubOnly
```

Avoid hiding these decisions in broad `restoreFoo` or `preserveFoo` helpers.
Small helpers are appropriate for repeated mechanics such as decoding
annotations, allocating keyed child objects, checking whether `save` is empty,
or converting low-level Kubernetes field shapes.

## Preservation Annotations

Preservation annotations exist only to make unrepresentable data survive a
round trip through a version that cannot express it natively.

It is acceptable for the annotation payload to include representable fields as
context. Restore code must explicitly ignore those fields unless they are needed
only to locate the unrepresentable data.

For compound objects with mixed representability, such as pod templates or job
specs, restore code must copy individual unrepresentable leaves. It must not
restore the whole compound object and then patch represented fields over it.

## Named Lists

For list-map fields, preserved data must be matched by the declared list-map
key, not by slice index.

For example, `v1beta1.spec.components[]` data is matched by `name`. If the live
object no longer contains that name, the preserved subobject is stale and must
be ignored. If a live object introduces a new name, it gets no preserved data
unless the annotation has a matching key.

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
