# Structural defaulting

Mutating admission mirrors the Go API type tree, keeps one semantic owner for
each default, and leaves the webhook handler as a narrow request boundary.

## Contract

- Mirror the Go API type tree. A parent owns defaults coordinating its scalar
  fields or children and delegates local child behavior to functions named after
  the exact child Go type.
- Keep webhook handlers narrow: verify the admission GVK and concrete Go type,
  obtain request-scoped inputs, call the root `default<Type>` function, and return.
  Put each field default in the structural function for the API type that declares
  that field.
- Apply defaults to the canonical API representation whenever conversion preserves
  the required field presence and semantics. Version-specific behavior is an
  exception for information that conversion cannot represent.
- Preserve explicit user values. Fill only absence defined by the API and never
  rewrite an explicit value to make it valid.
- Keep authoritative server-owned overwrites in the resource-root
  `default<Type>` function. Document why the server owns the field, when it
  replaces a submitted value, and cover that overwrite through the production
  admission chain.
- Do not infer or track value provenance. Treat the decoded object as authoritative:
  a user-supplied value and the same value applied by CRD schema defaulting are
  indistinguishable. Do not inspect the raw admission request, compare against a
  known default, consult annotations, or use another side channel to reconstruct
  whether a field was user supplied. If a rule needs to observe absence, leave the
  field unset until the owning layer applies the default or model presence explicitly
  in the API.

## Ownership

- Prefer CRD schema defaults for unconditional, context-free API defaults when
  applying them before admission preserves every presence distinction needed by
  later rules.
- Use mutating admission for conditional, cross-field, version-aware, or
  environment-derived defaults that must be persisted in the API object.
- Use reconciliation only for runtime state that cannot be known at admission.
  Do not make a controller repair an omitted field that admission promises to
  persist.
- Assign every default one owner. Do not apply the same default independently in
  CRD schema, mutating admission, and reconciliation.

## Structure

- Name each structural function after the exact Go API type it defaults, for
  example `defaultDynamoGraphDeploymentSpec`,
  `defaultDynamoComponentDeploymentSharedSpec`, and `defaultComponentRoleSpec`.
- A parent defaults its own scalar fields and calls child defaulters in API
  declaration order. Traverse every list element, including elements introduced
  by an UPDATE.
- Structural functions must be idempotent: applying them twice produces no
  additional mutation.
- Default only a field whose API-defined absence is observable. Preserve explicit
  zero, `false`, empty collections, empty objects, and empty strings whenever they
  are valid user choices. Use pointer presence or an explicit presence signal when
  zero values are meaningful.
- Apply operation-independent defaults on CREATE and UPDATE. A CREATE-only or
  UPDATE-only default is exceptional: document the API reason and cover the
  opposite operation in tests.
- For an optional child pointer, the parent checks for nil and creates the child
  only when the child object itself has a documented default. Otherwise it leaves
  the pointer absent and skips child traversal.
- Keep cross-field defaults at the lowest common API-type ancestor of the fields
  they coordinate.
- Separate environment-dependent resolution from mutation. A resolver reads
  external state and returns focused typed data; the structural function decides
  which absent API fields receive it.
- Do not compensate for an invalid explicit value or an unresolved required value.
  Leave those states visible so validating admission reports the exact problem.

## Signatures and files

- Keep registration, request decoding, GVK checks, and the call to the root
  traversal in `<owner>_handler.go`.
- Keep the API value first in a structural `default...` signature. Pass additional
  context as direct typed values or one type-specific options struct.
- Pure structural functions return no value. Return an error only for an operation
  that can genuinely fail; ordinary absence is not an error.
- Pass the admission operation only to the smallest function whose documented
  semantics differ between CREATE and UPDATE.
- Helpers and external resolvers must not mutate their inputs.
- Use standard `k8s.io/utils/ptr` helpers for pointer defaults and equality. Do not
  add one-line dereference, presence, or pointer-comparison helpers.
- Keep structural functions for the canonical API version in `<owner>.go` and
  lookup, derivation, sorting, normalization, and resolver helpers in
  `<owner>_helpers.go`.
- Put defaults for API types shared by multiple resources in `shared_<version>.go`
  and call the same function from every resource traversal.
- Put exceptional version-specific behavior in `<owner>_<version>.go` and document
  why conversion cannot carry the required semantics. Keep canonical function and
  options names unsuffixed; suffix non-canonical names with the Go API version, for
  example `defaultDynamoGraphDeploymentSpecV1alpha1` and
  `dynamoGraphDeploymentV1alpha1DefaultOptions`.

### Examples

Representative signatures, with function bodies omitted:

```go
// dynamographdeployment_handler.go
func (d *DGDDefaulter) Default(ctx context.Context, obj runtime.Object) error

// dynamographdeployment.go
func defaultDynamoGraphDeployment(
    dgd *nvidiacomv1beta1.DynamoGraphDeployment,
    opts dynamoGraphDeploymentDefaultOptions,
)

func defaultDynamoGraphDeploymentSpec(
    spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
    operation admissionv1.Operation,
)

// dynamographdeployment_v1alpha1.go
func defaultDynamoGraphDeploymentV1alpha1(
    dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
    opts dynamoGraphDeploymentV1alpha1DefaultOptions,
)

func defaultDynamoGraphDeploymentSpecV1alpha1(
    spec *nvidiacomv1alpha1.DynamoGraphDeploymentSpec,
    operation admissionv1.Operation,
)
```

## Tests

- Keep one table-driven admission-chain test per resource. Exercise source-version
  schema defaults first, convert through the production path when versions differ,
  then invoke the public mutating webhook handler.
- Prove idempotence by applying defaulting twice and comparing the complete object
  after the first and second pass.
- Cover absent values, meaningful explicit zero values, explicit non-default values,
  empty nested objects, newly added list elements, and shared nested types.
- Prove that a user-supplied value and an identical value applied by CRD schema
  defaulting produce the same webhook result.
- Cover CREATE and UPDATE explicitly when behavior differs by operation.
- When a resource has multiple served versions, cover conversion in both directions
  and assert that explicit values survive the round trip.
- For environment-derived defaults, test resolution separately from structural
  application and cover their composition through the production admission path.
- Assert the complete mutated object or a focused typed subtree. Do not test only a
  rendered patch or log message.
