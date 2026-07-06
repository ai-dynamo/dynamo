# Validation package

## Structural validation

- Follow the Kubernetes API validation style: typed validators compose
  `field.ErrorList` through `*field.Path` and report all independent errors.
- Validation must follow the Go API type tree. Every API struct with custom
  semantics has one validator named after its exact Go type, for example
  `validateDynamoGraphDeploymentSpec`,
  `validateDynamoComponentDeploymentSharedSpec`, and
  `validateKvTransferPolicy`.
- A parent validates its own scalar fields and calls child validators in API
  declaration order. Slice paths use `Index(i)` and map paths use `Key(key)`;
  sort map keys before emitting errors.
- Aggregate independent errors. Do not use the presence of an earlier error to
  skip a validation or lookup unless that operation actually depends on the
  earlier validation succeeding.
- Seed root validation with the actual top-level field paths, such as
  `field.NewPath("metadata")` and `field.NewPath("spec")`, matching upstream
  Kubernetes validation. Do not invent a synthetic resource path or start
  child validation from an empty path.
- Treat embedded `metav1.ObjectMeta` as a structural child. Object metadata
  and annotation rules belong in `validateObjectMeta`, called with the
  `metadata` path.
- Do not name validators after a policy or implementation detail. Start with
  the lowest common API-type ancestor of the fields a rule relates, then keep
  the rule there when it coordinates siblings or needs broad aggregation.
  A child may own a rule when its invalid field and most of its logic are local
  to that type, and the required ancestor facts are cheap and clear to pass
  explicitly. Helpers for lookup, sorting, normalization, or deriving context
  are not validators and must not use a `validate` name.
- For Kubernetes-owned nested types, delegate to their Kubernetes validator at
  the exact field path instead of reimplementing their schema validation.
- Before deleting or moving an existing validation rule, inventory it and map
  it explicitly to CRD schema, CEL, or a structural validator. Presence rules
  do not replace value rules; preserve any semantic gap in code or strengthen
  the schema and prove it with schema-admission coverage.
- File ownership follows the API type being validated, not the resource that
  currently reaches it. Keep resource-specific validators in that resource's
  file and validators for types shared by multiple resources in a
  `shared_<version>.go` file.
- Keep every structural `validate<Type>` function in that type and version's
  main validation file. Put only non-validator helpers in the matching
  `<owner>_helpers.go` file; do not move shared validators into a caller's file
  for convenience.
- Keep compatibility validators for a non-storage API version in a separate
  `<owner>_<version>.go` file. Keep validators for API types shared by multiple
  resources in the corresponding `shared_<version>.go` file.
- Use one `<owner>_helpers.go` file across API versions. Helpers do not get
  version-specific files; their typed signatures already make the applicable
  API version clear.

## Validator signatures and context

- Keep structural values first, followed by `fldPath`.
- Every structural `validate...` function returns `field.ErrorList`.
- Accumulate warnings during that same structural traversal through
  core request-scoped receiver methods named `warn` and `warnf`; do not return
  warnings through every validator signature or implement a second warning
  traversal.
- The primary API value and `fldPath` passed to a validator are non-nil
  invariants and must be documented on the function. Do not add defensive nil
  checks for required validator arguments.
- For an optional child pointer, the parent checks for nil and only then calls
  the child validator. Update parents likewise handle removal before calling a
  child update validator; when an old value may legitimately be absent, state
  that explicitly in the child validator's contract.
- Pass up to three ancestor-derived contextual values as direct, typed
  parameters.
- If a validator needs four or more such values, use one final, sparse,
  type-specific options struct named after that validator's API type.
- Define an options struct only when it is needed. It contains only data the
  current API value cannot derive for itself.
- Construct each child options struct afresh at the call site. Never copy,
  embed, mutate, or extend a parent options struct. Do not use a generic,
  accumulating validation-context bag.
- Keep request-wide immutable dependencies on the validator receiver: context,
  API reader/client, feature configuration, and caller identity. The receiver
  may also carry the warnings accumulated by `warn` and `warnf` because it is
  created once per request. Do not store the current API node, field path,
  derived traversal data, or accumulated errors on the receiver.
- Only structural `validate...` functions use validation receivers. Lookup,
  derivation, sorting, normalization, and comparison helpers are standalone
  functions with explicit dependencies. Core request-accumulation methods such
  as `warn` and `warnf` are the exception and stay with their receiver.
- Dependencies required by a validation path, including its context and
  manager/client, are non-nil construction invariants. Document and satisfy
  those invariants at the boundary; do not add nil fallbacks inside helpers.
- Update validators take `new`, `old`, and `fldPath` as their structural
  inputs. Apply the same direct-context/typed-options threshold afterward.
- Update rules consider both old and new state whenever removing or replacing
  a field could bypass a guard that applies while the field is present.
- When otherwise identical Go type names from another API version need a
  distinct validator, suffix the type name with the version, for example
  `validateVolumeMountV1alpha1`; do not prefix the version.
- Use the standard `k8s.io/utils/ptr` helpers such as `ptr.Deref` and
  `ptr.Equal` for simple pointer defaults and equality. Do not add one-line
  dereference or pointer-comparison helpers.

## Errors, warnings, and compatibility

- All `validate...` functions return `field.ErrorList`; do not return `error`,
  use `errors.Join`, or build field paths with `fmt.Sprintf`.
- Use typed Kubernetes errors (`field.Required`, `field.Invalid`,
  `field.Forbidden`, `field.NotSupported`, and immutable-field validation).
  The admission boundary converts the final error list to an API invalid error.
- Keep `field.Invalid` bad values compact and non-sensitive. Never attach a
  complete resource subtree, pod template, environment list, or other
  potentially secret-bearing value; use the offending scalar or
  `field.Forbidden` when no bad value is needed.
- Emit warnings from their structural owner through the request-scoped
  receiver during the same recursion that collects errors. Keep warning
  accumulation outside `field.ErrorList`; do not add a warning-only recursion.
- Keep storage-version and compatibility-version validation recursions
  separate. Conversion is a boundary with explicit conversion and fidelity
  tests; do not build a parallel cross-version validator.

## API types shared by multiple resources

- Give each shared API type one structural validator that every resource
  recursion reuses. Do not wrap it in a stateful per-type validator object or
  duplicate it under each caller.
- Keep the full shared-type subtree, including create and update validators, in
  the matching `shared_<version>.go` file.
- Shared-type validators use a shared request receiver. Resource-specific
  receivers embed that base so they can compose shared validation without
  attaching shared methods to a resource-specific receiver.
- Declare a validation receiver alongside the primary structural validation it
  supports, with its core accumulation methods beside it. Do not create a
  standalone file solely for receiver plumbing.
- Keep only dependencies and request accumulation needed by shared validation
  on the shared receiver. Resource-only state stays on the resource-specific
  receiver.
- Rules involving parent-only data stay with the parent validator. Pass parent
  facts into a child only when the child's rule remains locally understandable
  and the extraction cost is small.

## Tests

- Keep one table-driven admission-chain test per resource. Run the request's
  source schema and CEL validation first, convert only accepted requests, then
  invoke the webhook. Group related rows inside that table with comments; do
  not create parallel matrix tests for individual validation areas.
- Add every schema-, CEL-, create-webhook, and update-webhook regression as a
  row in that table. Use a separate focused unit test only for an internal
  contract that the effective admission chain cannot reach, such as a fatal
  defensive conversion failure.
- Assert typed errors and exact field paths, aggregation of independent errors,
  and deterministic ordering. Do not assert only rendered error strings.
- Add regression coverage for every legacy rule retained during a structural
  migration. When ownership moves to schema or CEL, exercise the real schema
  admission boundary rather than treating a unit-test assumption as proof.
- Every newly added nested API type must be directly checked by its parent or
  delegated to its exact-type validator.
