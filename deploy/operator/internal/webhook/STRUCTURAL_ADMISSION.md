# Structural admission

Structural admission is the operator's convention for custom-resource webhook
defaulting and validation. Both traversals mirror the Go API type tree, keep one
semantic owner for each rule, and leave the admission handler as a narrow request
boundary.

## Shared contract

- Keep generated CRD schema and CEL, mutating admission, validating admission,
  and reconciliation as distinct layers. Put each default or invariant in the
  earliest layer that has all required information, and do not duplicate it in
  another layer as a fallback.
- Mirror the Go API type tree in both defaulting and validation. A parent owns
  rules coordinating its scalar fields or children and delegates local child
  behavior to functions named after the exact child Go type.
- Keep admission handlers narrow: verify the admission GVK and concrete Go type,
  obtain request-scoped inputs, call one root structural traversal, and return
  the result. Do not put recursive API policy in handler methods.
- Treat conversion as an explicit boundary. Register admission for the canonical
  hub representation and dispatch foreign-version behavior from the converted
  object shape, including version envelopes. Do not dispatch semantic behavior
  from the client's original `requestKind` alone.
- Preserve explicit user values. Defaulting fills only absence defined by the API;
  validation rejects invalid explicit values. Neither layer silently rewrites an
  explicit value to make it valid.
- Keep request-wide immutable dependencies on the admission receiver. Do not put
  the current API node, traversal path, derived scratch values, accumulated
  mutations, or accumulated errors in a generic context bag.

## Defaulting migration status

Keep this table current whenever a resource defaulter is migrated.

| Resource | Status | Current implementation |
| --- | --- | --- |
| `DynamoGraphDeployment` | Transitional | `defaulting/dynamographdeployment_handler.go` plus shared helpers |
| `DynamoComponentDeployment` | Transitional | `defaulting/dynamocomponentdeployment_handler.go` plus shared helpers |
| `DynamoGraphDeploymentRequest` | Handler-local | `defaulting/dynamographdeploymentrequest_handler.go` |

A defaulter is structural only after its handler delegates to a root function and
nested API types are defaulted by their exact-type functions.

## Structural defaulting

- Name each structural function after the exact Go API type it defaults, for
  example `defaultDynamoGraphDeploymentSpec`,
  `defaultDynamoComponentDeploymentSharedSpec`, and `defaultComponentRoleSpec`.
- A parent defaults its own scalar fields and calls child defaulters in API
  declaration order. Traverse every list element, including elements introduced
  by an UPDATE. Sort map keys before behavior depends on traversal order.
- Structural defaulters intentionally mutate only the API value passed to them.
  They must be idempotent: a second pass over the result produces no additional
  mutation.
- Default only a field whose API-defined absence is observable. Preserve explicit
  zero, `false`, empty collections, empty objects, and empty strings whenever they
  are valid user choices. Use pointer presence or an explicit presence signal when
  zero values are meaningful.
- Apply operation-independent defaults on CREATE and UPDATE. A CREATE-only or
  UPDATE-only default is exceptional: document the API reason beside the rule and
  cover the opposite operation in tests.
- For an optional child pointer, the parent checks for nil and creates the child
  only when the child object itself has a documented default. Otherwise it leaves
  the pointer absent and skips child traversal.
- Keep cross-field defaults at the lowest common API-type ancestor. Pass a small
  typed ancestor fact into a child only when the child's local default remains
  understandable in isolation.
- Resolve environment-dependent inputs before applying them. A resolver is
  read-only and returns focused typed data; the structural defaulter decides which
  absent API fields receive that data. Do not mix Kubernetes discovery, policy,
  recursive traversal, and mutation in one helper.
- Preserve explicit values across representations. When native parameters or a
  foreign-version envelope can own a field, determine effective ownership before
  applying typed defaults and never overwrite the authoritative representation.
- Defaulting must not compensate for an invalid explicit value or an unresolved
  required value. Leave those states visible so validating admission reports the
  exact problem.

## Defaulting signatures and context

- Keep the API value first in a structural `default...` signature, followed by
  direct typed context or one final sparse options struct.
- Pure structural defaults return no value. Return an error only when the rule
  performs an operation that can genuinely fail; do not turn ordinary absence
  into an error or silently swallow a failed required resolution.
- Pass up to three ancestor-derived contextual values as direct typed parameters.
  At four or more, use one sparse type-specific options struct containing only
  facts the current API value cannot derive.
- Construct child options afresh at the call site. Never copy, embed, mutate, or
  extend a parent options struct, and do not introduce a generic defaulting context.
- Pass the admission operation only to the smallest function whose documented
  semantics actually differ between CREATE and UPDATE.
- Keep request-wide clients, readers, feature configuration, operator version,
  and caller identity on the root defaulter receiver. Prefer standalone functions
  for pure nested defaults and explicit resolver interfaces for external reads.
- Defaulting mutation is part of the function contract, but helper reads and
  resolvers must not mutate their inputs.
- Use standard `k8s.io/utils/ptr` helpers for pointer defaults and equality. Do not
  add one-line dereference, presence, or pointer-comparison helpers.

## Defaulting files and shared types

- Keep registration, request decoding, GVK checks, and the call to the root
  traversal in `<owner>_handler.go`.
- Keep every structural `default<Type>` function in the type and version's main
  `<owner>.go` file. Put only lookup, derivation, sorting, normalization, and
  resolver helpers in `<owner>_helpers.go`.
- Put defaults for API types shared by multiple resources in the corresponding
  `shared_<version>.go` file and call the same function from every resource
  traversal. Do not duplicate a shared default under each handler.
- Keep compatibility behavior for a non-storage API version in a separate
  `<owner>_<version>.go` file. Conversion fidelity remains independently tested.
- Use one `<owner>_helpers.go` file across API versions; typed signatures make the
  applicable version explicit.

## Default ownership

- Prefer CRD schema defaults for unconditional, context-free API defaults when
  applying them before admission does not destroy field-presence semantics needed
  by later rules.
- Use mutating admission for conditional, cross-field, version-aware, or
  environment-derived defaults that must be persisted in the API object.
- Use reconciliation only for runtime state that cannot be known at admission.
  Do not make a controller repair an omitted field that admission promises to
  persist.
- Assign every default one canonical owner and document its CREATE/UPDATE and
  conversion behavior. Never apply the same default independently in CRD schema,
  admission, and reconciliation.

## Defaulting tests

- Keep one table-driven admission-chain test per resource. Exercise the generated
  source-version schema defaults first, convert through the production path when
  versions differ, then invoke the public mutating webhook handler.
- Cover CREATE and UPDATE explicitly. For every operation-specific default, include
  the opposite operation and prove that its behavior is intentional.
- Prove idempotence by applying defaulting twice and comparing the complete object
  after the first and second pass.
- Cover absent values, every meaningful explicit zero value, explicit non-default
  values, empty nested objects, newly added list elements, and shared nested types.
- Cover conversion in both directions when a resource has multiple served versions.
  Assert that native and envelope-backed representations receive equivalent semantic
  defaults and round-trip without losing explicit values.
- For environment-derived defaults, test resolution separately from structural
  application, then cover the real admission composition with the production
  resolver boundary. Prove that partial explicit input resolves only missing fields.
- Assert the complete mutated object or a focused typed subtree. Do not test only a
  rendered patch or log message, and do not duplicate behavior already covered by
  the resource's admission table.

## Validation migration status

Keep this table current whenever a resource validator is migrated.

| Resource | Status | Canonical implementation |
| --- | --- | --- |
| `DynamoGraphDeployment` | Structural | `dynamographdeployment.go`, `dynamographdeployment_v1alpha1.go` |
| `DynamoComponentDeployment` | Structural | `dynamocomponentdeployment.go`, `dynamocomponentdeployment_v1alpha1.go` |
| `DynamoGraphDeploymentRequest` | Structural | `dynamographdeploymentrequest.go` |
| `DynamoModel` | Structural | `dynamomodel.go` |

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

## API-version and update ownership

- Run admission in production order: validate the submitted source-version
  schema and CEL, convert only accepted requests through the production path,
  then invoke the public create or update webhook handler.
- Validate shared semantics once in the storage-version recursion when typed
  conversion preserves every required input and the reported field path has
  the same meaning in both versions. Do not repeat that rule in the
  compatibility-version recursion.
- Keep compatibility-version validation only for rules that cannot be
  faithfully enforced after conversion, such as source-only fields,
  source-schema gaps, source-specific field paths, or source-specific warnings.
- Determine version ownership from the typed conversion contract, not from
  private preservation annotations. Validation code must never inspect, decode,
  branch on, or expose conversion payload annotations or payload shapes.
- Treat represented live values as authoritative, including nil, empty, false,
  and zero. Compatibility validation must not restore, supplement, or resurrect
  a represented field from stale preservation data.
- Run stateless new-state validation for both creates and updates. Update
  validators add only rules that genuinely depend on old state, such as
  immutability, ratcheting, ownership transitions, or removal bypasses; do not
  duplicate a static rule behind field-change or tuple-change detection.
- When a newly introduced static rule intentionally ratchets pre-existing
  violations, keep its decision in one shared function. The update adapter may
  suppress the new-state error only when the complete normalized rule inputs
  and the invalid field value are identical in the old and new objects. Do not
  approximate equality with a list of fields that happen to trigger the rule;
  cover an unrelated edit, every contract-input transition, repair, and
  rollback through the production admission chain.
- Before adding version-specific or update-specific validation, trace the
  public admission call graph and existing typed conversion. Add the rule at
  the lowest single structural owner that all applicable paths already reach.

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

## Validation tests

- Keep one table-driven admission-chain test per resource. Reuse the shared DGD/DCD
  harness in `shared_test.go`: run the generated source-version schema and CEL
  validators first, convert only accepted requests through the production path,
  then invoke the public create or update webhook handler. Extend the shared harness
  for new resources or versions; do not create a resource-local substitute.
- Group related rows inside that table with comments; do not create parallel matrix
  tests for individual validation areas.
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
