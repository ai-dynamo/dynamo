# Structural defaulting

Mutating admission mirrors the Go API type tree, keeps one semantic owner for
each default, and leaves the webhook handler as a narrow request boundary.

## Contract

- Keep CRD schema defaults, mutating admission, and reconciliation as distinct
  layers. Put each default in the earliest layer that has all required information,
  and do not duplicate it in another layer as a fallback.
- Mirror the Go API type tree. A parent owns defaults coordinating its scalar
  fields or children and delegates local child behavior to functions named after
  the exact child Go type.
- Keep webhook handlers narrow: verify the admission GVK and concrete Go type,
  obtain request-scoped inputs, call the root `default<Type>` function, and return.
  Put each field default in the structural function for the API type that declares
  that field.
- Treat conversion as an explicit boundary. Register admission for the canonical
  hub representation and dispatch foreign-version behavior from the converted
  object shape, including version envelopes. Do not dispatch semantic behavior
  from the client's original `requestKind` alone.
- Preserve explicit user values. Fill only absence defined by the API and never
  silently rewrite an explicit value to make it valid.

## Migration status

Keep this table current whenever a resource is migrated.

| Resource | Status | Current implementation |
| --- | --- | --- |
| `DynamoGraphDeployment` | Transitional | `defaulting/dynamographdeployment_handler.go` plus shared helpers |
| `DynamoComponentDeployment` | Transitional | `defaulting/dynamocomponentdeployment_handler.go` plus shared helpers |
| `DynamoGraphDeploymentRequest` | Handler-local | `defaulting/dynamographdeploymentrequest_handler.go` |

A resource reaches Structural status only after its handler delegates to a root
function and nested API types are handled by their exact-type functions.

## Structure

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
- Do not compensate for an invalid explicit value or an unresolved required value.
  Leave those states visible so validating admission reports the exact problem.

## Signatures and context

- Keep the API value first in a structural `default...` signature, followed by
  direct typed context or one final sparse options struct.
- Pure structural defaults return no value. Return an error only when the rule
  performs an operation that can genuinely fail; do not turn ordinary absence
  into an error or silently swallow a failed required resolution.
- Pass up to three ancestor-derived contextual values as direct typed parameters.
  At four or more, use one sparse type-specific options struct containing only
  facts the current API value cannot derive.
- Construct child options afresh at the call site. Never copy, embed, mutate, or
  extend a parent options struct, and do not introduce a generic context bag.
- Pass the admission operation only to the smallest function whose documented
  semantics actually differ between CREATE and UPDATE.
- Request-wide clients, readers, feature configuration, operator version, and
  caller identity may live on the root defaulter receiver or in a small typed
  request context. Prefer standalone functions for pure nested defaults and explicit
  resolver interfaces for external reads.
- Mutation is part of the structural function contract, but helper reads and
  resolvers must not mutate their inputs.
- Use standard `k8s.io/utils/ptr` helpers for pointer defaults and equality. Do not
  add one-line dereference, presence, or pointer-comparison helpers.

## Files and shared types

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

## Ownership

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

## Tests

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
