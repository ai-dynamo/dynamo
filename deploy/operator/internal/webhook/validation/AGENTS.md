# Validation package

Before changing validation code or tests in this package, read
[`STRUCTURAL_VALIDATION.md`](STRUCTURAL_VALIDATION.md) completely. Its architecture,
implementation, migration, and test requirements are mandatory for all new or
modified custom-resource validation.

- Do not extend a legacy error-based validator with new ad hoc patterns. Either
  express the rule in the CRD schema or CEL, or migrate the affected validation
  path to structural validation.
- Structural-validation migrations must reuse the shared DGD/DCD admission-chain
  test harness. Extend `shared_test.go` when the harness needs another resource;
  do not build a resource-local substitute that bypasses schema, CEL, conversion,
  or the public webhook handler.
- When a resource is migrated, update the migration status in
  `STRUCTURAL_VALIDATION.md` in the same change.
