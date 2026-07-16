# Validation package

Before changing validation code or tests in this package, read
[`STRUCTURAL_VALIDATION.md`](STRUCTURAL_VALIDATION.md) completely. Its architecture,
implementation, migration, and test requirements are mandatory for all new or
modified custom-resource validation.

- Do not extend a legacy error-based validator with new ad hoc patterns. Either
  express the rule in the CRD schema or CEL, or migrate the affected validation
  path to structural validation.
- When a resource is migrated, update the migration status in
  `STRUCTURAL_VALIDATION.md` in the same change.
