# Validation package

Before changing validation code or tests in this package, read
[`STRUCTURAL_VALIDATION.md`](STRUCTURAL_VALIDATION.md) completely. Its requirements
are mandatory.

## Storage and compatibility version ownership

- After typed conversion, the storage-version traversal validates every rule
  whose inputs it can represent, for requests from **every** served version.
- Alpha validation is supplemental, never an alternative complete validation
  pass. Keep only rules that need source-only information, information lost by
  conversion, or coverage missing from the submitted version's schema/CEL.
- Different submitted field paths do not justify copying a rule into Alpha or
  skipping the Beta rule. Adapt diagnostic paths in the storage-version owner.
- Never pass complete Beta objects or component maps into an Alpha validator
  to repeat a shared rule. Never compare whole Alpha specs to enforce a rule
  already owned by Beta.
- Every compatibility validator's Go doc must state the specific information
  or source-schema constraint that conversion cannot preserve for validation.
- Conversion does not run the destination version's schema or CEL. Inventory
  source-schema gaps explicitly; do not silently import different Beta API
  constraints into Alpha.
