# AGENTS.md

Before changing API code or API tests in this directory or any versioned API
package below it, read `CONVERSION.md`.

Every API type change in any version must update the corresponding conversion
code and conversion tests, or explicitly document why conversion is unaffected.

Follow its invariants for hub/spoke conversion, sparse annotation preservation,
live-source precedence, structural helper naming, and fuzz coverage.

Keep public API comments concise and focused on the field's contract, defaults,
and validation constraints. Put reconciliation mechanics and detailed rationale
in comments beside the implementation or in its scoped `AGENTS.md`, not in API
field descriptions.
