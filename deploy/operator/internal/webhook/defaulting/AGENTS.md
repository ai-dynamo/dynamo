# Defaulting package

## Feature-gated defaulting

**Gate-dependent defaults must use the request namespace's effective `features.Gate`.**

- Resolve gates through the shared `features.Resolver`. Do not read operator
  configuration booleans, environment variables, runtime capability detection,
  namespace Leases, or installation mode directly from a defaulter.
- Namespace Lease values override global values in either direction.
- Keep defaults that are not feature-dependent stable across namespaces.
- Test the global value and namespace overrides in both directions for every
  gate-dependent default.

The Grove-dependent `minAvailable` default follows this contract.
