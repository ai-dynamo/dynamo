# Rust Libraries

## Environment Variables In Tests

- Prefer not to test one environment variable mapping directly to one configuration field. That plumbing is usually tautological and does not justify process-global test state.
- Test pure parsing and configuration logic by passing values through a helper, lookup closure, builder, or explicit config object.
- Keep an environment-backed test only when it exercises a nontrivial observable contract such as precedence, interaction between settings, fallback behavior, startup lifecycle, or integration with an external component.
- When a test must modify the process environment, use `temp_env` for the complete setup, assertion, and cleanup scope. Never call `std::env::set_var` or `std::env::remove_var` directly in a test.
- Environment mutation and restoration are process-global. Ensure every test that can read or write the affected variables participates in the same crate-wide serialization mechanism; a module-local mutex is not sufficient.
- Do not combine environment mutation with a process-global `OnceLock`, `LazyLock`, or equivalent cache unless the test can reset or inject that cache. Prefer testing the uncached loader directly.
- Keep environment overrides scoped tightly and restore the caller's exact prior state, including when the test panics or returns early.
