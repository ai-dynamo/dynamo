# Project: Dynamo

- Build: `cd lib/bindings/python && VIRTUAL_ENV=../../.venv ../../.venv/bin/maturin develop --uv --features kv-indexer && cd ../../.. && uv pip install -e .`

## Codegen
- When adding/changing constants in `lib/runtime/src/metrics/prometheus_names.rs`, always regenerate the Python mirror: `cargo run -p dynamo-codegen --bin gen-python-prometheus-names`. This updates `lib/bindings/python/src/dynamo/prometheus_names.py`. Commit the regenerated file alongside the Rust change.

## Rust Patterns
- For static dispatch, prefer RPITIT (`-> impl Future<...> + Send`) over `#[async_trait]` to avoid heap allocation. Use `#[async_trait]` only when dynamic dispatch (`dyn Trait`) is required (e.g. framework traits like `AsyncEngine`/`Operator`).
- Prefer let chains (`if let ... && let ... && cond`) over nested `if let` blocks.
- Use `std::sync::LazyLock` for lazy statics — not `lazy_static!` or `once_cell`.
- Use `#[expect(lint)]` over `#[allow(lint)]` — it warns when the suppression becomes unnecessary.
- Use `.cast_signed()` / `.cast_unsigned()` for integer sign conversion — not `as`.
- Use precise capturing (`+ use<'a>`) when RPIT lifetime capture needs to be narrowed.
- Prefer `async || {}` closures when futures need to borrow from captures.
- Use inline `const { ... }` blocks for generic const initialization.
- Use associated type bounds (`T: Trait<Assoc: Bounds>`) instead of extra where clauses.
- Use `Duration::from_mins` / `from_hours` for timeout construction.
- Use C-string literals (`c"..."`) for FFI/PyO3 instead of `CStr::from_bytes_with_nul`.
- Use `RwLockWriteGuard::downgrade` for atomic write→read lock transitions.
- Use `Vec::extract_if` / `HashMap::extract_if` instead of retain+collect patterns.
- Use `HashMap::get_disjoint_mut` for safe multi-key mutable access.
- Use `Result::flatten` for `Result<Result<T, E>, E>` chains.
- Use `std::fmt::from_fn` for one-off `Display` formatting from closures.
- Use `Vec::pop_if` / `VecDeque::pop_front_if` for conditional pops.

## Git / macOS
- **NEVER push directly to main.** Always create a feature branch and push there. All changes go through PRs.
- `.gitattributes` marks `*.json.gz` as `text eol=lf`, causing perpetual dirty diffs on macOS for binary gz files. Fix with `.git/info/attributes`: `<path> binary`
- Pre-commit stash/unstash fails when dirty binary files exist in the working tree. Clean the tree (local binary override or renormalize) before committing.
- Never `git add --renormalize` a file you don't intend to commit — it silently stages it.
