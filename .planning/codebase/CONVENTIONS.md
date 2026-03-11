# Coding Conventions

**Analysis Date:** 2026-03-11

## Overview

This is a multi-language codebase (Rust, Python, TypeScript/JavaScript, C/C++). Primary focus is Rust (inference engine) and Python (frameworks/integrations). Conventions are enforced via pre-commit hooks and workspace-wide configurations.

## Naming Patterns

**Rust Files:**
- Modules: `snake_case` (e.g., `kvbm_engine`, `offload_pipeline`)
- Functions: `snake_case` (e.g., `enqueue_blocks`, `wait_confirmed`)
- Types/Structs: `PascalCase` (e.g., `OffloadEngine`, `CancellationToken`, `TransferResult`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- Traits: `PascalCase` (e.g., `Leader`, `Worker`, `ObjectBlockOps`)
- Test functions: `fn test_description_name()` - inline in modules with `#[cfg(test)]` blocks

**Python Files:**
- Modules: `snake_case` (e.g., `storage.py`, `label_injecting_collector.py`)
- Classes: `PascalCase` (e.g., `DynamoRouterConfig`, `TestGetFs`)
- Functions: `snake_case` (e.g., `parse_args()`, `get_fs()`, `validate()`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `TEST_MODELS`)
- Test classes: `Test` prefix with camelCase (e.g., `TestGetFs`, `TestGetMediaUrl`)
- Test methods: `test_description_name()` (e.g., `test_local_file_url()`)

**C/C++/CUDA Files:**
- Follow clang-format with fallback to style file (`.clang-format` in root)
- Naming: snake_case for functions/variables, PascalCase for types

## Code Style

**Formatting:**
- Python: Black formatter (line length 88)
- Rust: `cargo fmt` (automatic via Cargo)
- C/C++/CUDA: clang-format with custom `.clang-format` config (root directory)
- JavaScript/TypeScript: In TypeScript frontend (fern docs), standard tools

**Linting:**
- Python: ruff (fast linting replacing flake8), black (formatting), isort (import sorting)
  - Line length: 88 characters
  - Config: `pyproject.toml` → `[tool.ruff]` and `[tool.isort]`
- Rust: `cargo clippy` with default rules, `cargo machete` for unused dependencies
- Pre-commit: All languages checked via `.pre-commit-config.yaml`

**Configuration Files:**
- `pyproject.toml`: Python project config (ruff, isort, pytest, mypy, sphinx)
- `Cargo.toml`: Rust workspace root; individual crates in `lib/*/Cargo.toml`
- `.clang-format`: C/C++/CUDA formatting rules
- `.pre-commit-config.yaml`: isort, black, flake8, clang-format, codespell

## Import Organization

**Rust:**
Order (in `lib.rs` or module files):
1. External crate imports (`use anyhow::...`, `use tokio::...`)
2. Workspace dependencies (`use dynamo_runtime::...`, `use kvbm_config::...`)
3. Standard library (`use std::...`)
4. Relative module imports (`mod batch;`, `pub use handle::...`)
5. Inline test modules (`#[cfg(test)] mod tests { ... }`)

Example from `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/lib/runtime/src/lib.rs`:
```rust
use std::{collections::HashMap, sync::{Arc, OnceLock, Weak}};
use anyhow::{Context as ErrorContext, Error, Ok as OK, Result};
use async_once_cell::OnceCell;

pub mod config;
pub use config::RuntimeConfig;
pub mod engine;
pub use distributed::{DistributedRuntime, distributed_test_utils};
```

**Python:**
- isort profile: `black` (auto-sorts with compatibility with Black formatter)
- Known first-party: `dynamo`, `deploy` (config in `pyproject.toml`)
- Known third-party: `vllm`, `tensorrt_llm`, `sglang`, `aiconfigurator`
- Trailing comma on multi-line imports
- Example from `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/components/src/dynamo/router/args.py`:
```python
import argparse
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.llm import KvRouterConfig
```

**Path Aliases:**
- Python: Uses absolute imports (`from dynamo.common...`), no relative imports for cross-module
- Rust: `use` for named items; no special path alias system in core (workspace dependencies handled in Cargo)

## Error Handling

**Rust:**
- Primary: `anyhow::Result<T>` for functions that propagate errors
- Error creation: `anyhow::anyhow!(msg)` macro or `bail!(msg)` for early returns
- Custom errors: `thiserror` crate for domain-specific errors (use when error type needs to be caught specifically)
- Context: Use `.context("operation description")` for error chaining
- Workspace imports: `pub use anyhow::{Result, Error, Context as ErrorContext, bail as raise};` (see `lib/runtime/src/lib.rs`)
- Pattern: Early returns with `?` operator for error propagation

Example from `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/lib/kvbm-engine/src/offload/cancel_tests.rs`:
```rust
#[tokio::test]
async fn test_draining_countdown_to_confirmation() {
    let (token, updater) = CancellationToken::new();
    // ... test code
}
```

**Python:**
- Raise built-in exceptions: `ValueError`, `TypeError`, `RuntimeError`, `ImportError`
- Use `raise ... from e` for exception chaining (preserve context)
- No custom exception classes in codebase (use ValueError/RuntimeError)
- Validation in `validate()` methods that raise `ValueError` on constraint violation

Example from `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/components/src/dynamo/router/args.py`:
```python
def validate(self) -> None:
    """Validate config invariants."""
    if not self.endpoint:
        raise ValueError("endpoint is required (set --endpoint or DYN_ROUTER_ENDPOINT)")
```

## Logging

**Framework:** tracing (Rust) and logging (Python)

**Rust:**
- Use `tracing` macros: `trace!()`, `debug!()`, `info!()`, `warn!()`, `error!()`
- Structured logging with field syntax: `info!(field = value, "message")`
- Config in `tracing-subscriber` with JSON output and env-filter
- Entry point: `components/src/dynamo/*/main.py` or `lib/*/src/main.rs` initializes subscriber

**Python:**
- Use standard `logging` module with `getLogger(__name__)`
- Initialized via logger at module level: `_logger = logging.getLogger(__name__)`
- Access via `_logger.info()`, `_logger.debug()`, `_logger.error()`
- Configuration via `logging.basicConfig()` in main or conftest

Example from `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/tests/conftest.py`:
```python
import logging
_logger = logging.getLogger(__name__)
# Used in fixtures: _logger.info("message")
```

## Comments

**When to Comment:**
- Complex algorithms: Explain intent and approach
- Non-obvious workarounds: Document "WAR" (workaround) with context
- Safety considerations: Preconditions, invariants, unsafe blocks (Rust)
- Architectural decisions: Link to docs or issues

**Documentation Comments:**
- Rust: `///` for public items, module-level `//!` for crate/module docs
  - Example from `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/lib/kvbm-engine/src/offload/mod.rs`:
  ```rust
  //! Offload Engine for asynchronous block transfers between storage tiers.
  //!
  //! # Architecture
  //! ...
  ```
- Python: Docstrings for all public functions and classes
  - Style: One-liner for simple functions; multi-line for complex ones
  - Example: `def validate(self) -> None: """Validate config invariants."""`

**JSDoc/TSDoc:**
- Not heavily used; Python docstrings and Rust doc comments are primary
- TypeScript in fern (docs generation) uses standard TSDoc

## Function Design

**Size:** Keep functions focused on single responsibility
- Rust: Typical range 20-50 lines (larger functions broken into helpers)
- Python: Similar, with exception for generated/model code

**Parameters:**
- Rust: Use `&self`, `&mut self`, or owned types explicitly
  - Builder pattern for complex initialization (see `KvbmRuntimeBuilder`)
  - Trait objects for flexibility (e.g., `Arc<dyn Worker>`)
- Python: Type hints required for public APIs
  - Use `Optional[T]` for nullable, not bare `None`
  - Dataclass pattern for config (see `DynamoRouterConfig`)

**Return Values:**
- Rust: `Result<T>` for fallible operations, Option<T> for potentially absent values
- Python: Return `None` implicitly or explicitly; type hints indicate return type

## Module Design

**Exports:**
- Rust: Explicit `pub use` statements at module root
  - Example: `pub use engine::{OffloadEngine, OffloadEngineBuilder};`
  - Private by default; only re-export public API
- Python: No `__all__` declaration; all public names (no leading `_`) are exported by default

**Barrel Files:**
- Rust: `mod.rs` or inline modules accumulate re-exports
  - Example: `lib/kvbm-engine/src/offload/mod.rs` re-exports all public types
- Python: `__init__.py` files don't typically re-export; imports are explicit

## SPDX License Headers

**All files require:**
```
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Enforced via pre-commit (may be skipped for certain files like generated code).

## Async/Await Patterns

**Rust:**
- Async functions: `async fn name() -> Result<T>`
- Tokio runtime: All code runs on `#[tokio::main]` or test harness
- Async tests: `#[tokio::test]` attribute
- Cancellation: Use `tokio_util::sync::CancellationToken`

**Python:**
- Async functions: `async def name() -> T:` with type hints
- Pytest marker: `@pytest.mark.asyncio` for async test functions
- Asyncio mode: Auto (from `pyproject.toml` → `asyncio_mode = "auto"`)

---

*Convention analysis: 2026-03-11*
