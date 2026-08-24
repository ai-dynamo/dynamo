---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Code Quality
subtitle: Style, testing, validation, and commit conventions for Dynamo changes
---

Keep pull requests focused and validate the code paths they change. Maintainers assess correctness,
test coverage, architecture fit, readability, and how review feedback is addressed.

## Pre-commit Checks

The repository configures checks in
[`.pre-commit-config.yaml`](https://github.com/ai-dynamo/dynamo/blob/v1.4.1/.pre-commit-config.yaml).
Run them before pushing:

```bash
pre-commit run
```

To check the complete repository, run:

```bash
pre-commit run --all-files
```

## Commit Messages

Use a [Conventional Commit](https://www.conventionalcommits.org/) title. Add a scope when it makes
ownership or impact clearer.

| Type | Use |
| --- | --- |
| `feat` | New behavior or capability |
| `fix` | Bug fix |
| `docs` | Documentation-only change |
| `test` | Test addition or correction |
| `ci` | Continuous integration change |
| `refactor` | Internal change without intended behavior change |
| `perf` | Performance improvement |
| `chore` | Maintenance work |
| `build` | Build system or dependency change |
| `style` | Formatting-only change |
| `revert` | Revert of an earlier change |

Examples:

```text
feat(router): add weighted load balancing
fix(frontend): handle streaming timeout
docs: clarify the macOS build steps
test(planner): cover scaling policy boundaries
```

Every commit must also include DCO sign-off; see
[DCO and Licensing](dco-and-licensing.md).

## Language Conventions

Use the formatter and linter configured by the component you change.

| Language | Primary References and Tools |
| --- | --- |
| Python | [PEP 8](https://peps.python.org/pep-0008/), Ruff, and Black |
| Rust | [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/), `cargo fmt`, and `cargo clippy` |
| Go | [Effective Go](https://go.dev/doc/effective_go) and `gofmt` |

Follow patterns in the surrounding package. Do not introduce a new tool or convention in a focused
feature or bug-fix pull request unless the change specifically targets project-wide tooling.

## Testing

Run the smallest relevant test set while iterating, then expand validation before requesting
review. The correct command depends on the component.

Common starting points include:

```bash
pytest tests/
cargo test --locked --all-targets
cd deploy/operator && go test ./... -v
```

Use package-specific test instructions when a component provides them. Include the exact commands
and results in the pull request's **Validation** section.

## Quality Checklist

Before requesting review, confirm that the change:

- Solves one clearly stated problem.
- Includes tests for new behavior and bug fixes when practical.
- Builds without new warnings or errors.
- Contains no commented-out implementation or unrelated cleanup.
- Updates user or contributor documentation when behavior or workflows change.
- Passes relevant local checks.
- Uses signed-off commits.

CI is a final verification layer, not a replacement for targeted local testing.
