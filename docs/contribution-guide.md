---
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
subtitle: How to contribute to Dynamo
max-toc-depth: 3
---

# Contribution Guide

Dynamo is an open-source distributed inference platform, built by a growing community of contributors. The project is licensed under [Apache 2.0](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE) and developed in the open -- every contributor goes through the same review process, CI pipeline, and code standards, whether they work at NVIDIA or not.

With 200+ external contributors, 220+ merged community PRs, and new contributors joining every month, Dynamo is one of the fastest-growing open-source inference projects. Check out our [commit activity](https://github.com/ai-dynamo/dynamo/graphs/commit-activity) and [GitHub stars](https://github.com/ai-dynamo/dynamo/stargazers). Whether you're fixing a typo or building a major feature, this guide will help you get started.

Join the community:

- [CNCF Slack (`#ai-dynamo`)](https://communityinviter.com/apps/cloud-native/cncf) -- join CNCF Slack and find us in `#ai-dynamo`
- [Discord](https://discord.gg/nvidia-dynamo)
- [GitHub Discussions](https://github.com/ai-dynamo/dynamo/discussions)

## TL;DR

For experienced contributors:

1. Fork and clone the repo
2. For changes ≥100 lines or new features, [open an issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) first
3. Create a branch: `git checkout -b yourname/fix-router-timeout`
4. Make changes, run `pre-commit run`
5. Commit with DCO sign-off: `git commit -s -m "fix: description"`
6. Open a PR targeting `main`

---

## Ways to Contribute

### Report a Bug

Found something broken? [Open a bug report](https://github.com/ai-dynamo/dynamo/issues/new?template=bug_report.yml) with:

- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, GPU, Python version, Dynamo version)

### Improve Documentation

Documentation improvements are always welcome:

- Fixing typos or unclear explanations
- Adding examples or tutorials
- Improving API documentation

Small doc fixes can be submitted directly as PRs without an issue.

### Propose a Feature

Have an idea? [Open a feature request](https://github.com/ai-dynamo/dynamo/issues/new?template=feature_request.yml) to discuss it with maintainers before implementation.

### Contribute Code

Ready to write code? See the [Contribution Workflow](#contribution-workflow) section below.

### Help the Community

Not all contributions are code. You can also:

- Answer questions on Discord or CNCF Slack
- Review pull requests
- Share how you're using Dynamo -- blog posts, talks, or social media
- Star the [repository](https://github.com/ai-dynamo/dynamo)

---

## Getting Started

### Find an Issue

Browse [open issues](https://github.com/ai-dynamo/dynamo/issues) or look for:

| Issue Type | Description |
|------------|-------------|
| [Good First Issues](https://github.com/ai-dynamo/dynamo/labels/good-first-issue) | Beginner-friendly, with guidance |
| [Help Wanted](https://github.com/ai-dynamo/dynamo/labels/help-wanted) | Community contributions welcome |

### Fork and Clone

1. [Fork the repository](https://github.com/ai-dynamo/dynamo/fork) on GitHub
2. Clone your fork:

```bash
git clone https://github.com/YOUR-USERNAME/dynamo.git
cd dynamo
git remote add upstream https://github.com/ai-dynamo/dynamo.git
```

### Building from Source

> [!TIP]
> Full build instructions are included below. Expand the accordion to set up your local development environment.

<details>
<summary>Expand build instructions</summary>

#### 1. Install System Libraries

**Ubuntu:**

```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

**macOS:**

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake protobuf

# Verify Metal is accessible
xcrun -sdk macosx metal
```

#### 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### 3. Create a Python Virtual Environment

Install [uv](https://docs.astral.sh/uv/#installation) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

#### 4. Install Build Tools

```bash
uv pip install pip maturin
```

[Maturin](https://github.com/PyO3/maturin) is the Rust-Python bindings build tool.

#### 5. Build the Rust Bindings

```bash
cd lib/bindings/python
maturin develop --uv
```

#### 6. Install GPU Memory Service

```bash
cd ../../..
uv pip install -e lib/gpu_memory_service
```

#### 7. Install the Wheel

```bash
uv pip install -e .
```

#### 8. Verify the Build

```bash
python3 -m dynamo.frontend --help
```

> [!TIP]
> VSCode and Cursor users can use the [`.devcontainer`](https://github.com/ai-dynamo/dynamo/tree/main/.devcontainer) folder for a pre-configured development environment. See the [devcontainer README](https://github.com/ai-dynamo/dynamo/blob/main/.devcontainer/README.md) for details.

</details>

### Set Up Pre-commit Hooks

```bash
uv pip install pre-commit
pre-commit install
```

You're all set up! Get curious -- explore the codebase, experiment with the [examples](https://github.com/ai-dynamo/dynamo/tree/main/examples), and see how the pieces fit together. When you're ready, pick an issue from the [Good First Issues](https://github.com/ai-dynamo/dynamo/labels/good-first-issue) board or read on for the full contribution workflow.

---

## Contribution Workflow

The contribution process depends on the size and scope of your change. Even when not required, opening an issue is a great way to start a conversation with Dynamo maintainers before investing time in a PR.

| Size | Lines Changed | Example | What You Need |
|------|---------------|---------|---------------|
| **XS** | 1–10 | Typo fix, config tweak | Submit a PR directly |
| **S** | 10–100 | Small bug fix, doc improvement, focused refactor | Submit a PR directly |
| **M** | 100–200 | Feature addition, moderate refactor | [Open an issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) first |
| **L** | 200–500 | Multi-file feature, new component | [Open an issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) first |
| **XL** | 500–1000 | Major feature, cross-component change | [Open an issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) first |
| **XXL** | 1000+ | Architecture change | Requires a [DEP](https://github.com/ai-dynamo/enhancements) |

**Small changes (under 100 lines):** Submit a PR directly -- no issue needed. This includes typos, simple bug fixes, and formatting. If your PR addresses an existing approved issue, link it with "Fixes #123".

**Larger changes (≥100 lines):** [Open a Contribution Request](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) issue first and wait for the `approved-for-pr` label before submitting a PR.

**Architecture changes:** Changes that affect multiple components, introduce or modify public APIs, alter communication plane architecture, or affect backend integration contracts require a [Dynamo Enhancement Proposal (DEP)](https://github.com/ai-dynamo/enhancements). Open a DEP in the [`ai-dynamo/enhancements`](https://github.com/ai-dynamo/enhancements) repo before starting implementation.

### Submitting a Pull Request

1. **Create a GitHub Issue** (if required) — [Open a Contribution Request](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) and describe what you're solving, your proposed approach, estimated PR size, and files affected.

2. **Get Approval** — Wait for maintainers to review and apply the `approved-for-pr` label.

3. **Submit a Pull Request** — [Open a PR](https://github.com/ai-dynamo/dynamo/compare) that references the issue using GitHub keywords (e.g., "Fixes #123").

4. **Address Code Rabbit Review** — Respond to automated Code Rabbit suggestions, including nitpicks.

5. **Trigger CI Tests** — For external contributors, a maintainer must comment `/ok to test COMMIT-ID` to run the full CI suite, where `COMMIT-ID` is the short SHA of your latest commit. Fix any failing tests before requesting human review.

6. **Request Review** — Add the person who approved your issue as a reviewer. Check [CODEOWNERS](https://github.com/ai-dynamo/dynamo/blob/main/CODEOWNERS) for required approvers based on files modified.

> [!IMPORTANT]
> **AI-Generated Code:** While we encourage using AI tools, you must fully understand every change in your PR. Inability to explain submitted code will result in rejection.

### Branch Naming

Use a descriptive branch name that identifies you and the change:

```text
yourname/fix-description
```

Examples:

```text
jsmith/fix-router-timeout
jsmith/add-lora-support
```

---

## Code Style & Quality

Maintainers assess contribution quality based on code style, test coverage, architecture alignment, and review responsiveness. Consistent, high-quality contributions are the foundation for building trust in the project.

### Pre-commit Hooks

All PRs are checked against [pre-commit hooks](https://github.com/ai-dynamo/dynamo/blob/main/.pre-commit-config.yaml). After [installing pre-commit](#set-up-pre-commit-hooks), run checks locally:

```bash
pre-commit run --all-files
```

### Commit Message Conventions

Use [conventional commit](https://www.conventionalcommits.org/) prefixes:

| Prefix | Use For |
|--------|---------|
| `feat:` | New features |
| `fix:` | Bug fixes |
| `docs:` | Documentation changes |
| `refactor:` | Code refactoring (no behavior change) |
| `test:` | Adding or updating tests |
| `chore:` | Maintenance, dependency updates |
| `ci:` | CI/CD changes |
| `perf:` | Performance improvements |

Examples:

```text
feat(router): add weighted load balancing
fix(frontend): resolve streaming timeout on large responses
docs: update quickstart for macOS users
test(planner): add unit tests for scaling policy
```

### Language Conventions

| Language | Style Guide | Formatter |
|----------|-------------|-----------|
| **Python** | [PEP 8](https://peps.python.org/pep-0008/) | `black`, `ruff` |
| **Rust** | [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) | `cargo fmt`, `cargo clippy` |
| **Go** | [Effective Go](https://go.dev/doc/effective_go) | `gofmt` |

### Testing

Run the test suite before submitting a PR:

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest -m unit tests/

# Run a specific test file
pytest -s -v tests/test_example.py
```

For Rust components:

```bash
cargo test
```

For the Kubernetes operator (Go):

```bash
cd deploy/operator
go test ./... -v
```

### General Guidelines

- Keep PRs focused -- one concern per PR
- Write clean, well-documented code that future contributors can understand
- Include tests for new functionality and bug fixes
- Ensure clean builds (no warnings or errors)
- All tests must pass
- No commented-out code
- Respond to review feedback promptly and constructively

### Running GitHub Actions Locally

Use [act](https://nektosact.com/) to run workflows locally:

```bash
act -j pre-merge-rust
```

Or use the [GitHub Local Actions](https://marketplace.visualstudio.com/items?itemName=SanjulaGanepola.github-local-actions) VS Code extension.

---

## What to Expect

### Status Labels

| Status | What It Means |
|--------|---------------|
| `needs-triage` | We're reviewing your issue |
| `needs-info` | We need more details from you |
| `approved-for-pr` | Ready for implementation — submit a PR |
| `in-progress` | Someone is working on this |
| `blocked` | Waiting on external dependency |

### Response Times

We aim to:

- **Respond** to new issues within a few business days
- **Triage** high-priority issues within a week

Issues with no activity for 30 days may be auto-closed (can be reopened).

### Review Process

After you submit a PR and complete the steps in [Submitting a Pull Request](#submitting-a-pull-request):

1. The reviewer will provide feedback -- please respond to all comments within a reasonable timeframe
2. If changes are requested, address them and ping the reviewer for re-review
3. If your PR hasn't been reviewed within 7 days, feel free to ping the reviewer or leave a comment

### Good First Issues

Issues labeled `good-first-issue` are sized for new contributors. We provide extra guidance on these -- look for clear acceptance criteria and a suggested approach in the issue description.

---

## DCO & Licensing

### Developer Certificate of Origin

Dynamo requires all contributions to be signed off with the [Developer Certificate of Origin (DCO)](https://developercertificate.org/). This certifies that you have the right to submit your contribution under the project's [Apache 2.0 license](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE).

Each commit must include a sign-off line:

```text
Signed-off-by: Jane Smith &lt;jane.smith@email.com&gt;
```

Add this automatically with the `-s` flag:

```bash
git commit -s -m "fix: your descriptive message"
```

**Requirements:**

- Use your real name (no pseudonyms or anonymous contributions)
- Your `user.name` and `user.email` must be configured in git

**DCO Check Failed?** See our [DCO Troubleshooting Guide](https://github.com/ai-dynamo/dynamo/blob/main/DCO.md) for step-by-step instructions to fix it.

### License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE).

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All participants are expected to abide by our [Code of Conduct](https://github.com/ai-dynamo/dynamo/blob/main/CODE_OF_CONDUCT.md).

---

## Security

If you discover a security vulnerability, please follow the instructions in our [Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md). Do not open a public issue for security vulnerabilities.

---

## Getting Help

- **CNCF Slack**: [Join CNCF Slack](https://communityinviter.com/apps/cloud-native/cncf) and find us in `#ai-dynamo`
- **Discord**: [Join our community](https://discord.gg/nvidia-dynamo)
- **Discussions**: [GitHub Discussions](https://github.com/ai-dynamo/dynamo/discussions)
- **Documentation**: [docs.nvidia.com/dynamo](https://docs.nvidia.com/dynamo/)

Thank you for contributing to Dynamo!
