<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribution Guidelines

## External Contributors: Issue-First Workflow

Thank you for your interest in contributing to Dynamo. To help us review your work efficiently, please follow the workflow below.

### When Is a GitHub Issue Required?

**You can submit a PR directly without an issue if:**

- Your change is **<100 lines of code** AND addresses a simple, focused concern (typos, simple bug fixes, formatting)
- **OR** your PR addresses an **existing approved GitHub Issue** (link with "Fixes #123")

**You must create a GitHub Issue first for:**

- Changes ≥100 lines of code
- New features, architecture changes, or multi-component changes
- Any change that requires design discussion

**Note**: All PRs are triaged. If your PR lacks sufficient context or understanding, reviewers will reject the PR and ask you to first submit an issue and get approval before proceeding.

### Issue-First Workflow Steps

**If you are an external contributor and your change requires a GitHub Issue**, please follow this workflow. This process ensures that external contributions are well-aligned with the project's goals and reduces the likelihood of significant rework.

1. **Create a GitHub Issue First** – Before writing any code, [open a GitHub Issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) using the **Contribution Request** template.

2. **Identify the Problem** – Clearly explain the problem you are trying to solve, including any relevant context, error messages, or use cases.

3. **Recommend a Solution** – Propose your intended solution or approach in the issue. Include:
   - **Estimated PR size**: XS / S / M / L / XL / XXL (use your best judgment)
   - **Files affected**: Approximate number and which components/interfaces
   - **Type of change**:
     - **Bug fix**: Corrects existing behavior with minimal code changes; should include tests that capture the issue
     - **New feature**: Adds new behavior or capability; requires comprehensive tests for the new functionality
     - **Refactoring**: Restructures code without changing behavior; no new tests required
     - **Performance improvement**: Optimizes existing behavior; should include before/after benchmarks

   This helps the Dynamo team understand your plan, assess complexity, and provide early feedback. PRs that significantly exceed their stated size may be rejected.

4. **Get Approval** – Wait for the Dynamo team to review and approve both the problem statement and your proposed solution. This ensures alignment with the project's architecture and roadmap before you invest time in implementation. Once approved, a maintainer will apply the `approved-for-pr` label to your issue.

5. **Submit a Pull Request** – Once your issue has the `approved-for-pr` label, [submit a PR](https://github.com/ai-dynamo/dynamo/compare) that references the issue. Link your PR to the issue using [GitHub keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue) (e.g., "Fixes #123" or "Closes #123").

6. **Address Code Rabbit Review** – Wait for the automated Code Rabbit review to complete. Address its suggestions, including the nitpicks—they are often quite insightful and help improve code quality.

7. **Ensure CI Tests Pass** – Wait for all CI tests to pass. If any tests fail, investigate and fix the issues before requesting human review.

8. **Check CODEOWNERS** – Review the [CODEOWNERS](https://github.com/ai-dynamo/dynamo/blob/main/CODEOWNERS) file to identify which team members need to sign off on your PR based on the files you've modified.

9. **Request a Review** – Add whomever approved your GitHub Issue as a reviewer on your PR. Please also add [@dagil-nvidia](https://github.com/dagil-nvidia) for visibility.

> **Note on AI-Generated Code**: While Dynamo encourages the use of AI-generated code, it is the full responsibility of the submitter to understand every change in the PR. Failure to demonstrate sufficient understanding of the submitted code will result in rejection.

---

## General Contribution Guidelines

Contributions that fix documentation errors or that make small changes
to existing code can be contributed directly by following the rules
below and submitting an appropriate PR.

Contributions intended to add significant new functionality must
follow a more collaborative path described in the following
points. Before submitting a large PR that adds a major enhancement or
extension, be sure to submit a GitHub issue that describes the
proposed change so that the Dynamo team can provide feedback.

- As part of the GitHub issue discussion, a design for your change
  will be agreed upon. An up-front design discussion is required to
  ensure that your enhancement is done in a manner that is consistent
  with Dynamo's overall architecture.

- The Dynamo project is spread across multiple GitHub Repositories.
  The Dynamo team will provide guidance about how and where your enhancement
  should be implemented.

- Testing is a critical part of any Dynamo
  enhancement. You should plan on spending significant time on
  creating tests for your change. The Dynamo team will help you to
  design your testing so that it is compatible with existing testing
  infrastructure.

- If your enhancement provides a user visible feature then you need to
  provide documentation.

# Contribution Rules

- The code style convention is enforced by common formatting tools
  for a given language (such as clang-format for c++, black for python).
  See below on how to ensure your contributions conform. In general please follow
  the existing conventions in the relevant file, submodule, module,
  and project when you add new code or when you extend/fix existing
  functionality.

- Avoid introducing unnecessary complexity into existing code so that
  maintainability and readability are preserved.

- Try to keep code changes for each pull request (PR) as concise as possible:

  - Fillout PR template with clear description and mark applicable checkboxes

  - Avoid committing commented-out code.

  - Wherever possible, each PR should address a single concern. If
    there are several otherwise-unrelated things that should be fixed
    to reach a desired endpoint, it is perfectly fine to open several
    PRs and state in the description which PR depends on another
    PR. The more complex the changes are in a single PR, the more time
    it will take to review those changes.

  - Make sure that the build log is clean, meaning no warnings or
    errors should be present.

  - Make sure all tests pass.

- Dynamo's default build assumes recent versions of
  dependencies (CUDA, TensorFlow, PyTorch, TensorRT,
  etc.). Contributions that add compatibility with older versions of
  those dependencies will be considered, but NVIDIA cannot guarantee
  that all possible build configurations work, are not broken by
  future contributions, and retain highest performance.

- Make sure that you can contribute your work to open source (no
  license and/or patent conflict is introduced by your code).
  You must certify compliance with the
  [license terms](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE)
  and sign off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org)
  described below before your pull request (PR) can be merged.

- Thanks in advance for your patience as we review your contributions;
  we do appreciate them!

# Coding Convention

All pull requests are checked against the
[pre-commit hooks](https://github.com/pre-commit/pre-commit-hooks)
located [in the repository's top-level .pre-commit-config.yaml](https://github.com/ai-dynamo/dynamo/blob/main/.pre-commit-config.yaml).
The hooks do some sanity checking like linting and formatting.
These checks must pass to merge a change.

To run these locally, you can
[install pre-commit,](https://pre-commit.com/#install)
then run `pre-commit install` inside the cloned repo. When you
commit a change, the pre-commit hooks will run automatically.
If a fix is implemented by a pre-commit hook, adding the file again
and running `git commit` a second time will pass and successfully
commit.

# Running Github actions locally

To run the Github actions locally, you can use the `act` tool.
See [act usage](https://nektosact.com/introduction.html) for more information.

For example, to run the pre-merge-rust workflow locally, you can use the following command from terminal:
```
act -j pre-merge-rust
```

Also you can use vscode extension [GitHub Local Actions](https://marketplace.visualstudio.com/items?itemName=SanjulaGanepola.github-local-actions) to run the workflows from vscode.


# Developer Certificate of Origin

Dynamo is an open source product released under
the Apache 2.0 license (see either
[the Apache site](https://www.apache.org/licenses/LICENSE-2.0) or
the [LICENSE file](./LICENSE)). The Apache 2.0 license allows you
to freely use, modify, distribute, and sell your own products
that include Apache 2.0 licensed software.

We respect intellectual property rights of others and we want
to make sure all incoming contributions are correctly attributed
and licensed. A Developer Certificate of Origin (DCO) is a
lightweight mechanism to do that.

The DCO is a declaration attached to every contribution made by
every developer. In the commit message of the contribution,
the developer simply adds a `Signed-off-by` statement and thereby
agrees to the DCO, which you can find below or at [DeveloperCertificate.org](http://developercertificate.org/).

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

We require that every contribution to Dynamo is signed with
a Developer Certificate of Origin, this is verified by a required CI check.
Additionally, please use your real name.
We do not accept anonymous contributors nor those utilizing pseudonyms.

Each commit must include a DCO which looks like this

```
Signed-off-by: Jane Smith <jane.smith@email.com>
```
You may type this line on your own when writing your commit messages.
However, if your user.name and user.email are set in your git configs,
you can use `-s` or `--signoff` to add the `Signed-off-by` line to
the end of the commit message.

⚠️ **Contributor-Friendly DCO Guide:**
If your pull request fails the DCO check, don't worry! Check out our [DCO Troubleshooting Guide](DCO.md) for step-by-step instructions to fix it quickly.
