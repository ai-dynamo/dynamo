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

# Contributing to Dynamo

Thank you for your interest in contributing to Dynamo!

For the full walkthrough - setting up a fork, building from source, and what to expect during review - see the [Contribution Guide](https://docs.nvidia.com/dynamo/dev/contributing/contribution-guide) on the docs site.

How the project is governed is defined in [GOVERNANCE.md](GOVERNANCE.md): the contributor ladder, decision-making, SIGs, and conflict resolution. The rules below are the ones that decide whether your pull request can be reviewed and merged.

## Before You Open a Pull Request

Small changes go straight to a pull request. Typo fixes, documentation corrections, small bug fixes, and narrow configuration changes need no issue.

Open a [Contribution Request](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) first, and wait for the `approved-for-pr` label, when a change is over 100 core lines, spans multiple areas, changes a public API, or adds a dependency. **Core lines** means changed lines of code, excluding tests, documentation, and generated files. Trusted Contributors are exempt from the size trigger, but a Contribution Request is still required for the other three.

Changes to a public API, to communication plane architecture, to backend integration contracts, or spanning multiple areas need a [Dynamo Enhancement Proposal](https://github.com/ai-dynamo/enhancements) instead of a Contribution Request. Do not open both: a DEP settles whether the work is wanted and how it should be built, which is the Contribution Request's question and more. A DEP is sponsored by the Special Interest Group covering the affected areas, acting through a Co-Lead or an area Maintainer, reviewed by the Maintainers of every area it touches, and approved by a two-thirds vote of the Core Maintainers. Take the idea to the SIG before writing the proposal. A DEP is not limited to architectural changes: anything worth agreeing before it is built can have one, including changes to process and convention. Once it is approved, the pull requests implementing it link the DEP and skip the Contribution Request, though they still need their code owners and two approvals.

## Review and CI

Every commit must carry a signature that GitHub reports as verified. Unsigned commits block CI approval.

External contributors need a Maintainer to approve CI for the pull request's current head, either by commenting `/ok to test COMMIT-ID` or by updating the branch. Each new push creates a new head and needs approval again. Trusted Contributors receive automatic CI approval on every head.

Every pull request needs approval from at least two Maintainers other than the author. These are human reviews: AI-assisted review is a supplemental signal and does not count toward the threshold. GitHub auto-requests the [CODEOWNERS](CODEOWNERS) team that owns each file you touched.

## AI-Assisted Contributions

Use AI tooling if it helps, and understand every line you submit. You are the author of record, whatever produced the code, and being unable to explain a change is grounds for rejection.

Disclose substantial AI assistance in the pull request description. Fully automated submissions, opened without a human reviewing the content, are not accepted. See "AI-Assisted Contributions" in [GOVERNANCE.md](GOVERNANCE.md).

## Quick Links

- [Project Governance](GOVERNANCE.md)
- [Good first issues](https://github.com/ai-dynamo/dynamo/labels/good-first-issue)
- [Help wanted](https://github.com/ai-dynamo/dynamo/labels/help-wanted)
- [Open a bug report](https://github.com/ai-dynamo/dynamo/issues/new?template=bug_report.yml)
- [Propose a feature](https://github.com/ai-dynamo/dynamo/issues/new?template=feature_request.yml)
- [Enhancement Proposals](https://github.com/ai-dynamo/enhancements)
- [GitHub Discussions](https://github.com/ai-dynamo/dynamo/discussions)
- [Slack](https://ai-dynamo.org/slack)
- [Office Hours](https://www.youtube.com/playlist?list=PL5B692fm6--tgryKu94h2Zb7jTFM3Go4X)
- [Community Meetings](https://docs.google.com/document/d/1uR8xD_hlYGwV6QspvSc36k1H-wo1BUcVmFbHH9xlXd8/view) ([Youtube](https://www.youtube.com/@ai-dynamo-community)) -- Weekly (Wed 10:30 AM PT) development community meetings
- [Dynamo Day Recordings](https://nvevents.nvidia.com/dynamoday)

Dynamo requires all contributions to be signed off with the [Developer Certificate of Origin (DCO)](https://developercertificate.org/). This certifies that you have the right to submit your contribution under the project's [Apache 2.0 license](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE).

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE).
