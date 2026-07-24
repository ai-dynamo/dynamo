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

# Dynamo Project Governance

Dynamo is an open-source distributed inference framework. This document defines how the project is governed, how contributors advance, how decisions are made, and how the project listens to its community.

If you are new to Dynamo, start with the [Contribution Guide](https://docs.nvidia.com/dynamo/getting-started/contribution-guide). For questions about governance, open a [GitHub Discussion](https://github.com/ai-dynamo/dynamo/discussions).

## Values

1. **Collective Responsibility.** Contributors at every level share responsibility for the quality and direction of the project.
2. **Open Exchange of Ideas.** Ideas are judged on their merit, not their origin. Technical decisions are informed by open discussion.
3. **Transparency.** Governance actions - promotions, removals, disputes - are decided and recorded publicly with clear rationale.
4. **Iterative Velocity.** Process exists to enable progress, not constrain it. Fast iteration with continuous feedback benefits everyone - contributors see their input reflected sooner, and the project evolves faster.
5. **Technical Excellence.** Technical decisions prioritize long-term maintainability and quality.
6. **Ecosystem Compatibility.** Changes must not introduce incompatibilities between supported inference backends. The project strives for feature parity across the entire stack.

## Contributor Ladder

Throughout this document, an **area** is a subject-matter component of the repository whose ownership is defined in [CODEOWNERS](https://github.com/ai-dynamo/dynamo/blob/main/CODEOWNERS).

### Contributor

Anyone who has had at least one pull request merged.

**Eligibility**

- One or more pull requests merged into the repository.
- Every commit signed off under the [Developer Certificate of Origin](https://developercertificate.org/) (`Signed-off-by:`).

**Privileges**

*These apply to every fork contribution, first PR included.*

- Submit pull requests from a fork.
- Full CI requires a Maintainer to comment `/ok to test`.
- **[Contribution Request (CR) Issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml)** - required for any pull request that exceeds 100 core lines (changed lines of code, excluding tests, documentation, and generated files), spans multiple areas, changes a public API, or adds a dependency.
  - A Maintainer approves a CR by adding the `approved-for-pr` label.
  - Architectural changes also require a DEP, opened as a separate issue and linked from the CR (see [Contribution Requests and DEPs](#contribution-requests-and-deps)).
  - The CR captures the problem, proposed approach, and expected size before the work is done, so the pull request does not re-argue the design: a small change needs the PR template alone; a sized change opens a CR first.

### Trusted Contributor

A Contributor who has demonstrated sustained, quality contributions. There is no fixed PR count, but a trend takes more than one merged pull request to establish. Eligibility is a qualitative judgment against the criteria below, made by a sponsoring Maintainer who cites multiple merged pull requests as evidence of the trend.

**Eligibility**

- **Sustained Track Record.** A consistent trend of merged contributions within an area over time, not a single burst.
- **Code Quality.** Follows repository conventions and is readable; not copy-pasted or boilerplate-only.
- **Test Coverage.** New behavior ships with tests; feature pull requests are not test-free.
- **Architecture Alignment.** Fits existing patterns and respects area boundaries; no needless re-architecting.
- **Review Responsiveness.** Addresses review feedback constructively; no defensive churn or ignored comments.
- **Scope and Impact.** Contributions are substantive, not solely mechanical or neutral artifacts (dependency bumps, generated files, one-line typo fixes).
- **No Unresolved Negative Signals.** No pattern of reverted changes, copy-paste work passed off as substantive, automated low-effort submissions (raw scanner dumps, spell-check-only pull requests), or unaddressed review feedback.

**Privileges**

A Trusted Contributor keeps all Contributor privileges, and gains:

- May trigger CI on their own pull requests by commenting `/ok to test` - no need to wait for a Maintainer. (A plain Contributor always needs a Maintainer to comment it.)
- May review and approve pull requests within their area of expertise, but still cannot merge - merge authority begins at Maintainer.
- No longer needs a Contribution Request purely for size - the 100-core-line threshold is lifted. A CR is still required for structural changes: touching multiple areas, changing a public API, or adding a dependency.

### Maintainer

A Trusted Contributor who has earned merge authority within a specific area.

**Eligibility**

- **Volume and Tenure.** 10+ merged pull requests over 6+ months within the area.
- **Contribution Quality.** Meets every Trusted Contributor criterion above, sustained across the full record: consistent test coverage, architecture alignment, and a clean review history, with no unresolved negative signals. The sponsoring Maintainer must be willing to stake scoped merge authority on it.
- **Area Depth.** Demonstrated ownership of a specific area (defined by CODEOWNERS), not breadth alone.

**Privileges**

- Review and merge pull requests within their area.
- Trigger CI for any pull request.
- Nominate Contributors for Trusted Contributor status.

*"Maintainer" refers to this governance role. External Maintainers receive GitHub `write` access scoped to their area via CODEOWNERS.*

### Core Maintainer

A Maintainer who has demonstrated project-wide judgment and cross-area expertise.

**Eligibility**

- **Active Maintainer.** Current Maintainer in good standing (not emeritus), with a sustained record of merges and reviews.
- **Cross-Area Contributions.** Substantive contributions or reviews across two or more areas (defined by CODEOWNERS), not depth in a single area alone.
- **Project-Wide Judgment.** A track record of sound decisions on changes that affect the whole project - architectural review, DEP participation, or conflict resolution - such that other Core Maintainers would trust them to merge anywhere.

**Privileges**

- Review and approve pull requests in any area.
- Make architectural decisions affecting the entire project.
- Approve or veto Maintainer nominations.
- Nominate Maintainers for Core Maintainer status.

### Lead Core Maintainer

One Core Maintainer is designated as Lead Core Maintainer.

**Eligibility**

- An active Core Maintainer.

**Privileges**

- Serves as the final decision-maker when Core Maintainers cannot reach consensus - the deadlock breaker of last resort.

### Nominations and Promotions

- **Contributor:** Automatic on the first merged pull request. No nomination required.
- **Trusted Contributor:** Nominated by a Maintainer. Requires approval from at least one Core Maintainer.
- **Maintainer:** Nominated by a Core Maintainer. Two-thirds supermajority vote of Core Maintainers.
- **Core Maintainer:** Nominated by a Core Maintainer. Two-thirds supermajority vote of Core Maintainers.
- **Lead Core Maintainer:** Designated from among active Core Maintainers by a two-thirds supermajority vote; serves with no fixed term. May be replaced at any time by a three-quarters supermajority vote of active Core Maintainers, or may step down voluntarily. Inactivity triggers the standard six-month emeritus transition.
- Candidates do not vote on their own promotion. Supermajority thresholds are computed against the set of active Core Maintainers other than the candidate. The same exclusion applies to removal votes - the subject is excluded from the count.
- Roles belong to individuals, not their employers. Every rung is earned against the criteria above and is never granted by affiliation, and a contributor's standing does not change when their employer does.

DEP-sponsored areas may include an accelerated Maintainer appointment as part of the DEP approval.

All promotion decisions are posted publicly with reasoning.

### Removal

A Maintainer or Core Maintainer may be proposed for removal for cause (Code of Conduct violations, sustained misalignment, conflicts of interest, failure to fulfill responsibilities). Contributors may also be blocked for cause. The individual is given an opportunity to respond. Removal requires a two-thirds supermajority vote of Core Maintainers. The decision is posted publicly.

Contributors may resign voluntarily at any time.

### Inactivity and Emeritus

Maintainers and Core Maintainers inactive for six months may be moved to emeritus status after private outreach. Emeritus members retain recognition but lose active permissions. They may return within twelve months through an expedited process.

## How We Work

### Development Model

Dynamo favors iterative development and fast feedback. Maintainers are trusted to move changes forward within their areas of expertise, with the expectation that review and discussion continue after a change lands when useful. Features land early and evolve in the open, with contributors and maintainers shaping implementations together.

- Maintainers and Core Maintainers may merge changes within their area of expertise without requiring prior consensus beyond the approvals below.
- Every pull request requires approval from at least two Maintainers other than the author - human reviews only; AI-assisted review is a supplemental signal and does not count toward this threshold. No one merges their own work alone.
- Changes requiring a DEP - multi-area impact, public API changes, or communication plane architecture - go through the full DEP process before landing.
- When a pull request has unresolved objections but is considered important for the project, either the Lead Core Maintainer acting alone, or any two Core Maintainers acting together, may designate it as a Strategic Initiative and commit it. They provide a brief impact statement and assign an engineer to address community feedback in the next release cycle.

Community members are encouraged to open GitHub Issues or start discussions on merged features. Core Maintainers commit to addressing valid feedback - usability gaps, API concerns, performance issues - in a timely manner, and designs or APIs in newly landed features may evolve in response to what the community surfaces.

### Contribution Requests and DEPs

Two instruments gate larger changes, and they answer different questions:

- **A Contribution Request (CR) is permission to build.** A CR is a GitHub issue, opened from the [Contribution Request template](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) before a sized change lands. A Maintainer approves it by adding the `approved-for-pr` label, confirming the change is welcome before the work is invested.
- **A Dynamo Enhancement Proposal (DEP) is design consensus.** A DEP carries the formal design for architectural changes and requires Core Maintainer supermajority approval. It is opened as its own issue using the DEP template and linked from the CR.

A small change needs neither - the pull request template alone suffices. A sized change needs a CR. An architectural change needs both.

### AI-Assisted Contributions

AI tooling is welcome in the workflow; accountability is not transferable to it.

- Contributors may use AI assistance to write code, but must understand and stand behind every line they submit. The author of record is responsible for the change, whatever produced it.
- Substantial AI assistance is disclosed in the pull request description.
- Fully automated submissions - pull requests generated and opened without a human reviewing the content - are not accepted, and a pattern of low-effort automated submissions is a negative signal on the contributor ladder.
- On the review side, AI-assisted review is a supplemental signal only; it does not count toward the two-Maintainer approval threshold (see Development Model).

### Decision-Making

The decisions in scope for governance are: pull request approval, architectural changes (DEP), contributor advancement, governance amendments, and release packaging. Day-to-day Maintainer decisions (implementation choice within an area, refactoring inside CODEOWNERS scope, issue triage prioritization) sit with area Maintainers and do not require governance overhead.

The default decision process is lazy consensus. A change proposed by the people responsible for the affected area proceeds unless someone objects within a reasonable review window; silence is consent. An objection needs a stated reason and openness to an alternative, not just a veto. Explicit votes are reserved for the decisions that name them: contributor advancement, removals, DEP approval, and governance amendments.

- **Within an Area.** The area's Maintainers decide. If they cannot agree, the matter escalates to Core Maintainers.
- **Across Areas.** Core Maintainers decide, with input from affected Maintainers.
- **Project-Wide Architecture.** Requires a [Dynamo Enhancement Proposal (DEP)](https://github.com/ai-dynamo/dynamo/issues?q=is%3Aissue%20label%3A%22dep%3Adraft%22%2C%22dep%3Aproposed%22%2C%22dep%3Aapproved%22%2C%22dep%3Aimplementing%22%2C%22dep%3Acompleted%22%2C%22dep%3Adeferred%22%2C%22dep%3Asuperseeded%22) and Core Maintainer approval.
- **Release Packaging.** Core Maintainers decide what ships in a release.

A DEP is required when a change affects multiple areas, introduces or modifies a public API, alters communication plane architecture, or affects backend integration contracts. To propose a DEP, [open an issue](https://github.com/ai-dynamo/dynamo/issues/new/choose) using the DEP template.

### Conflict Resolution

1. Discuss in the relevant pull request or issue.
2. If unresolved, the area's Maintainers decide. Cross-area disputes go to Core Maintainers.
3. Any contributor may escalate to Core Maintainers by opening a GitHub issue. Core Maintainers will respond within seven business days.
4. If Core Maintainers cannot reach consensus, the Lead Core Maintainer makes the final determination and publicly articulates the reasoning.

## Special Interest Groups (SIGs)

Special Interest Groups (SIGs) are open, standing groups that coordinate work within one domain of the project: roadmap discussion, design review, and cross-area coordination. Anyone may join and participate in any SIG - no contributor-ladder standing is required.

SIGs coordinate and advise; they do not carry merge authority. Review and merge stay with the Maintainers of each area, and architectural changes still go through the DEP process. A SIG often spans multiple areas: the area teams (`@ai-dynamo/dynamo-<area>-codeowners`) anchor code review within the SIG's scope, while the SIG is where the roadmap and design conversation happens.

Each SIG has a **SIG Lead** accountable for its agenda, its reporting, and routing items into the governance process (a Contribution Request, a DEP, or escalation to Core Maintainers).

Core Maintainers create, merge, or retire SIGs as the project evolves. The current set, with each SIG's scope and CODEOWNERS groups, is maintained in [SIGS.md](SIGS.md).

## Governance Changes

Changes to this document require a pull request and approval by a two-thirds supermajority of Core Maintainers. The initial version takes effect at adoption, ratified by the Core Maintainers listed in [MAINTAINERS.md](MAINTAINERS.md).

The roster files - [MAINTAINERS.md](MAINTAINERS.md) and [SIGS.md](SIGS.md) - are not part of this document, and updating them is not a governance amendment. They record outcomes of processes defined here (promotion and removal votes, SIG lifecycle decisions): a roster pull request is opened by a Core Maintainer and links the decision it records.

## Code of Conduct and Security

All participants are expected to abide by the [Code of Conduct](https://github.com/ai-dynamo/dynamo/blob/main/CODE_OF_CONDUCT.md).

For matters that require confidentiality, Core Maintainers may charter a small conduct committee to handle Code of Conduct reports. Committee membership is public; deliberations are not. Outcomes are reported publicly with the minimum detail that confidentiality allows, and anyone involved in a report is recused from deciding it.

Security vulnerabilities should be reported according to the [Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md). Vulnerability and CVE response follows NVIDIA's product security process (PSIRT); it is not chartered by project governance.

## Current Core Maintainers

The current Core Maintainers, the Lead Core Maintainer designation, and the area-maintainer references are listed in [MAINTAINERS.md](MAINTAINERS.md).

## References

- [MAINTAINERS.md](MAINTAINERS.md)
- [SIGS.md](SIGS.md)
- [Contribution Guide](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/ai-dynamo/dynamo/blob/main/CODE_OF_CONDUCT.md)
- [Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md)
- [Dynamo GitHub Issues](https://github.com/ai-dynamo/dynamo/issues/new/choose)

---

*This governance model is designed for Dynamo's current scale; Core Maintainers review it quarterly for effectiveness. Major changes follow the governance amendment process above.*
