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
2. **Open Exchange of Ideas.** Ideas are judged on their merit. Where an idea came from, and who proposed it, carries no weight in the decision.
3. **Transparency.** Governance actions - promotions, removals, disputes - are decided and recorded publicly with clear rationale.
4. **Iterative Velocity.** We keep process light enough that it does not slow the work down. Fast iteration with continuous feedback benefits everyone: contributors see their input reflected sooner, and the project evolves faster.
5. **Technical Excellence.** Technical decisions prioritize long-term maintainability and quality.
6. **Ecosystem Compatibility.** Dynamo is neutral across inference engines: vLLM, SGLang, and TensorRT-LLM are peers. Changes must not privilege one supported backend over another or introduce incompatibilities between them, and the project strives for feature parity across the entire stack.

## Contributor Ladder

Throughout this document, an **area** is a subject-matter component of the repository that [CODEOWNERS](https://github.com/ai-dynamo/dynamo/blob/main/CODEOWNERS) assigns to an owning team.

### Contributor

Anyone who has had at least one pull request merged.

**Eligibility**

- **Merged Contribution.** One or more pull requests merged into the repository.
- **Signed Commits.** Every commit signed off under the [Developer Certificate of Origin](https://developercertificate.org/) (`Signed-off-by:`) and carrying a signature that GitHub reports as verified. Unsigned commits block CI approval.

**Privileges**

These apply to every fork contribution, the first pull request included.

- Submit pull requests from a fork.
- Request full CI, which a Maintainer approves for the pull request's current head by commenting `/ok to test` or by synchronizing the branch. Each new push creates a new head and needs approval again.
- Open a [Contribution Request (CR) issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) before any pull request that exceeds 100 core lines (changed lines of code, excluding tests, documentation, and generated files), spans multiple areas, changes a public API, or adds a dependency. Architectural changes also need a DEP. See [Contribution Requests and DEPs](#contribution-requests-and-deps) for how both work.

### Trusted Contributor

A Contributor who has demonstrated sustained, quality contributions. The volume and tenure bar below is a floor. Meeting it makes a contributor eligible; a sponsoring Maintainer weighs the remaining criteria and nominates, and a second Maintainer or a Core Maintainer approves.

**Eligibility**

- **Volume and Tenure.** 5+ merged pull requests over 2+ months. A single burst of work in one week does not meet this.
- **Code Quality.** Follows repository conventions and is readable; not copy-pasted or boilerplate-only.
- **Test Coverage.** New behavior ships with tests; feature pull requests are not test-free.
- **Architecture Alignment.** Fits existing patterns and respects area boundaries; no needless re-architecting.
- **Review Responsiveness.** Addresses review feedback constructively; no defensive churn or ignored comments.
- **Scope and Impact.** Contributions are substantive, not solely mechanical or neutral artifacts (dependency bumps, generated files, one-line typo fixes).
- **No Unresolved Negative Signals.** No pattern of reverted changes, copy-paste work passed off as substantive, automated low-effort submissions (raw scanner dumps, spell-check-only pull requests), or unaddressed review feedback.

**Privileges**

A Trusted Contributor keeps all Contributor privileges, and gains:

- Receive automatic CI approval. Once added to the approved-contributor list, the CI approval workflow posts `/ok to test` for each new head commit with no Maintainer action. Verified commit signatures are still required.
- Review and approve pull requests within their area of expertise, without merge authority - merge authority begins at Maintainer.
- Open sized pull requests without a Contribution Request - the 100-core-line threshold is lifted. A CR is still required for structural changes: touching multiple areas, changing a public API, or adding a dependency.

### Maintainer

A Trusted Contributor who has earned merge authority within a specific area.

**Eligibility**

- **Volume and Tenure.** 10+ merged pull requests over 6+ months within the area. New and partner-built areas may appoint their first Maintainers ahead of this bar under Area Bootstrap, below.
- **Contribution Quality.** Meets every Trusted Contributor criterion above, sustained across the full record: consistent test coverage, architecture alignment, and a clean review history, with no unresolved negative signals. The sponsoring Maintainer must be willing to stake scoped merge authority on it.
- **Area Depth.** Demonstrated ownership of a specific area (defined by CODEOWNERS). Working across many areas without owning one does not meet this.

**Privileges**

- Review and merge pull requests within their area.
- Trigger CI for any pull request.
- Nominate Contributors for Trusted Contributor status.

*"Maintainer" refers to this governance role. Internal Maintainers hold area authority through membership in their area's CODEOWNERS team. External Maintainers cannot join those org teams, so they are listed individually in CODEOWNERS on their area's paths, and repository `write` comes from the `external_maintainers` access team, orthogonal to the area teams. Branch protection requires review from the owners of each changed path, so in both cases a Maintainer's approval satisfies the merge gate only within their own area.*

### Core Maintainer

A Maintainer who has demonstrated project-wide judgment and cross-area expertise.

**Eligibility**

- **Active Maintainer.** Current Maintainer in good standing (not emeritus), with a sustained record of merges and reviews.
- **Cross-Area Contributions.** Substantive contributions or reviews across two or more areas (defined by CODEOWNERS). Deep work in one area alone does not meet this.
- **Project-Wide Judgment.** A track record of sound decisions on changes that affect the whole project - architectural review, DEP participation, or conflict resolution - such that other Core Maintainers would trust them to merge anywhere.

**Privileges**

- Review and approve pull requests in any area.
- Vote on Maintainer promotions, removals, DEP approval, and governance amendments.
- Propose a Maintainer to the Project Leaders for Core Maintainer appointment.
- Decide by majority what ships in a release.

### Nominations and Promotions

- **Contributor:** Automatic on the first merged pull request. No nomination required.
- **Trusted Contributor:** Nominated by a Maintainer. Requires approval from a second Maintainer or a Core Maintainer.
- **Maintainer:** Nominated by a Core Maintainer. Two-thirds supermajority vote of Core Maintainers.
- **Core Maintainer:** Eligibility is earned against the Core Maintainer criteria above: a sustained record of merges and reviews, substantive contributions or reviews across two or more areas, and demonstrated project-wide judgment. Appointed by the Project Leaders, acting together.
- Where a threshold names a vote of Core Maintainers, the Project Leaders vote with them and count toward the total. Candidates do not vote on their own promotion, and supermajority thresholds are computed against the set of active voters other than the candidate. The same exclusion applies to removal votes - the subject is excluded from the count.
- Roles on the contributor ladder belong to the individual. An employer holds no claim on them. Every rung is earned against the criteria above and is never granted by affiliation, and a contributor's standing does not change when their employer does.

**Area Bootstrap.** A new area, or one built largely by an outside organization, has no one who can meet a volume-and-tenure bar measured against work that did not exist yet. For these, Core Maintainers may appoint initial Maintainers ahead of the floor by the same two-thirds supermajority vote, recording publicly which part of the floor was not met and why the area warrants it. The quality criteria are never waived, and the standard floor applies to every appointment in that area afterward.

All promotion decisions are posted publicly with reasoning.

### Removal

A Maintainer, Core Maintainer, or Project Leader may be proposed for removal for cause (Code of Conduct violations, sustained misalignment, conflicts of interest, failure to fulfill responsibilities). A Core Maintainer or Project Leader may block a Contributor for the same causes. The individual is given seven business days to respond before the vote. Removal requires either a two-thirds supermajority vote of Core Maintainers, or the Project Leaders acting together. The decision is posted publicly.

Contributors may resign voluntarily at any time.

### Inactivity and Emeritus

A Maintainer, Core Maintainer, or Project Leader inactive for six months may be moved to emeritus status after private outreach. Emeritus members retain recognition but lose active permissions. They may return within twelve months through an expedited process.

## Project Leadership

A **Project Leader** sets engineering direction for the project from outside the contributor ladder. Project Leadership is the group of them.

**Appointment**

- The organization providing the project's engineering staffing appoints the Project Leaders. A Project Leader serves with no fixed term and may step down at any time. Removal for cause follows the Removal section, and inactivity triggers the standard six-month emeritus transition.

**Eligibility**

- **Engineering Leadership.** A Project Leader directs the project's engineering work and answers for the staffing behind it. A Project Leader who moves out of that position, including by changing employer, leaves the role.

**Privileges**

- A Project Leader reviews and approves pull requests in any area, and votes alongside the Core Maintainers on every decision that requires their vote.
- Project Leaders appoint Core Maintainers, and acting together may remove a Core Maintainer for any of the causes listed under Removal.
- Acting together, the Project Leaders make the final determination when a two-thirds vote of Core Maintainers fails to resolve a matter. If they do not agree, the change does not proceed.

## How We Work

### Development Model

Dynamo favors iterative development and fast feedback. The project trusts Maintainers to move changes forward within their areas of expertise, with the expectation that review and discussion continue after a change lands when useful. Features land early and evolve in the open, with contributors and maintainers shaping implementations together.

- Maintainers and Core Maintainers may merge changes within their area of expertise without requiring prior consensus beyond the approvals below.
- Every pull request requires approval from at least two Maintainers other than the author, counting Core Maintainers and Project Leaders. Human reviews only; AI-assisted review is a supplemental signal and does not count toward this threshold. No one merges their own work alone.
- Changes requiring a DEP - multi-area impact, public API changes, or communication plane architecture - go through the full DEP process before landing.
- When a pull request has unresolved objections but is considered important for the project, either a Project Leader acting alone, or any two Core Maintainers acting together, may designate it as a Strategic Initiative and commit it. They provide a brief impact statement and assign an engineer to address community feedback in the next release cycle. A Strategic Initiative cannot be used for a change that requires a DEP - a public API change, multi-area impact, or communication plane architecture. Those go through the DEP process regardless of how urgent the change is.

Community members may open GitHub Issues or start discussions on merged features. Core Maintainers commit to addressing valid feedback - usability gaps, API concerns, performance issues - within the next release cycle, and designs or APIs in newly landed features may evolve in response to what the community surfaces.

### Contribution Requests and DEPs

Two instruments gate larger changes:

- **A Contribution Request (CR) is permission to build.** A CR is a GitHub issue, opened from the [Contribution Request template](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) before a sized change lands. A Maintainer approves it by adding the `approved-for-pr` label, confirming the change is welcome before the work is invested. An area's Maintainers respond to a CR within seven business days, either approving it, asking for changes, or explaining why the change is not wanted.
- **A Dynamo Enhancement Proposal (DEP) is design consensus.** A DEP carries the formal design for architectural changes and requires Core Maintainer supermajority approval. The author opens it as its own issue using the DEP template and links it from the CR. Core Maintainers reach a decision within 30 days of a DEP being marked ready for review.

A small change needs neither, and the pull request template alone suffices. A sized change needs a CR, and an architectural change needs a CR and a DEP.

### AI-Assisted Contributions

AI tooling is welcome in the workflow.

- Contributors may use AI assistance to write code, but must understand and stand behind every line they submit. The author of record is responsible for the change, whatever produced it.
- Substantial AI assistance is disclosed in the pull request description.
- Fully automated submissions - pull requests generated and opened without a human reviewing the content - are not accepted, and a pattern of low-effort automated submissions is a negative signal on the contributor ladder.
- On the review side, AI-assisted review is a supplemental signal only; it does not count toward the two-Maintainer approval threshold (see Development Model).

### Decision-Making

The decisions in scope for governance are: pull request approval, architectural changes (DEP), contributor advancement, governance amendments, and release packaging. Day-to-day Maintainer decisions (implementation choice within an area, refactoring inside CODEOWNERS scope, issue triage prioritization) sit with area Maintainers and do not require governance overhead.

The default decision process is lazy consensus. A change proposed by the people responsible for the affected area proceeds unless someone objects within the review window, which is 72 hours unless the proposer sets a longer one; silence is consent. An objection needs a stated reason and openness to an alternative, not just a veto. Explicit votes are reserved for the decisions that name them: Maintainer promotions, removals, DEP approval, and governance amendments.

Disagreement escalates in defined steps. An objection that remains unresolved after the 72-hour review window and one synchronous discussion among the responsible Maintainers goes to a two-thirds vote of Core Maintainers. The Project Leaders make the final determination only after a vote fails to reach two-thirds.

- **Within an Area.** The area's Maintainers decide. If they cannot agree, the matter escalates to Core Maintainers.
- **Across Areas.** Core Maintainers decide, with input from affected Maintainers.
- **Project-Wide Architecture.** Requires a [Dynamo Enhancement Proposal (DEP)](https://github.com/ai-dynamo/dynamo/issues?q=is%3Aissue%20label%3A%22dep%3Adraft%22%2C%22dep%3Aproposed%22%2C%22dep%3Aapproved%22%2C%22dep%3Aimplementing%22%2C%22dep%3Acompleted%22%2C%22dep%3Adeferred%22%2C%22dep%3Asuperseeded%22) and Core Maintainer approval.
- **Release Packaging.** Core Maintainers decide what ships in a release.

A DEP is required when a change affects multiple areas, introduces or modifies a public API, alters communication plane architecture, or affects backend integration contracts. To propose a DEP, [open an issue](https://github.com/ai-dynamo/dynamo/issues/new/choose) using the DEP template.

### Conflict Resolution

1. Discuss in the relevant pull request or issue.
2. If unresolved, the area's Maintainers decide. Cross-area disputes go to Core Maintainers.
3. Any contributor may escalate to Core Maintainers by opening a GitHub issue. Core Maintainers will respond within seven business days.
4. If a two-thirds vote of Core Maintainers fails to resolve the matter, the Project Leaders make the final determination and publicly articulate the reasoning.

## Special Interest Groups (SIGs)

Special Interest Groups (SIGs) are open, standing groups that coordinate work within one domain of the project: roadmap discussion, design review, and cross-area coordination. Anyone may join and participate in any SIG - no contributor-ladder standing is required.

SIGs coordinate and advise; they do not carry merge authority. Review and merge stay with the Maintainers of each area, and architectural changes still go through the DEP process. A SIG often spans multiple areas: the area teams (`@ai-dynamo/dynamo-<area>-codeowners`) anchor code review within the SIG's scope, while the SIG is where the roadmap and design conversation happens.

Each SIG has a **SIG Lead** who runs its agenda, reports on its work, and routes items into the governance process (a Contribution Request, a DEP, or escalation to Core Maintainers).

Maintainers of an ecosystem repository that a SIG coordinates have the same standing in that SIG as the main repository's area Maintainers, and can serve as SIG Lead. Code review and merge authority stay with each repository's own owners.

Core Maintainers create, merge, or retire SIGs as the project evolves. [SIGS.md](SIGS.md) holds the current set, with each SIG's scope, Leads, and CODEOWNERS groups.

## Governance Changes

Changes to this document require a pull request and approval by a two-thirds supermajority of Core Maintainers. The initial version takes effect at adoption, ratified by the Core Maintainers listed in [MAINTAINERS.md](MAINTAINERS.md).

The roster files - [MAINTAINERS.md](MAINTAINERS.md), [SIGS.md](SIGS.md), and [CONTRIBUTORS.md](CONTRIBUTORS.md) - are not part of this document, and updating them is not a governance amendment. They record outcomes of processes defined here (promotion and removal votes, SIG lifecycle decisions): a Core Maintainer opens the roster pull request and links the decision it records.

## Code of Conduct and Security

All participants are expected to abide by the [Code of Conduct](https://github.com/ai-dynamo/dynamo/blob/main/CODE_OF_CONDUCT.md).

For matters that require confidentiality, Core Maintainers may charter a small conduct committee to handle Code of Conduct reports. Committee membership is public; deliberations are not. Outcomes are reported publicly with the minimum detail that confidentiality allows, and anyone involved in a report is recused from deciding it.

Security vulnerabilities should be reported according to the [Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md). Vulnerability and CVE response follows NVIDIA's product security process (PSIRT); it is not chartered by project governance.

## References

- [MAINTAINERS.md](MAINTAINERS.md)
- [SIGS.md](SIGS.md)
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- [Contribution Guide](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/ai-dynamo/dynamo/blob/main/CODE_OF_CONDUCT.md)
- [Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md)
- [Dynamo GitHub Issues](https://github.com/ai-dynamo/dynamo/issues/new/choose)

---

*This governance model fits Dynamo's current scale; Core Maintainers review it quarterly for effectiveness. Major changes follow the governance amendment process above.*
