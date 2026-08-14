<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
title: How Dynamo Is Governed
subtitle: Who decides what, and how contributors earn a say
-->

# How Dynamo Is Governed

Dynamo is an open-source project with a published governance model. This page explains what that model means for you as a contributor. The governance model itself is maintained in the repository, in [GOVERNANCE.md](https://github.com/ai-dynamo/dynamo/blob/main/GOVERNANCE.md), which is the authoritative text whenever this page and it disagree.

## The contributor ladder

Standing in the project is earned through contribution and belongs to you as an individual, never to your employer. It survives a job change.

| Role | How you get there | What it lets you do |
| :---- | :---- | :---- |
| Contributor | Your first merged pull request | Submit pull requests and request CI |
| Trusted Contributor | 5+ merged pull requests over 2+ months, nominated by a Maintainer and approved by a Core Maintainer | Automatic CI approval; review and approve in your area; the 100-line Contribution Request trigger is lifted |
| Maintainer | 10+ merged pull requests over 6+ months in one area, by two-thirds vote | Merge within your area; nominate Trusted Contributors |
| Core Maintainer | A cross-area record and project-wide judgment. A Core Maintainer nominates you, and Project Leadership appoints you | Merge anywhere, approve promotions, decide architecture |

The volume and tenure numbers are a floor, not a trigger. Meeting them makes you eligible, and a sponsoring Maintainer weighs code quality, test coverage, architecture fit, and review conduct to decide. Promotions are posted publicly with the reasoning.

Separate from the ladder is **Project Leadership**, held by the NVIDIA engineering leaders accountable for Dynamo's direction and the staffing behind it. Its members review in any area and appoint Core Maintainers from among those the body nominates. They do not vote on the decisions that go to a vote; they resolve the ones a vote fails to settle.

This is the one role tied to an employer, and the governance document says so plainly rather than treating it as earned like the rungs above. NVIDIA staffs and funds the project's engineering, and Project Leadership is where that commitment sits.

New areas, and areas built largely by an outside organization, can appoint their first Maintainers before anyone could meet a floor measured against work that did not exist yet. That provision is called Area Bootstrap, and when it is used, the part of the floor that was not met is named publicly.

The current rosters live in [MAINTAINERS.md](https://github.com/ai-dynamo/dynamo/blob/main/MAINTAINERS.md) and [CONTRIBUTORS.md](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTORS.md).

## How decisions get made

The default is lazy consensus. A proposal from the people responsible for an area proceeds unless someone objects within the review window, which is 72 hours unless the proposer sets a longer one, and silence counts as consent. An objection needs a stated reason and an openness to an alternative, not just a veto.

Explicit votes are reserved for the decisions that name them: advancement, removals, Dynamo Enhancement Proposal approval, and amendments to the governance document itself. The Core Maintainers are the electorate for all of them.

Where a decision lands depends on its blast radius. Within one area, that area's Maintainers decide. Across areas, Core Maintainers decide with input from the affected Maintainers. Project-wide architecture requires a Dynamo Enhancement Proposal.

Disagreement escalates in defined steps rather than stalling. An objection still unresolved after the review window and one synchronous discussion among the responsible Maintainers goes to a two-thirds vote of Core Maintainers. Only if that vote fails does Project Leadership make the final determination.

## Two instruments, two questions

A **Contribution Request** is permission to build. Open one before sized work so you know the change is welcome before you invest in it.

A **Dynamo Enhancement Proposal** is design consensus. It carries the formal design for a change. Architectural changes require one, and anything else worth agreeing before it is built may have one, including changes to process and convention. The Special Interest Group covering the affected areas sponsors it. The Maintainers of every area it touches review it, including areas outside that SIG's scope, so sponsoring never decides who reviews. Approval is a two-thirds vote of Core Maintainers, since a DEP is by definition a change no single area owns.

A small change needs neither. A sized change needs a Contribution Request. An architectural change needs a Dynamo Enhancement Proposal instead, which answers the same question the Contribution Request asks and more, so you do not open both. The thresholds that decide which bucket you are in are in [CONTRIBUTING.md](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTING.md), and [Enhancement Proposals](../contributing/enhancement-proposals.md) covers how a DEP is written and reviewed.

## Where coordination happens

Special Interest Groups are the open forum for roadmap discussion and design review in a given domain. Anyone can join one. See [Special Interest Groups](sigs.md).

If something goes wrong, raise it in the pull request or issue first. Unresolved matters go to the area's Maintainers, then to Core Maintainers, who respond within seven business days.
