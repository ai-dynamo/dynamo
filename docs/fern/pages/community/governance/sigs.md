<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
title: Special Interest Groups
subtitle: Find the group that works on what you care about
-->

# Special Interest Groups

A Special Interest Group (SIG) is an open, standing group that coordinates work within one domain: roadmap discussion, design review, and cross-area coordination. Anyone can join and participate. You do not need any standing on the contributor ladder to show up, ask a question, or argue about a design.

The current roster, with the code areas each SIG covers and who leads it, is maintained in [SIGS.md](https://github.com/ai-dynamo/dynamo/blob/main/SIGS.md).

## What a SIG does, and does not do

A SIG is where a design gets discussed before it becomes a pull request, where a roadmap gets argued, and where work that crosses several areas gets coordinated. It is the right place to raise "should we build this" and "has anyone thought about this."

A SIG does not merge code. Merge authority stays with the Maintainers of the affected area, as defined in [CODEOWNERS](https://github.com/ai-dynamo/dynamo/blob/main/CODEOWNERS). A SIG that reaches agreement on a design still sends the change through the normal review path, and a design that needs formal sign-off still needs a Dynamo Enhancement Proposal.

Each SIG has co-leads who are accountable for it: keeping discussion moving, representing the domain, and making sure decisions get recorded.

## How to participate

Every SIG has a channel in the ai-dynamo community Slack, named for the SIG. Joining the channel is joining the SIG.

Design discussion belongs in GitHub Issues rather than in chat, so the reasoning survives for whoever reads it in a year. The channel is for coordination and for pointing at the issue.

If a SIG holds a call, notes are posted to its channel: what was decided, what is still open, and who owns the follow-ups.

## Ecosystem projects

Several projects in the [ai-dynamo](https://github.com/ai-dynamo) organization have their own repositories, and each is coordinated by the SIG whose domain it serves. That SIG is where the project's roadmap and its integration with Dynamo get discussed, while code review and merge authority stay with the project's own owners. [SIGS.md](https://github.com/ai-dynamo/dynamo/blob/main/SIGS.md) maps each project to its SIG.

## Starting or retiring a SIG

SIGs are created, merged, and retired as the project's shape changes. The process is defined in [GOVERNANCE.md](https://github.com/ai-dynamo/dynamo/blob/main/GOVERNANCE.md). If you think a domain needs one and does not have one, raise it with the Core Maintainers.
