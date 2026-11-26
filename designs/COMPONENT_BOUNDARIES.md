<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Component boundaries

## Scope

These rules govern deployment interpretation, runtime configuration, interface
bindings, and DGD access by components and external integrations. They keep
configuration dependencies acyclic: the operator interprets the deployment and
supplies the inputs its components consume. Runtime feedback and control loops
remain possible through explicit interfaces.

## Invariants

1. **Deployment interpretation has one owner.** The owning operator interprets
   deployment fields, component and role PodTemplates, defaults, lowering, and
   wiring. Other consumers must not reproduce that interpretation.
2. **Components receive explicit runtime inputs.** A DGD-deployed component must
   not read its deploying DGD's spec or status, or private intermediate lowering
   objects, to discover its configuration, capabilities, or cross-component
   settings. Supply those facts through its own configuration and interface
   bindings. It must be usable without a DGD when those interfaces are supplied.
3. **External integrations consume semantic facts.** An external integration may
   read the DGD spec. For facts derived by the operator, it consumes published
   semantic status rather than reconstructing defaults, roles, or workload
   semantics. A copied PodTemplate or role PodSpec is not a semantic contract.
4. **Runtime interfaces remain explicit.** Components may inspect actual leaf
   resources, including peer Pods and Services, and use explicitly configured
   control interfaces. Those permissions do not authorize reading their own DGD
   as an implicit configuration channel.
5. **Permissions reinforce the boundary.** Grant access to the resources and
   interfaces the component consumes. Remove access to its deploying DGD and
   intermediate lowering objects as the corresponding dependencies are migrated.

## Review checks

- Trace each new configuration fact from its authoritative producer to every
  consumer. Is it passed explicitly, or recovered by reading the deployment?
- Check DGD clients, watches, role/PodTemplate inspection, and new RBAC grants.
  Distinguish a deployed component from an external integration and from a
  legitimate leaf-resource observer.
- For an operator-derived fact, find the semantic status field or explicit
  runtime input. If none exists, introduce the missing interface at its owner.
- Check that the documented runtime interfaces allow use without a DGD.
- Check that scaling or feedback uses an explicit control interface and does not
  introduce a configuration dependency back into the deploying object.

## Migration exceptions

Planner's existing DGD reads for model, GPU, replica, and Retune configuration
are migration debt. Replace them incrementally with explicit inputs and
interfaces; do not expand them to new configuration responsibilities.

Planner's direct patches of component replica counts are a temporary exception.
The target is a scaling adapter explicitly referenced in Planner startup
configuration. Retire Planner's DGD access as these paths migrate. This rule set
requires no immediate breaking removal of those existing paths.

## Decision source

Derived from [DEP #19488](https://github.com/ai-dynamo/dynamo/issues/19488).
The DEP carries its approval status and discussion; this document is the rule
set proposed for adoption with this documentation change.
