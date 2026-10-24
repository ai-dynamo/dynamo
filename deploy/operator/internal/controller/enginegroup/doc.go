/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package enginegroup owns Kubernetes reconciliation, runtime resolution, status,
// and DGD child integration for independently resizable engine worlds.
// The Kubernetes-independent membership coordinator and adapter contracts remain
// in internal/enginegroup; they do not depend on this controller package.
//
// The private, versioned checkpoint owns coordinator recovery state. Public status
// projects operations and observed membership; it is never decoded into effect authority.
// New desired levels are checkpointed before a later reconcile may apply them.
// The unpublished status.reconciliation schema is not automatically migrated:
// an initialized world without valid recovery authority remains fail-closed.
//
// Grove recreation preserves the logical group but starts a new physical lifetime.
// Protected Pods retain kubelet-confirmed terminal evidence until it is durably
// checkpointed. The old clique must disappear and every old allocation must be
// proven stopped before rebinding. Missing evidence and unreachable nodes stay
// fail-closed; no infrastructure fencing or survivor adoption is inferred.
// Fresh formation uses lifetime-scoped adapter journals and a serving check,
// then converges to the preserved desired size rather than the creation seed.
//
// DGD deletion requests child deletion and retains native capacity until the
// child finishes retirement. Only the child installs its effect-owning finalizer.
// A supported backend uses the same durable workflow to drain, commit empty
// membership, and release exact incarnations. Unsupported retirement remains
// explicitly blocked; resource deletion is never treated as process-stop proof.
package enginegroup
