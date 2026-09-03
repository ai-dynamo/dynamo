/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package lws installs LeaderWorkerSet admission for cluster-backed operator
// tests without running the LeaderWorkerSet controllers.
//
// Unlike the Grove adapter beside it, this package does not mock admission
// behavior: Setup registers the production upstream LWS handlers. Only the
// focused MutatingWebhookConfiguration and ValidatingWebhookConfiguration are
// constructed locally from the upstream webhook markers so the test harness
// can point them at its in-process webhook manager.
package lws
