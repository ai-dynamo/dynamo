/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package grove provides the temporary Grove admission integration used by
// cluster-backed operator tests.
//
// The harness does not run Grove controllers, but PodCliqueSets still pass
// through mutating and validating admission at Grove's production webhook
// paths. The handlers in this package are deliberately transparent: native CRD
// schema validation, CEL rules, and OpenAPI defaulting continue to run in the
// Kubernetes API server, while Grove's internal semantic validation is not
// copied into Dynamo.
//
// Replace this package with Grove's public operator/webhook/podcliqueset.Setup
// function once a release containing that API is available.
package grove
