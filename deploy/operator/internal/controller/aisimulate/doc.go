/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Package aisimulate manipulates the native AISimulate input: the JSON document an AISimulate
// run executes.
//
// The package knows only that contract, as a map or as raw JSON, and nothing of the Kubernetes API
// that produces it. Lowering a DynamoGraphDeploymentRequest into an underlay is the Request
// reconciler's job; this package then adds the derived defaults that no API field carries (the
// request count, a bounded trial budget for a KV-load range search, and goal.strict_sla when an
// SLA bound is set) and applies the caller's parameters last, as a JSON Merge Patch (RFC 7386):
// native objects merge recursively, arrays and scalars replace the value, and null deletes it. The
// parameters are authoritative: they win on every path they name, including the derived defaults,
// and are applied once.
//
// Admission does not call this package: the parameters are opaque there, and the validation of the
// resulting input belongs to the schema of the AISimulate release the Request selects. A stored
// input is decoded with DecodeInput and never built again.
//
// The document stays unstructured (map[string]any) because its schema belongs to that release, not
// to this package. Reads and writes go through the helpers of
// k8s.io/apimachinery/pkg/apis/meta/v1/unstructured.
package aisimulate
