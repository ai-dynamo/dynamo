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

// Package aisimulate lowers a v1beta2 DynamoGraphDeploymentRequest spec into the native
// AISimulate input that a Run executes.
//
// Lowering sits below the Request reconciler. Admission calls Parameters to check that
// spec.aisimulate.parameters is an object, and Lower to check the effective input, for
// example that it names the hardware. The reconciler freezes the output of Lower into the Run,
// so admission and the Run see the same bytes.
//
// Lower works in three steps. The typed spec (model, backends, hardware, workload, objective)
// is first translated into the native input, the "underlay". The derived defaults that no API
// field carries are then added to it: the request count, a bounded trial budget for a
// KV-load range search, and goal.strict_sla when an SLA bound is set. Last,
// spec.aisimulate.parameters is applied on top as a JSON Merge Patch (RFC 7386): native
// objects merge recursively, arrays and scalars replace the value, and null deletes it. The
// parameters are authoritative: they win on every path they name, including the derived
// defaults, and are applied once, last.
//
// The document stays unstructured (map[string]any) because its schema belongs to the
// AISimulate release the Request selects, not to this package. Reads and writes go
// through the helpers of k8s.io/apimachinery/pkg/apis/meta/v1/unstructured.
package aisimulate
