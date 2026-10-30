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

// Package aisimulate lowers a v1beta2 NativeSearchInput snapshot into the public
// AISimulate recommendation input consumed by `aisimulate recommend --stack dynamo`.
//
// Admission and reconciliation use the same lowering. Admission runs it after structural
// validation so malformed overrides and unmappable typed values fail before persistence. The
// Request reconciler snapshots the accepted input into an immutable Run. The Run reconciler
// lowers that snapshot once, records the result in status.aisimulate.input and passes the same
// document to the Job.
//
// Lower starts with the public engine-role underlay required for AISimulate's default
// aggregated-and-disaggregated search. It then translates the typed spec (model, backends,
// hardware, workload, objective) into that input. The derived defaults that no API field
// carries are added next: the request count, a bounded trial budget for a
// KV-load range search, and optimization.strict_sla when an SLA bound is set. Last,
// spec.aisimulate.inputOverrides is applied on top as a JSON Merge Patch (RFC 7386): upstream
// objects merge recursively, arrays and scalars replace the value, and null deletes it. The
// input overrides are authoritative: they win on every path they name, including the derived
// defaults, and are applied once, last.
//
// The document stays unstructured (map[string]any) because its schema belongs to the
// AISimulate release bundled in the operator's Planner image, not to this package. Reads and writes go
// through the helpers of k8s.io/apimachinery/pkg/apis/meta/v1/unstructured.
package aisimulate
