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

// Package dgdrinput builds the effective AISimulate input of a v1beta2
// DynamoGraphDeploymentRequest.
//
// The typed spec (model, backends, hardware, workload, objective) is first
// translated into AISimulate's native input, the "underlay". spec.aisimulate.parameters
// is then applied on top of it as a JSON Merge Patch (RFC 7386): native objects merge
// recursively, arrays and scalars replace the underlay value, and null deletes it.
// Finally the derived defaults that cannot live in the CRD are added: the request
// count, a bounded trial budget for a KV-load range search, and goal.strict_sla.
//
// Everything here is a pure function of the spec. The admission webhooks use it to
// validate the complete merged input, and the Request controller uses it to freeze the
// effective input into the Run, so both see the same bytes.
//
// The input spec must already carry the CRD's structural defaults; this package does not
// repeat them. It targets the native sweeper input (search_space, workload, goal, sweep)
// of the AISimulate version the API describes and is version-bound: the version-to-schema
// registry that validates unknown native keys is a separate piece.
package dgdrinput
