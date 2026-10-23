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

// Package dgdrrunpublisher is the sidecar that turns the Sweeper's pod-local desired
// state snapshot into Kubernetes state: it creates/deletes DynamoGraphDeploymentCandidates
// and updates the ordered candidate references and progress on
// DynamoGraphDeploymentRun.status.
//
// It is a controller-runtime controller for exactly one DynamoGraphDeploymentRun, with
// two loops that meet at one workqueue key (the Run's namespace/name):
//
//  1. The input observer (Observer) checks the snapshot file the Sweeper writes into a
//     shared emptyDir. On a semantic change it atomically stores the new desired state
//     (InputStore) and enqueues the key. Unchanged input enqueues nothing and backs the
//     check interval off; a change resets it.
//  2. The Reconciler consumes the key. Informer events for the Run, its candidates and
//     this Pod enqueue the same key. It reads the latest stored input and the cached
//     cluster state, then creates missing candidates with server-side apply, completes
//     candidate status (its metrics), updates the Run's published-candidate count, progress
//     time and ordered references only where they differ, and finally deletes obsolete candidates. Failed writes are
//     retried by the workqueue with backoff.
//
// The publisher's own writes come back as watch events and the cache can trail them. The
// Reconciler keeps no record of what it wrote: it decides from the current observations,
// constrains its status writes to the resourceVersion it observed (a stale decision
// conflicts and is retried through the workqueue); see Reconciler.
//
// The manager's cache covers only this Run, its candidates (by the run-UID label) and the
// publisher's own Pod, so there is no API-server polling and no spec comparison:
// candidate specs are immutable once created.
//
// The sidecar is a regular container of the same Job pod as the Sweeper. The Job only
// completes once the publisher has exited, so a successful exit acknowledges that the
// final snapshot was reconciled. The run controller derives the terminal Completed
// condition from the Job result and the publisher's bounded container termination
// message; the publisher never writes a phase.
//
// Termination (see Reconciler):
//
//	terminal Succeeded snapshot reconciled             -> exit 0
//	terminal Failed snapshot reconciled                -> exit ExitRunFailed
//	Sweeper exited non-zero, no terminal snapshot      -> reconcile last snapshot, exit 0
//	                                                      (the Sweeper's exit fails the Job)
//	Sweeper exited zero, no terminal snapshot          -> reconcile last snapshot,
//	                                                      exit ExitProtocolViolation
//	invalid or regressing snapshot, Sweeper exited     -> exit ExitProtocolViolation
//	final/last snapshot cannot be reconciled           -> exit ExitReconcileFailed
package dgdrrunpublisher
