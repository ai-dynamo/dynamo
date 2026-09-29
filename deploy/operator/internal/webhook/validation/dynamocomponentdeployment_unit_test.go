/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"context"
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sptr "k8s.io/utils/ptr"
)

// TestElasticEPSingleReplicaAppliesToLeadersOnly covers a defect that made the feature
// unusable and that no leader-only test could see.
//
// A follower is a deep copy of its leader, so it carries the same
// --enable-elastic-ep --data-parallel-backend ray flags. An unscoped single-replica rule
// therefore reads a scaled follower as a second leader and rejects it -- blocking the one
// operation the feature exists to perform. The rule's own message says "capacity is added
// by scaling followers rather than leaders", which is precisely what it was preventing.
//
// Caught on dynamo-aws-gb300, not in unit tests: patching a follower DCD to replicas 3
// was refused by the admission webhook. Every unit test until then had exercised the
// leader path only.
//
// EP16 at TP4 is one leader plus three followers, so a follower must be able to hold a
// replica count above one for the feature to work at all.
//
// Mutation check: dropping !isElasticEPFollower from the guard in
// validateDynamoComponentDeploymentSpec fails the "a follower scaled to three is accepted"
// subtest -- the only one above the rule's own replicas <= 1 early return.
func TestElasticEPSingleReplicaAppliesToLeadersOnly(t *testing.T) {
	elasticArgs := []string{
		"python3 -m dynamo.vllm --model test --enable-elastic-ep --enable-eplb " +
			"--data-parallel-backend ray --data-parallel-size 4",
	}

	newDCD := func(follower bool, replicas int32) *nvidiacomv1beta1.DynamoComponentDeployment {
		dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "mydgd-decode", Namespace: "default"},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				BackendFramework: "vllm",
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "Decode",
					ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
					Replicas:      k8sptr.To(replicas),
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:    consts.MainContainerName,
								Image:   "nvcr.io/nvidia/ai-dynamo/vllm-runtime:test",
								Command: []string{"/bin/sh", "-c"},
								Args:    elasticArgs,
							}},
						},
					},
				},
			},
		}
		if follower {
			dcd.Name = "mydgd-decode-flw"
			dcd.Annotations = map[string]string{
				consts.KubeAnnotationElasticEPFollower: consts.KubeLabelValueTrue,
			}
		}
		return dcd
	}

	tests := []struct {
		name       string
		follower   bool
		replicas   int32
		wantReject bool
	}{
		{
			// The rule's actual purpose: two leaders would share one <leader>-ray DNS
			// name, each running its own Ray head.
			name:       "a leader above one replica is rejected",
			follower:   false,
			replicas:   2,
			wantReject: true,
		},
		{
			name:     "a single-replica leader is accepted",
			follower: false,
			replicas: 1,
		},
		{
			// EP16 at TP4. This is the case the cluster refused.
			name:     "a follower scaled to three is accepted",
			follower: true,
			replicas: 3,
		},
		{
			name:     "a follower scaled to one is accepted",
			follower: true,
			replicas: 1,
		},
		{
			name:     "a follower at rest is accepted",
			follower: true,
			replicas: 0,
		},
	}

	// Gate on: the single-replica rule only runs at all when the PoC is enabled.
	ctx := features.WithGate(context.Background(), features.Gates{ElasticEPRayPoC: true})
	validator := NewDynamoComponentDeploymentValidator()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := validator.Validate(ctx, newDCD(tt.follower, tt.replicas))

			if tt.wantReject {
				if err == nil || !k8serrors.IsInvalid(err) {
					t.Fatalf("error = %v, want an invalid field error", err)
				}
				// Assert on the rule, not merely on "some error": an unrelated
				// validation failure would otherwise make this subtest vacuous.
				if !strings.Contains(err.Error(), "supports a single leader replica") {
					t.Fatalf("rejected for the wrong reason, want the single-leader-replica rule: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("a follower must be able to carry the replica count that is its whole "+
					"purpose; EP16 at TP4 needs one leader and three followers: %v", err)
			}
		})
	}

	t.Run("the rule does not run at all when the gate is off", func(t *testing.T) {
		offCtx := features.WithGate(context.Background(), features.Gates{})
		if _, err := validator.Validate(offCtx, newDCD(false, 2)); err != nil {
			t.Fatalf("gate off must not apply a rule that describes only the PoC topology: %v", err)
		}
	})

	// A cluster-wide gate can be switched on long after a component was admitted, and
	// admission never re-runs on a flip. Without a ratchet, enabling the gate freezes
	// every already-accepted component with replicas > 1 against ANY edit, not just a
	// replica change -- because the stateless rule fires again on every UPDATE.
	//
	// Mutation check: replacing validateElasticEPSingleReplicaRatcheted with the
	// unratcheted validateElasticEPSingleReplica fails the unrelated-edit subtest.
	t.Run("ratchets an unchanged pre-existing violation", func(t *testing.T) {
		for _, tt := range []struct {
			name       string
			oldRep     int32
			newRep     int32
			mutate     func(*nvidiacomv1beta1.DynamoComponentDeployment)
			wantReject bool
		}{
			{
				// The freeze case: replicas stay at 2, something unrelated changes.
				name:   "an unrelated edit is allowed while replicas stay violating",
				oldRep: 2, newRep: 2,
				mutate: func(d *nvidiacomv1beta1.DynamoComponentDeployment) {
					d.Spec.PodTemplate.Spec.Containers[0].Image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:newer"
				},
			},
			{
				name:   "changing the replica count re-asserts the rule",
				oldRep: 2, newRep: 3,
				wantReject: true,
			},
			{
				// The ratchet must not become a licence to introduce a new violation.
				name:   "a fresh violation is still rejected",
				oldRep: 1, newRep: 2,
				wantReject: true,
			},
			{
				name:   "returning to a single replica is always allowed",
				oldRep: 2, newRep: 1,
			},
		} {
			t.Run(tt.name, func(t *testing.T) {
				oldDCD := newDCD(false, tt.oldRep)
				newObj := newDCD(false, tt.newRep)
				if tt.mutate != nil {
					tt.mutate(newObj)
				}
				_, err := validator.ValidateUpdate(ctx, oldDCD, newObj, runtimeVersionSourceV1Beta1)

				if tt.wantReject {
					if err == nil || !strings.Contains(err.Error(), "supports a single leader replica") {
						t.Fatalf("error = %v, want the single-leader-replica rule", err)
					}
					return
				}
				if err != nil {
					t.Fatalf("enabling a cluster-wide gate must not freeze an existing object "+
						"against unrelated edits: %v", err)
				}
			})
		}
	})
}
