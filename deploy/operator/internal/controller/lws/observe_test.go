/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"context"
	"errors"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

func TestObserveLeaderAndFollowerCurrentFailure(t *testing.T) {
	tests := []struct {
		name         string
		mutate       func(*fixture)
		wantPod      string
		wantExitCode int32
	}{
		{
			name: "failed leader with healthy follower",
			mutate: func(f *fixture) {
				healthyMain(f.workerPod, f.readyLTT)
			},
			wantPod:      "decode-0",
			wantExitCode: 23,
		},
		{
			name: "failed follower with healthy leader",
			mutate: func(f *fixture) {
				healthyMain(f.leaderPod, f.readyLTT)
			},
			wantPod:      "decode-0-1",
			wantExitCode: 137,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFixture()
			tt.mutate(f)
			reader := f.reader(f.allObjects()...)

			t.Log("Observe one failed role under matching ownership")
			got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
			require.NoError(t, err)

			t.Log("Verify exactly one CurrentFailure witness for that role")
			var failure *Observation
			for i := range got {
				obs := got[i]
				if obs.Verdict == VerdictCurrentFailure {
					require.Nil(t, failure, "unexpected second CurrentFailure: %#v", got)
					failure = &got[i]
					continue
				}
				assert.Equal(t, VerdictNoCurrentFailureEvidence, obs.Verdict)
			}
			require.NotNil(t, failure)
			assert.Equal(t, ReasonCurrentMainNonzeroExit, failure.Reason)
			assert.Equal(t, tt.wantPod, failure.PodName)
			require.NotNil(t, failure.ExitCode)
			assert.Equal(t, tt.wantExitCode, *failure.ExitCode)
			require.NotNil(t, failure.GroupIndex)
			assert.Equal(t, 0, *failure.GroupIndex)
			assert.Equal(t, f.leaderPod.UID, failure.LeaderUID)
			assert.Equal(t, ReadOriginManagerCache, failure.ReadOrigin)
		})
	}

	t.Run("simultaneous failures keep only deterministic leader witness", func(t *testing.T) {
		f := newFixture()
		reader := f.reader(f.allObjects()...)

		t.Log("Observe failed leader and follower under one incarnation")
		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)

		t.Log("Verify dedupe keeps the deterministic leader witness")
		failures := 0
		var witness string
		var exitCode int32
		for _, obs := range got {
			if obs.Verdict != VerdictCurrentFailure {
				continue
			}
			failures++
			witness = obs.PodName
			require.NotNil(t, obs.ExitCode)
			exitCode = *obs.ExitCode
		}
		assert.Equal(t, 1, failures)
		assert.Equal(t, "decode-0", witness)
		assert.Equal(t, int32(23), exitCode)
	})
}

func TestObserveDoesNotMutateInputs(t *testing.T) {
	f := newFixture()
	reader := f.reader(f.allObjects()...)
	dcdBefore := f.dcd.DeepCopy()
	lwsBefore := f.lws.DeepCopy()

	t.Log("Observe without changing the caller's root objects")
	_, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
	require.NoError(t, err)
	assert.Equal(t, dcdBefore, f.dcd)
	assert.Equal(t, lwsBefore, f.lws)
}

func TestObserveIdentityAndRevisionUnknown(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*fixture)
		pod    string
		reason string
	}{
		{
			name: "owning STS template revision mismatch",
			mutate: func(f *fixture) {
				f.leaderSTS.Spec.Template.Labels[leaderworkersetv1.RevisionKey] = "other"
			},
			pod:    "leader",
			reason: ReasonOwningSTSTemplateRevisionMismatch,
		},
		{
			name: "leader STS template revision mismatch for follower",
			mutate: func(f *fixture) {
				healthyMain(f.leaderPod, f.readyLTT)
				f.leaderSTS.Spec.Template.Labels[leaderworkersetv1.RevisionKey] = "other"
			},
			pod:    "worker",
			reason: ReasonLeaderSTSTemplateRevisionMismatch,
		},
		{
			name: "STS UID mismatch",
			mutate: func(f *fixture) {
				f.leaderSTS.UID = "replaced-sts"
			},
			pod:    "leader",
			reason: ReasonSTSIdentityMismatch,
		},
		{
			name: "missing controller owner",
			mutate: func(f *fixture) {
				f.leaderPod.OwnerReferences = nil
			},
			pod:    "leader",
			reason: ReasonMissingOwner,
		},
		{
			name: "non-controller owner",
			mutate: func(f *fixture) {
				// GetControllerOf ignores Controller=false, so the Pod has no controller owner.
				f.leaderPod.OwnerReferences[0].Controller = ptr.To(false)
			},
			pod:    "leader",
			reason: ReasonMissingOwner,
		},
		{
			name: "wrong owner API kind",
			mutate: func(f *fixture) {
				f.leaderPod.OwnerReferences[0].Kind = "ReplicaSet"
			},
			pod:    "leader",
			reason: ReasonOwnerAPIKindMismatch,
		},
		{
			name: "LWS observed generation lag",
			mutate: func(f *fixture) {
				f.lws.Status.ObservedGeneration = 1
			},
			pod:    "none",
			reason: ReasonLWSObservedGenerationLag,
		},
		{
			name: "deleting dependency STS",
			mutate: func(f *fixture) {
				f.leaderSTS.DeletionTimestamp = &f.finished
				f.leaderSTS.Finalizers = []string{"test.nvidia.com/hold"}
			},
			pod:    "leader",
			reason: ReasonDependencyDeleting,
		},
		{
			name: "missing STS in cache",
			mutate: func(f *fixture) {
				// Drop the owning STS object from the reader below.
			},
			pod:    "leader-no-sts",
			reason: ReasonSTSCacheMiss,
		},
		{
			name: "cross-group leader for follower",
			mutate: func(f *fixture) {
				healthyMain(f.leaderPod, f.readyLTT)
				f.leaderPod.Labels[leaderworkersetv1.GroupIndexLabelKey] = "1"
			},
			pod:    "worker",
			reason: ReasonGroupIndexInvalid,
		},
		{
			name: "future FinishedAt",
			mutate: func(f *fixture) {
				f.leaderPod.Status.ContainerStatuses[0].State.Terminated.FinishedAt = metav1.NewTime(f.now.Add(time.Minute))
			},
			pod:    "leader",
			reason: ReasonFinishedAtInFuture,
		},
		{
			name: "missing FinishedAt",
			mutate: func(f *fixture) {
				f.leaderPod.Status.ContainerStatuses[0].State.Terminated.FinishedAt = metav1.Time{}
			},
			pod:    "leader",
			reason: ReasonMissingFinishedAt,
		},
		{
			name: "missing Ready LastTransitionTime",
			mutate: func(f *fixture) {
				f.leaderPod.Status.Conditions = []corev1.PodCondition{{
					Type:   corev1.PodReady,
					Status: corev1.ConditionFalse,
				}}
			},
			pod:    "leader",
			reason: ReasonMissingReadyLastTransitionTime,
		},
		{
			name: "Ready LTT before Pod creation",
			mutate: func(f *fixture) {
				f.leaderPod.Status.Conditions = []corev1.PodCondition{{
					Type:               corev1.PodReady,
					Status:             corev1.ConditionFalse,
					LastTransitionTime: metav1.NewTime(f.createdAt.Add(-time.Second)),
				}}
			},
			pod:    "leader",
			reason: ReasonReadyLTTBeforeCreation,
		},
		{
			name: "Ready LTT beyond clock skew",
			mutate: func(f *fixture) {
				f.leaderPod.Status.Conditions = []corev1.PodCondition{{
					Type:               corev1.PodReady,
					Status:             corev1.ConditionFalse,
					LastTransitionTime: metav1.NewTime(f.now.Add(time.Minute)),
				}}
			},
			pod:    "leader",
			reason: ReasonReadyLTTInFuture,
		},
		{
			name: "FinishedAt before Pod creation",
			mutate: func(f *fixture) {
				f.leaderPod.Status.ContainerStatuses[0].State.Terminated.FinishedAt = metav1.NewTime(f.createdAt.Add(-time.Second))
			},
			pod:    "leader",
			reason: ReasonFinishedAtBeforeCreation,
		},
		{
			name: "missing owning STS template revision",
			mutate: func(f *fixture) {
				// Pod metadata still carries revision; owning STS template does not.
				delete(f.leaderSTS.Spec.Template.Labels, leaderworkersetv1.RevisionKey)
			},
			pod:    "leader",
			reason: ReasonOwningSTSTemplateRevisionMismatch,
		},
		{
			name: "root deleting",
			mutate: func(f *fixture) {
				f.lws.DeletionTimestamp = &f.finished
				f.lws.Finalizers = []string{"test.nvidia.com/hold"}
			},
			pod:    "none",
			reason: ReasonRootDeleting,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFixture()
			tt.mutate(f)

			objects := []client.Object{f.dcd, f.lws}
			switch tt.pod {
			case "leader":
				objects = append(objects, f.leaderSTS, f.leaderPod)
			case "leader-no-sts":
				objects = append(objects, f.leaderPod)
			case "worker":
				objects = append(objects, f.leaderSTS, f.workerSTS, f.leaderPod, f.workerPod)
			}

			reader := f.reader(objects...)

			t.Log("Observe the mutated fixture")
			got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
			require.NoError(t, err)
			require.NotEmpty(t, got)

			t.Log("Verify at least one Unknown observation with the expected reason")
			found := false
			for _, obs := range got {
				assert.NotEqual(t, VerdictCurrentFailure, obs.Verdict)
				if obs.Verdict == VerdictUnknown && obs.Reason == tt.reason {
					found = true
				}
			}
			assert.True(t, found, "observations=%#v", got)
		})
	}
}

func TestObserveNoCurrentFailureEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*fixture)
		reason string
	}{
		{
			name: "no current termination while ReadyTrue",
			mutate: func(f *fixture) {
				healthyMain(f.leaderPod, f.readyLTT)
			},
			reason: ReasonNoCurrentMainTerminated,
		},
		{
			name: "ReadyTrue with current nonzero exit",
			mutate: func(f *fixture) {
				f.leaderPod.Status.Conditions = []corev1.PodCondition{{
					Type:               corev1.PodReady,
					Status:             corev1.ConditionTrue,
					LastTransitionTime: f.readyLTT,
				}}
			},
			reason: ReasonReadyTrue,
		},
		{
			name: "exit zero",
			mutate: func(f *fixture) {
				f.leaderPod.Status.ContainerStatuses[0].State.Terminated.ExitCode = 0
			},
			reason: ReasonMainExitZero,
		},
		{
			name: "no current termination",
			mutate: func(f *fixture) {
				f.leaderPod.Status.ContainerStatuses[0].State = corev1.ContainerState{
					Waiting: &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"},
				}
				f.leaderPod.Status.ContainerStatuses[0].LastTerminationState = corev1.ContainerState{
					Terminated: &corev1.ContainerStateTerminated{ExitCode: 23, FinishedAt: f.finished},
				}
			},
			reason: ReasonNoCurrentMainTerminated,
		},
		{
			name: "deleting pod",
			mutate: func(f *fixture) {
				f.leaderPod.DeletionTimestamp = &f.finished
				f.leaderPod.Finalizers = []string{"test.nvidia.com/hold"}
			},
			reason: ReasonPodDeleting,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFixture()
			tt.mutate(f)
			reader := f.reader(f.dcd, f.lws, f.leaderSTS, f.leaderPod)

			t.Log("Observe a Pod that cannot supply current failure evidence")
			got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
			require.NoError(t, err)
			require.Len(t, got, 1)
			assert.Equal(t, VerdictNoCurrentFailureEvidence, got[0].Verdict)
			assert.Equal(t, tt.reason, got[0].Reason)
			if tt.reason == ReasonNoCurrentMainTerminated {
				assert.Nil(t, got[0].ExitCode)
			}
		})
	}
}

func TestObserveIndexOverflowAndMissingIdentity(t *testing.T) {
	t.Run("group index overflow does not wrap into range", func(t *testing.T) {
		f := newFixture()
		f.leaderPod.Labels[leaderworkersetv1.GroupIndexLabelKey] = "4294967296"
		reader := f.reader(f.dcd, f.lws, f.leaderSTS, f.leaderPod)

		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)
		require.Len(t, got, 1)
		assert.Equal(t, VerdictUnknown, got[0].Verdict)
		assert.Equal(t, ReasonGroupIndexOutOfRange, got[0].Reason)
		assert.Nil(t, got[0].GroupIndex)
	})

	t.Run("worker index overflow does not wrap into range", func(t *testing.T) {
		f := newFixture()
		f.leaderPod.Labels[leaderworkersetv1.WorkerIndexLabelKey] = "4294967296"
		reader := f.reader(f.dcd, f.lws, f.leaderSTS, f.leaderPod)

		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)
		require.Len(t, got, 1)
		assert.Equal(t, VerdictUnknown, got[0].Verdict)
		assert.Equal(t, ReasonWorkerIndexOutOfRange, got[0].Reason)
	})

	t.Run("missing creation timestamp", func(t *testing.T) {
		f := newFixture()
		f.leaderPod.CreationTimestamp = metav1.Time{}
		reader := f.reader(f.dcd, f.lws, f.leaderSTS, f.leaderPod)

		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)
		require.Len(t, got, 1)
		assert.Equal(t, VerdictUnknown, got[0].Verdict)
		assert.Equal(t, ReasonMissingCreationTimestamp, got[0].Reason)
	})

	t.Run("missing Pod UID", func(t *testing.T) {
		f := newFixture()
		f.leaderPod.UID = ""
		reader := f.reader(f.dcd, f.lws, f.leaderSTS, f.leaderPod)

		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)
		require.Len(t, got, 1)
		assert.Equal(t, VerdictUnknown, got[0].Verdict)
		assert.Equal(t, ReasonMissingPodUID, got[0].Reason)
		assert.Nil(t, got[0].GroupIndex)
		assert.Nil(t, got[0].ExitCode)
	})

	t.Run("no indexed pods visible", func(t *testing.T) {
		f := newFixture()
		reader := f.reader(f.dcd, f.lws)

		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)
		require.Len(t, got, 1)
		assert.Equal(t, VerdictUnknown, got[0].Verdict)
		assert.Equal(t, ReasonNoPodsVisible, got[0].Reason)
		assert.Nil(t, got[0].GroupIndex)
		assert.Empty(t, got[0].PodName)
	})
}

func TestObserveRootEpochRecheck(t *testing.T) {
	t.Run("changed LWS generation invalidates CurrentFailure", func(t *testing.T) {
		f := newFixture()
		base := f.reader(f.allObjects()...)
		reader := interceptor.NewClient(base, interceptor.Funcs{
			Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				if err := c.Get(ctx, key, obj, opts...); err != nil {
					return err
				}
				if current, ok := obj.(*leaderworkersetv1.LeaderWorkerSet); ok {
					current.Generation = f.lws.Generation + 1
					current.Status.ObservedGeneration = current.Generation
				}
				return nil
			},
		})

		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)
		require.NotEmpty(t, got)
		for _, obs := range got {
			assert.Equal(t, VerdictUnknown, obs.Verdict)
			assert.Equal(t, ReasonRootEpochChanged, obs.Reason)
		}
	})

	t.Run("DCD cache miss after dependency reads", func(t *testing.T) {
		f := newFixture()
		base := f.reader(f.allObjects()...)
		reader := interceptor.NewClient(base, interceptor.Funcs{
			Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				if _, ok := obj.(*nvidiacomv1beta1.DynamoComponentDeployment); ok {
					return apierrors.NewNotFound(nvidiacomv1beta1.GroupVersion.WithResource("dynamocomponentdeployments").GroupResource(), key.Name)
				}
				return c.Get(ctx, key, obj, opts...)
			},
		})

		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)
		require.NotEmpty(t, got)
		assert.Equal(t, VerdictUnknown, got[0].Verdict)
		assert.Equal(t, ReasonRootCacheMiss, got[0].Reason)
	})

	t.Run("root Get forbidden remains an error", func(t *testing.T) {
		f := newFixture()
		base := f.reader(f.allObjects()...)
		reader := interceptor.NewClient(base, interceptor.Funcs{
			Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				if _, ok := obj.(*nvidiacomv1beta1.DynamoComponentDeployment); ok {
					return errors.New("forbidden")
				}
				return c.Get(ctx, key, obj, opts...)
			},
		})

		_, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.ErrorContains(t, err, "forbidden")
	})

	t.Run("same UID and generation owner change invalidates CurrentFailure", func(t *testing.T) {
		f := newFixture()
		base := f.reader(f.allObjects()...)
		reader := interceptor.NewClient(base, interceptor.Funcs{
			Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				if err := c.Get(ctx, key, obj, opts...); err != nil {
					return err
				}
				if current, ok := obj.(*leaderworkersetv1.LeaderWorkerSet); ok {
					// Keep identity epoch; only retarget the controller owner UID.
					current.OwnerReferences = []metav1.OwnerReference{{
						APIVersion: nvidiacomv1beta1.GroupVersion.String(),
						Kind:       nvidiacomv1beta1.DynamoComponentDeploymentGVK.Kind,
						Name:       f.dcd.Name,
						UID:        "other-dcd-uid",
						Controller: ptr.To(true),
					}}
				}
				return nil
			},
		})

		t.Log("Observe after a same-epoch owner retarget")
		got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
		require.NoError(t, err)
		require.NotEmpty(t, got)
		for _, obs := range got {
			assert.NotEqual(t, VerdictCurrentFailure, obs.Verdict)
			assert.Equal(t, VerdictUnknown, obs.Verdict)
			assert.Equal(t, ReasonDCDIdentityMismatch, obs.Reason)
		}
	})
}

func TestObserveDedupesSameIncarnation(t *testing.T) {
	f := newFixture()
	healthyMain(f.leaderPod, f.readyLTT)

	secondWorker := f.workerPod.DeepCopy()
	secondWorker.Name = "decode-0-1-extra"
	secondWorker.UID = types.UID("worker-pod-uid-2")
	failMain(secondWorker, 23, f.finished, f.readyLTT)

	reader := f.reader(f.dcd, f.lws, f.leaderSTS, f.workerSTS, f.leaderPod, f.workerPod, secondWorker)

	t.Log("Observe two failed followers under one leader incarnation")
	got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
	require.NoError(t, err)

	t.Log("Verify one deterministic CurrentFailure witness remains")
	failures := 0
	var witness string
	for _, obs := range got {
		if obs.Verdict == VerdictCurrentFailure {
			failures++
			witness = obs.PodName
		}
	}
	assert.Equal(t, 1, failures)
	assert.Equal(t, "decode-0-1", witness)
}

func TestObserveDistinctGroupsAreNotDeduped(t *testing.T) {
	f := newFixture()
	f.lws.Spec.Replicas = ptr.To(int32(2))

	otherLeaderSTS := f.leaderSTS.DeepCopy()
	otherLeaderSTS.Name = "decode-group1-sts"
	otherLeaderSTS.UID = "leader-sts-uid-1"

	otherLeader := basePod("decode-1", "leader-pod-uid-1", "1", "0", f.createdAt)
	otherLeader.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: appsAPIVersion(),
		Kind:       statefulSetKind,
		Name:       otherLeaderSTS.Name,
		UID:        otherLeaderSTS.UID,
		Controller: ptr.To(true),
	}}
	failMain(otherLeader, 23, f.finished, f.readyLTT)
	otherLeaderSTS.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: leaderWorkerSetAPIVersion,
		Kind:       leaderWorkerSetKind,
		Name:       f.lws.Name,
		UID:        f.lws.UID,
		Controller: ptr.To(true),
	}}

	reader := f.reader(f.dcd, f.lws, f.leaderSTS, otherLeaderSTS, f.leaderPod, otherLeader)

	t.Log("Observe failed leaders in two groups")
	got, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
	require.NoError(t, err)

	t.Log("Verify distinct groups keep separate CurrentFailure results")
	failures := 0
	for _, obs := range got {
		if obs.Verdict == VerdictCurrentFailure {
			failures++
		}
	}
	assert.Equal(t, 2, failures)
}

func TestObserveReturnsListErrors(t *testing.T) {
	f := newFixture()
	reader := fakeReaderWithListError(f)

	t.Log("Propagate a real indexed list failure")
	_, err := Observe(t.Context(), reader, f.dcd, f.lws, f.now)
	require.ErrorContains(t, err, "forbidden")
}

func fakeReaderWithListError(f *fixture) client.Client {
	base := f.reader(f.dcd, f.lws)
	return interceptor.NewClient(base, interceptor.Funcs{
		List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
			return errors.New("forbidden")
		},
	})
}

func appsAPIVersion() string {
	return "apps/v1"
}
