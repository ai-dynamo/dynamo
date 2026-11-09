/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

const (
	testNamespace = "inference"
	testRevision  = "rev-aaaa"
)

type fixture struct {
	now       time.Time
	createdAt metav1.Time
	finished  metav1.Time
	readyLTT  metav1.Time

	dcd       *nvidiacomv1beta1.DynamoComponentDeployment
	lws       *leaderworkersetv1.LeaderWorkerSet
	leaderSTS *appsv1.StatefulSet
	workerSTS *appsv1.StatefulSet
	leaderPod *corev1.Pod
	workerPod *corev1.Pod
}

func newFixture() *fixture {
	now := time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC)
	createdAt := metav1.NewTime(now.Add(-2 * time.Minute))
	finished := metav1.NewTime(now.Add(-30 * time.Second))
	readyLTT := metav1.NewTime(now.Add(-45 * time.Second))

	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "decode",
			Namespace:  testNamespace,
			UID:        "dcd-uid",
			Generation: 2,
		},
	}
	lws := &leaderworkersetv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "decode",
			Namespace:  testNamespace,
			UID:        "lws-uid",
			Generation: 2,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(dcd, nvidiacomv1beta1.DynamoComponentDeploymentGVK),
			},
		},
		Spec: leaderworkersetv1.LeaderWorkerSetSpec{
			Replicas: ptr.To(int32(1)),
			LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
				Size: ptr.To(int32(2)),
			},
		},
		Status: leaderworkersetv1.LeaderWorkerSetStatus{ObservedGeneration: 2},
	}

	leaderSTS := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "decode",
			Namespace: testNamespace,
			UID:       "leader-sts-uid",
			Labels: map[string]string{
				leaderworkersetv1.SetNameLabelKey: "decode",
			},
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(lws, leaderWorkerSetGVK()),
			},
		},
		Spec: appsv1.StatefulSetSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{leaderworkersetv1.RevisionKey: testRevision},
				},
			},
		},
	}

	leaderPod := basePod("decode-0", "leader-pod-uid", "0", "0", createdAt)
	leaderPod.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(leaderSTS, appsv1.SchemeGroupVersion.WithKind(statefulSetKind)),
	}
	failMain(leaderPod, 23, finished, readyLTT)

	workerSTS := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "decode-0",
			Namespace: testNamespace,
			UID:       "worker-sts-uid",
			Labels: map[string]string{
				leaderworkersetv1.SetNameLabelKey:    "decode",
				leaderworkersetv1.GroupIndexLabelKey: "0",
			},
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(leaderPod, corev1.SchemeGroupVersion.WithKind(podKind)),
			},
		},
		Spec: appsv1.StatefulSetSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{leaderworkersetv1.RevisionKey: testRevision},
				},
			},
		},
	}

	workerPod := basePod("decode-0-1", "worker-pod-uid", "0", "1", createdAt)
	workerPod.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(workerSTS, appsv1.SchemeGroupVersion.WithKind(statefulSetKind)),
	}
	failMain(workerPod, 137, finished, readyLTT)

	return &fixture{
		now:       now,
		createdAt: createdAt,
		finished:  finished,
		readyLTT:  readyLTT,
		dcd:       dcd,
		lws:       lws,
		leaderSTS: leaderSTS,
		workerSTS: workerSTS,
		leaderPod: leaderPod,
		workerPod: workerPod,
	}
}

func (f *fixture) reader(objects ...client.Object) client.Client {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = nvidiacomv1beta1.AddToScheme(s)
	_ = leaderworkersetv1.AddToScheme(s)
	return fake.NewClientBuilder().
		WithScheme(s).
		WithIndex(&corev1.Pod{}, SetNameIndex, SetNameIndexValues).
		WithObjects(objects...).
		Build()
}

func (f *fixture) allObjects() []client.Object {
	return []client.Object{f.dcd, f.lws, f.leaderSTS, f.workerSTS, f.leaderPod, f.workerPod}
}

func leaderWorkerSetGVK() schema.GroupVersionKind {
	return schema.GroupVersionKind{
		Group:   "leaderworkerset.x-k8s.io",
		Version: "v1",
		Kind:    leaderWorkerSetKind,
	}
}

func basePod(name, uid, group, worker string, createdAt metav1.Time) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         testNamespace,
			UID:               types.UID(uid),
			ResourceVersion:   "10",
			CreationTimestamp: createdAt,
			Labels: map[string]string{
				leaderworkersetv1.SetNameLabelKey:     "decode",
				leaderworkersetv1.GroupIndexLabelKey:  group,
				leaderworkersetv1.WorkerIndexLabelKey: worker,
				leaderworkersetv1.RevisionKey:         testRevision,
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
}

func failMain(pod *corev1.Pod, exit int32, finished, readyLTT metav1.Time) {
	pod.Status.Conditions = []corev1.PodCondition{{
		Type:               corev1.PodReady,
		Status:             corev1.ConditionFalse,
		LastTransitionTime: readyLTT,
	}}
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
		Name: commonconsts.MainContainerName,
		State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
			ExitCode:   exit,
			Reason:     "Error",
			FinishedAt: finished,
		}},
	}}
}

func healthyMain(pod *corev1.Pod, readyLTT metav1.Time) {
	pod.Status.Conditions = []corev1.PodCondition{{
		Type:               corev1.PodReady,
		Status:             corev1.ConditionTrue,
		LastTransitionTime: readyLTT,
	}}
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
		Name: commonconsts.MainContainerName,
		State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{
			StartedAt: readyLTT,
		}},
	}}
}
