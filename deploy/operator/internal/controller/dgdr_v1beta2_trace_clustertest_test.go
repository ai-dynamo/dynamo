//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	semver "github.com/Masterminds/semver/v3"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/aisimulate"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/clusterenv"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"
)

// The writer gives a trace file an owner the Job's users do not have and the trace group, so the
// sweeper can read it only through the supplemental group of the Run Job's pod.
const (
	clusterTestTraceUID  = 4321
	clusterTestTraceGID  = dgdrRunTraceGroup
	clusterTestTraceMode = "640"
)

// clusterTestTraceFile is the claim and claim-relative path a trace Request reads.
type clusterTestTraceFile struct {
	claim, path string
}

// clusterTestTraceWriter gzips the fixture to the claim file and gives it the requested owner and mode.
const clusterTestTraceWriter = `
import gzip, os, shutil, sys
source, target, uid, gid, mode = sys.argv[1:]
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(source, "rb") as src, gzip.GzipFile(target, "wb", mtime=0) as dst:
    shutil.copyfileobj(src, dst)
os.chown(target, int(uid), int(gid))
os.chmod(target, int(mode, 8))
`

// clusterTestTraceLinker places a symlink at the claim file that points outside the claim.
const clusterTestTraceLinker = `
import os, sys
link, target = sys.argv[1:]
os.makedirs(os.path.dirname(link), exist_ok=True)
os.symlink(target, link)
`

// clusterTestTraceStat reports the claim file's owner, mode and digest as its termination message.
const clusterTestTraceStat = `
import hashlib, os, sys
st = os.lstat(sys.argv[1])
with open(sys.argv[1], "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
with open("/dev/termination-log", "w") as log:
    log.write(f"{st.st_uid} {st.st_gid} {st.st_mode & 0o7777:o} {digest}")
`

// createClusterTestTraceClaim creates the static filesystem claim of trace in env's namespace. Kind
// installs no dynamic provisioner the test may rely on, so the claim binds a PersistentVolume of its
// own on one node, which is deleted with the test.
func createClusterTestTraceClaim(t *testing.T, env *clusterenv.TestEnv, trace clusterTestTraceFile) {
	t.Helper()
	ctx := t.Context()

	t.Log("Pin the trace volume to one node, so every pod that mounts it sees the same files")
	var nodes corev1.NodeList
	require.NoError(t, env.Client().List(ctx, &nodes))
	require.NotEmpty(t, nodes.Items, "the cluster has no nodes")
	hostname := nodes.Items[0].Labels[corev1.LabelHostname]
	require.NotEmpty(t, hostname, "node %q has no %s label", nodes.Items[0].Name, corev1.LabelHostname)

	t.Log("Create the PersistentVolume and the claim bound to it")
	volume := &corev1.PersistentVolume{
		Name: "dgdr-trace-" + env.Namespace(),
		Spec: corev1.PersistentVolumeSpec{
			Capacity:                      corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("16Mi")},
			AccessModes:                   []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
			PersistentVolumeReclaimPolicy: corev1.PersistentVolumeReclaimRetain,
			ClaimRef:                      &corev1.ObjectReference{Namespace: env.Namespace(), Name: trace.claim},
			PersistentVolumeSource: corev1.PersistentVolumeSource{HostPath: &corev1.HostPathVolumeSource{
				Path: "/var/local/dynamo-clustertest/" + env.Namespace(),
				Type: ptr.To(corev1.HostPathDirectoryOrCreate),
			}},
			NodeAffinity: &corev1.VolumeNodeAffinity{Required: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
				MatchExpressions: []corev1.NodeSelectorRequirement{{
					Key: corev1.LabelHostname, Operator: corev1.NodeSelectorOpIn, Values: []string{hostname},
				}},
			}}}},
		},
	}
	require.NoError(t, env.Client().Create(ctx, volume))
	t.Cleanup(func() {
		if err := env.Client().Delete(context.Background(), volume); client.IgnoreNotFound(err) != nil {
			t.Errorf("delete trace PersistentVolume %q: %v", volume.Name, err)
		}
	})
	claim := &corev1.PersistentVolumeClaim{
		Name: trace.claim, Namespace: env.Namespace(),
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes:      []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
			VolumeMode:       ptr.To(corev1.PersistentVolumeFilesystem),
			StorageClassName: new(""),
			VolumeName:       volume.Name,
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("16Mi")},
			},
		},
	}
	require.NoError(t, env.Client().Create(ctx, claim))
}

// runClusterTestTraceJob runs script with args in image, with claim mounted at /claim (read-only
// unless writable) and the files of configMap, if named, at /fixture. It waits for the Job to
// succeed and returns its termination message.
func runClusterTestTraceJob(
	t *testing.T,
	env *clusterenv.TestEnv,
	name, image, claim, configMap string,
	writable bool,
	script string,
	args ...string,
) string {
	t.Helper()
	ctx := t.Context()

	t.Logf("Run the %s Job as root with the trace claim mounted", name)
	pod := corev1.PodSpec{
		RestartPolicy:   corev1.RestartPolicyNever,
		SecurityContext: &corev1.PodSecurityContext{RunAsUser: ptr.To[int64](0), RunAsGroup: ptr.To[int64](0)},
		Containers: []corev1.Container{{
			Name:            "trace",
			Image:           image,
			ImagePullPolicy: corev1.PullIfNotPresent,
			Command:         append([]string{"python3", "-c", script}, args...),
			VolumeMounts:    []corev1.VolumeMount{{Name: "claim", MountPath: "/claim", ReadOnly: !writable}},
		}},
		Volumes: []corev1.Volume{{
			Name:                  "claim",
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: claim, ReadOnly: !writable},
		}},
	}
	if configMap != "" {
		pod.Containers[0].VolumeMounts = append(pod.Containers[0].VolumeMounts, corev1.VolumeMount{Name: "fixture", MountPath: "/fixture", ReadOnly: true})
		pod.Volumes = append(pod.Volumes, corev1.Volume{
			Name:      "fixture",
			ConfigMap: &corev1.ConfigMapVolumeSource{Name: configMap},
		})
	}
	job := &batchv1.Job{
		Name: name, Namespace: env.Namespace(),
		Spec: batchv1.JobSpec{BackoffLimit: ptr.To[int32](0), Template: corev1.PodTemplateSpec{Spec: pod}},
	}
	require.NoError(t, env.Client().Create(ctx, job))

	t.Logf("Wait for the %s Job to succeed", name)
	require.Eventually(t, func() bool {
		require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(job), job))
		require.Zero(t, job.Status.Failed, "%s Job failed: %+v", name, job.Status.Conditions)
		return job.Status.Succeeded == 1
	}, 5*time.Minute, time.Second)

	t.Logf("Read the %s Job's termination message", name)
	var pods corev1.PodList
	require.NoError(t, env.Client().List(ctx, &pods, client.InNamespace(env.Namespace()),
		client.MatchingLabels{batchv1.JobNameLabel: name}))
	require.Len(t, pods.Items, 1)
	require.Len(t, pods.Items[0].Status.ContainerStatuses, 1)
	terminated := pods.Items[0].Status.ContainerStatuses[0].State.Terminated
	require.NotNil(t, terminated, "%s pod did not terminate", name)
	return terminated.Message
}

// writeClusterTestTrace gzips the fixture at fixture into trace and returns the file's owner, mode
// and digest as the claim holds it.
func writeClusterTestTrace(t *testing.T, env *clusterenv.TestEnv, trace clusterTestTraceFile, fixture, image string) string {
	t.Helper()
	ctx := t.Context()

	t.Log("Carry the checked-in trace fixture into the cluster")
	content, err := os.ReadFile(fixture)
	require.NoError(t, err)
	require.NoError(t, env.Client().Create(ctx, &corev1.ConfigMap{
		Name: "trace-fixture", Namespace: env.Namespace(),
		Data: map[string]string{"trace.jsonl": string(content)},
	}))

	runClusterTestTraceJob(t, env, "trace-writer", image, trace.claim, "trace-fixture", true, clusterTestTraceWriter,
		"/fixture/trace.jsonl", filepath.Join("/claim", trace.path),
		fmt.Sprint(clusterTestTraceUID), fmt.Sprint(clusterTestTraceGID), clusterTestTraceMode)
	return statClusterTestTrace(t, env, trace, "trace-before", image)
}

// statClusterTestTrace returns the owner, mode and digest of trace as its claim holds it, read by a
// Job named name.
func statClusterTestTrace(t *testing.T, env *clusterenv.TestEnv, trace clusterTestTraceFile, name, image string) string {
	t.Helper()
	stat := runClusterTestTraceJob(t, env, name, image, trace.claim, "", false, clusterTestTraceStat,
		filepath.Join("/claim", trace.path))
	require.Regexp(t, fmt.Sprintf(`^%d %d %s [0-9a-f]{64}$`, clusterTestTraceUID, clusterTestTraceGID, clusterTestTraceMode), stat)
	return stat
}

// requireClusterTestTraceInput checks that run replayed trace through its fixed mount path.
func requireClusterTestTraceInput(t *testing.T, run *nvidiacomv1beta2.DynamoGraphDeploymentRun, trace clusterTestTraceFile) {
	t.Helper()
	require.NotNil(t, run.Status.AISimulate, "completed Run has no AISimulate status")
	var input struct {
		Traffic struct {
			Source map[string]any `json:"source"`
			Load   map[string]any `json:"load"`
		} `json:"traffic"`
	}
	require.NoError(t, yaml.Unmarshal(run.Status.AISimulate.Input.Raw, &input))
	require.Equal(t, map[string]any{
		"type": "trace", "format": "dynamo", "paths": []any{aisimulate.TracePath(trace.path)},
	}, input.Traffic.Source)
	require.Equal(t, "trace_timestamps", input.Traffic.Load["type"])
}

func TestClusterDynamoGraphDeploymentRequestV1beta2RefusesATraceSymlinkThatLeavesItsClaim(t *testing.T) {
	ctx := t.Context()
	plannerImage := clusterenv.RequireEnv(t, "DYNAMO_CLUSTERTEST_PROFILER_IMAGE")
	operatorImage := clusterenv.RequireEnv(t, "DYNAMO_CLUSTERTEST_OPERATOR_IMAGE")

	t.Log("Create an isolated namespace in the explicitly unlocked cluster")
	env := clusterTestEnv.RunT(t)
	env.BlockReplicaSets()

	t.Log("Place a symlink at the trace path that points at a file outside the claim")
	trace := clusterTestTraceFile{claim: "linked-traces", path: "retune/escape.jsonl.gz"}
	createClusterTestTraceClaim(t, env, trace)
	runClusterTestTraceJob(t, env, "trace-linker", plannerImage, trace.claim, "", true, clusterTestTraceLinker,
		filepath.Join("/claim", trace.path), "/etc/passwd")

	t.Log("Apply a Request that reads the linked trace path through production admission")
	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{
		Name: "aisimulate-trace-symlink", Namespace: env.Namespace(),
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
			DynamoGraphDeploymentRequestNativeSpec: nvidiacomv1beta2.DynamoGraphDeploymentRequestNativeSpec{
				NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
					Model:    &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-32B"},
					Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM},
					Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
						SKU: new("h200_sxm"), Budget: new(int32(2)),
					}},
					Workload: &nvidiacomv1beta2.WorkloadSpec{Trace: &nvidiacomv1beta2.TraceWorkloadSpec{
						Format: nvidiacomv1beta2.TraceFormatDynamo,
						Source: nvidiacomv1beta2.TraceSourceSpec{PVC: &nvidiacomv1beta2.TracePVCSource{
							ClaimName: trace.claim, Path: trace.path,
						}},
					}},
				},
			},
		},
	}
	require.NoError(t, env.Client().Create(ctx, request))

	t.Log("Watch every Run transition before starting the production controllers")
	watchClient, err := client.NewWithWatch(env.RESTConfig(), client.Options{Scheme: newClusterTestScheme()})
	require.NoError(t, err)
	runWatch, err := watchClient.Watch(ctx, &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}, client.InNamespace(env.Namespace()))
	require.NoError(t, err)
	defer runWatch.Stop()
	startClusterTestDGDRV1beta2Controllers(t, env, plannerImage, operatorImage)

	t.Log("The kubelet refuses the subPath, so the sweeper waits and the open Run reports it verbatim")
	var waiting nvidiacomv1beta2.DynamoGraphDeploymentRun
	awaitClusterTestRun(t, runWatch, "the Run did not report its waiting sweeper", func(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) bool {
		waiting = *run.DeepCopy()
		condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
		return condition != nil && condition.Reason == dgdrRunReasonContainerWaiting
	})
	condition := meta.FindStatusCondition(waiting.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	require.Equal(t, metav1.ConditionFalse, condition.Status)
	require.NotNil(t, waiting.Status.JobRef)
	require.Nil(t, waiting.Status.CompletionTime)

	t.Log("The sweeper never started: its pod reports the kubelet's waiting state, which the Run carries verbatim")
	var pods corev1.PodList
	require.NoError(t, env.Client().List(ctx, &pods, client.InNamespace(env.Namespace()),
		client.MatchingLabels{batchv1.JobNameLabel: waiting.Status.JobRef.Name}))
	require.Len(t, pods.Items, 1)
	var sweeper *corev1.ContainerStatus
	for i := range pods.Items[0].Status.ContainerStatuses {
		if pods.Items[0].Status.ContainerStatuses[i].Name == dgdrRunSweeperContainer {
			sweeper = &pods.Items[0].Status.ContainerStatuses[i]
		}
	}
	require.NotNil(t, sweeper)
	require.NotNil(t, sweeper.State.Waiting)
	require.Nil(t, sweeper.State.Running)
	require.Nil(t, sweeper.LastTerminationState.Terminated)
	require.Zero(t, sweeper.RestartCount)
	require.Contains(t, condition.Message, fmt.Sprintf("container %q is waiting: %s: ", dgdrRunSweeperContainer, sweeper.State.Waiting.Reason))

	t.Log("The waiting container alone neither ends the Run nor makes the operator delete its Job")
	require.Never(t, func() bool {
		job := &batchv1.Job{}
		err := env.Client().Get(ctx, client.ObjectKey{Namespace: env.Namespace(), Name: waiting.Status.JobRef.Name}, job)
		run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
		require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(&waiting), run))
		return err != nil || job.DeletionTimestamp != nil ||
			meta.IsStatusConditionTrue(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	}, 20*time.Second, time.Second)

	t.Log("The Run's finalizer protects the claim; deleting the claim cancels the Run")
	claim := &corev1.PersistentVolumeClaim{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKey{Namespace: env.Namespace(), Name: trace.claim}, claim))
	require.Contains(t, claim.Finalizers, dgdrRunTraceClaimFinalizer(waiting.UID))
	require.NoError(t, env.Client().Delete(ctx, claim))
	var canceled nvidiacomv1beta2.DynamoGraphDeploymentRun
	awaitClusterTestRun(t, runWatch, "the Run was not canceled with its claim", func(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) bool {
		canceled = *run.DeepCopy()
		return meta.IsStatusConditionTrue(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	})

	t.Log("The Job and its pod were gone before the Run failed with TraceSourceDeleted")
	var jobs batchv1.JobList
	require.NoError(t, env.Client().List(ctx, &jobs, client.InNamespace(env.Namespace())))
	for i := range jobs.Items {
		require.False(t, metav1.IsControlledBy(&jobs.Items[i], &canceled), "Job %q of the canceled Run still exists", jobs.Items[i].Name)
	}
	require.NoError(t, env.Client().List(ctx, &pods, client.InNamespace(env.Namespace()),
		client.MatchingLabels{batchv1.JobNameLabel: waiting.Status.JobRef.Name}))
	require.Empty(t, pods.Items)
	condition = meta.FindStatusCondition(canceled.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	require.Equal(t, nvidiacomv1beta2.DGDRRunPhaseFailed, canceled.Status.Phase)
	require.Equal(t, dgdrRunReasonTraceSourceDeleted, condition.Reason)

	t.Log("No Candidate was published, and the completed Run releases the claim")
	var candidates nvidiacomv1beta2.DynamoGraphDeploymentCandidateList
	require.NoError(t, env.Client().List(ctx, &candidates, client.InNamespace(env.Namespace())))
	require.Empty(t, candidates.Items)
	require.Eventually(t, func() bool {
		stored := &corev1.PersistentVolumeClaim{}
		err := env.Client().Get(ctx, client.ObjectKeyFromObject(claim), stored)
		if apierrors.IsNotFound(err) {
			return true
		}
		require.NoError(t, err)
		return !slices.Contains(stored.Finalizers, dgdrRunTraceClaimFinalizer(waiting.UID))
	}, time.Minute, time.Second, "the canceled Run did not release its claim")
}

// awaitClusterTestRun reads Run events from runWatch until done accepts one, failing with message
// after five minutes. No Run may ever succeed: none of them may read a trace outside its claim.
func awaitClusterTestRun(
	t *testing.T,
	runWatch watch.Interface,
	message string,
	done func(*nvidiacomv1beta2.DynamoGraphDeploymentRun) bool,
) {
	t.Helper()
	deadline := time.NewTimer(5 * time.Minute)
	defer deadline.Stop()
	for {
		select {
		case event, ok := <-runWatch.ResultChan():
			require.True(t, ok, "Run watch closed: %s", message)
			require.NotEqual(t, watch.Error, event.Type, "Run watch failed")
			run, ok := event.Object.(*nvidiacomv1beta2.DynamoGraphDeploymentRun)
			require.Truef(t, ok, "Run watch returned %T", event.Object)
			require.NotEqual(t, nvidiacomv1beta2.DGDRRunPhaseSucceeded, run.Status.Phase, "Run read a trace outside its claim")
			if done(run) {
				return
			}
		case <-deadline.C:
			t.Fatal(message)
		}
	}
}

// startClusterTestDGDRV1beta2Controllers starts the production Request and Run controllers for env.
func startClusterTestDGDRV1beta2Controllers(t *testing.T, env *clusterenv.TestEnv, plannerImage, operatorImage string) {
	t.Helper()
	operatorVersion, err := clusterTestOperatorVersion()
	require.NoError(t, err)
	parsedOperatorVersion, err := semver.NewVersion(operatorVersion)
	require.NoError(t, err)
	dynamoVersion := fmt.Sprintf("%d.%d.%d", parsedOperatorVersion.Major(), parsedOperatorVersion.Minor(), parsedOperatorVersion.Patch())

	t.Log("Start the production Request, Run, AISimulate Job, and publisher chain")
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{}}
	env.StartManager(func(mgr ctrl.Manager) error {
		setupOptions := SetupOptions{Config: operatorConfig, RuntimeConfig: runtimeConfig}
		if err := SetupDynamoGraphDeploymentRequestV1beta2(mgr, setupOptions); err != nil {
			return err
		}
		return SetupDynamoGraphDeploymentRunV1beta2(mgr, DynamoGraphDeploymentRunSetupOptions{
			SetupOptions:            setupOptions,
			PlannerImage:            plannerImage,
			DynamoVersion:           dynamoVersion,
			OperatorImage:           operatorImage,
			OperatorImagePullPolicy: corev1.PullIfNotPresent,
		})
	})
}
