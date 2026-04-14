/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"context"
	"fmt"
	"path/filepath"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	GMSInitContainerName      = "gms-init"
	GMSWeightsServerContainer = "gms-server"
	GMSKVCacheServerContainer = "gms-kv-cache"
	GMSLoaderContainer        = "gms-loader"
	GMSSaverContainer         = "gms-saver"
	GMSSocketsVolumeName      = "gms-sockets"
	GMSControlVolumeName      = "gms-control"
	GMSSocketDir              = "/var/run/nvidia-gms"
	GMSControlDir             = "/tmp/gms-control"
)

const gmsInitCommand = `
import os
from pathlib import Path

for path in (os.environ["GMS_SOCKET_DIR"], os.environ["GMS_CONTROL_DIR"]):
    Path(path).mkdir(parents=True, exist_ok=True)
`

const gmsServerCommand = `
import os
import signal
import subprocess
import sys
import time

import pynvml


STOP_FILE = os.path.join(os.environ["GMS_CONTROL_DIR"], "checkpoint-done")


def devices():
    pynvml.nvmlInit()
    try:
        return list(range(pynvml.nvmlDeviceGetCount()))
    finally:
        pynvml.nvmlShutdown()


processes = [
    subprocess.Popen([
        "gpu-memory-service",
        "--device",
        str(device),
        "--tag",
        os.environ["GMS_SERVER_TAG"],
    ])
    for device in devices()
]

if not processes:
    raise SystemExit("no nvidia devices found")


def shutdown(*_args):
    for process in processes:
        if process.poll() is None:
            process.terminate()


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

while True:
    stop_requested = os.path.exists(STOP_FILE)
    if stop_requested:
        shutdown()
    running = False
    for process in processes:
        code = process.poll()
        if code is None:
            running = True
            continue
        if stop_requested:
            continue
        shutdown()
        sys.exit(code)
    if not running:
        sys.exit(0)
    time.sleep(1)
`

const gmsSaveCommand = `
import json
import os
import ssl
import subprocess
import time
import urllib.request
from pathlib import Path

import pynvml
from gpu_memory_service.common.utils import get_socket_path


SERVICE_TOKEN = open(
    "/var/run/secrets/kubernetes.io/serviceaccount/token",
    encoding="utf-8",
).read().strip()
SERVICE_CA = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
POD_API_URL = (
    "https://"
    + os.environ["KUBERNETES_SERVICE_HOST"]
    + ":"
    + os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    + f"/api/v1/namespaces/{os.environ['POD_NAMESPACE']}/pods/{os.environ['POD_NAME']}"
)
SSL_CONTEXT = ssl.create_default_context(cafile=SERVICE_CA)
STOP_FILE = Path(os.environ["GMS_CONTROL_DIR"]) / "checkpoint-done"


def devices():
    pynvml.nvmlInit()
    try:
        return list(range(pynvml.nvmlDeviceGetCount()))
    finally:
        pynvml.nvmlShutdown()


def checkpoint_pod():
    request = urllib.request.Request(
        POD_API_URL,
        headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
    )
    with urllib.request.urlopen(request, context=SSL_CONTEXT, timeout=5) as response:
        return json.load(response)


def checkpoint_pod_ready(pod):
    status = pod.get("status") or {}
    if str(status.get("phase", "")).strip() != "Running":
        return False
    for condition in status.get("conditions") or []:
        if condition.get("type") == "Ready" and str(condition.get("status", "")).strip() == "True":
            return True
    return False


def main_terminated(pod):
    status = pod.get("status") or {}
    for container in status.get("containerStatuses") or []:
        if container.get("name") != "main":
            continue
        return bool((container.get("state") or {}).get("terminated"))
    return False


print("Waiting for checkpoint pod Ready=True before GMS save", flush=True)
while True:
    try:
        pod = checkpoint_pod()
    except Exception:
        time.sleep(1)
        continue
    if checkpoint_pod_ready(pod):
        break
    if main_terminated(pod):
        raise SystemExit("main container terminated before GMS save could start")
    time.sleep(1)
print("Checkpoint pod is Ready; starting GMS save", flush=True)

try:
    device_ids = devices()
    if not device_ids:
        raise SystemExit("no nvidia devices found")
    for device in device_ids:
        socket_path = get_socket_path(device, "weights")
        while not os.path.exists(socket_path):
            time.sleep(1)
        output_dir = os.path.join(os.environ["GMS_CHECKPOINT_DIR"], "gms", f"device-{device}")
        subprocess.run([
            "python3",
            "-m",
            "gpu_memory_service.cli.storage_runner",
            "save",
            "--output-dir",
            output_dir,
            "--device",
            str(device),
        ], check=True)
finally:
    STOP_FILE.write_text("done", encoding="utf-8")
`

const gmsLoadCommand = `
import os
import subprocess
import time

import pynvml
from gpu_memory_service.common.utils import get_socket_path


def devices():
    pynvml.nvmlInit()
    try:
        return list(range(pynvml.nvmlDeviceGetCount()))
    finally:
        pynvml.nvmlShutdown()


device_ids = devices()
if not device_ids:
    raise SystemExit("no nvidia devices found")

for device in device_ids:
    socket_path = get_socket_path(device, "weights")
    while not os.path.exists(socket_path):
        time.sleep(1)
    input_dir = os.path.join(os.environ["GMS_CHECKPOINT_DIR"], "gms", f"device-{device}")
    subprocess.run([
        "python3",
        "-m",
        "gpu_memory_service.cli.storage_runner",
        "load",
        "--input-dir",
        input_dir,
        "--device",
        str(device),
    ], check=True)

while True:
    time.sleep(3600)
`

func ResolveGMSCheckpointStorage(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	checkpointID string,
	artifactVersion string,
) (snapshotprotocol.Storage, error) {
	if reader == nil {
		return snapshotprotocol.Storage{}, fmt.Errorf("checkpoint client is required")
	}

	daemonSets := &appsv1.DaemonSetList{}
	if err := reader.List(
		ctx,
		daemonSets,
		ctrlclient.InNamespace(namespace),
		ctrlclient.MatchingLabels{snapshotprotocol.SnapshotAgentLabelKey: snapshotprotocol.SnapshotAgentLabelValue},
	); err != nil {
		return snapshotprotocol.Storage{}, fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}

	storage, err := snapshotprotocol.DiscoverStorageFromDaemonSets(namespace, daemonSets.Items)
	if err != nil {
		return snapshotprotocol.Storage{}, err
	}
	return snapshotprotocol.ResolveCheckpointStorage(checkpointID, artifactVersion, storage)
}

func EnsureGMSRestoreSidecars(podSpec *corev1.PodSpec, mainContainer *corev1.Container) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	ensureGMSSharedVolumes(podSpec)
	applyGMSSocketEnv(mainContainer)
	ensureVolumeMount(mainContainer, corev1.VolumeMount{Name: GMSSocketsVolumeName, MountPath: GMSSocketDir})
	ensureInitContainer(podSpec, mainContainer.Image)

	weightsServer := gmsServerContainer(mainContainer.Image, GMSWeightsServerContainer, "weights")
	copyGMSDeviceClaims(mainContainer, &weightsServer)
	ensureGMSContainer(podSpec, weightsServer)

	kvCacheServer := gmsServerContainer(mainContainer.Image, GMSKVCacheServerContainer, "kv_cache")
	copyGMSDeviceClaims(mainContainer, &kvCacheServer)
	ensureGMSContainer(podSpec, kvCacheServer)

	loader := gmsHelperContainer(mainContainer.Image, GMSLoaderContainer, gmsLoadCommand)
	copyGMSDeviceClaims(mainContainer, &loader)
	ensureGMSContainer(podSpec, loader)
}

func EnsureGMSRestoreHelperMounts(podSpec *corev1.PodSpec, storage snapshotprotocol.Storage) {
	loader := findContainer(podSpec, GMSLoaderContainer)
	if loader == nil {
		return
	}
	ensureCheckpointVolume(podSpec, storage.PVCName)
	ensureVolumeMount(loader, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	setEnv(loader, "GMS_CHECKPOINT_DIR", resolveGMSArtifactDir(storage))
}

func EnsureGMSCheckpointJobSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
) error {
	if podSpec == nil || mainContainer == nil {
		return nil
	}
	if len(mainContainer.Resources.Claims) == 0 {
		return fmt.Errorf("gms sidecars require main container resource claims")
	}
	if storage.PVCName == "" || storage.BasePath == "" || storage.Location == "" {
		return fmt.Errorf("gms checkpoint jobs require resolved checkpoint storage")
	}

	ensureGMSSharedVolumes(podSpec)
	applyGMSSocketEnv(mainContainer)
	ensureVolumeMount(mainContainer, corev1.VolumeMount{Name: GMSSocketsVolumeName, MountPath: GMSSocketDir})
	ensureInitContainer(podSpec, mainContainer.Image)

	weightsServer := gmsServerContainer(mainContainer.Image, GMSWeightsServerContainer, "weights")
	copyGMSDeviceClaims(mainContainer, &weightsServer)
	ensureGMSContainer(podSpec, weightsServer)

	kvCacheServer := gmsServerContainer(mainContainer.Image, GMSKVCacheServerContainer, "kv_cache")
	copyGMSDeviceClaims(mainContainer, &kvCacheServer)
	ensureGMSContainer(podSpec, kvCacheServer)

	saver := gmsHelperContainer(mainContainer.Image, GMSSaverContainer, gmsSaveCommand)
	copyGMSDeviceClaims(mainContainer, &saver)
	ensureCheckpointVolume(podSpec, storage.PVCName)
	ensureVolumeMount(&saver, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	setEnv(&saver, "GMS_CHECKPOINT_DIR", resolveGMSArtifactDir(storage))
	ensureGMSContainer(podSpec, saver)
	return nil
}

func resolveGMSArtifactDir(storage snapshotprotocol.Storage) string {
	checkpointRoot := filepath.Dir(filepath.Dir(storage.Location))
	artifactVersion := filepath.Base(storage.Location)
	return filepath.Join(checkpointRoot, "gms", "versions", artifactVersion)
}

func gmsServerContainer(image string, name string, tag string) corev1.Container {
	container := corev1.Container{
		Name:    name,
		Image:   image,
		Command: []string{"python3", "-c", gmsServerCommand},
		Env: []corev1.EnvVar{
			{Name: "GMS_SERVER_TAG", Value: tag},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: GMSSocketsVolumeName, MountPath: GMSSocketDir},
			{Name: GMSControlVolumeName, MountPath: GMSControlDir},
		},
	}
	applyGMSSocketEnv(&container)
	return container
}

func gmsHelperContainer(image string, name string, script string) corev1.Container {
	container := corev1.Container{
		Name:    name,
		Image:   image,
		Command: []string{"python3", "-c", script},
		Env: []corev1.EnvVar{
			{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: GMSSocketsVolumeName, MountPath: GMSSocketDir},
			{Name: GMSControlVolumeName, MountPath: GMSControlDir},
		},
	}
	applyGMSSocketEnv(&container)
	return container
}

func ensureGMSSharedVolumes(podSpec *corev1.PodSpec) {
	ensureVolume(podSpec, corev1.Volume{Name: GMSSocketsVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}})
	ensureVolume(podSpec, corev1.Volume{Name: GMSControlVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}})
}

func ensureInitContainer(podSpec *corev1.PodSpec, image string) {
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == GMSInitContainerName {
			return
		}
	}
	container := corev1.Container{
		Name:    GMSInitContainerName,
		Image:   image,
		Command: []string{"python3", "-c", gmsInitCommand},
		VolumeMounts: []corev1.VolumeMount{
			{Name: GMSSocketsVolumeName, MountPath: GMSSocketDir},
			{Name: GMSControlVolumeName, MountPath: GMSControlDir},
		},
	}
	applyGMSSocketEnv(&container)
	podSpec.InitContainers = append(podSpec.InitContainers, container)
}

func applyGMSSocketEnv(container *corev1.Container) {
	setEnv(container, "TMPDIR", GMSSocketDir)
	setEnv(container, "GMS_SOCKET_DIR", GMSSocketDir)
	setEnv(container, "GMS_CONTROL_DIR", GMSControlDir)
}

func copyGMSDeviceClaims(mainContainer *corev1.Container, container *corev1.Container) {
	if mainContainer == nil || container == nil || len(mainContainer.Resources.Claims) == 0 {
		return
	}
	container.Resources.Claims = append([]corev1.ResourceClaim{}, mainContainer.Resources.Claims...)
}

func ensureCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
	if pvcName == "" {
		return
	}
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == snapshotprotocol.CheckpointVolumeName {
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: snapshotprotocol.CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: pvcName},
		},
	})
}

func ensureVolume(podSpec *corev1.PodSpec, volume corev1.Volume) {
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == volume.Name {
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, volume)
}

func ensureVolumeMount(container *corev1.Container, mount corev1.VolumeMount) {
	for i := range container.VolumeMounts {
		if container.VolumeMounts[i].Name == mount.Name && container.VolumeMounts[i].MountPath == mount.MountPath {
			return
		}
	}
	container.VolumeMounts = append(container.VolumeMounts, mount)
}

func setEnv(container *corev1.Container, name string, value string) {
	for i := range container.Env {
		if container.Env[i].Name != name {
			continue
		}
		container.Env[i].Value = value
		container.Env[i].ValueFrom = nil
		return
	}
	container.Env = append(container.Env, corev1.EnvVar{Name: name, Value: value})
}

func ensureGMSContainer(podSpec *corev1.PodSpec, container corev1.Container) {
	if findContainer(podSpec, container.Name) != nil {
		return
	}
	podSpec.Containers = append(podSpec.Containers, container)
}

func findContainer(podSpec *corev1.PodSpec, name string) *corev1.Container {
	if podSpec == nil {
		return nil
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == name {
			return &podSpec.Containers[i]
		}
	}
	return nil
}
