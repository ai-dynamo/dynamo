package criu

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	criulib "github.com/checkpoint-restore/go-criu/v8"
	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// RestoreLogFilename is the CRIU restore log filename (also used by executor/restore.go).
const RestoreLogFilename = "restore.log"

const (
	netNsPath                = "/proc/1/ns/net"
	placeholderFDDir         = "/proc/1/fd"
	hostMountPath            = "/host"
	hostRunRoot              = "/host/run"
	hostContainerdRunRoot    = "/host/run/containerd"
	hostRunNetnsRoot         = "/host/run/netns"
	snapshotControlMountPath = "/snapshot-control"
)

var kubeletPodsRoot = "/host/var/lib/kubelet/pods"
var findHostRunNetnsTarget = findCurrentHostRunNetnsTarget
var restoreHostRunScratchRoot = "/tmp/dynamo-criu-restore-host-run"

// RestoreMountContext identifies restore-time Kubernetes mounts that CRIU
// must remap from checkpoint-time pod paths.
type RestoreMountContext struct {
	PodUID        string
	ContainerName string
}

type restoreExtMountStats struct {
	remappedSnapshotControl int
	remappedKubeletVolumes  int
	remappedHostNetns       int
	remappedHostRunParent   int
	hostNetnsTarget         string
	hostRunScratchTarget    string
}

type restoreExternalMount struct {
	source                  string
	target                  string
	remappedSnapshotControl bool
	remappedKubeletVolume   bool
	remappedHostNetns       bool
	remappedHostRunParent   bool
}

// ExecuteRestore opens the image/work directory FDs, configures inherited
// resources, and calls go-criu Restore. Returns the namespace-relative PID.
func ExecuteRestore(
	criuOpts *criurpc.CriuOpts,
	m *types.CheckpointManifest,
	checkpointPath string,
	log logr.Logger,
) (int32, error) {
	settings := m.CRIUDump.CRIU

	// Open image dir FD
	imageDir, imageDirFD, err := openPathForCRIU(checkpointPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()
	criuOpts.ImagesDirFd = proto.Int32(imageDirFD)

	// Open work dir FD
	if settings.WorkDir != "" {
		if err := os.MkdirAll(settings.WorkDir, 0755); err != nil {
			return 0, fmt.Errorf("failed to create CRIU work directory: %w", err)
		}
		workDirFile, workDirFD, err := openPathForCRIU(settings.WorkDir)
		if err != nil {
			return 0, fmt.Errorf("failed to open CRIU work directory: %w", err)
		}
		defer workDirFile.Close()
		criuOpts.WorkDirFd = proto.Int32(workDirFD)
	}

	c := criulib.MakeCriu()
	if _, err := os.Stat(settings.BinaryPath); err != nil {
		return 0, fmt.Errorf("criu binary not found at %s: %w", settings.BinaryPath, err)
	}
	c.SetCriuPath(settings.BinaryPath)

	netNsFile, err := os.Open(netNsPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open net NS at %s: %w", netNsPath, err)
	}
	defer netNsFile.Close()
	c.AddInheritFd("extNetNs", netNsFile)

	inheritedFiles := registerInheritFDs(c, m.K8s.StdioFDs, log)
	defer closeFiles(inheritedFiles)

	notify := &restoreNotify{log: log}
	log.V(1).Info("Executing go-criu Restore call")
	if err := c.Restore(criuOpts, notify); err != nil {
		log.Error(err, "go-criu Restore returned error")
		logging.LogRestoreErrors(checkpointPath, settings.WorkDir, log)
		return 0, fmt.Errorf("CRIU restore failed: %w", err)
	}

	return notify.restoredPID, nil
}

// BuildRestoreOpts assembles CriuOpts for a CRIU restore from the checkpoint manifest.
// ImagesDirFd and WorkDirFd are left unset — ExecuteRestore opens them at restore time.
func BuildRestoreOpts(m *types.CheckpointManifest, checkpointPath string, cgroupRoot string, mountCtx RestoreMountContext, log logr.Logger) (*criurpc.CriuOpts, error) {
	extMounts, stats, err := buildRestoreExtMounts(m, mountCtx)
	if err != nil {
		return nil, err
	}
	log.V(1).Info("Generated external mount map set", "ext_mount_count", len(extMounts))
	if stats.remappedSnapshotControl > 0 {
		log.Info("Remapped snapshot-control external mounts for CRIU restore",
			"count", stats.remappedSnapshotControl,
			"restore_pod_uid", mountCtx.PodUID,
			"restore_container", mountCtx.ContainerName,
		)
	}
	if stats.remappedKubeletVolumes > 0 {
		log.Info("Remapped kubelet pod volume external mounts for CRIU restore",
			"count", stats.remappedKubeletVolumes,
			"restore_pod_uid", mountCtx.PodUID,
		)
	}
	if stats.remappedHostNetns > 0 {
		log.Info("Remapped host netns external mounts for CRIU restore",
			"count", stats.remappedHostNetns,
			"target", stats.hostNetnsTarget,
		)
	}
	if stats.remappedHostRunParent > 0 {
		log.Info("Remapped host run parent external mount for CRIU restore",
			"count", stats.remappedHostRunParent,
			"target", stats.hostRunScratchTarget,
		)
	}

	settings := m.CRIUDump.CRIU
	criuOpts := &criurpc.CriuOpts{
		LogFile:  proto.String(RestoreLogFilename),
		Root:     proto.String("/"),
		ExtMnt:   extMounts,
		External: restoreNVIDIAExternalDevices(m.CRIUDump.External),
	}
	if len(criuOpts.External) > 0 {
		log.Info("Mapped external NVIDIA devices for CRIU restore", "external", criuOpts.External)
	}
	if err := applyCommonSettings(criuOpts, &settings); err != nil {
		return nil, err
	}

	// Restore-only options
	criuOpts.RstSibling = proto.Bool(settings.RstSibling)
	criuOpts.MntnsCompatMode = proto.Bool(settings.MntnsCompatMode)
	criuOpts.EvasiveDevices = proto.Bool(settings.EvasiveDevices)
	criuOpts.ForceIrmap = proto.Bool(settings.ForceIrmap)

	if cgroupRoot != "" && shouldSetCgroupRoot(criuOpts.GetManageCgroupsMode()) {
		criuOpts.CgRoot = []*criurpc.CgroupRoot{
			{Path: proto.String(cgroupRoot)},
		}
	}

	criuConfPath := filepath.Join(checkpointPath, criuConfFilename)
	if _, err := os.Stat(criuConfPath); err == nil {
		criuOpts.ConfigFile = proto.String(criuConfPath)
	}

	return criuOpts, nil
}

func buildRestoreExtMounts(m *types.CheckpointManifest, mountCtx RestoreMountContext) ([]*criurpc.ExtMountMap, restoreExtMountStats, error) {
	if len(m.CRIUDump.ExtMnt) == 0 {
		return nil, restoreExtMountStats{}, fmt.Errorf("checkpoint manifest is missing criuDump.extMnt")
	}

	restoreMap := map[string]string{"/": "."}
	snapshotControlTarget := resolveSnapshotControlExternalTarget(mountCtx)
	stats := restoreExtMountStats{
		hostNetnsTarget:      findHostRunNetnsTarget(),
		hostRunScratchTarget: restoreHostRunScratchRoot,
	}
	for _, val := range m.CRIUDump.ExtMnt {
		if val == "" || val == "/" {
			continue
		}
		mount := resolveRestoreExternalMount(val, mountCtx, snapshotControlTarget, stats.hostNetnsTarget)
		if mount.remappedSnapshotControl {
			stats.remappedSnapshotControl++
		}
		if mount.remappedKubeletVolume {
			stats.remappedKubeletVolumes++
		}
		if mount.remappedHostNetns {
			stats.remappedHostNetns++
		}
		if mount.remappedHostRunParent {
			stats.remappedHostRunParent++
		}
		restoreMap[mount.source] = mount.target
	}
	return toExtMountMaps(restoreMap), stats, nil
}

func PrepareRestoreMountpoints(m *types.CheckpointManifest, mountCtx RestoreMountContext, log logr.Logger) error {
	mountpoints := remappedRestoreMountpoints(m, mountCtx)
	if len(mountpoints) == 0 {
		return nil
	}
	if _, ok := mountpoints[restoreHostRunScratchRoot]; ok {
		if err := os.RemoveAll(restoreHostRunScratchRoot); err != nil {
			return fmt.Errorf("failed to reset restore host-run scratch tree %s: %w", restoreHostRunScratchRoot, err)
		}
	}

	if err := remountHost(false); err != nil {
		return fmt.Errorf("failed to remount %s read-write for restore mountpoint preparation: %w", hostMountPath, err)
	}
	prepared, prepareErr := createRestoreMountpoints(mountpoints)
	remountErr := remountHost(true)
	if remountErr != nil {
		log.Error(remountErr, "Failed to remount host root read-only after restore mountpoint preparation")
	}
	if prepareErr != nil {
		return prepareErr
	}
	if prepared > 0 {
		log.Info("Prepared remapped external mountpoints for CRIU restore",
			"count", prepared,
			"restore_pod_uid", mountCtx.PodUID,
			"restore_container", mountCtx.ContainerName,
		)
	}
	return remountErr
}

func remappedRestoreMountpoints(m *types.CheckpointManifest, mountCtx RestoreMountContext) map[string]string {
	mountpoints := map[string]string{}
	if m == nil || len(m.CRIUDump.ExtMnt) == 0 {
		return mountpoints
	}

	snapshotControlTarget := resolveSnapshotControlExternalTarget(mountCtx)
	hostNetnsTarget := findHostRunNetnsTarget()
	hostRunParentSeen := false
	var hostRunMounts []restoreExternalMount
	var kubeletMounts []restoreExternalMount
	var containerdRootfsMounts []string
	for _, val := range m.CRIUDump.ExtMnt {
		if val == "" || val == "/" {
			continue
		}
		mount := resolveRestoreExternalMount(val, mountCtx, snapshotControlTarget, hostNetnsTarget)
		cleanPath := filepath.Clean(mount.source)
		if cleanPath == hostRunRoot {
			hostRunParentSeen = true
		}
		if strings.HasPrefix(cleanPath, hostRunRoot+string(os.PathSeparator)) {
			hostRunMounts = append(hostRunMounts, mount)
		}
		if strings.HasPrefix(cleanPath, filepath.Clean(kubeletPodsRoot)+string(os.PathSeparator)) {
			kubeletMounts = append(kubeletMounts, mount)
		}
		if isHostContainerdRootfsMountpoint(cleanPath) {
			containerdRootfsMounts = append(containerdRootfsMounts, cleanPath)
		}
		if strings.HasPrefix(cleanPath, hostMountPath+string(os.PathSeparator)) &&
			cleanPath != hostRunRoot &&
			!isHostRunNetnsMountpoint(cleanPath) &&
			(mount.target != mount.source || isHostContainerdMountpoint(cleanPath)) {
			mountpoints[cleanPath] = mount.target
		}
	}
	if hostRunParentSeen {
		mountpoints[restoreHostRunScratchRoot] = hostRunRoot
		addHostRunScratchMountpoints(mountpoints, hostRunMounts)
	}
	addNestedRootfsMountpoints(mountpoints, containerdRootfsMounts, kubeletMounts)
	addNestedRootfsMountpoints(mountpoints, containerdRootfsMounts, nonContainerdHostRunMounts(hostRunMounts))
	return mountpoints
}

func resolveRestoreExternalMount(path string, mountCtx RestoreMountContext, snapshotControlTarget string, hostNetnsTarget string) restoreExternalMount {
	mount := restoreExternalMount{source: path, target: path}
	if remappedTarget, ok := remapSnapshotControlExternalMount(path, snapshotControlTarget); ok {
		mount.target = remappedTarget
		mount.remappedSnapshotControl = remappedTarget != path
	}
	if remappedTarget, ok := remapKubeletPodVolumeExternalMount(path, mountCtx); ok {
		mount.target = remappedTarget
		mount.remappedKubeletVolume = remappedTarget != path
	}
	if remappedTarget, ok := remapHostRunNetnsExternalMount(path, hostNetnsTarget); ok {
		mount.target = remappedTarget
		mount.remappedHostNetns = remappedTarget != path
	}
	if remappedTarget, ok := remapHostRunExternalMount(path, restoreHostRunScratchRoot); ok {
		mount.target = remappedTarget
		mount.remappedHostRunParent = remappedTarget != path
	}
	return mount
}

func addHostRunScratchMountpoints(mountpoints map[string]string, mounts []restoreExternalMount) {
	for _, mount := range mounts {
		cleanPath := filepath.Clean(mount.source)
		rel, ok := strings.CutPrefix(cleanPath, hostRunRoot+string(os.PathSeparator))
		if !ok {
			continue
		}
		mountpoints[filepath.Join(restoreHostRunScratchRoot, rel)] = mount.target
	}
}

func addNestedRootfsMountpoints(mountpoints map[string]string, rootfsMounts []string, mounts []restoreExternalMount) {
	for _, rootfs := range rootfsMounts {
		for _, mount := range mounts {
			rel, ok := rootfsRelativeExternalMountPath(mount.source)
			if !ok {
				continue
			}
			mountpoints[filepath.Join(rootfs, rel)] = mount.target
		}
	}
}

func nonContainerdHostRunMounts(mounts []restoreExternalMount) []restoreExternalMount {
	filtered := make([]restoreExternalMount, 0, len(mounts))
	for _, mount := range mounts {
		if !isHostContainerdMountpoint(mount.source) {
			filtered = append(filtered, mount)
		}
	}
	return filtered
}

func rootfsRelativeExternalMountPath(path string) (string, bool) {
	cleanPath := filepath.Clean(path)
	if rel, ok := strings.CutPrefix(cleanPath, hostMountPath+string(os.PathSeparator)); ok {
		return rel, true
	}
	if strings.HasPrefix(cleanPath, string(os.PathSeparator)) {
		return strings.TrimPrefix(cleanPath, string(os.PathSeparator)), true
	}
	return "", false
}

func createRestoreMountpoints(mountpoints map[string]string) (int, error) {
	paths := make([]string, 0, len(mountpoints))
	for path := range mountpoints {
		paths = append(paths, path)
	}
	sort.Strings(paths)

	prepared := 0
	for _, path := range paths {
		target := mountpoints[path]
		info, err := os.Stat(target)
		if err != nil {
			if isHostContainerdMountpoint(path) {
				if err := os.MkdirAll(path, 0755); err != nil {
					return prepared, fmt.Errorf("failed to create restore mountpoint directory %s: %w", path, err)
				}
				prepared++
			}
			continue
		}
		if info.IsDir() {
			if err := os.MkdirAll(path, 0755); err != nil {
				return prepared, fmt.Errorf("failed to create restore mountpoint directory %s: %w", path, err)
			}
			prepared++
			continue
		}
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return prepared, fmt.Errorf("failed to create restore mountpoint parent %s: %w", filepath.Dir(path), err)
		}
		f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, info.Mode().Perm())
		if err != nil {
			return prepared, fmt.Errorf("failed to create restore mountpoint file %s: %w", path, err)
		}
		if err := f.Close(); err != nil {
			return prepared, fmt.Errorf("failed to close restore mountpoint file %s: %w", path, err)
		}
		prepared++
	}
	return prepared, nil
}

func remountHost(readOnly bool) error {
	flags := uintptr(syscall.MS_REMOUNT | syscall.MS_BIND)
	if readOnly {
		flags |= syscall.MS_RDONLY
	}
	return syscall.Mount(hostMountPath, hostMountPath, "", flags, "")
}

func isHostContainerdMountpoint(path string) bool {
	return strings.HasPrefix(filepath.Clean(path), hostContainerdRunRoot+string(os.PathSeparator))
}

func isHostContainerdRootfsMountpoint(path string) bool {
	cleanPath := filepath.Clean(path)
	return strings.HasPrefix(cleanPath, hostContainerdRunRoot+string(os.PathSeparator)) &&
		strings.HasSuffix(cleanPath, string(os.PathSeparator)+"rootfs")
}

func isHostRunNetnsMountpoint(path string) bool {
	return strings.HasPrefix(filepath.Clean(path), hostRunNetnsRoot+string(os.PathSeparator))
}

func resolveSnapshotControlExternalTarget(mountCtx RestoreMountContext) string {
	const fallback = snapshotControlMountPath
	podUID := strings.TrimSpace(mountCtx.PodUID)
	containerName := strings.TrimSpace(mountCtx.ContainerName)
	if podUID == "" || containerName == "" {
		return fallback
	}

	matches, err := filepath.Glob(filepath.Join(
		kubeletPodsRoot,
		podUID,
		"volume-subpaths",
		"snapshot-control",
		containerName,
		"*",
	))
	if err != nil || len(matches) == 0 {
		return fallback
	}
	sort.Strings(matches)
	for _, match := range matches {
		info, err := os.Stat(match)
		if err == nil && info.IsDir() {
			return match
		}
	}
	return fallback
}

func remapSnapshotControlExternalMount(path string, target string) (string, bool) {
	cleanPath := filepath.Clean(path)
	if cleanPath == snapshotControlMountPath {
		return cleanPath, false
	}
	if strings.Contains(cleanPath, "/volume-subpaths/snapshot-control/") {
		return target, true
	}
	if strings.HasPrefix(cleanPath, "/host/run/containerd/io.containerd.runtime.v2.task/") &&
		strings.HasSuffix(cleanPath, "/rootfs/snapshot-control") {
		return target, true
	}
	return "", false
}

func remapKubeletPodVolumeExternalMount(path string, mountCtx RestoreMountContext) (string, bool) {
	podUID := strings.TrimSpace(mountCtx.PodUID)
	if podUID == "" {
		return "", false
	}
	cleanPath := filepath.Clean(path)
	rel, ok := strings.CutPrefix(cleanPath, filepath.Clean(kubeletPodsRoot)+string(os.PathSeparator))
	if !ok {
		return "", false
	}

	parts := strings.Split(rel, string(os.PathSeparator))
	if len(parts) < 3 {
		return "", false
	}

	switch parts[1] {
	case "volumes":
		return remapKubeletPodVolume(parts, podUID)
	case "volume-subpaths":
		return remapKubeletPodVolumeSubpath(parts, podUID, mountCtx.ContainerName)
	default:
		return "", false
	}
}

func remapKubeletPodVolume(parts []string, podUID string) (string, bool) {
	if len(parts) < 4 {
		return "", false
	}
	targetParts := append([]string{kubeletPodsRoot, podUID, "volumes"}, parts[2:]...)
	target := filepath.Join(targetParts...)
	if _, err := os.Stat(target); err == nil {
		return target, true
	}

	plugin := parts[2]
	volumeName := parts[3]
	if plugin != "kubernetes.io~projected" || !strings.HasPrefix(volumeName, "kube-api-access-") {
		return "", false
	}

	matches, err := filepath.Glob(filepath.Join(kubeletPodsRoot, podUID, "volumes", plugin, "kube-api-access-*"))
	if err != nil || len(matches) == 0 {
		return "", false
	}
	sort.Strings(matches)
	for _, match := range matches {
		target := filepath.Join(append([]string{match}, parts[4:]...)...)
		if _, err := os.Stat(target); err == nil {
			return target, true
		}
	}
	return "", false
}

func remapKubeletPodVolumeSubpath(parts []string, podUID string, containerName string) (string, bool) {
	if len(parts) < 5 {
		return "", false
	}
	if strings.TrimSpace(containerName) == "" {
		containerName = parts[3]
	}
	targetParts := append([]string{kubeletPodsRoot, podUID, "volume-subpaths", parts[2], containerName}, parts[4:]...)
	target := filepath.Join(targetParts...)
	if _, err := os.Stat(target); err == nil {
		return target, true
	}

	matches, err := filepath.Glob(filepath.Join(kubeletPodsRoot, podUID, "volume-subpaths", parts[2], containerName, "*"))
	if err != nil || len(matches) != 1 {
		return "", false
	}
	target = filepath.Join(append([]string{matches[0]}, parts[5:]...)...)
	if _, err := os.Stat(target); err == nil {
		return target, true
	}
	return "", false
}

func remapHostRunNetnsExternalMount(path string, target string) (string, bool) {
	if target != "" && isHostRunNetnsMountpoint(path) {
		return target, true
	}
	return "", false
}

func remapHostRunExternalMount(path string, target string) (string, bool) {
	if filepath.Clean(path) == hostRunRoot {
		return target, true
	}
	return "", false
}

func findCurrentHostRunNetnsTarget() string {
	netInfo, err := os.Stat(netNsPath)
	if err != nil {
		return ""
	}

	matches, err := filepath.Glob(filepath.Join(hostRunNetnsRoot, "*"))
	if err != nil || len(matches) == 0 {
		return ""
	}
	sort.Strings(matches)
	for _, match := range matches {
		info, err := os.Stat(match)
		if err == nil && os.SameFile(netInfo, info) {
			return match
		}
	}
	return ""
}

func registerInheritFDs(c *criulib.Criu, stdioFDs []string, log logr.Logger) []*os.File {
	if len(stdioFDs) == 0 {
		log.V(1).Info("No stdio FD descriptors in manifest, skipping inherit-fd setup")
		return nil
	}

	var openFiles []*os.File
	for i, target := range stdioFDs {
		if !strings.Contains(target, "pipe:") {
			continue
		}
		// stdin (fd 0) is a read-end pipe; stdout/stderr (fd 1, 2) are write-end
		openMode := os.O_WRONLY
		if i == 0 {
			openMode = os.O_RDONLY
		}
		fdPath := fmt.Sprintf("%s/%d", placeholderFDDir, i)
		f, err := os.OpenFile(fdPath, openMode, 0)
		if err != nil {
			log.V(1).Info("Failed to open placeholder stdio FD, skipping", "fd", i, "target", target, "error", err)
			continue
		}
		openFiles = append(openFiles, f)
		c.AddInheritFd(target, f)
	}

	log.V(1).Info("Registered inherited stdio pipes", "count", len(openFiles))
	return openFiles
}

func closeFiles(files []*os.File) {
	for _, file := range files {
		if file != nil {
			file.Close()
		}
	}
}

type restoreNotify struct {
	criulib.NoNotify
	restoredPID int32
	log         logr.Logger
}

func (n *restoreNotify) PreRestore() error {
	n.log.V(1).Info("CRIU pre-restore")
	return nil
}

func (n *restoreNotify) PostRestore(pid int32) error {
	n.restoredPID = pid
	n.log.Info("CRIU post-restore: process restored", "pid", pid)
	return nil
}
