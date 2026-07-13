package runtime

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	rootfsDiffFilename   = "rootfs-diff.tar"
	deletedFilesFilename = "deleted-files.json"
)

// RootfsDiffApplyStats contains low-overhead restore measurements.
type RootfsDiffApplyStats struct {
	SizeBytes        int64
	StatDuration     time.Duration
	ExtractDuration  time.Duration
	ChildRusage      ChildRusageStats
	CgroupBefore     CgroupResourceSnapshot
	CgroupAfter      CgroupResourceSnapshot
	CgroupDelta      CgroupResourceDelta
	CgroupReadErrors []string
}

// ChildRusageStats contains wait4 resource usage for the tar extraction child.
type ChildRusageStats struct {
	UserDuration               time.Duration
	SystemDuration             time.Duration
	MaxRSSKiB                  int64
	MinorFaults                int64
	MajorFaults                int64
	BlockInputOperations       int64
	BlockOutputOperations      int64
	VoluntaryContextSwitches   int64
	InvoluntaryContextSwitches int64
}

// CgroupResourceSnapshot contains low-cost cgroup-v2 counters relevant to rootfs extraction.
type CgroupResourceSnapshot struct {
	Path                string
	MemoryCurrentBytes  uint64
	MemoryEvents        map[string]uint64
	MemoryDirectReclaim map[string]uint64
	IOTotals            map[string]uint64
	CPUStat             map[string]uint64
}

// CgroupResourceDelta is the signed difference between two cgroup snapshots.
type CgroupResourceDelta struct {
	MemoryCurrentBytes  int64
	MemoryEvents        map[string]int64
	MemoryDirectReclaim map[string]int64
	IOTotals            map[string]int64
	CPUStat             map[string]int64
}

// DeletedFilesApplyStats contains low-overhead restore measurements.
type DeletedFilesApplyStats struct {
	SizeBytes      int64
	Entries        int
	Removed        int
	Skipped        int
	ReadDuration   time.Duration
	ParseDuration  time.Duration
	RemoveDuration time.Duration
}

// GetRootFS returns the container's root filesystem path via /host/proc.
func GetRootFS(pid int) (string, error) {
	rootPath := fmt.Sprintf("%s/%d/root", HostProcPath, pid)
	if _, err := os.Stat(rootPath); err != nil {
		return "", fmt.Errorf("rootfs not accessible at %s: %w", rootPath, err)
	}
	return rootPath, nil
}

// GetOverlayUpperDir extracts the overlay upperdir from mountinfo.
func GetOverlayUpperDir(pid int) (string, error) {
	mountInfo, err := ReadMountInfo(pid)
	if err != nil {
		return "", fmt.Errorf("failed to parse mountinfo: %w", err)
	}

	for _, mount := range mountInfo {
		if mount.MountPoint != "/" || mount.FSType != "overlay" {
			continue
		}

		for _, opt := range strings.Split(mount.VFSOptions, ",") {
			if strings.HasPrefix(opt, "upperdir=") {
				return strings.TrimPrefix(opt, "upperdir="), nil
			}
		}
	}

	return "", fmt.Errorf("overlay upperdir not found for pid %d", pid)
}

// CaptureRootfsDiff captures the overlay upperdir to a tar file.
func CaptureRootfsDiff(upperDir, checkpointDir string, exclusions types.OverlaySettings, bindMountDests []string) (string, error) {
	if upperDir == "" {
		return "", fmt.Errorf("upperdir is empty")
	}

	rootfsDiffPath := filepath.Join(checkpointDir, rootfsDiffFilename)

	tarArgs := []string{"--xattrs"}
	for _, excl := range buildExclusions(exclusions) {
		tarArgs = append(tarArgs, "--exclude="+excl)
	}
	for _, dest := range bindMountDests {
		tarArgs = append(tarArgs, "--exclude=."+dest)
	}
	tarArgs = append(tarArgs, "-C", upperDir, "-cf", rootfsDiffPath, ".")

	cmd := exec.Command("tar", tarArgs...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("tar failed: %w (output: %s)", err, string(output))
	}

	return rootfsDiffPath, nil
}

// buildExclusions merges exclusion lists and normalizes paths for tar --exclude patterns.
func buildExclusions(s types.OverlaySettings) []string {
	exclusions := append([]string(nil), s.Exclusions...)
	for i, p := range exclusions {
		if strings.HasPrefix(p, "*") {
			continue
		}
		p = strings.TrimPrefix(p, ".")
		p = strings.TrimPrefix(p, "/")
		exclusions[i] = "./" + p
	}
	return exclusions
}

// CaptureDeletedFiles finds whiteout files and saves them to a JSON file.
func CaptureDeletedFiles(upperDir, checkpointDir string) (bool, error) {
	if upperDir == "" {
		return false, nil
	}

	whiteouts, err := findWhiteoutFiles(upperDir)
	if err != nil {
		return false, fmt.Errorf("failed to find whiteout files: %w", err)
	}

	if len(whiteouts) == 0 {
		return false, nil
	}

	deletedFilesPath := filepath.Join(checkpointDir, deletedFilesFilename)
	data, err := json.Marshal(whiteouts)
	if err != nil {
		return false, fmt.Errorf("failed to marshal whiteouts: %w", err)
	}

	if err := os.WriteFile(deletedFilesPath, data, 0644); err != nil {
		return false, fmt.Errorf("failed to write deleted files: %w", err)
	}

	return true, nil
}

// ApplyRootfsDiff extracts rootfs-diff.tar into the target root.
func ApplyRootfsDiff(checkpointPath, targetRoot string, log logr.Logger) error {
	_, err := ApplyRootfsDiffWithStats(checkpointPath, targetRoot, log)
	return err
}

// ApplyRootfsDiffWithStats extracts rootfs-diff.tar and reports restore measurements.
func ApplyRootfsDiffWithStats(checkpointPath, targetRoot string, log logr.Logger) (RootfsDiffApplyStats, error) {
	var stats RootfsDiffApplyStats
	rootfsDiffPath := filepath.Join(checkpointPath, rootfsDiffFilename)
	statStart := time.Now()
	info, err := os.Stat(rootfsDiffPath)
	stats.StatDuration = time.Since(statStart)
	if os.IsNotExist(err) {
		log.V(1).Info("No rootfs-diff.tar, skipping")
		return stats, nil
	}
	if err == nil {
		stats.SizeBytes = info.Size()
	}

	// --skip-old-files: silently skip files that already exist in the restore target.
	// The rootfs diff only contains overlay upperdir changes (runtime-generated files
	// like triton caches, tmp files) — base image files should not be overwritten.
	log.Info("Applying rootfs diff", "target", targetRoot, "size_bytes", stats.SizeBytes)
	cmd := exec.Command("tar", "--skip-old-files", "-C", targetRoot, "-xf", rootfsDiffPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cgroupBefore, beforeErr := readCgroupResourceSnapshot()
	if beforeErr != nil {
		stats.CgroupReadErrors = append(stats.CgroupReadErrors, "before: "+beforeErr.Error())
	} else {
		stats.CgroupBefore = cgroupBefore
	}
	extractStart := time.Now()
	err = cmd.Run()
	stats.ExtractDuration = time.Since(extractStart)
	if cmd.ProcessState != nil {
		if usage, ok := cmd.ProcessState.SysUsage().(*syscall.Rusage); ok {
			stats.ChildRusage = childRusageStats(usage)
		}
	}
	cgroupAfter, afterErr := readCgroupResourceSnapshot()
	if afterErr != nil {
		stats.CgroupReadErrors = append(stats.CgroupReadErrors, "after: "+afterErr.Error())
	} else {
		stats.CgroupAfter = cgroupAfter
		if beforeErr == nil {
			stats.CgroupDelta = diffCgroupResourceSnapshots(cgroupBefore, cgroupAfter)
		}
	}
	log.Info("Rootfs diff extraction resource usage",
		"child_rusage", stats.ChildRusage,
		"cgroup_before", stats.CgroupBefore,
		"cgroup_after", stats.CgroupAfter,
		"cgroup_delta", stats.CgroupDelta,
		"cgroup_read_errors", stats.CgroupReadErrors,
	)
	if err != nil {
		return stats, fmt.Errorf("tar extract failed: %w", err)
	}
	return stats, nil
}

func childRusageStats(usage *syscall.Rusage) ChildRusageStats {
	return ChildRusageStats{
		UserDuration:               timevalDuration(usage.Utime),
		SystemDuration:             timevalDuration(usage.Stime),
		MaxRSSKiB:                  usage.Maxrss,
		MinorFaults:                usage.Minflt,
		MajorFaults:                usage.Majflt,
		BlockInputOperations:       usage.Inblock,
		BlockOutputOperations:      usage.Oublock,
		VoluntaryContextSwitches:   usage.Nvcsw,
		InvoluntaryContextSwitches: usage.Nivcsw,
	}
}

func timevalDuration(value syscall.Timeval) time.Duration {
	return time.Duration(value.Sec)*time.Second + time.Duration(value.Usec)*time.Microsecond
}

func readCgroupResourceSnapshot() (CgroupResourceSnapshot, error) {
	path, err := selfCgroupV2Path()
	if err != nil {
		return CgroupResourceSnapshot{}, err
	}

	memoryCurrent, err := readUintFile(filepath.Join(path, "memory.current"))
	if err != nil {
		return CgroupResourceSnapshot{}, err
	}
	memoryEvents, err := readFlatCounters(filepath.Join(path, "memory.events"))
	if err != nil {
		return CgroupResourceSnapshot{}, err
	}
	memoryStat, err := readFlatCounters(filepath.Join(path, "memory.stat"))
	if err != nil {
		return CgroupResourceSnapshot{}, err
	}
	memoryDirectReclaim := make(map[string]uint64)
	for name, value := range memoryStat {
		if strings.Contains(name, "direct") || strings.HasPrefix(name, "allocstall") {
			memoryDirectReclaim[name] = value
		}
	}
	ioTotals, err := readIOStatTotals(filepath.Join(path, "io.stat"))
	if err != nil {
		return CgroupResourceSnapshot{}, err
	}
	cpuStat, err := readFlatCounters(filepath.Join(path, "cpu.stat"))
	if err != nil {
		return CgroupResourceSnapshot{}, err
	}

	return CgroupResourceSnapshot{
		Path:                path,
		MemoryCurrentBytes:  memoryCurrent,
		MemoryEvents:        memoryEvents,
		MemoryDirectReclaim: memoryDirectReclaim,
		IOTotals:            ioTotals,
		CPUStat:             cpuStat,
	}, nil
}

func selfCgroupV2Path() (string, error) {
	data, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return "", fmt.Errorf("read /proc/self/cgroup: %w", err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "0::") {
			continue
		}
		return filepath.Join("/sys/fs/cgroup", strings.TrimPrefix(line, "0::/")), nil
	}
	return "", fmt.Errorf("cgroup v2 entry not found")
}

func readUintFile(path string) (uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read %s: %w", path, err)
	}
	value, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parse %s: %w", path, err)
	}
	return value, nil
}

func readFlatCounters(path string) (map[string]uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	counters := make(map[string]uint64)
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			return nil, fmt.Errorf("parse %s: unexpected line %q", path, line)
		}
		value, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse %s counter %s: %w", path, fields[0], err)
		}
		counters[fields[0]] = value
	}
	return counters, nil
}

func readIOStatTotals(path string) (map[string]uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	totals := make(map[string]uint64)
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		fields := strings.Fields(line)
		for _, field := range fields[1:] {
			name, rawValue, ok := strings.Cut(field, "=")
			if !ok {
				return nil, fmt.Errorf("parse %s: unexpected field %q", path, field)
			}
			value, err := strconv.ParseUint(rawValue, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("parse %s counter %s: %w", path, name, err)
			}
			totals[name] += value
		}
	}
	return totals, nil
}

func diffCgroupResourceSnapshots(before, after CgroupResourceSnapshot) CgroupResourceDelta {
	return CgroupResourceDelta{
		MemoryCurrentBytes:  int64(after.MemoryCurrentBytes) - int64(before.MemoryCurrentBytes),
		MemoryEvents:        diffCounters(before.MemoryEvents, after.MemoryEvents),
		MemoryDirectReclaim: diffCounters(before.MemoryDirectReclaim, after.MemoryDirectReclaim),
		IOTotals:            diffCounters(before.IOTotals, after.IOTotals),
		CPUStat:             diffCounters(before.CPUStat, after.CPUStat),
	}
}

func diffCounters(before, after map[string]uint64) map[string]int64 {
	delta := make(map[string]int64, len(after))
	for name, value := range after {
		delta[name] = int64(value) - int64(before[name])
	}
	return delta
}

// ApplyDeletedFiles removes files marked as deleted in the checkpoint.
func ApplyDeletedFiles(checkpointPath, targetRoot string, log logr.Logger) error {
	_, err := ApplyDeletedFilesWithStats(checkpointPath, targetRoot, log)
	return err
}

// ApplyDeletedFilesWithStats removes deleted files and reports restore measurements.
func ApplyDeletedFilesWithStats(checkpointPath, targetRoot string, log logr.Logger) (DeletedFilesApplyStats, error) {
	var stats DeletedFilesApplyStats
	deletedFilesPath := filepath.Join(checkpointPath, deletedFilesFilename)
	readStart := time.Now()
	data, err := os.ReadFile(deletedFilesPath)
	stats.ReadDuration = time.Since(readStart)
	if os.IsNotExist(err) {
		return stats, nil
	}
	if err != nil {
		return stats, fmt.Errorf("failed to read deleted files: %w", err)
	}
	stats.SizeBytes = int64(len(data))

	var deletedFiles []string
	parseStart := time.Now()
	err = json.Unmarshal(data, &deletedFiles)
	stats.ParseDuration = time.Since(parseStart)
	if err != nil {
		return stats, fmt.Errorf("failed to parse deleted files: %w", err)
	}
	stats.Entries = len(deletedFiles)

	removeStart := time.Now()
	targetRootAbs, err := filepath.Abs(targetRoot)
	if err != nil {
		stats.RemoveDuration = time.Since(removeStart)
		return stats, fmt.Errorf("failed to resolve target root %s: %w", targetRoot, err)
	}
	targetRootPrefix := targetRootAbs + string(os.PathSeparator)
	for _, f := range deletedFiles {
		if f == "" {
			stats.Skipped++
			continue
		}
		target := filepath.Join(targetRoot, f)
		targetAbs, err := filepath.Abs(target)
		if err != nil || (targetAbs != targetRootAbs && !strings.HasPrefix(targetAbs, targetRootPrefix)) {
			log.V(1).Info("Skipping out-of-root deleted file entry", "entry", f)
			stats.Skipped++
			continue
		}
		if _, err := os.Stat(target); os.IsNotExist(err) {
			stats.Skipped++
			continue
		} else if err != nil {
			log.V(1).Info("Could not stat deleted file target", "path", target, "error", err)
			stats.Skipped++
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.V(1).Info("Could not delete file", "path", target, "error", err)
			stats.Skipped++
			continue
		}
		stats.Removed++
	}
	stats.RemoveDuration = time.Since(removeStart)
	log.Info("Deleted files applied",
		"count", stats.Removed,
		"entries", stats.Entries,
		"skipped", stats.Skipped,
	)
	return stats, nil
}

// findWhiteoutFiles finds overlay whiteout files in the upperdir.
func findWhiteoutFiles(upperDir string) ([]string, error) {
	var whiteouts []string

	err := filepath.Walk(upperDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		name := info.Name()
		if strings.HasPrefix(name, ".wh.") {
			relPath, err := filepath.Rel(upperDir, path)
			if err != nil {
				return fmt.Errorf("failed to compute relative path for %s: %w", path, err)
			}
			dir := filepath.Dir(relPath)
			deletedFile := strings.TrimPrefix(name, ".wh.")
			deletedPath := deletedFile
			if dir != "." {
				deletedPath = filepath.Join(dir, deletedFile)
			}
			whiteouts = append(whiteouts, deletedPath)
		}
		return nil
	})

	return whiteouts, err
}
