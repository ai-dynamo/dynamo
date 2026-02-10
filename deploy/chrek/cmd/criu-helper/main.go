package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/criulog"
	criu "github.com/checkpoint-restore/go-criu/v7"
	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
	"google.golang.org/protobuf/proto"
)

const (
	defaultCRIULogLevel   = 5
	defaultWorkDirPerm    = 0755
	mountinfoBatchSize    = 100
	mountinfoMinFields    = 5
	mountinfoMountPtIndex = 4
)

// criu-helper is executed via nsenter inside the target container's namespaces.
// It performs remount and CRIU restore using go-criu library.

func main() {
	checkpointPath := flag.String("checkpoint", "", "Path to checkpoint directory (required)")
	logLevel := flag.Int("log-level", defaultCRIULogLevel, "CRIU log level (default: 5 for debug)")
	logFile := flag.String("log-file", "restore.log", "CRIU log filename")
	workDir := flag.String("work-dir", "", "CRIU work directory")
	libDir := flag.String("lib-dir", "", "CRIU plugin library directory")
	cgroupRoot := flag.String("cgroup-root", "", "Cgroup path to restore into (absolute path from host)")
	tcpEstablished := flag.Bool("tcp-established", false, "Restore TCP established connections")
	tcpClose := flag.Bool("tcp-close", false, "Close TCP connections instead of restoring")
	remount := flag.Bool("remount", true, "Remount /proc/sys as rw/ro around restore")
	verbose := flag.Bool("verbose", false, "Verbose logging")
	inheritFd := flag.Bool("inherit-fd", true, "Redirect stdout/stderr to container logging (default: true)")

	flag.Parse()

	// Setup logging - quiet by default to not clutter agent logs
	log := logrus.New()
	log.SetOutput(os.Stderr)
	log.SetFormatter(&logrus.TextFormatter{
		DisableTimestamp: false,
		DisableColors:    true,
	})
	if *verbose {
		log.SetLevel(logrus.DebugLevel)
	} else {
		// Default: only warnings and errors (agent already logs the important stuff)
		log.SetLevel(logrus.WarnLevel)
	}

	// Validate required flags
	if *checkpointPath == "" {
		log.Fatal("--checkpoint is required")
	}

	log.WithFields(logrus.Fields{
		"checkpoint":  *checkpointPath,
		"log_level":   *logLevel,
		"work_dir":    *workDir,
		"lib_dir":     *libDir,
		"cgroup_root": *cgroupRoot,
		"remount":     *remount,
	}).Info("CRIU helper starting")

	// Remount /proc/sys as read-write if requested
	if *remount {
		log.Info("Remounting /proc/sys as read-write")
		if err := unix.Mount("proc", "/proc/sys", "", unix.MS_REMOUNT, ""); err != nil {
			log.WithError(err).Warn("Failed to remount /proc/sys as rw (continuing anyway)")
		} else {
			log.Debug("Successfully remounted /proc/sys as rw")
			// Ensure we restore it to read-only on exit
			defer func() {
				log.Info("Remounting /proc/sys as read-only")
				if err := unix.Mount("proc", "/proc/sys", "", unix.MS_REMOUNT|unix.MS_RDONLY, ""); err != nil {
					log.WithError(err).Warn("Failed to remount /proc/sys as ro")
				}
			}()
		}
	}

	// Perform CRIU restore using go-criu library
	// With SOFT mode, CRIU automatically moves the restored process into the target cgroup
	pid, err := performRestore(*checkpointPath, *logLevel, *logFile, *workDir, *libDir, *cgroupRoot, *tcpEstablished, *tcpClose, *inheritFd, log)
	if err != nil {
		log.WithError(err).Fatal("CRIU restore failed")
	}

	// Output PID to stdout for parent process to capture
	fmt.Fprintf(os.Stdout, "RESTORED_PID=%d\n", pid)
	log.WithField("pid", pid).Info("CRIU restore completed successfully")
}

// performRestore executes CRIU restore using the go-criu library.
func performRestore(checkpointPath string, logLevel int, logFile string, workDir string, libDir string, cgroupRoot string, tcpEstablished bool, tcpClose bool, inheritFd bool, log *logrus.Logger) (int, error) {
	startTime := time.Now()

	imageDir, netNsFile, workDirFile, err := openRestoreFDs(checkpointPath, workDir, log)
	if err != nil {
		return 0, err
	}
	defer imageDir.Close()
	defer netNsFile.Close()
	if workDirFile != nil {
		defer workDirFile.Close()
	}

	inheritFds := buildInheritFds(netNsFile, log)
	workDirFd := getWorkDirFd(workDirFile)
	extMounts := generateExtMounts(log)
	// cgroupRoot is now passed from the agent (read from host before nsenter)
	// No longer extracting it inside criu-helper which runs after nsenter

	criuOpts := buildCRIUOpts(checkpointPath, imageDir, netNsFile, workDirFd, logLevel, logFile, tcpEstablished, tcpClose, cgroupRoot, inheritFds, extMounts)
	criuOpts.InheritFd = inheritFds

	return executeCRIURestore(criuOpts, checkpointPath, workDir, logFile, startTime, log)
}

// openRestoreFDs opens all required file descriptors for restore.
func openRestoreFDs(checkpointPath, workDir string, log *logrus.Logger) (*os.File, *os.File, *os.File, error) {
	imageDir, err := os.Open(checkpointPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to open checkpoint directory: %w", err)
	}

	if _, err := unix.FcntlInt(imageDir.Fd(), unix.F_SETFD, 0); err != nil {
		imageDir.Close()
		return nil, nil, nil, fmt.Errorf("failed to clear CLOEXEC on images dir: %w", err)
	}
	log.WithField("fd", imageDir.Fd()).Debug("Opened checkpoint directory")

	netNsFile, err := os.Open("/proc/1/ns/net")
	if err != nil {
		imageDir.Close()
		return nil, nil, nil, fmt.Errorf("failed to open /proc/1/ns/net: %w", err)
	}

	if _, err := unix.FcntlInt(netNsFile.Fd(), unix.F_SETFD, 0); err != nil {
		imageDir.Close()
		netNsFile.Close()
		return nil, nil, nil, fmt.Errorf("failed to clear CLOEXEC on netns fd: %w", err)
	}

	var workDirFile *os.File
	if workDir != "" {
		if err := os.MkdirAll(workDir, defaultWorkDirPerm); err == nil {
			workDirFile, err = os.Open(workDir)
			if err == nil {
				if _, err := unix.FcntlInt(workDirFile.Fd(), unix.F_SETFD, 0); err != nil {
					log.WithError(err).Warn("Failed to clear CLOEXEC on work dir")
					workDirFile.Close()
					workDirFile = nil
				} else {
					log.WithField("fd", workDirFile.Fd()).Debug("Opened work directory")
				}
			}
		}
	}

	return imageDir, netNsFile, workDirFile, nil
}

// buildInheritFds creates the InheritFd list for external network namespace.
func buildInheritFds(netNsFile *os.File, log *logrus.Logger) []*criurpc.InheritFd {
	inheritFds := []*criurpc.InheritFd{
		{
			Key: proto.String("extNetNs"),
			Fd:  proto.Int32(int32(netNsFile.Fd())),
		},
	}
	log.WithField("netns_fd", netNsFile.Fd()).Info("Network namespace fd prepared")
	return inheritFds
}

// getWorkDirFd returns the work directory FD or -1 if not set.
func getWorkDirFd(workDirFile *os.File) int32 {
	if workDirFile != nil {
		return int32(workDirFile.Fd())
	}
	return -1
}

// generateExtMounts generates external mount mappings with error handling.
func generateExtMounts(log *logrus.Logger) []*criurpc.ExtMountMap {
	extMounts, err := generateExtMountMaps()
	if err != nil {
		log.WithError(err).Warn("Failed to generate ExtMnt")
		return nil
	}
	log.WithField("count", len(extMounts)).Info("Generated ExtMnt mappings")
	return extMounts
}

// buildCRIUOpts constructs the CRIU options structure.
func buildCRIUOpts(checkpointPath string, imageDir, netNsFile *os.File, workDirFd int32, logLevel int, logFile string, tcpEstablished, tcpClose bool, cgroupRoot string, inheritFds []*criurpc.InheritFd, extMounts []*criurpc.ExtMountMap) *criurpc.CriuOpts {
	// Use SOFT mode to enable cgroup management with --cgroup-root
	// SOFT mode allows CRIU to call cgroup_move_into() during restore
	// This moves the restored process into the target cgroup BEFORE memory allocation
	cgMode := criurpc.CriuCgMode_SOFT
	criuOpts := &criurpc.CriuOpts{
		ImagesDirFd: proto.Int32(int32(imageDir.Fd())),
		LogLevel:    proto.Int32(int32(logLevel)),
		LogFile:     proto.String(logFile),

		// Root filesystem - use current container's root
		Root: proto.String("/"),

		// Restore in detached mode
		RstSibling: proto.Bool(true),

		// Mount namespace compatibility mode
		MntnsCompatMode: proto.Bool(true),

		// Network namespace is provided via InheritFd (not JoinNs) because it was
		// marked as external during checkpoint. CRIU expects to receive external
		// namespaces via InheritFd with the matching key ("extNetNs").

		// TCP options
		ShellJob:       proto.Bool(true),
		TcpEstablished: proto.Bool(tcpEstablished),
		TcpClose:       proto.Bool(tcpClose),

		// External Unix socket handling
		ExtUnixSk: proto.Bool(true),

		// Cgroup management - FULL mode with cgroup root redirection
		ManageCgroups:     proto.Bool(true),
		ManageCgroupsMode: &cgMode,

		// Device and inode handling
		EvasiveDevices: proto.Bool(true),
		ForceIrmap:     proto.Bool(true),

		// External mount mappings
		ExtMnt: extMounts,
	}

	// Set --cgroup-root to redirect restored process to placeholder's cgroup
	// This tells CRIU to restore under this path instead of the original checkpoint's path
	if cgroupRoot != "" {
		criuOpts.CgRoot = []*criurpc.CgroupRoot{
			{
				// No controller specified = applies to unified cgroup (cgroup v2)
				Path: proto.String(cgroupRoot),
			},
		}
	}

	// Add work directory if specified
	if workDirFd >= 0 {
		criuOpts.WorkDirFd = proto.Int32(workDirFd)
	}

	criuConfigPath := filepath.Join(checkpointPath, "criu.conf")
	criuOpts.ConfigFile = proto.String(criuConfigPath)

	return criuOpts
}

// executeCRIURestore runs the CRIU restore operation and returns the restored PID.
func executeCRIURestore(criuOpts *criurpc.CriuOpts, checkpointPath, workDir, logFile string, startTime time.Time, log *logrus.Logger) (int, error) {
	c := criu.MakeCriu()
	notify := &notifyHandler{log: log}

	logFields := logrus.Fields{
		"images_dir_fd":    criuOpts.GetImagesDirFd(),
		"root":             criuOpts.GetRoot(),
		"tcp_established":  criuOpts.GetTcpEstablished(),
		"tcp_close":        criuOpts.GetTcpClose(),
		"shell_job":        criuOpts.GetShellJob(),
		"rst_sibling":      criuOpts.GetRstSibling(),
		"mntns_compat":     criuOpts.GetMntnsCompatMode(),
		"work_dir_fd":      criuOpts.GetWorkDirFd(),
		"config_file":      criuOpts.GetConfigFile(),
		"cgroup_mode":      criuOpts.GetManageCgroupsMode(),
		"inherit_fd_count": len(criuOpts.GetInheritFd()),
	}
	if len(criuOpts.GetCgRoot()) > 0 {
		logFields["cgroup_root"] = criuOpts.GetCgRoot()[0].GetPath()
	}
	log.WithFields(logFields).Info("Executing CRIU restore")

	// Log InheritFd details if present
	if len(criuOpts.GetInheritFd()) > 0 {
		for i, ifd := range criuOpts.GetInheritFd() {
			log.WithFields(logrus.Fields{
				"index": i,
				"key":   ifd.GetKey(),
				"fd":    ifd.GetFd(),
			}).Info("InheritFd entry")
		}
	}

	if err := c.Restore(criuOpts, notify); err != nil {
		log.WithError(err).Error("CRIU restore failed")
		criulog.LogErrorsWithWorkDir(checkpointPath, workDir, logFile, log)
		return 0, fmt.Errorf("CRIU restore failed: %w", err)
	}

	duration := time.Since(startTime)
	log.WithFields(logrus.Fields{
		"restored_pid": notify.restoredPID,
		"duration":     duration,
	}).Info("CRIU restore completed")

	if notify.restoredPID > 0 {
		return int(notify.restoredPID), nil
	}

	return 0, fmt.Errorf("could not determine restored process PID")
}

// notifyHandler implements criu.Notify interface
type notifyHandler struct {
	log         *logrus.Logger
	restoredPID int32
}

func (n *notifyHandler) PreDump() error {
	n.log.Debug("CRIU notification: PreDump")
	return nil
}

func (n *notifyHandler) PostDump() error {
	n.log.Debug("CRIU notification: PostDump")
	return nil
}

func (n *notifyHandler) PreRestore() error {
	n.log.Debug("CRIU notification: PreRestore")
	return nil
}

func (n *notifyHandler) PostRestore(pid int32) error {
	n.log.WithField("pid", pid).Info("CRIU notification: PostRestore")
	n.restoredPID = pid
	return nil
}

func (n *notifyHandler) NetworkLock() error {
	n.log.Debug("CRIU notification: NetworkLock")
	return nil
}

func (n *notifyHandler) NetworkUnlock() error {
	n.log.Debug("CRIU notification: NetworkUnlock")
	return nil
}

func (n *notifyHandler) SetupNamespaces(pid int32) error {
	n.log.WithField("pid", pid).Debug("CRIU notification: SetupNamespaces")
	return nil
}

func (n *notifyHandler) PostSetupNamespaces() error {
	n.log.Debug("CRIU notification: PostSetupNamespaces")
	return nil
}

func (n *notifyHandler) PostResume() error {
	n.log.Debug("CRIU notification: PostResume")
	return nil
}

// generateExtMountMaps generates external mount mappings for CRIU restore.
// It parses /proc/1/mountinfo (the restore container's mounts) and creates
// ExtMountMap entries for each mount point.
func generateExtMountMaps() ([]*criurpc.ExtMountMap, error) {
	var maps []*criurpc.ExtMountMap
	addedMounts := make(map[string]bool)

	// Add root filesystem mapping first
	maps = append(maps, &criurpc.ExtMountMap{
		Key: proto.String("/"),
		Val: proto.String("."),
	})
	addedMounts["/"] = true

	// Parse /proc/1/mountinfo for all current mount points
	file, err := os.Open("/proc/1/mountinfo")
	if err != nil {
		return nil, fmt.Errorf("failed to open mountinfo: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < mountinfoMinFields {
			continue
		}
		mountPoint := fields[mountinfoMountPtIndex]

		if mountPoint == "/" || addedMounts[mountPoint] {
			continue
		}

		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(mountPoint),
			Val: proto.String(mountPoint),
		})
		addedMounts[mountPoint] = true
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading mountinfo: %w", err)
	}

	return maps, nil
}
