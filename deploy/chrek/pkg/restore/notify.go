package restore

import (
	criu "github.com/checkpoint-restore/go-criu/v7"
	"github.com/sirupsen/logrus"
)

// RestoreNotify implements criu.Notify for restore callbacks.
// It captures the restored process PID from the PostRestore callback.
type RestoreNotify struct {
	criu.NoNotify // Embed no-op implementation for all methods

	// RestoredPID is the PID of the restored process, set by PostRestore callback
	RestoredPID int32

	// log is the logger for notification events
	log *logrus.Entry
}

// NewRestoreNotify creates a new RestoreNotify handler.
func NewRestoreNotify(log *logrus.Entry) *RestoreNotify {
	return &RestoreNotify{
		log: log,
	}
}

// PreRestore is called before CRIU starts the restore operation.
func (n *RestoreNotify) PreRestore() error {
	if n.log != nil {
		n.log.Debug("CRIU pre-restore notification")
	}
	return nil
}

// PostRestore is called after CRIU completes the restore operation.
// The pid parameter contains the PID of the restored process.
func (n *RestoreNotify) PostRestore(pid int32) error {
	n.RestoredPID = pid
	if n.log != nil {
		n.log.WithField("pid", pid).Info("CRIU post-restore notification: process restored")
	}
	return nil
}

// PostResume is called after the restored process has resumed execution.
func (n *RestoreNotify) PostResume() error {
	if n.log != nil {
		n.log.Debug("CRIU post-resume notification")
	}
	return nil
}
