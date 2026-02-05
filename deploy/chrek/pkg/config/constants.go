// constants.go defines shared constants used across checkpoint and restore packages.
package config

const (
	// DevShmDirName is the directory name for captured /dev/shm contents.
	// Used by both checkpoint (to save) and restore (to load) packages.
	DevShmDirName = "dev-shm"
)
