// Package restore provides CRIU restore operations.
package restore

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/checkpoint-restore/go-criu/v7/crit"
	"github.com/checkpoint-restore/go-criu/v7/crit/images/fdinfo"
	"github.com/checkpoint-restore/go-criu/v7/crit/images/regfile"
	remap_file_path "github.com/checkpoint-restore/go-criu/v7/crit/images/remap-file-path"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"
)

// CreateLinkRemapStubs parses CRIU images to find remapped files and creates
// the link_remap stub files needed for CRIU restore.
//
// Background: When a file is unlink()'d but a process still has an open FD to it,
// CRIU handles this via "link remapping":
//
//   - During dump: CRIU creates a hardlink link_remap.<id> -> original_file
//   - During restore: CRIU does linkat(link_remap.<id>, original_path) to recreate it
//
// The link_remap file only exists on the original node's filesystem. For cross-node
// restore, we must create stub files so CRIU can hardlink from them.
//
// Without these stubs, CRIU fails with:
//
//	"Can't link <path>/link_remap.X -> <path>/original: No such file or directory"
func CreateLinkRemapStubs(checkpointPath string, log *logrus.Entry) error {
	// 1. Parse remap-fpath.img to find files that need remapping
	remapPath := filepath.Join(checkpointPath, "remap-fpath.img")
	remaps, err := parseRemapFpath(remapPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Debug("No remap-fpath.img found, no link_remap stubs needed")
			return nil
		}
		return fmt.Errorf("failed to parse remap-fpath.img: %w", err)
	}

	if len(remaps) == 0 {
		log.Debug("No file remaps found in checkpoint")
		return nil
	}

	// 2. Parse file info to build ID -> fileInfo mapping
	// Try reg-files.img first (older CRIU format), fall back to files.img (newer format)
	regFilesPath := filepath.Join(checkpointPath, "reg-files.img")
	filesPath := filepath.Join(checkpointPath, "files.img")

	var fileMap map[uint32]fileInfo
	var parseErr error

	// Try reg-files.img first (older CRIU format)
	fileMap, parseErr = parseRegFilesWithMode(regFilesPath)
	if parseErr != nil {
		log.WithError(parseErr).Debug("Could not parse reg-files.img, trying files.img")
		// Fall back to files.img (newer format)
		fileMap, parseErr = parseFilesImgWithMode(filesPath)
		if parseErr != nil {
			log.WithError(parseErr).WithField("remap_count", len(remaps)).Warn(
				"Found remap entries but could not parse reg-files.img or files.img — link_remap stubs will not be created")
			return fmt.Errorf("found %d remap entries but could not build file map: %w", len(remaps), parseErr)
		}
	}

	// 3. Create link_remap stub files for all remapped files
	var created []string
	for _, remap := range remaps {
		// Look up the original file by ID
		origInfo, ok := fileMap[remap.origID]
		if !ok {
			log.WithField("orig_id", remap.origID).Debug("Original file ID not found in file map, skipping")
			continue
		}

		// Look up the remap file path by remap ID
		// This is the link_remap.XXX file that CRIU will hardlink FROM
		remapInfo, ok := fileMap[remap.remapID]
		var remapName string
		var mode os.FileMode

		if ok {
			remapName = remapInfo.name
			mode = remapInfo.mode
		} else {
			// If we can't find the remap file in fileMap, construct it
			// CRIU creates link_remap files in the same directory as the original
			// with format: link_remap.<remap_id>
			dir := filepath.Dir(origInfo.name)
			if !strings.HasPrefix(dir, "/") {
				dir = "/" + dir
			}
			remapName = filepath.Join(dir, fmt.Sprintf("link_remap.%d", remap.remapID))
			// Use original file's mode since we don't have the remap file's mode
			mode = origInfo.mode
			log.WithFields(logrus.Fields{
				"orig_id":    remap.origID,
				"remap_id":   remap.remapID,
				"orig_path":  origInfo.name,
				"remap_path": remapName,
				"mode":       fmt.Sprintf("%04o", mode),
			}).Debug("Constructed link_remap path from remap ID")
		}

		// Normalize path
		if !strings.HasPrefix(remapName, "/") {
			remapName = "/" + remapName
		}

		// Check if the link_remap file already exists
		if _, err := os.Stat(remapName); err == nil {
			log.WithField("remap_file", remapName).Debug("Link remap file already exists")
			continue
		}

		// Create the link_remap stub file with correct permissions
		// CRIU will hardlink FROM this file TO the original path
		if err := createLinkRemapStub(remapName, mode); err != nil {
			log.WithError(err).WithFields(logrus.Fields{
				"remap_file": remapName,
				"target":     origInfo.name,
				"mode":       fmt.Sprintf("%04o", mode),
			}).Warn("Failed to create link_remap stub")
			continue
		}

		created = append(created, filepath.Base(remapName))
		log.WithFields(logrus.Fields{
			"remap_file": remapName,
			"target":     origInfo.name,
			"mode":       fmt.Sprintf("%04o", mode),
		}).Debug("Created link_remap stub file")
	}

	if len(created) > 0 {
		log.WithFields(logrus.Fields{
			"count":       len(created),
			"remap_files": created,
		}).Info("Created link_remap stub files for CRIU restore")
	} else {
		log.Debug("No link_remap stubs needed")
	}

	return nil
}

// fileInfo holds file metadata from CRIU checkpoint images
type fileInfo struct {
	name string
	mode os.FileMode
}

// remapEntry represents a file remap entry from CRIU
type remapEntry struct {
	origID    uint32
	remapID   uint32
	remapType int32
}

// parseRemapFpath parses the remap-fpath.img file
func parseRemapFpath(path string) ([]remapEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read and validate magic number using go-criu's ReadMagic
	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read magic: %w", err)
	}
	if magic != "REMAP_FPATH" {
		return nil, fmt.Errorf("unexpected magic: %s (expected REMAP_FPATH)", magic)
	}

	var entries []remapEntry
	sizeBuf := make([]byte, 4)

	for {
		// Read entry size
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read entry data: %w", err)
		}

		// Parse protobuf
		entry := &remap_file_path.RemapFilePathEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to unmarshal entry: %w", err)
		}

		entries = append(entries, remapEntry{
			origID:    entry.GetOrigId(),
			remapID:   entry.GetRemapId(),
			remapType: int32(entry.GetRemapType()),
		})
	}

	return entries, nil
}

// parseRegFilesWithMode parses the reg-files.img file and returns a map of ID -> fileInfo
func parseRegFilesWithMode(path string) (map[uint32]fileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read and validate magic number using go-criu's ReadMagic
	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read magic: %w", err)
	}
	if magic != "REG_FILES" {
		return nil, fmt.Errorf("unexpected magic: %s (expected REG_FILES)", magic)
	}

	fileMap := make(map[uint32]fileInfo)
	sizeBuf := make([]byte, 4)

	for {
		// Read entry size
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read entry data: %w", err)
		}

		// Parse protobuf
		entry := &regfile.RegFileEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to unmarshal entry: %w", err)
		}

		// Convert CRIU mode (includes file type bits) to os.FileMode
		// CRIU stores the full st_mode, we need just the permission bits
		mode := os.FileMode(entry.GetMode() & 0777)
		if mode == 0 {
			mode = 0600 // Default to owner read/write if mode not set
		}

		fileMap[entry.GetId()] = fileInfo{
			name: entry.GetName(),
			mode: mode,
		}
	}

	return fileMap, nil
}

// parseFilesImgWithMode parses the files.img file and returns a map of ID -> fileInfo
// This is the newer CRIU format where file info is embedded in FileEntry messages
func parseFilesImgWithMode(path string) (map[uint32]fileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read and validate magic number using go-criu's ReadMagic
	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read magic: %w", err)
	}
	if magic != "FILES" {
		return nil, fmt.Errorf("unexpected magic: %s (expected FILES)", magic)
	}

	fileMap := make(map[uint32]fileInfo)
	sizeBuf := make([]byte, 4)

	for {
		// Read entry size
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read entry data: %w", err)
		}

		// Parse protobuf as FileEntry
		entry := &fdinfo.FileEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to unmarshal entry: %w", err)
		}

		// Extract fileinfo from embedded RegFileEntry if present
		if entry.GetReg() != nil {
			reg := entry.GetReg()
			// Convert CRIU mode to os.FileMode (permission bits only)
			mode := os.FileMode(reg.GetMode() & 0777)
			if mode == 0 {
				mode = 0600 // Default to owner read/write if mode not set
			}

			fileMap[entry.GetId()] = fileInfo{
				name: reg.GetName(),
				mode: mode,
			}
		}
	}

	return fileMap, nil
}

// createLinkRemapStub creates an empty stub file for CRIU link_remap.
// The file is created with the specified mode to match what CRIU expects.
func createLinkRemapStub(path string, mode os.FileMode) error {
	// Ensure parent directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Create file with the specified mode
	// CRIU validates the file mode matches what was recorded at checkpoint time
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer f.Close()

	// Write 32 bytes of zeros as stub content
	// This provides a minimal valid file for CRIU to hardlink from
	stub := make([]byte, 32)
	if _, err := f.Write(stub); err != nil {
		return fmt.Errorf("failed to write stub data: %w", err)
	}

	return nil
}
