package cuda

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"
)

const (
	VMMInterposeEnv       = "DYN_SNAPSHOT_CUDA_VMM_INTERPOSE"
	VMMRedisAddressEnv    = "DYN_SNAPSHOT_CUDA_VMM_REDIS_ADDR"
	VMMRedisPasswordEnv   = "DYN_SNAPSHOT_CUDA_VMM_REDIS_PASSWORD"
	VMMRedisRDBPathEnv    = "DYN_SNAPSHOT_CUDA_VMM_REDIS_RDB_PATH"
	VMMRedisRestoreCmdEnv = "DYN_SNAPSHOT_CUDA_VMM_REDIS_RESTORE_COMMAND"

	vmmControlMount     = "/snapshot-control"
	vmmSocketPrefix     = "cuda-vmm-"
	vmmArtifactName     = "cuda-vmm.rdb"
	vmmRestorePoison    = "dynamo:cuda-vmm:restore-poison"
	vmmProtocolMagic    = 0x44564d4d
	vmmProtocolVersion  = 2
	vmmHeaderSize       = 256
	vmmRecordSize       = 96
	vmmPlacementSize    = 40
	vmmMessageSize      = 96
	vmmParticipantSize  = 33
	vmmGPUUUIDSize      = 16
	vmmTimeout          = 30 * time.Second
	vmmMaximumPayload   = 1 << 40
	vmmMaximumRedisBulk = 512 << 20
	vmmMaximumRESPLine  = 4 << 10
	vmmMaximumRightsFDs = 253

	vmmInspect         = 1
	vmmReadOwner       = 2
	vmmDetachImporters = 3
	vmmDetachOwners    = 4
	vmmRestoreOwner    = 5
	vmmRestoreImporter = 6
	vmmIdentify        = 7
	vmmQueryPlacement  = 8

	vmmOwner    = 1
	vmmImporter = 2

	vmmAllocation = 1

	vmmApplicationHandleLive = 1
)

type VMMProcess struct {
	ObservedPID  int
	NamespacePID int
	SocketPath   string
	Participant  string
}

func verifyVMMHealth(ctx context.Context, processes []VMMProcess) error {
	for _, process := range processes {
		if _, err := identifyVMMProcess(process); err != nil {
			return fmt.Errorf(
				"final CUDA VMM shim health verification failed for participant %q: %w",
				process.Participant,
				err,
			)
		}
	}
	return nil
}

type vmmExecutionPlacement map[string]map[string]int32

func currentVMMNode() (string, error) {
	node := strings.TrimSpace(os.Getenv("NODE_NAME"))
	if node == "" {
		return "", errors.New(
			"NODE_NAME is required for CUDA VMM identity placement validation",
		)
	}
	return node, nil
}

func isLowerHex(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && hex.EncodeToString(decoded) == value
}

func vmmDetachPlan(ledger vmmLedger) []vmmDetach {
	var plan []vmmDetach
	for _, resource := range ledger.Resources {
		for _, importer := range resource.Importers {
			plan = append(plan, vmmDetach{
				resourceID:  resource.ID,
				participant: importer.Participant,
				role:        vmmImporter,
			})
		}
	}
	for _, resource := range ledger.Resources {
		plan = append(plan, vmmDetach{
			resourceID:  resource.ID,
			participant: resource.Owner.Participant,
			role:        vmmOwner,
		})
	}
	return plan
}

type vmmDetach struct {
	resourceID  uint64
	participant string
	role        uint32
}

type vmmRestorePlan struct {
	resourceID    uint64
	resourceIndex int
	process       VMMProcess
	payload       []byte
}

func ValidateVMMProcessSet(expected, current []int) error {
	expected = append([]int(nil), expected...)
	current = append([]int(nil), current...)
	slices.Sort(expected)
	slices.Sort(current)
	if !slices.Equal(expected, current) {
		return fmt.Errorf(
			"CUDA process set changed at the VMM checkpoint boundary: expected=%v current=%v",
			expected, current,
		)
	}
	return nil
}

func writeVMMDigestPart(digest io.Writer, value []byte) {
	var size [8]byte
	binary.LittleEndian.PutUint64(size[:], uint64(len(value)))
	_, _ = digest.Write(size[:])
	_, _ = digest.Write(value)
}

func writeVMMDigestObject(digest io.Writer, objectID uint64, contents []byte) {
	var id [8]byte
	binary.LittleEndian.PutUint64(id[:], objectID)
	_, _ = digest.Write(id[:])
	writeVMMDigestPart(digest, contents)
}

type vmmHeader struct {
	Operation      uint16
	Status         int32
	Count          uint32
	AllocationUUID [16]byte
	ObjectID       uint64
	PayloadSize    uint64
	Message        string
	Participant    string
}

type vmmCaptureRecord struct {
	allocationUUID [16]byte
	address        uint64
	size           uint64
	offset         uint64
	role           uint32
	kind           uint32
	handleType     uint32
	flags          uint32
	device         int32
	gpuUUID        string
	properties     []byte
	access         []byte
	accessCount    uint32
	accessSize     uint32
}

type vmmMapping struct {
	Participant           string `json:"participant"`
	Address               uint64 `json:"address"`
	Size                  uint64 `json:"size"`
	GPUUUID               string `json:"gpuUUID"`
	RequestedHandleType   uint32 `json:"requestedHandleType"`
	ApplicationHandleLive bool   `json:"applicationHandleLive"`
	Properties            []byte `json:"properties,omitempty"`
	Access                []byte `json:"access"`
	AccessCount           uint32 `json:"accessCount"`
	AccessSize            uint32 `json:"accessSize"`
}

type vmmResource struct {
	ID        uint64       `json:"id"`
	Kind      string       `json:"kind"`
	Owner     vmmMapping   `json:"owner"`
	Importers []vmmMapping `json:"importers"`

	captureUUID [16]byte
}

type vmmPlacement struct {
	Node     string   `json:"node"`
	GPUUUIDs []string `json:"gpuUUIDs"`
}

type vmmParticipant struct {
	ID        string       `json:"id"`
	Placement vmmPlacement `json:"placement"`
}

type vmmLedger struct {
	Version      uint32           `json:"version"`
	Generation   string           `json:"generation"`
	Participants []vmmParticipant `json:"participants"`
	Resources    []vmmResource    `json:"resources"`
}

type vmmRedisConfig struct {
	address    string
	password   string
	rdbPath    string
	restoreCmd string
}

// DetectVMMInterpose requires every CUDA process to have been launched through
// the scoped preload launcher.
func DetectVMMInterpose(procRoot string, pids []int) (bool, error) {
	enabled := 0
	for _, pid := range pids {
		content, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "environ"))
		if err != nil {
			return false, fmt.Errorf("read CUDA process %d environment: %w", pid, err)
		}
		for _, entry := range strings.Split(string(content), "\x00") {
			if entry == VMMInterposeEnv+"=1" {
				enabled++
				break
			}
		}
	}
	if enabled != 0 && enabled != len(pids) {
		return false, fmt.Errorf(
			"CUDA VMM shim is scoped to %d of %d CUDA processes; all participants are required",
			enabled, len(pids),
		)
	}
	return enabled != 0, nil
}

func NewVMMGeneration() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", fmt.Errorf("generate CUDA VMM generation: %w", err)
	}
	return hex.EncodeToString(value[:]), nil
}

func CheckpointVMMProcesses(
	procRoot string,
	observedPIDs []int,
	namespacePIDs []int,
) ([]VMMProcess, error) {
	if len(observedPIDs) != len(namespacePIDs) {
		return nil, fmt.Errorf(
			"CUDA PID mapping count mismatch: observed=%d namespace=%d",
			len(observedPIDs), len(namespacePIDs),
		)
	}
	processes := make([]VMMProcess, 0, len(observedPIDs))
	for index, observedPID := range observedPIDs {
		namespacePID := namespacePIDs[index]
		path := filepath.Join(
			procRoot,
			strconv.Itoa(observedPID),
			"root",
			strings.TrimPrefix(vmmControlMount, "/"),
			fmt.Sprintf("%s%d.sock", vmmSocketPrefix, namespacePID),
		)
		if err := requireVMMSocket(path); err != nil {
			return nil, fmt.Errorf("CUDA process %d VMM endpoint: %w", observedPID, err)
		}
		process, err := identifyVMMProcess(VMMProcess{
			ObservedPID:  observedPID,
			NamespacePID: namespacePID,
			SocketPath:   path,
		})
		if err != nil {
			return nil, fmt.Errorf("identify CUDA process %d VMM endpoint: %w", observedPID, err)
		}
		processes = append(processes, process)
	}
	return processes, nil
}

func RestoreVMMProcesses(
	observedPIDs []int,
	namespacePIDs []int,
) ([]VMMProcess, error) {
	return restoreVMMProcesses(vmmControlMount, observedPIDs, namespacePIDs)
}

func restoreVMMProcesses(
	controlDir string,
	observedPIDs []int,
	namespacePIDs []int,
) ([]VMMProcess, error) {
	if len(observedPIDs) != len(namespacePIDs) {
		return nil, fmt.Errorf(
			"CUDA PID mapping count mismatch: observed=%d namespace=%d",
			len(observedPIDs), len(namespacePIDs),
		)
	}
	processes := make([]VMMProcess, 0, len(observedPIDs))
	for index, observedPID := range observedPIDs {
		namespacePID := namespacePIDs[index]
		path := filepath.Join(
			controlDir,
			fmt.Sprintf("%s%d.sock", vmmSocketPrefix, namespacePID),
		)
		if err := requireVMMSocket(path); err != nil {
			return nil, fmt.Errorf(
				"restored CUDA process %d (namespace PID %d) VMM endpoint: %w",
				observedPID, namespacePID, err,
			)
		}
		process, err := identifyVMMProcess(VMMProcess{
			ObservedPID:  observedPID,
			NamespacePID: namespacePID,
			SocketPath:   path,
		})
		if err != nil {
			return nil, fmt.Errorf(
				"identify restored CUDA process %d (namespace PID %d) VMM endpoint: %w",
				observedPID, namespacePID, err,
			)
		}
		processes = append(processes, process)
	}
	return processes, nil
}

func identifyVMMProcess(process VMMProcess) (VMMProcess, error) {
	response, err := exchangeVMM(
		context.Background(),
		process,
		vmmHeader{Operation: vmmIdentify},
		nil,
		-1,
		vmmResponseFDNone,
	)
	if err != nil {
		return VMMProcess{}, err
	}
	if len(response.payload) != 0 || response.header.Participant == "" {
		return VMMProcess{}, errors.New("VMM identify returned incomplete logical identity")
	}
	process.Participant = response.header.Participant
	return process, nil
}

func requireVMMSocket(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("required launcher-scoped socket %q: %w", path, err)
	}
	if info.Mode()&os.ModeSocket == 0 {
		return fmt.Errorf("%q is not a Unix socket", path)
	}
	return nil
}

// ValidateVMMArtifact rejects manifest/artifact disagreement before CRIU restore.
func ValidateVMMArtifact(checkpointDir string, enabled bool) error {
	path := filepath.Join(checkpointDir, vmmArtifactName)
	info, err := os.Lstat(path)
	if !enabled {
		if err == nil {
			return fmt.Errorf("CUDA VMM RDB artifact %q is present without manifest opt-in", path)
		}
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("inspect CUDA VMM RDB artifact %q: %w", path, err)
	}
	if err != nil {
		return fmt.Errorf("inspect required CUDA VMM RDB artifact %q: %w", path, err)
	}
	if !info.Mode().IsRegular() || info.Size() == 0 {
		return fmt.Errorf("required CUDA VMM RDB artifact %q must be a regular non-empty file", path)
	}
	return nil
}

// PrepareVMM captures the complete POSIX-FD VMM ledger, detaches every managed
// mapping, and exports the dedicated Redis RDB into the checkpoint artifact.
func PrepareVMM(
	ctx context.Context,
	processes []VMMProcess,
	generation string,
	checkpointDir string,
	log logr.Logger,
) (string, error) {
	if generation == "" {
		return "", errors.New("CUDA VMM generation is required")
	}
	config, err := loadVMMRedisConfig(false)
	if err != nil {
		return "", err
	}
	sourceNode, err := currentVMMNode()
	if err != nil {
		return "", err
	}
	ledger, err := inspectVMM(ctx, processes, generation, sourceNode)
	if err != nil {
		return "", err
	}
	client, err := dialVMMRedis(ctx, config)
	if err != nil {
		return "", err
	}
	defer client.Close()
	if err := client.expectOK("FLUSHDB"); err != nil {
		return "", fmt.Errorf("clear dedicated CUDA VMM Redis database: %w", err)
	}
	encoded, err := json.Marshal(ledger)
	if err != nil {
		return "", fmt.Errorf("encode CUDA VMM ledger: %w", err)
	}
	if err := client.set(vmmRedisKey(generation, "ledger"), encoded); err != nil {
		return "", err
	}
	if err := client.set(vmmRedisKey(generation, "state"), []byte("capturing")); err != nil {
		return "", err
	}
	digest := sha256.New()
	writeVMMDigestPart(digest, encoded)
	for index := range ledger.Resources {
		resource := &ledger.Resources[index]
		owner, err := findVMMProcess(processes, resource.Owner.Participant)
		if err != nil {
			return "", err
		}
		response, err := exchangeVMM(
			ctx,
			owner,
			vmmHeader{
				Operation:      vmmReadOwner,
				AllocationUUID: resource.captureUUID,
				ObjectID:       resource.ID,
			},
			nil,
			-1,
			vmmResponseFDNone,
		)
		if err != nil {
			return "", fmt.Errorf("capture CUDA VMM resource %d bytes: %w", resource.ID, err)
		}
		if response.header.PayloadSize != resource.Owner.Size ||
			uint64(len(response.payload)) != resource.Owner.Size {
			return "", fmt.Errorf(
				"CUDA VMM resource %d returned %d bytes, want %d",
				resource.ID, len(response.payload), resource.Owner.Size,
			)
		}
		if err := client.set(
			vmmRedisKey(generation, fmt.Sprintf("resource:%d", resource.ID)),
			response.payload,
		); err != nil {
			return "", err
		}
		writeVMMDigestObject(digest, resource.ID, response.payload)
	}
	stateDigest := hex.EncodeToString(digest.Sum(nil))
	if err := client.set(vmmRedisKey(generation, "digest"), []byte(stateDigest)); err != nil {
		return "", err
	}
	for _, detach := range vmmDetachPlan(ledger) {
		process, err := findVMMProcess(processes, detach.participant)
		if err != nil {
			return "", err
		}
		resource := ledger.Resources[detach.resourceID-1]
		operation := uint16(vmmDetachImporters)
		role := "importer"
		if detach.role == vmmOwner {
			operation = vmmDetachOwners
			role = "owner"
		}
		if _, err := exchangeVMM(
			ctx,
			process,
			vmmHeader{
				Operation:      operation,
				AllocationUUID: resource.captureUUID,
				ObjectID:       resource.ID,
			},
			nil,
			-1,
			vmmResponseFDNone,
		); err != nil {
			return "", fmt.Errorf(
				"detach CUDA VMM resource %d %s participant %s: %w",
				resource.ID, role, detach.participant, err,
			)
		}
	}
	if err := client.set(vmmRedisKey(generation, "state"), []byte("detached")); err != nil {
		return "", err
	}
	if err := client.expectOK("SAVE"); err != nil {
		return "", fmt.Errorf("SAVE dedicated CUDA VMM Redis database: %w", err)
	}
	if err := copyVMMArtifact(
		config.rdbPath,
		filepath.Join(checkpointDir, vmmArtifactName),
	); err != nil {
		return "", err
	}
	log.Info(
		"Captured launcher-scoped CUDA POSIX-FD VMM state",
		"generation", generation,
		"resources", len(ledger.Resources),
	)
	return stateDigest, nil
}

// RestoreVMM imports and validates the RDB artifact while CUDA processes remain
// locked, then calls unlock immediately before recreating owners and importers.
func RestoreVMM(
	ctx context.Context,
	processes []VMMProcess,
	generation string,
	expectedDigest string,
	checkpointDir string,
	unlock func() error,
	log logr.Logger,
) (retErr error) {
	if generation == "" || expectedDigest == "" {
		return errors.New("CUDA VMM generation and state digest are required")
	}
	if unlock == nil {
		return errors.New("CUDA VMM restore unlock callback is required")
	}
	config, err := loadVMMRedisConfig(true)
	if err != nil {
		return err
	}
	poisonClient, err := dialVMMRedis(ctx, config)
	if err != nil {
		return err
	}
	if err := poisonClient.expectOK("FLUSHDB"); err != nil {
		poisonClient.Close()
		return fmt.Errorf("clear dedicated CUDA VMM Redis database before restore: %w", err)
	}
	if err := poisonClient.set(vmmRestorePoison, []byte(generation)); err != nil {
		poisonClient.Close()
		return fmt.Errorf("poison dedicated CUDA VMM Redis database before restore: %w", err)
	}
	if err := poisonClient.Close(); err != nil {
		return fmt.Errorf("close CUDA VMM Redis poison connection: %w", err)
	}
	artifact := filepath.Join(checkpointDir, vmmArtifactName)
	command := exec.CommandContext(ctx, config.restoreCmd, artifact)
	command.Env = os.Environ()
	if output, err := command.CombinedOutput(); err != nil {
		return fmt.Errorf(
			"load CUDA VMM Redis RDB with %q: %w (output: %s)",
			config.restoreCmd, err, strings.TrimSpace(string(output)),
		)
	}
	client, err := dialVMMRedis(ctx, config)
	if err != nil {
		return err
	}
	defer client.Close()
	poison, err := client.getOptional(vmmRestorePoison)
	if err != nil {
		return err
	}
	if poison != nil {
		return fmt.Errorf("CUDA VMM Redis restore command left the pre-restore poison in place")
	}
	state, err := client.get(vmmRedisKey(generation, "state"))
	if err != nil {
		return err
	}
	if string(state) != "detached" {
		return fmt.Errorf("CUDA VMM Redis generation %s state is %q, want detached", generation, state)
	}
	encoded, err := client.get(vmmRedisKey(generation, "ledger"))
	if err != nil {
		return err
	}
	var ledger vmmLedger
	if err := json.Unmarshal(encoded, &ledger); err != nil {
		return fmt.Errorf("decode CUDA VMM Redis ledger: %w", err)
	}
	if ledger.Version != 1 || ledger.Generation != generation {
		return fmt.Errorf(
			"CUDA VMM Redis ledger is version/generation %d/%q, want 1/%q",
			ledger.Version, ledger.Generation, generation,
		)
	}
	if err := validateVMMLedger(ledger); err != nil {
		return fmt.Errorf("validate CUDA VMM Redis ledger: %w", err)
	}
	currentNode, err := currentVMMNode()
	if err != nil {
		return err
	}
	placement, err := validateRestoredVMMProcesses(ctx, processes, ledger, currentNode)
	if err != nil {
		return err
	}
	storedDigest, err := client.get(vmmRedisKey(generation, "digest"))
	if err != nil {
		return err
	}
	if string(storedDigest) != expectedDigest {
		return fmt.Errorf(
			"CUDA VMM Redis generation %s digest is %q, want %q",
			generation, storedDigest, expectedDigest,
		)
	}
	digest := sha256.New()
	writeVMMDigestPart(digest, encoded)
	contentsByResource := make([][]byte, len(ledger.Resources))
	for index, resource := range ledger.Resources {
		if resource.Kind != "allocation" {
			return fmt.Errorf("CUDA VMM resource %d has unsupported kind %q", resource.ID, resource.Kind)
		}
		contents, err := client.get(vmmRedisKey(generation, fmt.Sprintf("resource:%d", resource.ID)))
		if err != nil {
			return err
		}
		if uint64(len(contents)) != resource.Owner.Size {
			return fmt.Errorf(
				"CUDA VMM Redis resource %d has %d bytes, want %d",
				resource.ID, len(contents), resource.Owner.Size,
			)
		}
		contentsByResource[index] = contents
		writeVMMDigestObject(digest, resource.ID, contents)
	}
	actualDigest := hex.EncodeToString(digest.Sum(nil))
	if actualDigest != expectedDigest {
		return fmt.Errorf(
			"CUDA VMM Redis generation %s content digest is %q, want %q",
			generation, actualDigest, expectedDigest,
		)
	}
	brokerFDs := make([]int, len(ledger.Resources))
	for index := range brokerFDs {
		brokerFDs[index] = -1
	}
	defer func() {
		for index := range brokerFDs {
			if err := closeVMMFD(&brokerFDs[index]); err != nil {
				retErr = errors.Join(
					retErr,
					fmt.Errorf(
						"close CUDA VMM resource %d broker FD: %w",
						ledger.Resources[index].ID,
						err,
					),
				)
			}
		}
	}()
	ownerPlans := make([]vmmRestorePlan, 0, len(ledger.Resources))
	var importerPlans []vmmRestorePlan
	for index, resource := range ledger.Resources {
		owner, err := findVMMProcess(processes, resource.Owner.Participant)
		if err != nil {
			return err
		}
		payload := encodeVMMRestoreRecord(
			resource.ID,
			vmmOwner,
			resource.Owner,
			placement[resource.Owner.Participant][resource.Owner.GPUUUID],
			contentsByResource[index],
		)
		contentsByResource[index] = nil
		ownerPlans = append(ownerPlans, vmmRestorePlan{
			resourceID:    resource.ID,
			resourceIndex: index,
			process:       owner,
			payload:       payload,
		})
		for _, mapping := range resource.Importers {
			importer, err := findVMMProcess(processes, mapping.Participant)
			if err != nil {
				return err
			}
			importerPlans = append(importerPlans, vmmRestorePlan{
				resourceID:    resource.ID,
				resourceIndex: index,
				process:       importer,
				payload: encodeVMMRestoreRecord(
					resource.ID,
					vmmImporter,
					mapping,
					placement[mapping.Participant][mapping.GPUUUID],
					nil,
				),
			})
		}
	}
	if err := unlock(); err != nil {
		return fmt.Errorf("unlock CUDA processes before VMM replay: %w", err)
	}
	for _, plan := range ownerPlans {
		response, err := exchangeVMM(
			ctx,
			plan.process,
			vmmHeader{Operation: vmmRestoreOwner, ObjectID: plan.resourceID},
			plan.payload,
			-1,
			vmmResponseFDRequired,
		)
		if err != nil {
			return fmt.Errorf("restore CUDA VMM resource %d owner: %w", plan.resourceID, err)
		}
		brokerFDs[plan.resourceIndex] = response.fd
	}
	for _, plan := range importerPlans {
		if _, err := exchangeVMM(
			ctx,
			plan.process,
			vmmHeader{Operation: vmmRestoreImporter, ObjectID: plan.resourceID},
			plan.payload,
			brokerFDs[plan.resourceIndex],
			vmmResponseFDNone,
		); err != nil {
			return fmt.Errorf(
				"restore CUDA VMM resource %d importer participant %s: %w",
				plan.resourceID, plan.process.Participant, err,
			)
		}
	}
	for index := range brokerFDs {
		if err := closeVMMFD(&brokerFDs[index]); err != nil {
			return fmt.Errorf("close CUDA VMM resource %d broker FD: %w", ledger.Resources[index].ID, err)
		}
	}
	if err := verifyVMMHealth(ctx, processes); err != nil {
		return err
	}
	if err := client.set(vmmRedisKey(generation, "state"), []byte("restored")); err != nil {
		return err
	}
	log.Info(
		"Restored launcher-scoped CUDA POSIX-FD VMM state",
		"generation", generation,
		"resources", len(ledger.Resources),
	)
	return nil
}

func validateVMMLedger(ledger vmmLedger) error {
	participants := make(map[string]struct{}, len(ledger.Participants))
	participantGPUs := make(map[string]map[string]struct{}, len(ledger.Participants))
	for _, participant := range ledger.Participants {
		if participant.ID == "" || participant.Placement.Node == "" ||
			len(participant.Placement.GPUUUIDs) == 0 {
			return errors.New("ledger contains incomplete participant placement")
		}
		if len(participant.ID) != 32 || !isLowerHex(participant.ID) {
			return fmt.Errorf("ledger contains invalid participant ID %q", participant.ID)
		}
		if _, duplicate := participants[participant.ID]; duplicate {
			return fmt.Errorf("ledger contains duplicate participant %q", participant.ID)
		}
		participants[participant.ID] = struct{}{}
		gpus := make(map[string]struct{}, len(participant.Placement.GPUUUIDs))
		for _, gpuUUID := range participant.Placement.GPUUUIDs {
			if parseGPUUUID(gpuUUID) == nil {
				return fmt.Errorf(
					"ledger participant %q contains invalid GPU UUID %q",
					participant.ID, gpuUUID,
				)
			}
			gpus[gpuUUID] = struct{}{}
		}
		participantGPUs[participant.ID] = gpus
	}
	for index, resource := range ledger.Resources {
		if resource.ID != uint64(index+1) || resource.Kind != "allocation" ||
			resource.Owner.Participant == "" || len(resource.Importers) == 0 {
			return fmt.Errorf("ledger resource %d is incomplete or unsupported", resource.ID)
		}
		mappings := append([]vmmMapping{resource.Owner}, resource.Importers...)
		seen := make(map[string]struct{}, len(mappings))
		for _, mapping := range mappings {
			if _, ok := participants[mapping.Participant]; !ok {
				return fmt.Errorf(
					"ledger resource %d references unknown participant %q",
					resource.ID, mapping.Participant,
				)
			}
			if _, ok := participantGPUs[mapping.Participant][mapping.GPUUUID]; !ok {
				return fmt.Errorf(
					"ledger resource %d mapping GPU %q is outside participant %q placement",
					resource.ID, mapping.GPUUUID, mapping.Participant,
				)
			}
			if _, duplicate := seen[mapping.Participant]; duplicate {
				return fmt.Errorf(
					"ledger resource %d repeats participant %q",
					resource.ID, mapping.Participant,
				)
			}
			seen[mapping.Participant] = struct{}{}
			if mapping.Address == 0 || mapping.Size == 0 ||
				mapping.GPUUUID == "" || mapping.RequestedHandleType != 1 ||
				mapping.AccessCount == 0 || mapping.AccessSize == 0 ||
				len(mapping.Access) !=
					int(mapping.AccessCount)*int(mapping.AccessSize) {
				return fmt.Errorf(
					"ledger resource %d has incomplete mapping for participant %q",
					resource.ID, mapping.Participant,
				)
			}
		}
	}
	if len(ledger.Resources) == 0 {
		return errors.New("ledger contains no allocation resources")
	}
	return nil
}

func inspectVMM(
	ctx context.Context,
	processes []VMMProcess,
	generation string,
	sourceNode string,
) (vmmLedger, error) {
	type captured struct {
		process VMMProcess
		record  vmmCaptureRecord
	}
	var captures []captured
	for _, process := range processes {
		response, err := exchangeVMM(
			ctx,
			process,
			vmmHeader{Operation: vmmInspect},
			nil,
			-1,
			vmmResponseFDNone,
		)
		if err != nil {
			return vmmLedger{}, fmt.Errorf("inspect CUDA process %d VMM state: %w", process.ObservedPID, err)
		}
		records, err := decodeVMMRecords(response.header.Count, response.payload)
		if err != nil {
			return vmmLedger{}, fmt.Errorf("decode CUDA process %d VMM state: %w", process.ObservedPID, err)
		}
		for _, record := range records {
			captures = append(captures, captured{process: process, record: record})
		}
	}
	grouped := make(map[[16]byte][]captured)
	for _, capture := range captures {
		id := capture.record.allocationUUID
		grouped[id] = append(grouped[id], capture)
	}
	ids := make([][16]byte, 0, len(grouped))
	for id := range grouped {
		ids = append(ids, id)
	}
	slices.SortFunc(ids, func(left, right [16]byte) int {
		return bytes.Compare(left[:], right[:])
	})
	if strings.TrimSpace(sourceNode) == "" {
		return vmmLedger{}, errors.New("CUDA VMM source node identity is required")
	}
	participantNodes := make(map[string]string, len(processes))
	for _, process := range processes {
		if process.Participant == "" {
			return vmmLedger{}, fmt.Errorf(
				"CUDA process %d has incomplete logical participant identity",
				process.ObservedPID,
			)
		}
		if _, duplicate := participantNodes[process.Participant]; duplicate {
			return vmmLedger{}, fmt.Errorf(
				"duplicate CUDA VMM participant identity %q",
				process.Participant,
			)
		}
		participantNodes[process.Participant] = sourceNode
	}
	participants := make(map[string]vmmParticipant, len(captures))
	ledger := vmmLedger{Version: 1, Generation: generation}
	for index, id := range ids {
		resource := vmmResource{
			ID:          uint64(index + 1),
			Kind:        "allocation",
			captureUUID: id,
		}
		for _, capture := range grouped[id] {
			if capture.record.kind != vmmAllocation {
				return vmmLedger{}, fmt.Errorf(
					"CUDA VMM capture has unsupported object kind %d",
					capture.record.kind,
				)
			}
			mapping := mappingFromCapture(capture.process.Participant, capture.record)
			participant, ok := participants[capture.process.Participant]
			if !ok {
				participant = vmmParticipant{
					ID: capture.process.Participant,
					Placement: vmmPlacement{
						Node: participantNodes[capture.process.Participant],
					},
				}
			}
			participant.Placement.GPUUUIDs = append(
				participant.Placement.GPUUUIDs,
				mapping.GPUUUID,
			)
			participants[capture.process.Participant] = participant
			switch capture.record.role {
			case vmmOwner:
				if resource.Owner.Participant != "" {
					return vmmLedger{}, fmt.Errorf(
						"CUDA VMM resource %d has multiple owners", resource.ID,
					)
				}
				resource.Owner = mapping
			case vmmImporter:
				resource.Importers = append(resource.Importers, mapping)
			default:
				return vmmLedger{}, fmt.Errorf(
					"CUDA VMM capture has unknown role %d", capture.record.role,
				)
			}
		}
		if resource.Owner.Participant == "" || len(resource.Importers) == 0 {
			return vmmLedger{}, fmt.Errorf(
				"CUDA VMM resource %d requires one owner and at least one importer",
				resource.ID,
			)
		}
		slices.SortFunc(resource.Importers, func(left, right vmmMapping) int {
			return strings.Compare(left.Participant, right.Participant)
		})
		resourceParticipants := map[string]struct{}{resource.Owner.Participant: {}}
		for _, importer := range resource.Importers {
			if importer.Size != resource.Owner.Size {
				return vmmLedger{}, fmt.Errorf(
					"CUDA VMM resource %d importer participant %s maps %d bytes, owner maps %d",
					resource.ID, importer.Participant, importer.Size, resource.Owner.Size,
				)
			}
			if _, duplicate := resourceParticipants[importer.Participant]; duplicate {
				return vmmLedger{}, fmt.Errorf(
					"CUDA VMM resource %d has multiple mappings in participant %s",
					resource.ID, importer.Participant,
				)
			}
			resourceParticipants[importer.Participant] = struct{}{}
		}
		ledger.Resources = append(ledger.Resources, resource)
	}
	if len(ledger.Resources) == 0 {
		return vmmLedger{}, errors.New("no CUDA POSIX-FD VMM sharing graph discovered")
	}
	participantIDs := make([]string, 0, len(participants))
	for id := range participants {
		participantIDs = append(participantIDs, id)
	}
	slices.Sort(participantIDs)
	for _, id := range participantIDs {
		participant := participants[id]
		slices.Sort(participant.Placement.GPUUUIDs)
		participant.Placement.GPUUUIDs =
			slices.Compact(participant.Placement.GPUUUIDs)
		ledger.Participants = append(ledger.Participants, participant)
	}
	if err := validateVMMLedger(ledger); err != nil {
		return vmmLedger{}, err
	}
	return ledger, nil
}

func mappingFromCapture(participant string, record vmmCaptureRecord) vmmMapping {
	return vmmMapping{
		Participant:           participant,
		Address:               record.address,
		Size:                  record.size,
		GPUUUID:               record.gpuUUID,
		RequestedHandleType:   record.handleType,
		ApplicationHandleLive: record.flags&vmmApplicationHandleLive != 0,
		Properties:            append([]byte(nil), record.properties...),
		Access:                append([]byte(nil), record.access...),
		AccessCount:           record.accessCount,
		AccessSize:            record.accessSize,
	}
}

func readVMMRESPLine(reader *bufio.Reader) (string, error) {
	line := make([]byte, 0, 64)
	for {
		next, err := reader.ReadByte()
		if err != nil {
			return "", fmt.Errorf("unterminated Redis response line: %w", err)
		}
		switch next {
		case '\r':
			terminator, err := reader.ReadByte()
			if err != nil {
				return "", fmt.Errorf("unterminated Redis response line: %w", err)
			}
			if terminator != '\n' {
				return "", errors.New("malformed Redis response line terminator")
			}
			return string(line), nil
		case '\n':
			return "", errors.New("malformed Redis response line terminator")
		default:
			line = append(line, next)
			if len(line) > vmmMaximumRESPLine {
				return "", fmt.Errorf(
					"Redis response line exceeds %d bytes",
					vmmMaximumRESPLine,
				)
			}
		}
	}
}

func validateRestoredVMMProcesses(
	ctx context.Context,
	processes []VMMProcess,
	ledger vmmLedger,
	currentNode string,
) (vmmExecutionPlacement, error) {
	expected := make(map[string]vmmParticipant, len(ledger.Participants))
	for _, participant := range ledger.Participants {
		if participant.ID == "" || participant.Placement.Node == "" {
			return nil, errors.New("CUDA VMM ledger has incomplete participant placement")
		}
		if _, duplicate := expected[participant.ID]; duplicate {
			return nil, fmt.Errorf("CUDA VMM ledger has duplicate participant %q", participant.ID)
		}
		expected[participant.ID] = participant
	}
	if strings.TrimSpace(currentNode) == "" {
		return nil, errors.New("current CUDA VMM node identity is required")
	}
	actual := make(map[string]VMMProcess, len(processes))
	for _, process := range processes {
		if _, duplicate := actual[process.Participant]; duplicate {
			return nil, fmt.Errorf(
				"multiple restored shims claim participant %q",
				process.Participant,
			)
		}
		actual[process.Participant] = process
	}
	for id, participant := range expected {
		_, ok := actual[id]
		if !ok {
			return nil, fmt.Errorf("CUDA VMM participant %q has no restored shim endpoint", id)
		}
		if currentNode != participant.Placement.Node {
			return nil, fmt.Errorf(
				"CUDA VMM participant %q target node is %q, source was %q",
				id, currentNode, participant.Placement.Node,
			)
		}
	}
	placement := make(vmmExecutionPlacement, len(expected))
	for _, process := range processes {
		response, err := exchangeVMM(
			ctx,
			process,
			vmmHeader{Operation: vmmQueryPlacement},
			nil,
			-1,
			vmmResponseFDNone,
		)
		if err != nil {
			return nil, fmt.Errorf(
				"query CUDA VMM participant %q current GPU placement: %w",
				process.Participant,
				err,
			)
		}
		ordinals, err := decodeVMMPlacement(
			response.header.Count,
			response.payload,
		)
		if err != nil {
			return nil, fmt.Errorf(
				"decode CUDA VMM participant %q current GPU placement: %w",
				process.Participant,
				err,
			)
		}
		participant, managed := expected[process.Participant]
		if !managed {
			if len(ordinals) != 0 {
				return nil, fmt.Errorf(
					"unexpected restored CUDA VMM participant %q has %d detached managed placements",
					process.Participant,
					len(ordinals),
				)
			}
			continue
		}
		want := make(map[string]struct{}, len(participant.Placement.GPUUUIDs))
		for _, gpuUUID := range participant.Placement.GPUUUIDs {
			want[gpuUUID] = struct{}{}
		}
		if len(ordinals) != len(want) {
			return nil, fmt.Errorf(
				"CUDA VMM participant %q current GPU placement count is %d, want %d",
				process.Participant,
				len(ordinals),
				len(want),
			)
		}
		placement[process.Participant] = make(map[string]int32, len(ordinals))
		for sourceUUID, current := range ordinals {
			if _, ok := want[sourceUUID]; !ok {
				return nil, fmt.Errorf(
					"CUDA VMM participant %q reported unexpected source GPU %q",
					process.Participant,
					sourceUUID,
				)
			}
			if sourceUUID != current.uuid {
				return nil, fmt.Errorf(
					"CUDA VMM participant %q ordinal %d currently maps GPU %q, source was %q",
					process.Participant,
					current.ordinal,
					current.uuid,
					sourceUUID,
				)
			}
			placement[process.Participant][sourceUUID] = current.ordinal
		}
	}
	return placement, nil
}

func decodeVMMPlacement(
	count uint32,
	payload []byte,
) (map[string]struct {
	uuid    string
	ordinal int32
}, error) {
	if uint64(count)*vmmPlacementSize != uint64(len(payload)) {
		return nil, errors.New("invalid VMM placement payload length")
	}
	result := make(map[string]struct {
		uuid    string
		ordinal int32
	}, count)
	for range count {
		source := formatGPUUUID(payload[8:24])
		current := formatGPUUUID(payload[24:40])
		if source == "" || current == "" {
			return nil, errors.New("invalid VMM placement GPU UUID")
		}
		if _, duplicate := result[source]; duplicate {
			return nil, fmt.Errorf("duplicate VMM source GPU UUID %q", source)
		}
		result[source] = struct {
			uuid    string
			ordinal int32
		}{
			uuid:    current,
			ordinal: int32(binary.LittleEndian.Uint32(payload[0:4])),
		}
		payload = payload[vmmPlacementSize:]
	}
	return result, nil
}

func findVMMProcess(processes []VMMProcess, participant string) (VMMProcess, error) {
	for _, process := range processes {
		if process.Participant == participant {
			return process, nil
		}
	}
	return VMMProcess{}, fmt.Errorf("CUDA VMM participant %q is not available", participant)
}

func decodeVMMRecords(count uint32, payload []byte) ([]vmmCaptureRecord, error) {
	records := make([]vmmCaptureRecord, 0, count)
	for range count {
		if len(payload) < vmmRecordSize {
			return nil, errors.New("truncated VMM record")
		}
		record := vmmCaptureRecord{
			address:     binary.LittleEndian.Uint64(payload[24:32]),
			size:        binary.LittleEndian.Uint64(payload[32:40]),
			offset:      binary.LittleEndian.Uint64(payload[40:48]),
			role:        binary.LittleEndian.Uint32(payload[48:52]),
			kind:        binary.LittleEndian.Uint32(payload[52:56]),
			handleType:  binary.LittleEndian.Uint32(payload[56:60]),
			flags:       binary.LittleEndian.Uint32(payload[60:64]),
			device:      int32(binary.LittleEndian.Uint32(payload[64:68])),
			accessCount: binary.LittleEndian.Uint32(payload[72:76]),
			accessSize:  binary.LittleEndian.Uint32(payload[76:80]),
			gpuUUID:     formatGPUUUID(payload[80:96]),
		}
		copy(record.allocationUUID[:], payload[0:16])
		propertiesSize := binary.LittleEndian.Uint32(payload[68:72])
		metadataSize := uint64(vmmRecordSize) + uint64(propertiesSize) +
			uint64(record.accessCount)*uint64(record.accessSize)
		if metadataSize > uint64(len(payload)) {
			return nil, errors.New("invalid VMM record lengths")
		}
		record.properties = append(
			[]byte(nil),
			payload[vmmRecordSize:vmmRecordSize+int(propertiesSize)]...,
		)
		record.access = append(
			[]byte(nil),
			payload[vmmRecordSize+int(propertiesSize):int(metadataSize)]...,
		)
		if record.allocationUUID == ([16]byte{}) || record.address == 0 ||
			record.size == 0 || record.offset != 0 || record.accessCount == 0 ||
			record.accessSize == 0 || record.kind == 0 || record.gpuUUID == "" {
			return nil, errors.New("unsupported incomplete VMM mapping metadata")
		}
		payload = payload[metadataSize:]
		records = append(records, record)
	}
	if len(payload) != 0 {
		return nil, errors.New("trailing bytes after VMM records")
	}
	return records, nil
}

func encodeVMMRestoreRecord(
	objectID uint64,
	role uint32,
	mapping vmmMapping,
	deviceOrdinal int32,
	contents []byte,
) []byte {
	payload := make(
		[]byte,
		vmmRecordSize+len(mapping.Properties)+len(mapping.Access)+len(contents),
	)
	binary.LittleEndian.PutUint64(payload[16:24], objectID)
	binary.LittleEndian.PutUint64(payload[24:32], mapping.Address)
	binary.LittleEndian.PutUint64(payload[32:40], mapping.Size)
	binary.LittleEndian.PutUint32(payload[48:52], role)
	binary.LittleEndian.PutUint32(payload[52:56], vmmAllocation)
	binary.LittleEndian.PutUint32(payload[56:60], mapping.RequestedHandleType)
	if mapping.ApplicationHandleLive {
		binary.LittleEndian.PutUint32(payload[60:64], vmmApplicationHandleLive)
	}
	binary.LittleEndian.PutUint32(payload[64:68], uint32(deviceOrdinal))
	binary.LittleEndian.PutUint32(payload[68:72], uint32(len(mapping.Properties)))
	binary.LittleEndian.PutUint32(payload[72:76], mapping.AccessCount)
	binary.LittleEndian.PutUint32(payload[76:80], mapping.AccessSize)
	copy(payload[80:96], parseGPUUUID(mapping.GPUUUID))
	cursor := vmmRecordSize
	copy(payload[cursor:], mapping.Properties)
	cursor += len(mapping.Properties)
	copy(payload[cursor:], mapping.Access)
	cursor += len(mapping.Access)
	copy(payload[cursor:], contents)
	return payload
}

func formatGPUUUID(value []byte) string {
	if len(value) != vmmGPUUUIDSize || bytes.Equal(value, make([]byte, vmmGPUUUIDSize)) {
		return ""
	}
	encoded := hex.EncodeToString(value)
	return strings.Join([]string{
		encoded[0:8],
		encoded[8:12],
		encoded[12:16],
		encoded[16:20],
		encoded[20:32],
	}, "-")
}

func parseGPUUUID(value string) []byte {
	decoded, err := hex.DecodeString(strings.ReplaceAll(value, "-", ""))
	if err != nil || len(decoded) != vmmGPUUUIDSize {
		return nil
	}
	return decoded
}

type vmmResponseFDExpectation uint8

const (
	vmmResponseFDNone vmmResponseFDExpectation = iota
	vmmResponseFDRequired
)

type vmmResponse struct {
	header  vmmHeader
	payload []byte
	fd      int
}

func closeVMMFD(fd *int) error {
	owned := *fd
	*fd = -1
	if owned < 0 {
		return nil
	}
	return unix.Close(owned)
}

func closeVMMFDs(fds []int) error {
	var result error
	for index := range fds {
		if err := closeVMMFD(&fds[index]); err != nil {
			result = errors.Join(result, fmt.Errorf("close received VMM FD: %w", err))
		}
	}
	return result
}

func exchangeVMM(
	ctx context.Context,
	process VMMProcess,
	request vmmHeader,
	payload []byte,
	passedFD int,
	responseFD vmmResponseFDExpectation,
) (result vmmResponse, retErr error) {
	result.fd = -1
	dialer := net.Dialer{Timeout: vmmTimeout}
	connection, err := dialer.DialContext(ctx, "unix", process.SocketPath)
	if err != nil {
		return result, err
	}
	defer connection.Close()
	unixConnection, ok := connection.(*net.UnixConn)
	if !ok {
		return result, errors.New("VMM endpoint is not a Unix connection")
	}
	if deadline, ok := ctx.Deadline(); ok {
		_ = unixConnection.SetDeadline(deadline)
	} else {
		_ = unixConnection.SetDeadline(time.Now().Add(vmmTimeout))
	}
	request.PayloadSize = uint64(len(payload))
	encoded := encodeVMMHeader(request)
	var ancillary []byte
	if passedFD >= 0 {
		ancillary = unix.UnixRights(passedFD)
	}
	written, _, err := unixConnection.WriteMsgUnix(encoded, ancillary, nil)
	if err != nil {
		return result, err
	}
	if err := writeVMMBytes(unixConnection, encoded[written:]); err != nil {
		return result, err
	}
	if len(payload) != 0 {
		if err := writeVMMBytes(unixConnection, payload); err != nil {
			return result, err
		}
	}
	responseBuffer := make([]byte, vmmHeaderSize+64*1024)
	ancillary = make([]byte, unix.CmsgSpace(4*vmmMaximumRightsFDs))
	size, ancillarySize, flags, _, err := unixConnection.ReadMsgUnix(responseBuffer, ancillary)
	if err != nil {
		return result, err
	}
	receivedFDs, ancillaryErr := receiveVMMFDs(ancillary[:ancillarySize])
	defer func() {
		if err := closeVMMFDs(receivedFDs); err != nil {
			retErr = errors.Join(retErr, err)
		}
	}()
	if ancillaryErr != nil {
		return result, ancillaryErr
	}
	if flags&unix.MSG_CTRUNC != 0 {
		return result, errors.New("truncated VMM response ancillary data")
	}
	if size == 0 {
		return result, errors.New("empty VMM response header")
	}
	if size < vmmHeaderSize {
		if _, err := io.ReadFull(unixConnection, responseBuffer[size:vmmHeaderSize]); err != nil {
			return result, fmt.Errorf("short VMM response header: %w", err)
		}
		size = vmmHeaderSize
	}
	response, err := decodeVMMHeader(responseBuffer[:vmmHeaderSize])
	if err != nil {
		return result, err
	}
	result.header = response
	if response.Operation != request.Operation {
		return result, fmt.Errorf(
			"VMM response operation %d does not match request %d",
			response.Operation, request.Operation,
		)
	}
	if response.Status != 0 {
		return result, errors.New(response.Message)
	}
	switch responseFD {
	case vmmResponseFDNone:
		if len(receivedFDs) != 0 {
			return result, fmt.Errorf(
				"VMM operation %d returned %d forbidden response FDs",
				request.Operation,
				len(receivedFDs),
			)
		}
	case vmmResponseFDRequired:
		if len(receivedFDs) != 1 {
			return result, fmt.Errorf(
				"VMM operation %d returned %d response FDs, want exactly one",
				request.Operation,
				len(receivedFDs),
			)
		}
	default:
		return result, fmt.Errorf("invalid VMM response FD expectation %d", responseFD)
	}
	if response.PayloadSize > vmmMaximumPayload {
		return result, fmt.Errorf("VMM response payload %d exceeds limit", response.PayloadSize)
	}
	responsePayload := make([]byte, int(response.PayloadSize))
	payloadPrefix := responseBuffer[vmmHeaderSize:size]
	if uint64(len(payloadPrefix)) > response.PayloadSize {
		return result, errors.New("VMM response exceeds declared payload size")
	}
	copy(responsePayload, payloadPrefix)
	if _, err := io.ReadFull(unixConnection, responsePayload[len(payloadPrefix):]); err != nil {
		return result, err
	}
	result.payload = responsePayload
	if responseFD == vmmResponseFDRequired {
		result.fd = receivedFDs[0]
		receivedFDs[0] = -1
	}
	return result, nil
}

func receiveVMMFDs(ancillary []byte) ([]int, error) {
	var receivedFDs []int
	for len(ancillary) != 0 {
		if len(ancillary) < unix.CmsgLen(0) {
			return receivedFDs, errors.New("malformed VMM ancillary data")
		}
		header, data, remainder, err := unix.ParseOneSocketControlMessage(ancillary)
		if err != nil {
			return receivedFDs, err
		}
		ancillary = remainder
		if header.Level != unix.SOL_SOCKET || header.Type != unix.SCM_RIGHTS {
			continue
		}
		message := unix.SocketControlMessage{Header: header, Data: data}
		fds, err := unix.ParseUnixRights(&message)
		if err != nil {
			return receivedFDs, err
		}
		receivedFDs = append(receivedFDs, fds...)
		if len(data)%4 != 0 {
			return receivedFDs, errors.New("malformed VMM SCM_RIGHTS data")
		}
	}
	return receivedFDs, nil
}

func writeVMMBytes(writer io.Writer, content []byte) error {
	for len(content) != 0 {
		written, err := writer.Write(content)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		content = content[written:]
	}
	return nil
}

func encodeVMMHeader(header vmmHeader) []byte {
	buffer := make([]byte, vmmHeaderSize)
	binary.LittleEndian.PutUint32(buffer[0:4], vmmProtocolMagic)
	binary.LittleEndian.PutUint16(buffer[4:6], vmmProtocolVersion)
	binary.LittleEndian.PutUint16(buffer[6:8], header.Operation)
	binary.LittleEndian.PutUint32(buffer[8:12], uint32(header.Status))
	binary.LittleEndian.PutUint32(buffer[12:16], header.Count)
	copy(buffer[24:40], header.AllocationUUID[:])
	binary.LittleEndian.PutUint64(buffer[40:48], header.ObjectID)
	binary.LittleEndian.PutUint64(buffer[48:56], header.PayloadSize)
	copy(buffer[56:56+vmmMessageSize], header.Message)
	copy(buffer[152:152+vmmParticipantSize], header.Participant)
	return buffer
}

func decodeVMMHeader(buffer []byte) (vmmHeader, error) {
	if len(buffer) != vmmHeaderSize ||
		binary.LittleEndian.Uint32(buffer[0:4]) != vmmProtocolMagic ||
		binary.LittleEndian.Uint16(buffer[4:6]) != vmmProtocolVersion {
		return vmmHeader{}, errors.New("invalid VMM response protocol")
	}
	header := vmmHeader{
		Operation:   binary.LittleEndian.Uint16(buffer[6:8]),
		Status:      int32(binary.LittleEndian.Uint32(buffer[8:12])),
		Count:       binary.LittleEndian.Uint32(buffer[12:16]),
		ObjectID:    binary.LittleEndian.Uint64(buffer[40:48]),
		PayloadSize: binary.LittleEndian.Uint64(buffer[48:56]),
		Message:     strings.TrimRight(string(buffer[56:56+vmmMessageSize]), "\x00"),
		Participant: strings.TrimRight(
			string(buffer[152:152+vmmParticipantSize]),
			"\x00",
		),
	}
	copy(header.AllocationUUID[:], buffer[24:40])
	return header, nil
}

func loadVMMRedisConfig(requireRestore bool) (vmmRedisConfig, error) {
	config := vmmRedisConfig{
		address:    strings.TrimSpace(os.Getenv(VMMRedisAddressEnv)),
		password:   os.Getenv(VMMRedisPasswordEnv),
		rdbPath:    strings.TrimSpace(os.Getenv(VMMRedisRDBPathEnv)),
		restoreCmd: strings.TrimSpace(os.Getenv(VMMRedisRestoreCmdEnv)),
	}
	if config.address == "" {
		return config, fmt.Errorf("%s is required for the dedicated job-scoped Redis endpoint", VMMRedisAddressEnv)
	}
	if !requireRestore && config.rdbPath == "" {
		return config, fmt.Errorf(
			"%s is required and must be the Redis RDB path visible to the snapshot coordinator",
			VMMRedisRDBPathEnv,
		)
	}
	if requireRestore && config.restoreCmd == "" {
		return config, fmt.Errorf(
			"%s is required to load the checkpoint RDB into the dedicated Redis endpoint",
			VMMRedisRestoreCmdEnv,
		)
	}
	return config, nil
}

func copyVMMArtifact(source, destination string) error {
	input, err := os.Open(source)
	if err != nil {
		return fmt.Errorf("open Redis RDB %q after SAVE: %w", source, err)
	}
	defer input.Close()
	sourceInfo, err := input.Stat()
	if err != nil {
		return fmt.Errorf("inspect Redis RDB %q after SAVE: %w", source, err)
	}
	if !sourceInfo.Mode().IsRegular() || sourceInfo.Size() == 0 {
		return fmt.Errorf("Redis RDB source %q after SAVE must be a regular non-empty file", source)
	}
	if destinationInfo, err := os.Stat(destination); err == nil &&
		os.SameFile(sourceInfo, destinationInfo) {
		return fmt.Errorf(
			"Redis RDB source %q must be outside the checkpoint artifact path",
			source,
		)
	}
	output, err := os.OpenFile(destination, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if err != nil {
		return fmt.Errorf("create CUDA VMM RDB artifact: %w", err)
	}
	if _, err := io.Copy(output, input); err != nil {
		output.Close()
		return fmt.Errorf("copy CUDA VMM RDB artifact: %w", err)
	}
	if err := output.Close(); err != nil {
		return fmt.Errorf("close CUDA VMM RDB artifact: %w", err)
	}
	return nil
}

func vmmRedisKey(generation, suffix string) string {
	return "dynamo:cuda-vmm:" + generation + ":" + suffix
}

type vmmRedisClient struct {
	connection net.Conn
	reader     *bufio.Reader
}

func dialVMMRedis(ctx context.Context, config vmmRedisConfig) (*vmmRedisClient, error) {
	connection, err := (&net.Dialer{Timeout: vmmTimeout}).DialContext(ctx, "tcp", config.address)
	if err != nil {
		return nil, fmt.Errorf("connect dedicated CUDA VMM Redis at %s: %w", config.address, err)
	}
	client := &vmmRedisClient{connection: connection, reader: bufio.NewReader(connection)}
	if config.password != "" {
		if err := client.expectOK("AUTH", config.password); err != nil {
			connection.Close()
			return nil, fmt.Errorf("authenticate dedicated CUDA VMM Redis: %w", err)
		}
	}
	return client, nil
}

func (client *vmmRedisClient) Close() error {
	return client.connection.Close()
}

func (client *vmmRedisClient) set(key string, value []byte) error {
	response, err := client.command([]byte("SET"), []byte(key), value)
	if err != nil {
		return fmt.Errorf("write CUDA VMM Redis key %q: %w", key, err)
	}
	if string(response) != "OK" {
		return fmt.Errorf("write CUDA VMM Redis key %q returned %q", key, response)
	}
	return nil
}

func (client *vmmRedisClient) get(key string) ([]byte, error) {
	response, err := client.getOptional(key)
	if err != nil {
		return nil, err
	}
	if response == nil {
		return nil, fmt.Errorf("CUDA VMM Redis key %q is missing", key)
	}
	return response, nil
}

func (client *vmmRedisClient) getOptional(key string) ([]byte, error) {
	response, err := client.command([]byte("GET"), []byte(key))
	if err != nil {
		return nil, fmt.Errorf("read CUDA VMM Redis key %q: %w", key, err)
	}
	return response, nil
}

func (client *vmmRedisClient) expectOK(command string, args ...string) error {
	encoded := make([][]byte, 0, len(args)+1)
	encoded = append(encoded, []byte(command))
	for _, argument := range args {
		encoded = append(encoded, []byte(argument))
	}
	response, err := client.command(encoded...)
	if err != nil {
		return err
	}
	if string(response) != "OK" {
		return fmt.Errorf("%s returned %q", command, response)
	}
	return nil
}

func (client *vmmRedisClient) command(arguments ...[]byte) ([]byte, error) {
	if err := client.connection.SetDeadline(time.Now().Add(vmmTimeout)); err != nil {
		return nil, err
	}
	var request bytes.Buffer
	fmt.Fprintf(&request, "*%d\r\n", len(arguments))
	for _, argument := range arguments {
		fmt.Fprintf(&request, "$%d\r\n", len(argument))
		request.Write(argument)
		request.WriteString("\r\n")
	}
	if err := writeVMMBytes(client.connection, request.Bytes()); err != nil {
		return nil, err
	}
	prefix, err := client.reader.ReadByte()
	if err != nil {
		return nil, err
	}
	line, err := readVMMRESPLine(client.reader)
	if err != nil {
		return nil, err
	}
	switch prefix {
	case '+':
		return []byte(line), nil
	case '-':
		return nil, errors.New(line)
	case '$':
		size, err := strconv.Atoi(line)
		if err != nil {
			return nil, err
		}
		if size == -1 {
			return nil, nil
		}
		if size < -1 || size > vmmMaximumRedisBulk {
			return nil, fmt.Errorf("invalid Redis bulk length %d", size)
		}
		value := make([]byte, size+2)
		if _, err := io.ReadFull(client.reader, value); err != nil {
			return nil, err
		}
		if !bytes.Equal(value[size:], []byte("\r\n")) {
			return nil, errors.New("invalid Redis bulk terminator")
		}
		return value[:size], nil
	default:
		return nil, fmt.Errorf("unsupported Redis response prefix %q", prefix)
	}
}
