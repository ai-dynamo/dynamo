package cuda

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"
)

func TestDetectVMMInterposeRequiresUniformLauncherScope(t *testing.T) {
	procRoot := t.TempDir()
	writeEnvironment := func(pid int, content string) {
		t.Helper()
		path := filepath.Join(procRoot, strconv.Itoa(pid))
		if err := os.MkdirAll(path, 0700); err != nil {
			t.Fatal(err)
		}

		if err := os.WriteFile(filepath.Join(path, "environ"), []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
	writeEnvironment(10, VMMInterposeEnv+"=1\x00")
	writeEnvironment(11, "PATH=/bin\x00")

	if _, err := DetectVMMInterpose(procRoot, []int{10, 11}); err == nil {
		t.Fatal("mixed launcher scope must be rejected")
	}
	writeEnvironment(11, VMMInterposeEnv+"=1\x00")
	enabled, err := DetectVMMInterpose(procRoot, []int{10, 11})
	if err != nil || !enabled {
		t.Fatalf("uniform launcher scope = %t, %v", enabled, err)
	}
}

func TestRestoreVMMProcessesUsesNamespacePIDSocket(t *testing.T) {
	controlDir := t.TempDir()
	const (
		observedPID  = 356
		namespacePID = 1
		participant  = "11111111111111111111111111111111"
	)
	socketPath := filepath.Join(
		controlDir,
		fmt.Sprintf("%s%d.sock", vmmSocketPrefix, namespacePID),
	)
	serveVMMProcess(t, socketPath, participant, nil)

	processes, err := restoreVMMProcesses(
		controlDir,
		[]int{observedPID},
		[]int{namespacePID},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(processes) != 1 {
		t.Fatalf("restored VMM process count = %d, want 1", len(processes))
	}
	process := processes[0]
	if process.ObservedPID != observedPID {
		t.Fatalf("observed PID = %d, want %d", process.ObservedPID, observedPID)
	}
	if process.NamespacePID != namespacePID {
		t.Fatalf("namespace PID = %d, want %d", process.NamespacePID, namespacePID)
	}
	if process.SocketPath != socketPath {
		t.Fatalf("socket path = %q, want %q", process.SocketPath, socketPath)
	}
	if process.Participant != participant {
		t.Fatalf("participant = %q, want %q", process.Participant, participant)
	}
}

func TestRestoreVMMProcessesRejectsPIDCountMismatch(t *testing.T) {
	_, err := RestoreVMMProcesses([]int{356, 362}, []int{1})
	if err == nil || !strings.Contains(err.Error(), "PID mapping count mismatch") {
		t.Fatalf("PID count mismatch error = %v", err)
	}
}

func TestValidateVMMArtifact(t *testing.T) {
	t.Run("disabled rejects artifact", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(
			filepath.Join(dir, vmmArtifactName),
			[]byte("rdb"),
			0o600,
		); err != nil {
			t.Fatal(err)
		}
		if err := ValidateVMMArtifact(dir, false); err == nil {
			t.Fatal("disabled VMM manifest accepted an RDB artifact")
		}
	})
	t.Run("enabled requires regular non-empty artifact", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, vmmArtifactName)
		if err := ValidateVMMArtifact(dir, true); err == nil {
			t.Fatal("missing VMM artifact was accepted")
		}
		if err := os.WriteFile(path, nil, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := ValidateVMMArtifact(dir, true); err == nil {
			t.Fatal("empty VMM artifact was accepted")
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := ValidateVMMArtifact(dir, true); err == nil {
			t.Fatal("directory VMM artifact was accepted")
		}
	})
	t.Run("enabled accepts regular non-empty artifact", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(
			filepath.Join(dir, vmmArtifactName),
			[]byte("rdb"),
			0o600,
		); err != nil {
			t.Fatal(err)
		}
		if err := ValidateVMMArtifact(dir, true); err != nil {
			t.Fatal(err)
		}
	})
}

func TestCopyVMMArtifactRequiresRegularNonEmptySource(t *testing.T) {
	destination := filepath.Join(t.TempDir(), vmmArtifactName)
	empty := filepath.Join(t.TempDir(), "empty.rdb")
	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := copyVMMArtifact(empty, destination); err == nil {
		t.Fatal("empty Redis RDB source was accepted")
	}
	if err := copyVMMArtifact(t.TempDir(), destination); err == nil {
		t.Fatal("non-regular Redis RDB source was accepted")
	}
	source := filepath.Join(t.TempDir(), "dump.rdb")
	if err := os.WriteFile(source, []byte("rdb"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := copyVMMArtifact(source, destination); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(content, []byte("rdb")) {
		t.Fatalf("artifact content = %q", content)
	}
}

func TestVMMStateDigestBindsMetadataAndBytes(t *testing.T) {
	digest := func(metadata, contents []byte) string {
		hash := sha256.New()
		writeVMMDigestPart(hash, metadata)
		writeVMMDigestObject(hash, 1, contents)
		return hex.EncodeToString(hash.Sum(nil))
	}
	baseline := digest([]byte("ledger"), []byte("contents"))
	if baseline == digest([]byte("changed"), []byte("contents")) {
		t.Fatal("digest did not bind ledger metadata")
	}
	if baseline == digest([]byte("ledger"), []byte("changed")) {
		t.Fatal("digest did not bind allocation contents")
	}
}

func TestRestoreVMMRejectsNoOpRestoreCommand(t *testing.T) {
	generation := "00112233445566778899aabbccddeeff"
	t.Setenv(VMMRedisAddressEnv, serveVMMRestoreRedis(t, map[string][]byte{
		vmmRestorePoison: []byte(generation),
	}, nil))
	t.Setenv(VMMRedisRDBPathEnv, "")
	t.Setenv(VMMRedisRestoreCmdEnv, "/bin/true")
	err := RestoreVMM(
		context.Background(),
		nil,
		generation,
		"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
		t.TempDir(),
		func() error {
			t.Fatal("preflight failure invoked unlock callback")
			return nil
		},
		logr.Discard(),
	)
	if err == nil || !strings.Contains(err.Error(), "poison") {
		t.Fatalf("RestoreVMM no-op error = %v, want poison rejection", err)
	}
}

func TestRestoreVMMUnlockBoundary(t *testing.T) {
	const (
		generation   = "00112233445566778899aabbccddeeff"
		sourceUUID   = "00112233-4455-6677-8899-aabbccddeeff"
		participantA = "11111111111111111111111111111111"
		participantB = "22222222222222222222222222222222"
	)
	contents := [][]byte{{1, 2, 3, 4}, {5, 6, 7, 8}}
	mapping := func(participant string, address uint64) vmmMapping {
		return vmmMapping{
			Participant:         participant,
			Address:             address,
			Size:                4,
			GPUUUID:             sourceUUID,
			RequestedHandleType: 1,
			Access:              []byte{1, 2, 3, 4},
			AccessCount:         1,
			AccessSize:          4,
		}
	}
	ledger := vmmLedger{
		Version:    1,
		Generation: generation,
		Participants: []vmmParticipant{
			{ID: participantA, Placement: vmmPlacement{Node: "node-a", GPUUUIDs: []string{sourceUUID}}},
			{ID: participantB, Placement: vmmPlacement{Node: "node-a", GPUUUIDs: []string{sourceUUID}}},
		},
		Resources: []vmmResource{
			{
				ID: 1, Kind: "allocation", Owner: mapping(participantA, 0x1000),
				Importers: []vmmMapping{mapping(participantB, 0x2000)},
			},
			{
				ID: 2, Kind: "allocation", Owner: mapping(participantB, 0x3000),
				Importers: []vmmMapping{mapping(participantA, 0x4000)},
			},
		},
	}
	encoded, err := json.Marshal(ledger)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.New()
	writeVMMDigestPart(digest, encoded)
	for index, content := range contents {
		writeVMMDigestObject(digest, uint64(index+1), content)
	}
	expectedDigest := hex.EncodeToString(digest.Sum(nil))
	t.Setenv(VMMRedisRDBPathEnv, "")
	t.Setenv(VMMRedisRestoreCmdEnv, "/bin/true")
	t.Setenv("NODE_NAME", "node-a")
	preflight := []string{
		"placement:1:owner,2:importer",
		"placement:2:owner,1:importer",
		"content:1", "content:2",
	}

	tests := []struct {
		name          string
		corruptSecond bool
		unlockError   error
		failOwner     uint64
		wantError     string
		replayEvents  []string
		noUnlock      bool
	}{
		{
			name: "success unlocks once after all preflight",
			replayEvents: []string{
				"owner:1", "owner:2", "importer:1", "importer:2",
				"health:1:owner,2:importer",
				"health:2:owner,1:importer",
				"redis-restored",
			},
		},
		{
			name:          "late content preflight failure does not unlock",
			corruptSecond: true,
			wantError:     "content digest",
			noUnlock:      true,
		},
		{
			name:        "unlock failure sends no owner replay",
			unlockError: errors.New("unlock failed"),
			wantError:   "unlock CUDA processes before VMM replay",
		},
		{
			name:         "second owner failure after unlock fails restore",
			failOwner:    2,
			wantError:    "owner replay failed",
			replayEvents: []string{"owner:1", "owner:2"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			events := make(chan string, 20)
			loaded := map[string][]byte{
				vmmRedisKey(generation, "state"):  []byte("detached"),
				vmmRedisKey(generation, "ledger"): encoded,
				vmmRedisKey(generation, "digest"): []byte(expectedDigest),
			}
			for index, content := range contents {
				loaded[vmmRedisKey(generation, fmt.Sprintf("resource:%d", index+1))] =
					append([]byte(nil), content...)
			}
			if test.corruptSecond {
				loaded[vmmRedisKey(generation, "resource:2")][0] ^= 0xff
			}
			t.Setenv(VMMRedisAddressEnv, serveVMMRestoreRedis(t, loaded, events))

			processes := []VMMProcess{
				serveVMMRestoreProcess(
					t, participantA, sourceUUID, contents, 1, test.failOwner, events,
				),
				serveVMMRestoreProcess(
					t, participantB, sourceUUID, contents, 2, test.failOwner, events,
				),
			}
			err := RestoreVMM(
				context.Background(),
				processes,
				generation,
				expectedDigest,
				t.TempDir(),
				func() error {
					events <- "unlock"
					return test.unlockError
				},
				logr.Discard(),
			)
			if err != nil && test.wantError == "" {
				t.Fatal(err)
			}
			if test.wantError != "" && (err == nil || !strings.Contains(err.Error(), test.wantError)) {
				t.Fatalf("RestoreVMM error = %v, want %q", err, test.wantError)
			}
			gotEvents := make([]string, 0, len(events))
			for range len(events) {
				gotEvents = append(gotEvents, <-events)
			}
			wantEvents := slices.Clone(preflight)
			if !test.noUnlock {
				wantEvents = append(wantEvents, "unlock")
			}
			wantEvents = append(wantEvents, test.replayEvents...)
			if !slices.Equal(gotEvents, wantEvents) {
				t.Fatalf("restore events = %v, want %v", gotEvents, wantEvents)
			}
		})
	}
}

func TestVMMDetachPlanOrdersImportersBeforeOwners(t *testing.T) {
	plan := vmmDetachPlan(vmmLedger{Resources: []vmmResource{
		{
			ID:        1,
			Owner:     vmmMapping{Participant: "owner-a"},
			Importers: []vmmMapping{{Participant: "importer-a"}, {Participant: "importer-b"}},
		},
		{
			ID:        2,
			Owner:     vmmMapping{Participant: "owner-b"},
			Importers: []vmmMapping{{Participant: "importer-c"}},
		},
	}})
	want := []vmmDetach{
		{resourceID: 1, participant: "importer-a", role: vmmImporter},
		{resourceID: 1, participant: "importer-b", role: vmmImporter},
		{resourceID: 2, participant: "importer-c", role: vmmImporter},
		{resourceID: 1, participant: "owner-a", role: vmmOwner},
		{resourceID: 2, participant: "owner-b", role: vmmOwner},
	}
	if fmt.Sprint(plan) != fmt.Sprint(want) {
		t.Fatalf("detach plan = %#v, want %#v", plan, want)
	}
}

func TestInspectVMMRejectsUnsupportedGraph(t *testing.T) {
	process := serveVMMInspect(t, []vmmCaptureRecord{{
		allocationUUID: [16]byte{1},
		address:        0x1000,
		size:           4096,
		role:           vmmImporter,
		access:         []byte{1, 2, 3, 4},
		accessCount:    1,
		accessSize:     4,
	}})
	_, err := inspectVMM(
		context.Background(),
		[]VMMProcess{process},
		"generation",
		"node-a",
	)
	if err == nil {
		t.Fatal("importer without owner must be rejected")
	}
}

func TestDecodeVMMRecordsRejectsZeroAllocationUUID(t *testing.T) {
	payload := encodeVMMRecordsForTest([]vmmCaptureRecord{{
		address:     0x1000,
		size:        4096,
		role:        vmmOwner,
		properties:  []byte{1},
		access:      []byte{1},
		accessCount: 1,
		accessSize:  1,
	}})
	if _, err := decodeVMMRecords(1, payload); err == nil {
		t.Fatal("zero allocation UUID was accepted")
	}
}

func TestInspectVMMBuildsDeterministicGraph(t *testing.T) {
	const ownerID = "11111111111111111111111111111111"
	const importerID = "22222222222222222222222222222222"
	owner := serveVMMInspect(t, []vmmCaptureRecord{{
		allocationUUID: [16]byte{9, 8},
		address:        0x2000,
		size:           4096,
		role:           vmmOwner,
		properties:     []byte{1, 2},
		access:         []byte{3, 4},
		accessCount:    1,
		accessSize:     2,
	}})
	owner.Participant = ownerID
	importer := serveVMMInspect(t, []vmmCaptureRecord{{
		allocationUUID: [16]byte{9, 8},
		address:        0x4000,
		size:           4096,
		role:           vmmImporter,
		access:         []byte{6, 7},
		accessCount:    1,
		accessSize:     2,
	}})
	importer.Participant = importerID

	ledger, err := inspectVMM(
		context.Background(),
		[]VMMProcess{importer, owner},
		"generation",
		"node-a",
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(ledger.Resources) != 1 || ledger.Resources[0].ID != 1 ||
		ledger.Resources[0].Kind != "allocation" ||
		ledger.Resources[0].Owner.Participant != ownerID ||
		len(ledger.Resources[0].Importers) != 1 ||
		ledger.Resources[0].Importers[0].Participant != importerID ||
		len(ledger.Participants) != 2 ||
		ledger.Participants[0].Placement.Node != "node-a" {
		t.Fatalf("unexpected graph: %#v", ledger)
	}

	encoded, err := json.Marshal(ledger)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip vmmLedger
	if err := json.Unmarshal(encoded, &roundTrip); err != nil {
		t.Fatal(err)
	}
	reencoded, err := json.Marshal(roundTrip)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, reencoded) {
		t.Fatalf("ledger encoding is not stable:\n%s\n%s", encoded, reencoded)
	}
	for _, forbidden := range []string{
		`"pid"`, `"fd"`, `"realHandle"`, `"context"`, `"allocationUUID"`,
	} {
		if strings.Contains(strings.ToLower(string(encoded)), strings.ToLower(forbidden)) {
			t.Fatalf("persistent ledger contains forbidden runtime identity %s: %s", forbidden, encoded)
		}
	}
}

func TestInspectVMMGraphEncodingIsIndependentOfDiscoveryOrder(t *testing.T) {
	inspect := func(reverse bool) []byte {
		t.Helper()
		specifications := []struct {
			participant string
			record      vmmCaptureRecord
		}{
			{
				participant: "44444444444444444444444444444444",
				record: vmmCaptureRecord{
					allocationUUID: [16]byte{9, 8}, address: 0x1000, size: 4096,
					role: vmmOwner, properties: []byte{1}, access: []byte{2},
					accessCount: 1, accessSize: 1,
				},
			},
			{
				participant: "33333333333333333333333333333333",
				record: vmmCaptureRecord{
					allocationUUID: [16]byte{9, 8}, address: 0x2000, size: 4096,
					role: vmmImporter, access: []byte{3},
					accessCount: 1, accessSize: 1,
				},
			},
			{
				participant: "22222222222222222222222222222222",
				record: vmmCaptureRecord{
					allocationUUID: [16]byte{1, 2}, address: 0x3000, size: 8192,
					role: vmmOwner, properties: []byte{4}, access: []byte{5},
					accessCount: 1, accessSize: 1,
				},
			},
			{
				participant: "11111111111111111111111111111111",
				record: vmmCaptureRecord{
					allocationUUID: [16]byte{1, 2}, address: 0x4000, size: 8192,
					role: vmmImporter, access: []byte{6},
					accessCount: 1, accessSize: 1,
				},
			},
		}
		var processes []VMMProcess
		for _, specification := range specifications {
			process := serveVMMInspect(
				t,
				[]vmmCaptureRecord{specification.record},
			)
			process.Participant = specification.participant
			processes = append(processes, process)
		}
		if reverse {
			slices.Reverse(processes)
		}
		ledger, err := inspectVMM(
			context.Background(),
			processes,
			"generation",
			"node-a",
		)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(ledger)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}

	forward := inspect(false)
	reverse := inspect(true)
	if !bytes.Equal(forward, reverse) {
		t.Fatalf("ledger depends on process discovery order:\n%s\n%s", forward, reverse)
	}
}

func TestInspectVMMKeepsIndependentOwnerUUIDsSeparate(t *testing.T) {
	const (
		ownerA    = "11111111111111111111111111111111"
		importerA = "22222222222222222222222222222222"
		ownerB    = "33333333333333333333333333333333"
		importerB = "44444444444444444444444444444444"
	)
	mapping := func(uuid [16]byte, role uint32, address uint64) vmmCaptureRecord {
		record := vmmCaptureRecord{
			allocationUUID: uuid,
			address:        address,
			size:           4096,
			role:           role,
			access:         []byte{1},
			accessCount:    1,
			accessSize:     1,
		}
		if role == vmmOwner {
			record.properties = []byte{1}
		}
		return record
	}
	process := func(participant string, record vmmCaptureRecord) VMMProcess {
		result := serveVMMInspect(t, []vmmCaptureRecord{record})
		result.Participant = participant
		return result
	}
	uuidA := [16]byte{1}
	uuidB := [16]byte{2}
	ledger, err := inspectVMM(
		context.Background(),
		[]VMMProcess{
			process(ownerB, mapping(uuidB, vmmOwner, 0x3000)),
			process(importerA, mapping(uuidA, vmmImporter, 0x2000)),
			process(importerB, mapping(uuidB, vmmImporter, 0x4000)),
			process(ownerA, mapping(uuidA, vmmOwner, 0x1000)),
		},
		"generation",
		"node-a",
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(ledger.Resources) != 2 ||
		ledger.Resources[0].Owner.Participant != ownerA ||
		ledger.Resources[1].Owner.Participant != ownerB {
		t.Fatalf("independent UUID resources were merged or reordered: %#v", ledger.Resources)
	}

	duplicateOwner := process(ownerB, mapping(uuidA, vmmOwner, 0x3000))
	_, err = inspectVMM(
		context.Background(),
		[]VMMProcess{
			process(ownerA, mapping(uuidA, vmmOwner, 0x1000)),
			duplicateOwner,
			process(importerA, mapping(uuidA, vmmImporter, 0x2000)),
		},
		"generation",
		"node-a",
	)
	if err == nil || !strings.Contains(err.Error(), "multiple owners") {
		t.Fatalf("duplicate owner UUID error = %v", err)
	}
}

func TestValidateRestoredVMMProcessesRequiresIdentityPlacement(t *testing.T) {
	const participant = "11111111111111111111111111111111"
	const sourceUUID = "00112233-4455-6677-8899-aabbccddeeff"
	ledger := vmmLedger{Participants: []vmmParticipant{{
		ID: participant,
		Placement: vmmPlacement{
			Node:     "node-a",
			GPUUUIDs: []string{sourceUUID},
		},
	}}}
	tests := []struct {
		name        string
		currentNode string
		currentUUID string
		wantError   string
	}{
		{
			name:        "exact identity",
			currentNode: "node-a",
			currentUUID: sourceUUID,
		},
		{
			name:        "same ordinal different UUID",
			currentNode: "node-a",
			currentUUID: "ffeeddcc-bbaa-9988-7766-554433221100",
			wantError:   "currently maps GPU",
		},
		{
			name:        "different current node",
			currentNode: "node-b",
			currentUUID: sourceUUID,
			wantError:   "target node",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			process := serveVMMPlacement(
				t,
				participant,
				7,
				sourceUUID,
				test.currentUUID,
			)

			placement, err := validateRestoredVMMProcesses(
				context.Background(),
				[]VMMProcess{process},
				ledger,
				test.currentNode,
			)
			if test.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantError) {
					t.Fatalf("placement error = %v, want %q", err, test.wantError)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := placement[participant][sourceUUID]; got != 7 {
				t.Fatalf("validated ordinal = %d, want 7", got)
			}
		})
	}
}

func TestValidateRestoredVMMProcessesParticipantSet(t *testing.T) {
	const (
		participant = "11111111111111111111111111111111"
		extra       = "22222222222222222222222222222222"
		sourceUUID  = "00112233-4455-6677-8899-aabbccddeeff"
	)
	ledger := vmmLedger{Participants: []vmmParticipant{{
		ID: participant,
		Placement: vmmPlacement{
			Node:     "node-a",
			GPUUUIDs: []string{sourceUUID},
		},
	}}}
	valid := func(t *testing.T) VMMProcess {
		return serveVMMPlacement(t, participant, 7, sourceUUID, sourceUUID)
	}
	empty := func(t *testing.T, id string) VMMProcess {
		return serveVMMPlacementResponse(t, id, 0, nil)
	}
	tests := []struct {
		name          string
		processes     func(*testing.T) []VMMProcess
		wantError     string
		wantPlacement bool
	}{
		{
			name: "extra empty placement is ignored",
			processes: func(t *testing.T) []VMMProcess {
				return []VMMProcess{valid(t), empty(t, extra)}
			},
			wantPlacement: true,
		},
		{
			name: "extra managed placement is rejected",
			processes: func(t *testing.T) []VMMProcess {
				return []VMMProcess{
					valid(t),
					serveVMMPlacement(t, extra, 7, sourceUUID, sourceUUID),
				}
			},
			wantError: "unexpected restored CUDA VMM participant",
		},
		{
			name: "missing expected endpoint is rejected",
			processes: func(t *testing.T) []VMMProcess {
				return []VMMProcess{empty(t, extra)}
			},
			wantError: "has no restored shim endpoint",
		},
		{
			name: "expected empty placement is rejected",
			processes: func(t *testing.T) []VMMProcess {
				return []VMMProcess{empty(t, participant)}
			},
			wantError: "current GPU placement count is 0, want 1",
		},
		{
			name: "duplicate participant is rejected",
			processes: func(*testing.T) []VMMProcess {
				return []VMMProcess{
					{Participant: participant},
					{Participant: participant},
				}
			},
			wantError: "multiple restored shims claim participant",
		},
		{
			name: "extra placement query error is rejected",
			processes: func(t *testing.T) []VMMProcess {
				return []VMMProcess{
					valid(t),
					{
						Participant: extra,
						SocketPath:  filepath.Join(t.TempDir(), "missing.sock"),
					},
				}
			},
			wantError: "query CUDA VMM participant",
		},
		{
			name: "extra placement decode error is rejected",
			processes: func(t *testing.T) []VMMProcess {
				return []VMMProcess{
					valid(t),
					serveVMMPlacementResponse(t, extra, 1, nil),
				}
			},
			wantError: "decode CUDA VMM participant",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			placement, err := validateRestoredVMMProcesses(
				context.Background(),
				test.processes(t),
				ledger,
				"node-a",
			)
			if test.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantError) {
					t.Fatalf("placement error = %v, want %q", err, test.wantError)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !test.wantPlacement {
				return
			}
			if len(placement) != 1 {
				t.Fatalf("execution placement has %d participants, want 1", len(placement))
			}
			if _, ok := placement[extra]; ok {
				t.Fatal("extra empty participant was included in execution placement")
			}
			if got := placement[participant][sourceUUID]; got != 7 {
				t.Fatalf("validated ordinal = %d, want 7", got)
			}
		})
	}
}

func TestCurrentVMMNodeFailsClosed(t *testing.T) {
	t.Setenv("NODE_NAME", "")
	if _, err := currentVMMNode(); err == nil {
		t.Fatal("missing current node identity was accepted")
	}
	t.Setenv("NODE_NAME", "node-a")
	if node, err := currentVMMNode(); err != nil || node != "node-a" {
		t.Fatalf("current node = %q, %v", node, err)
	}
}

func TestRedisStoreContract(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	done := make(chan error, 1)
	go func() {
		connection, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		defer connection.Close()
		reader := bufio.NewReader(connection)
		command, err := readRESPCommand(reader)
		if err != nil {
			done <- err
			return
		}
		if len(command) != 3 || string(command[0]) != "SET" ||
			string(command[1]) != "binary" ||
			!bytes.Equal(command[2], []byte{0, 1, 2, 0, 3}) {
			done <- fmt.Errorf("unexpected command: %q", command)
			return
		}
		_, err = io.WriteString(connection, "+OK\r\n")
		done <- err
	}()

	client, err := dialVMMRedis(
		context.Background(),
		vmmRedisConfig{address: listener.Addr().String()},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if err := client.set("binary", []byte{0, 1, 2, 0, 3}); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestRedisRejectsInvalidResponseLines(t *testing.T) {
	tests := []struct {
		name     string
		response string
	}{
		{name: "bulk length below nil", response: "$-2\r\n"},
		{name: "bulk length above limit", response: "$536870913\r\n"},
		{
			name:     "overlong simple response",
			response: "+" + strings.Repeat("a", vmmMaximumRESPLine+1) + "\r\n",
		},
		{
			name:     "overlong error response",
			response: "-" + strings.Repeat("a", vmmMaximumRESPLine+1) + "\r\n",
		},
		{
			name:     "overlong bulk length",
			response: "$" + strings.Repeat("1", vmmMaximumRESPLine+1) + "\r\n",
		},
		{name: "unterminated response", response: "+OK"},
		{name: "malformed terminator", response: "+OK\n"},
		{name: "malformed bulk length", response: "$1x\r\n"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server, client := net.Pipe()
			defer server.Close()
			done := make(chan struct{})
			go func() {
				defer close(done)
				reader := bufio.NewReader(server)
				_, _ = readRESPCommand(reader)
				_, _ = io.WriteString(server, test.response)
				_ = server.Close()
			}()
			redis := &vmmRedisClient{
				connection: client,
				reader:     bufio.NewReader(client),
			}
			if _, err := redis.command([]byte("GET"), []byte("key")); err == nil {
				t.Fatalf("Redis response %q was accepted", test.response)
			}
			_ = client.Close()
			<-done
		})
	}
}

func TestExchangeVMMPreservesPayloadCoalescedWithHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vmm.sock")
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	want := []byte("coalesced payload")
	go func() {
		connection, err := listener.Accept()
		if err != nil {
			return
		}
		defer connection.Close()
		request := make([]byte, vmmHeaderSize)
		if _, err := io.ReadFull(connection, request); err != nil {
			return
		}
		response := encodeVMMHeader(vmmHeader{
			Operation:   vmmIdentify,
			PayloadSize: uint64(len(want)),
		})
		response = append(response, want...)
		_ = writeVMMBytes(connection, response)
	}()

	response, err := exchangeVMM(
		context.Background(),
		VMMProcess{SocketPath: path},
		vmmHeader{Operation: vmmIdentify},
		nil,
		-1,
		vmmResponseFDNone,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(response.payload, want) {
		t.Fatalf("payload = %q, want %q", response.payload, want)
	}
}

func TestExchangeVMMClosesReceivedFDOnInvalidResponse(t *testing.T) {
	tests := []struct {
		name     string
		response []byte
	}{
		{name: "invalid header", response: make([]byte, vmmHeaderSize)},
		{
			name: "valid success forbids FD",
			response: encodeVMMHeader(vmmHeader{
				Operation: vmmIdentify,
			}),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testExchangeVMMClosesReceivedFD(t, test.response)
		})
	}
}

func testExchangeVMMClosesReceivedFD(t *testing.T, response []byte) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "vmm.sock")
	listener, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	fd, err := unix.MemfdCreate("vmm-test", unix.MFD_CLOEXEC)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Close(fd)
	var status unix.Stat_t
	if err := unix.Fstat(fd, &status); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		connection, err := listener.AcceptUnix()
		if err != nil {
			return
		}
		defer connection.Close()
		request := make([]byte, vmmHeaderSize)
		if _, err := io.ReadFull(connection, request); err != nil {
			return
		}
		_, _, _ = connection.WriteMsgUnix(response, unix.UnixRights(fd), nil)
	}()
	before := countMatchingFDs(t, status.Dev, status.Ino)
	_, err = exchangeVMM(
		context.Background(),
		VMMProcess{SocketPath: path},
		vmmHeader{Operation: vmmIdentify},
		nil,
		-1,
		vmmResponseFDNone,
	)
	<-done
	if err == nil {
		t.Fatal("response with invalid FD contract was accepted")
	}
	if after := countMatchingFDs(t, status.Dev, status.Ino); after != before {
		t.Fatalf("matching FD count = %d, want %d", after, before)
	}
}

func TestExchangeVMMRejectsMissingRequiredFD(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vmm.sock")
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		connection, err := listener.Accept()
		if err != nil {
			return
		}
		defer connection.Close()
		request := make([]byte, vmmHeaderSize)
		if _, err := io.ReadFull(connection, request); err != nil {
			return
		}
		_ = writeVMMBytes(connection, encodeVMMHeader(vmmHeader{
			Operation: vmmRestoreOwner,
		}))
	}()

	if _, err := exchangeVMM(
		context.Background(),
		VMMProcess{SocketPath: path},
		vmmHeader{Operation: vmmRestoreOwner},
		nil,
		-1,
		vmmResponseFDRequired,
	); err == nil {
		t.Fatal("missing required response FD was accepted")
	}
}

func TestCloseVMMFDClearsOwnershipBeforeReturningError(t *testing.T) {
	fd := 1 << 30
	if err := closeVMMFD(&fd); err == nil {
		t.Fatal("closing an invalid owned FD returned no error")
	}
	if fd != -1 {
		t.Fatalf("owned FD slot = %d, want -1", fd)
	}
	if err := closeVMMFD(&fd); err != nil {
		t.Fatalf("cleared owned FD was retried: %v", err)
	}
}

func TestReceiveVMMFDsPreservesOwnershipOnPartialParseFailure(t *testing.T) {
	fd, err := unix.MemfdCreate("vmm-partial-rights", unix.MFD_CLOEXEC)
	if err != nil {
		t.Fatal(err)
	}
	ancillary := append(unix.UnixRights(fd), 0)
	received, err := receiveVMMFDs(ancillary)
	if err == nil || len(received) != 1 {
		t.Fatalf("partial ancillary parse = %v, %v", received, err)
	}
	if err := closeVMMFDs(received); err != nil {
		t.Fatal(err)
	}
	if received[0] != -1 {
		t.Fatalf("received FD slot = %d, want -1", received[0])
	}
}

func serveVMMPlacement(
	t *testing.T,
	participant string,
	ordinal int32,
	sourceUUID string,
	currentUUID string,
) VMMProcess {
	t.Helper()
	payload := make([]byte, vmmPlacementSize)
	binary.LittleEndian.PutUint32(payload[0:4], uint32(ordinal))
	copy(payload[8:24], parseGPUUUID(sourceUUID))
	copy(payload[24:40], parseGPUUUID(currentUUID))
	return serveVMMPlacementResponse(t, participant, 1, payload)
}

func serveVMMPlacementResponse(
	t *testing.T,
	participant string,
	count uint32,
	payload []byte,
) VMMProcess {
	t.Helper()
	return serveVMMProcess(t, "", participant, func(
		request vmmResponse,
	) (vmmResponse, error) {
		if request.header.Operation != vmmQueryPlacement {
			return vmmResponse{}, fmt.Errorf(
				"unexpected VMM operation %d", request.header.Operation,
			)
		}
		return vmmResponse{
			header:  vmmHeader{Operation: vmmQueryPlacement, Count: count},
			payload: payload,
			fd:      -1,
		}, nil
	})
}

func countMatchingFDs(t *testing.T, dev uint64, ino uint64) int {
	t.Helper()
	entries, err := os.ReadDir("/proc/self/fd")
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, entry := range entries {
		fd, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		var status unix.Stat_t
		if unix.Fstat(fd, &status) == nil && status.Dev == dev && status.Ino == ino {
			count++
		}
	}
	return count
}

func serveVMMInspect(
	t *testing.T,
	records []vmmCaptureRecord,
) VMMProcess {
	t.Helper()
	return serveVMMProcess(
		t,
		"",
		"11111111111111111111111111111111",
		func(request vmmResponse) (vmmResponse, error) {
			if request.header.Operation != vmmInspect {
				return vmmResponse{}, fmt.Errorf(
					"unexpected VMM operation %d", request.header.Operation,
				)
			}
			payload := encodeVMMRecordsForTest(records)
			return vmmResponse{
				header:  vmmHeader{Operation: vmmInspect, Count: uint32(len(records))},
				payload: payload,
				fd:      -1,
			}, nil
		},
	)
}

func encodeVMMRecordsForTest(records []vmmCaptureRecord) []byte {
	var payload []byte
	for _, record := range records {
		if record.kind == 0 {
			record.kind = vmmAllocation
		}
		if record.gpuUUID == "" {
			record.gpuUUID = "00112233-4455-6677-8899-aabbccddeeff"
		}
		if record.handleType == 0 {
			record.handleType = 1
		}
		encoded := make([]byte, vmmRecordSize+len(record.properties)+len(record.access))
		copy(encoded[0:16], record.allocationUUID[:])
		binary.LittleEndian.PutUint64(encoded[24:32], record.address)
		binary.LittleEndian.PutUint64(encoded[32:40], record.size)
		binary.LittleEndian.PutUint32(encoded[48:52], record.role)
		binary.LittleEndian.PutUint32(encoded[52:56], record.kind)
		binary.LittleEndian.PutUint32(encoded[56:60], record.handleType)
		binary.LittleEndian.PutUint32(encoded[60:64], record.flags)
		binary.LittleEndian.PutUint32(encoded[64:68], uint32(record.device))
		binary.LittleEndian.PutUint32(encoded[68:72], uint32(len(record.properties)))
		binary.LittleEndian.PutUint32(encoded[72:76], record.accessCount)
		binary.LittleEndian.PutUint32(encoded[76:80], record.accessSize)
		copy(encoded[80:96], parseGPUUUID(record.gpuUUID))
		copy(encoded[vmmRecordSize:], record.properties)
		copy(encoded[vmmRecordSize+len(record.properties):], record.access)
		payload = append(payload, encoded...)
	}
	return payload
}

func readRESPCommand(reader *bufio.Reader) ([][]byte, error) {
	line, err := reader.ReadString('\n')
	if err != nil {
		return nil, err
	}
	count, err := strconv.Atoi(line[1 : len(line)-2])
	if err != nil {
		return nil, err
	}
	command := make([][]byte, count)
	for index := range command {
		line, err = reader.ReadString('\n')
		if err != nil {
			return nil, err
		}
		size, err := strconv.Atoi(line[1 : len(line)-2])
		if err != nil {
			return nil, err
		}
		value := make([]byte, size+2)
		if _, err := io.ReadFull(reader, value); err != nil {
			return nil, err
		}
		command[index] = value[:size]
	}
	return command, nil
}

func serveVMMRestoreRedis(
	t *testing.T,
	loaded map[string][]byte,
	events chan<- string,
) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		run := func() error {
			poison, err := listener.Accept()
			if err != nil {
				return err
			}
			reader := bufio.NewReader(poison)
			for _, want := range []string{"FLUSHDB", "SET"} {
				command, err := readRESPCommand(reader)
				if err != nil || len(command) == 0 || string(command[0]) != want {
					return fmt.Errorf("poison Redis command = %q, want %s: %w", command, want, err)
				}
				if _, err := io.WriteString(poison, "+OK\r\n"); err != nil {
					return err
				}
			}
			if err := poison.Close(); err != nil {
				return err
			}
			connection, err := listener.Accept()
			if err != nil {
				return err
			}
			defer connection.Close()
			reader = bufio.NewReader(connection)
			for {
				command, err := readRESPCommand(reader)
				if errors.Is(err, io.EOF) {
					return nil
				}
				if err != nil {
					return err
				}
				switch string(command[0]) {
				case "GET":
					key := string(command[1])
					if marker := ":resource:"; events != nil && strings.Contains(key, marker) {
						events <- "content:" + key[strings.LastIndex(key, marker)+len(marker):]
					}
					value, ok := loaded[key]
					if !ok {
						_, err = io.WriteString(connection, "$-1\r\n")
					} else {
						_, err = fmt.Fprintf(connection, "$%d\r\n", len(value))
						if err == nil {
							err = writeVMMBytes(connection, append(append([]byte(nil), value...), '\r', '\n'))
						}
					}
				case "SET":
					loaded[string(command[1])] = append([]byte(nil), command[2]...)
					if events != nil && strings.HasSuffix(string(command[1]), ":state") &&
						string(command[2]) == "restored" {
						events <- "redis-restored"
					}
					_, err = io.WriteString(connection, "+OK\r\n")
				default:
					err = fmt.Errorf("unexpected Redis command: %q", command)
				}
				if err != nil {
					return err
				}
			}
		}
		done <- run()
	}()
	t.Cleanup(func() {
		_ = listener.Close()
		if err := <-done; err != nil && !errors.Is(err, net.ErrClosed) {
			t.Errorf("restore Redis server: %v", err)
		}
	})
	return listener.Addr().String()
}

func serveVMMRestoreProcess(
	t *testing.T,
	participant string,
	sourceUUID string,
	contents [][]byte,
	ownerResource uint64,
	failOwner uint64,
	events chan<- string,
) VMMProcess {
	t.Helper()
	roleSummary := fmt.Sprintf("%d:owner,%d:importer", ownerResource, 3-ownerResource)
	return serveVMMProcess(t, "", participant, func(request vmmResponse) (vmmResponse, error) {
		response := vmmResponse{
			header: vmmHeader{
				Operation:   request.header.Operation,
				Participant: participant,
			},
			fd: -1,
		}
		switch request.header.Operation {
		case vmmQueryPlacement:
			placement := make([]byte, vmmPlacementSize)
			copy(placement[8:24], parseGPUUUID(sourceUUID))
			copy(placement[24:40], parseGPUUUID(sourceUUID))
			response.header.Count = 1
			response.payload = placement
			events <- "placement:" + roleSummary
			return response, nil
		case vmmRestoreOwner:
			if err := validateVMMRestoreRequest(
				request, ownerResource, vmmOwner,
				contents[ownerResource-1],
			); err != nil {
				return response, err
			}
			events <- fmt.Sprintf("owner:%d", ownerResource)
			if failOwner == ownerResource {
				response.header.Status = 1
				response.header.Message = "owner replay failed"
				return response, nil
			}
			var err error
			response.fd, err = unix.MemfdCreate("vmm-restore-owner", unix.MFD_CLOEXEC)
			return response, err
		case vmmRestoreImporter:
			importerResource := uint64(3) - ownerResource
			if err := validateVMMRestoreRequest(
				request, importerResource, vmmImporter, nil,
			); err != nil {
				return response, err
			}
			events <- fmt.Sprintf("importer:%d", importerResource)
			return response, nil
		case vmmIdentify:
			events <- "health:" + roleSummary
			return response, nil
		default:
			return response, fmt.Errorf("unexpected VMM operation %d", request.header.Operation)
		}
	})
}

func validateVMMRestoreRequest(
	request vmmResponse,
	resourceID uint64,
	wantRole uint32,
	wantContents []byte,
) error {
	if request.header.ObjectID != resourceID || len(request.payload) < vmmRecordSize {
		return fmt.Errorf("invalid resource %d replay header", resourceID)
	}
	if binary.LittleEndian.Uint64(request.payload[16:24]) != resourceID ||
		binary.LittleEndian.Uint32(request.payload[48:52]) != wantRole {
		return fmt.Errorf("invalid resource %d replay payload", resourceID)
	}
	metadataSize := vmmRecordSize +
		int(binary.LittleEndian.Uint32(request.payload[68:72])) +
		int(binary.LittleEndian.Uint32(request.payload[72:76]))*
			int(binary.LittleEndian.Uint32(request.payload[76:80]))
	if metadataSize > len(request.payload) ||
		!bytes.Equal(request.payload[metadataSize:], wantContents) {
		return fmt.Errorf("invalid resource %d replay contents", resourceID)
	}
	if (request.fd >= 0) != (wantRole == vmmImporter) {
		return fmt.Errorf("invalid resource %d replay FD", resourceID)
	}
	return nil
}

func serveVMMProcess(
	t *testing.T,
	path string,
	participant string,
	handler func(vmmResponse) (vmmResponse, error),
) VMMProcess {
	t.Helper()
	if path == "" {
		path = filepath.Join(t.TempDir(), "vmm.sock")
	}
	listener, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		for {
			connection, err := listener.AcceptUnix()
			if err != nil {
				done <- err
				return
			}
			request, err := readVMMTestRequest(connection)
			if err == nil {
				response := vmmResponse{
					header: vmmHeader{Operation: vmmIdentify, Participant: participant},
					fd:     -1,
				}
				if handler != nil {
					response, err = handler(request)
				}
				if err == nil {
					err = writeVMMTestResponse(connection, response)
				}
				err = errors.Join(err, closeVMMFD(&response.fd))
			}
			err = errors.Join(err, closeVMMFD(&request.fd), connection.Close())
			if err != nil {
				done <- err
				return
			}
		}
	}()
	t.Cleanup(func() {
		_ = listener.Close()
		if err := <-done; err != nil && !errors.Is(err, net.ErrClosed) {
			t.Errorf("VMM process %s: %v", participant, err)
		}
	})
	return VMMProcess{
		ObservedPID:  1,
		NamespacePID: 1,
		SocketPath:   path,
		Participant:  participant,
	}
}

func readVMMTestRequest(connection *net.UnixConn) (vmmResponse, error) {
	request := vmmResponse{fd: -1}
	encoded := make([]byte, vmmHeaderSize)
	ancillary := make([]byte, unix.CmsgSpace(4))
	size, ancillarySize, flags, _, err := connection.ReadMsgUnix(encoded, ancillary)
	if err != nil {
		return request, err
	}
	fds, err := receiveVMMFDs(ancillary[:ancillarySize])
	if err != nil {
		return request, errors.Join(err, closeVMMFDs(fds))
	}
	if flags&unix.MSG_CTRUNC != 0 || len(fds) > 1 {
		return request, errors.Join(
			fmt.Errorf("invalid test VMM request FDs: flags=%d count=%d", flags, len(fds)),
			closeVMMFDs(fds),
		)
	}
	if len(fds) == 1 {
		request.fd = fds[0]
	}
	if _, err := io.ReadFull(connection, encoded[size:]); err != nil {
		return request, err
	}
	request.header, err = decodeVMMHeader(encoded)
	if err != nil {
		return request, err
	}
	request.payload = make([]byte, request.header.PayloadSize)
	if _, err := io.ReadFull(connection, request.payload); err != nil {
		return request, err
	}
	return request, nil
}

func writeVMMTestResponse(connection *net.UnixConn, response vmmResponse) error {
	response.header.PayloadSize = uint64(len(response.payload))
	encoded := append(encodeVMMHeader(response.header), response.payload...)
	if response.fd >= 0 {
		written, _, err := connection.WriteMsgUnix(
			encoded, unix.UnixRights(response.fd), nil,
		)
		if err != nil {
			return err
		}
		if err := writeVMMBytes(connection, encoded[written:]); err != nil {
			return err
		}
		return nil
	}
	return writeVMMBytes(connection, encoded)
}
