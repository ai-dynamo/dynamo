package cuda

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
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
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
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

func TestValidateVMMLedgerMulticastGraph(t *testing.T) {
	const (
		owner       = "11111111111111111111111111111111"
		importer    = "22222222222222222222222222222222"
		ownerGPU    = "00112233-4455-6677-8899-aabbccddeeff"
		importerGPU = "ffeeddcc-bbaa-9988-7766-554433221100"
		size        = uint64(4096)
	)
	mapping := func(participant, gpu string) vmmMapping {
		return vmmMapping{
			Participant:         participant,
			Address:             0x1000,
			Size:                size,
			GPUUUID:             gpu,
			RequestedHandleType: vmmHandleFabric,
			Access:              []byte{1},
			AccessCount:         1,
			AccessSize:          1,
		}
	}
	valid := func() vmmLedger {
		allocationOwner := mapping(owner, ownerGPU)
		allocationImporter := mapping(importer, importerGPU)
		multicastOwner := mapping(owner, ownerGPU)
		multicastOwner.Multicast = &vmmMulticastMapping{
			BackingResourceID: 1,
			BackingRole:       vmmOwner,
			Size:              size,
			BindAPI:           vmmMulticastBindMem,
		}
		multicastImporter := mapping(importer, importerGPU)
		multicastImporter.Multicast = &vmmMulticastMapping{
			BackingResourceID: 1,
			BackingRole:       vmmImporter,
			Size:              size,
			BindAPI:           vmmMulticastBindMemV2,
		}
		return vmmLedger{
			Version: vmmLedgerVersion,
			Participants: []vmmParticipant{
				{
					ID: owner,
					Placement: vmmPlacement{
						Node: "node-a", GPUUUIDs: []string{ownerGPU},
					},
				},
				{
					ID: importer,
					Placement: vmmPlacement{
						Node: "node-a", GPUUUIDs: []string{importerGPU},
					},
				},
			},
			Resources: []vmmResource{
				{
					ID:        1,
					Kind:      "allocation",
					Owner:     allocationOwner,
					Importers: []vmmMapping{allocationImporter},
				},
				{
					ID:        2,
					Kind:      "multicast",
					Owner:     multicastOwner,
					Importers: []vmmMapping{multicastImporter},
					Multicast: &vmmMulticastProperties{
						HandleTypes: vmmHandleFabric,
						NumDevices:  2,
						Size:        size,
					},
				},
			},
		}
	}
	if err := validateVMMLedger(valid()); err != nil {
		t.Fatalf("valid multicast graph rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*vmmLedger)
	}{
		{
			name: "unknown backing resource",
			mutate: func(ledger *vmmLedger) {
				ledger.Resources[1].Owner.Multicast.BackingResourceID = 3
			},
		},
		{
			name: "duplicate member GPU",
			mutate: func(ledger *vmmLedger) {
				ledger.Resources[1].Importers[0].GPUUUID = ownerGPU
				ledger.Participants[1].Placement.GPUUUIDs[0] = ownerGPU
			},
		},
		{
			name: "partial bind",
			mutate: func(ledger *vmmLedger) {
				ledger.Resources[1].Owner.Multicast.Size--
			},
		},
		{
			name: "nonzero multicast offset",
			mutate: func(ledger *vmmLedger) {
				ledger.Resources[1].Owner.Multicast.MulticastOffset = 1
			},
		},
		{
			name: "duplicate participant bind role",
			mutate: func(ledger *vmmLedger) {
				ledger.Resources[1].Importers[0].Multicast.BackingRole = vmmOwner
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ledger := valid()
			test.mutate(&ledger)
			if err := validateVMMLedger(ledger); err == nil {
				t.Fatal("malformed multicast graph was accepted")
			}
		})
	}
}

func TestExchangeVMMRejectsNonemptySuccessBeforeReadingPayload(t *testing.T) {
	const participant = "11111111111111111111111111111111"
	path := filepath.Join(t.TempDir(), "vmm.sock")
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	releasePeer := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
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
			Operation:      vmmSetPlacement,
			Count:          1,
			AllocationUUID: [16]byte{1},
			ObjectID:       1,
			PayloadSize:    1,
			Message:        "unexpected",
			Participant:    participant,
		}))
		<-releasePeer
	}()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err = exchangeVMM(
		ctx,
		VMMProcess{SocketPath: path, Participant: participant},
		vmmHeader{Operation: vmmSetPlacement},
		nil,
		-1,
		vmmResponseEmpty,
	)
	close(releasePeer)
	<-done
	if err == nil || !strings.Contains(err.Error(), "nonempty success response") {
		t.Fatalf("nonempty SET_PLACEMENT response error = %v", err)
	}
}

func TestVMMPlacementPayloadSizeLimit(t *testing.T) {
	maximumCount := vmmMaximumRedisBulk / vmmPlacementSize
	size, err := vmmPlacementPayloadSize(maximumCount)
	if err != nil || size != maximumCount*vmmPlacementSize {
		t.Fatalf("maximum placement payload size = %d, %v", size, err)
	}
	if _, err := vmmPlacementPayloadSize(maximumCount + 1); err == nil {
		t.Fatal("placement payload above the C limit was accepted")
	}
}

func TestSetRestoredVMMPlacementEmptyExtraEndpoint(t *testing.T) {
	const (
		participant = "11111111111111111111111111111111"
		extra       = "22222222222222222222222222222222"
		sourceUUID  = "00112233-4455-6677-8899-aabbccddeeff"
	)
	managed := serveVMMProcess(t, "", participant, func(
		request vmmResponse,
	) (vmmResponse, error) {
		if err := validateVMMPlacementRequest(
			request, sourceUUID, sourceUUID, 0,
		); err != nil {
			return vmmResponse{}, err
		}
		return vmmResponse{
			header: vmmHeader{Operation: vmmSetPlacement},
			fd:     -1,
		}, nil
	})
	extraEndpoint := func(t *testing.T, reject bool) VMMProcess {
		return serveVMMProcess(t, "", extra, func(
			request vmmResponse,
		) (vmmResponse, error) {
			if request.header.Operation != vmmSetPlacement ||
				request.header.Count != 0 ||
				len(request.payload) != 0 {
				return vmmResponse{}, errors.New("extra endpoint received nonempty setup")
			}
			response := vmmResponse{
				header: vmmHeader{Operation: vmmSetPlacement},
				fd:     -1,
			}
			if reject {
				response.header.Status = -1
				response.header.Message = "detached CUDA placement metadata is inconsistent"
			}
			return response, nil
		})
	}
	payload, err := encodeVMMPlacement([]types.VMMPlacement{{
		SourceGPUUUID: sourceUUID,
		TargetGPUUUID: sourceUUID,
		TargetOrdinal: 0,
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name      string
		reject    bool
		wantError bool
	}{
		{name: "zero-managed endpoint accepts empty setup"},
		{name: "managed endpoint rejects empty setup", reject: true, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := setRestoredVMMPlacement(
				context.Background(),
				[]vmmPlacementSetup{
					{process: managed, count: 1, payload: payload},
					{process: extraEndpoint(t, test.reject)},
				},
			)
			if (err != nil) != test.wantError {
				t.Fatalf("set placement error = %v, wantError=%t", err, test.wantError)
			}
		})
	}
}

func TestValidateVMMResponseShapeRejectsInvalidShapeBeforeAllocation(t *testing.T) {
	tests := []struct {
		name        string
		expectation vmmResponseExpectation
		response    vmmHeader
		fdCount     int
	}{
		{
			name:        "oversized fabric payload",
			expectation: vmmResponseFabricBroker,
			response: vmmHeader{
				HandleType:  vmmHandleFabric,
				PayloadSize: vmmFabricHandleSize + 1,
			},
		},
		{
			name:        "POSIX payload",
			expectation: vmmResponsePOSIXBroker,
			response: vmmHeader{
				HandleType:  vmmHandlePOSIX,
				PayloadSize: 1,
			},
			fdCount: 1,
		},
		{
			name:        "wrong handle type",
			expectation: vmmResponseFabricBroker,
			response:    vmmHeader{HandleType: vmmHandlePOSIX},
		},
		{
			name:        "empty response count",
			expectation: vmmResponseEmpty,
			response:    vmmHeader{Count: 1},
		},
		{
			name:        "empty response object metadata",
			expectation: vmmResponseEmpty,
			response: vmmHeader{
				ObjectID:       1,
				AllocationUUID: [16]byte{1},
				Message:        "unexpected",
			},
		},
		{
			name:        "empty response FD",
			expectation: vmmResponseEmpty,
			fdCount:     1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := validateVMMResponseShape(
				vmmRestoreOwner,
				test.response,
				test.expectation,
				test.fdCount,
			)
			if err == nil {
				t.Fatal("invalid typed response declaration was accepted")
			}
		})
	}
}

func TestRestoreVMMFabricTransport(t *testing.T) {
	const (
		generation   = "00112233445566778899aabbccddeeff"
		sourceUUID   = "00112233-4455-6677-8899-aabbccddeeff"
		participantA = "11111111111111111111111111111111"
		participantB = "22222222222222222222222222222222"
	)
	contents := []byte{1, 2, 3, 4}
	raw := bytes.Repeat([]byte{0xa5}, vmmFabricHandleSize)
	mapping := func(participant string, address uint64) vmmMapping {
		return vmmMapping{
			Participant:         participant,
			Address:             address,
			Size:                uint64(len(contents)),
			GPUUUID:             sourceUUID,
			RequestedHandleType: vmmHandleFabric,
			Access:              []byte{1, 2, 3, 4},
			AccessCount:         1,
			AccessSize:          4,
		}
	}
	ledger := vmmLedger{
		Version:    vmmLedgerVersion,
		Generation: generation,
		Participants: []vmmParticipant{
			{
				ID: participantA,
				Placement: vmmPlacement{
					Node: "node-a", GPUUUIDs: []string{sourceUUID},
				},
			},
			{
				ID: participantB,
				Placement: vmmPlacement{
					Node: "node-a", GPUUUIDs: []string{sourceUUID},
				},
			},
		},
		Resources: []vmmResource{{
			ID:        1,
			Kind:      "allocation",
			Owner:     mapping(participantA, 0x1000),
			Importers: []vmmMapping{mapping(participantB, 0x2000)},
		}},
	}
	encoded, err := json.Marshal(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte(base64.StdEncoding.EncodeToString(raw))) {
		t.Fatal("FABRIC raw broker bytes appeared in durable ledger JSON")
	}
	digest := sha256.New()
	writeVMMDigestPart(digest, encoded)
	writeVMMDigestObject(digest, 1, contents)
	expectedDigest := hex.EncodeToString(digest.Sum(nil))
	events := make(chan string, 8)
	owner := serveVMMProcess(
		t, "", participantA,
		func(request vmmResponse) (vmmResponse, error) {
			switch request.header.Operation {
			case vmmSetPlacement:
				if err := validateVMMPlacementRequest(
					request, sourceUUID, sourceUUID, 0,
				); err != nil {
					return vmmResponse{}, err
				}
				return vmmResponse{
					header: vmmHeader{Operation: vmmSetPlacement},
					fd:     -1,
				}, nil
			case vmmFinalizeRestore:
				return vmmResponse{
					header: vmmHeader{Operation: vmmFinalizeRestore},
					fd:     -1,
				}, nil
			case vmmRestoreOwner:
				if request.header.HandleType != vmmHandleFabric ||
					request.fd >= 0 {
					return vmmResponse{}, errors.New("invalid FABRIC owner request shape")
				}
				if err := validateVMMRestoreRequest(
					request, 1, vmmOwner, 0, contents,
				); err != nil {
					return vmmResponse{}, err
				}
				events <- "owner"
				return vmmResponse{
					header: vmmHeader{
						Operation:  vmmRestoreOwner,
						HandleType: vmmHandleFabric,
						ObjectKind: vmmAllocation,
					},
					payload: append([]byte(nil), raw...),
					fd:      -1,
				}, nil
			case vmmIdentify:
				return vmmResponse{
					header: vmmHeader{
						Operation:   vmmIdentify,
						Participant: participantA,
					},
					fd: -1,
				}, nil
			default:
				return vmmResponse{}, fmt.Errorf(
					"unexpected owner operation %d", request.header.Operation,
				)
			}
		},
	)
	importer := serveVMMProcess(
		t, "", participantB,
		func(request vmmResponse) (vmmResponse, error) {
			switch request.header.Operation {
			case vmmSetPlacement:
				if err := validateVMMPlacementRequest(
					request, sourceUUID, sourceUUID, 0,
				); err != nil {
					return vmmResponse{}, err
				}
				return vmmResponse{
					header: vmmHeader{Operation: vmmSetPlacement},
					fd:     -1,
				}, nil
			case vmmRestoreImporter:
				if request.header.HandleType != vmmHandleFabric ||
					request.fd >= 0 ||
					len(request.payload) < vmmRecordSize+vmmFabricHandleSize ||
					!bytes.Equal(
						request.payload[len(request.payload)-vmmFabricHandleSize:],
						raw,
					) {
					return vmmResponse{}, errors.New(
						"invalid FABRIC importer request shape",
					)
				}
				metadata := request.payload[:len(request.payload)-vmmFabricHandleSize]
				request.payload = metadata
				if err := validateVMMRestoreRequest(
					request, 1, vmmImporter, 0, nil,
				); err != nil {
					return vmmResponse{}, err
				}
				events <- "importer"
				return vmmResponse{
					header: vmmHeader{Operation: vmmRestoreImporter},
					fd:     -1,
				}, nil
			case vmmFinalizeRestore:
				return vmmResponse{
					header: vmmHeader{Operation: vmmFinalizeRestore},
					fd:     -1,
				}, nil
			case vmmIdentify:
				return vmmResponse{
					header: vmmHeader{
						Operation:   vmmIdentify,
						Participant: participantB,
					},
					fd: -1,
				}, nil
			default:
				return vmmResponse{}, fmt.Errorf(
					"unexpected importer operation %d",
					request.header.Operation,
				)
			}
		},
	)
	loaded := map[string][]byte{
		vmmRedisKey(generation, "state"):      []byte("detached"),
		vmmRedisKey(generation, "ledger"):     encoded,
		vmmRedisKey(generation, "digest"):     []byte(expectedDigest),
		vmmRedisKey(generation, "resource:1"): contents,
	}
	t.Setenv(VMMRedisAddressEnv, serveVMMRestoreRedis(t, loaded, nil))
	t.Setenv(VMMRedisRDBPathEnv, "")
	t.Setenv(VMMRedisRestoreCmdEnv, "/bin/true")
	t.Setenv("NODE_NAME", "node-a")
	if err := RestoreVMM(
		context.Background(),
		[]VMMProcess{owner, importer},
		generation,
		expectedDigest,
		t.TempDir(),
		[]types.VMMPlacement{{
			SourceGPUUUID: sourceUUID,
			TargetGPUUUID: sourceUUID,
			TargetOrdinal: 0,
		}},
		func() error {
			events <- "unlock"
			return nil
		},
		logr.Discard(),
	); err != nil {
		t.Fatal(err)
	}
	got := []string{<-events, <-events, <-events}
	want := []string{"unlock", "owner", "importer"}
	if !slices.Equal(got, want) {
		t.Fatalf("FABRIC restore events = %v, want %v", got, want)
	}
}

func TestVMMProtocolV5ObjectKindLayout(t *testing.T) {
	encoded := encodeVMMHeader(vmmHeader{
		Operation:  vmmRestoreOwner,
		HandleType: vmmHandleFabric,
		ObjectKind: vmmMulticast,
	})
	if got := binary.LittleEndian.Uint16(encoded[4:6]); got != 5 {
		t.Fatalf("protocol version = %d, want 5", got)
	}
	if got := binary.LittleEndian.Uint32(encoded[16:20]); got != vmmHandleFabric {
		t.Fatalf("header handle type = %d, want %d", got, vmmHandleFabric)
	}
	if got := binary.LittleEndian.Uint32(encoded[20:24]); got != vmmMulticast {
		t.Fatalf("header object kind = %d, want %d", got, vmmMulticast)
	}
	if len(encoded) != vmmHeaderSize {
		t.Fatalf("header size = %d, want %d", len(encoded), vmmHeaderSize)
	}
	decoded, err := decodeVMMHeader(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.HandleType != vmmHandleFabric {
		t.Fatalf(
			"decoded handle type = %d, want %d",
			decoded.HandleType, vmmHandleFabric,
		)
	}
	if decoded.ObjectKind != vmmMulticast {
		t.Fatalf(
			"decoded object kind = %d, want %d",
			decoded.ObjectKind, vmmMulticast,
		)
	}
	encoded[255] = 1
	if _, err := decodeVMMHeader(encoded); err == nil {
		t.Fatal("nonzero final reserved header byte was accepted")
	}
}

func TestValidateVMMLedgerExactHandleTypes(t *testing.T) {
	const (
		owner        = "11111111111111111111111111111111"
		importer     = "22222222222222222222222222222222"
		sourceUUID   = "00112233-4455-6677-8899-aabbccddeeff"
		resourceID   = uint64(1)
		resourceSize = uint64(4096)
	)
	mapping := func(participant string, handleType uint32) vmmMapping {
		return vmmMapping{
			Participant:         participant,
			Address:             0x1000,
			Size:                resourceSize,
			GPUUUID:             sourceUUID,
			RequestedHandleType: handleType,
			Access:              []byte{1},
			AccessCount:         1,
			AccessSize:          1,
		}
	}
	ledger := func(ownerType, importerType uint32) vmmLedger {
		return vmmLedger{
			Version: vmmLedgerVersion,
			Participants: []vmmParticipant{
				{
					ID: owner,
					Placement: vmmPlacement{
						Node: "node-a", GPUUUIDs: []string{sourceUUID},
					},
				},
				{
					ID: importer,
					Placement: vmmPlacement{
						Node: "node-a", GPUUUIDs: []string{sourceUUID},
					},
				},
			},
			Resources: []vmmResource{{
				ID:        resourceID,
				Kind:      "allocation",
				Owner:     mapping(owner, ownerType),
				Importers: []vmmMapping{mapping(importer, importerType)},
			}},
		}
	}
	for _, handleType := range []uint32{vmmHandlePOSIX, vmmHandleFabric} {
		if err := validateVMMLedger(ledger(handleType, handleType)); err != nil {
			t.Fatalf("exact handle type %d rejected: %v", handleType, err)
		}
	}
	if err := validateVMMLedger(ledger(9, 9)); err == nil {
		t.Fatal("combined POSIX|FABRIC handle type was accepted")
	}
	if err := validateVMMLedger(
		ledger(vmmHandlePOSIX, vmmHandleFabric),
	); err == nil {
		t.Fatal("mixed owner/importer handle types were accepted")
	}
	atLimit := ledger(vmmHandlePOSIX, vmmHandlePOSIX)
	atLimit.Resources[0].Owner.Size = vmmMaximumRedisBulk
	atLimit.Resources[0].Importers[0].Size = vmmMaximumRedisBulk
	if err := validateVMMLedger(atLimit); err != nil {
		t.Fatalf("allocation at the Redis bulk limit was rejected: %v", err)
	}
	for _, role := range []string{"owner", "importer"} {
		t.Run(role+" above allocation limit", func(t *testing.T) {
			aboveLimit := ledger(vmmHandlePOSIX, vmmHandlePOSIX)
			if role == "owner" {
				aboveLimit.Resources[0].Owner.Size =
					vmmMaximumRedisBulk + 1
			} else {
				aboveLimit.Resources[0].Importers[0].Size =
					vmmMaximumRedisBulk + 1
			}
			if err := validateVMMLedger(aboveLimit); err == nil {
				t.Fatal("mapping above the Redis bulk limit was accepted")
			}
		})
	}
}

func TestVMMBrokerResultShapesAndCleanup(t *testing.T) {
	t.Run("fabric", func(t *testing.T) {
		raw := bytes.Repeat([]byte{0xa5}, vmmFabricHandleSize)
		response := vmmResponse{
			header: vmmHeader{
				HandleType: vmmHandleFabric,
				ObjectKind: vmmAllocation,
			},
			payload: raw,
			fd:      -1,
		}
		result, err := takeVMMBrokerResult(
			&response, vmmHandleFabric, vmmAllocation,
		)
		if err != nil {
			t.Fatal(err)
		}
		if result.fd >= 0 || len(result.bytes) != vmmFabricHandleSize {
			t.Fatalf("FABRIC broker result = %#v", result)
		}
		if err := closeVMMBrokerResult(&result); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(raw, make([]byte, vmmFabricHandleSize)) {
			t.Fatal("FABRIC broker cleanup did not wipe raw bytes")
		}
	})
	t.Run("posix rejects payload and closes fd", func(t *testing.T) {
		fd, err := unix.MemfdCreate("vmm-broker-shape", unix.MFD_CLOEXEC)
		if err != nil {
			t.Fatal(err)
		}
		payload := []byte{1}
		response := vmmResponse{
			header: vmmHeader{
				HandleType: vmmHandlePOSIX,
				ObjectKind: vmmAllocation,
			},
			payload: payload,
			fd:      fd,
		}
		if _, err := takeVMMBrokerResult(
			&response, vmmHandlePOSIX, vmmAllocation,
		); err == nil {
			t.Fatal("POSIX broker response with payload was accepted")
		}
		if payload[0] != 0 || response.fd != -1 {
			t.Fatal("invalid POSIX broker result was not cleaned")
		}
		if _, err := unix.FcntlInt(uintptr(fd), unix.F_GETFD, 0); err == nil {
			t.Fatal("invalid POSIX broker FD remained open")
		}
	})
}

func TestVMMFabricImporterTransportIsTransient(t *testing.T) {
	metadata := []byte{1, 2, 3}
	raw := bytes.Repeat([]byte{0x7b}, vmmFabricHandleSize)
	payload, fd, err := vmmImporterTransport(metadata, vmmBrokerResult{
		handleType: vmmHandleFabric,
		fd:         -1,
		bytes:      raw,
	})
	if err != nil {
		t.Fatal(err)
	}
	if fd >= 0 || len(payload) != len(metadata)+vmmFabricHandleSize ||
		!bytes.Equal(payload[:len(metadata)], metadata) ||
		!bytes.Equal(payload[len(metadata):], raw) {
		t.Fatalf("FABRIC importer transport has invalid shape: fd=%d len=%d", fd, len(payload))
	}
	payload[0] = 0xff
	payload[len(metadata)] = 0
	if metadata[0] != 1 || raw[0] != 0x7b {
		t.Fatal("FABRIC importer transport aliases durable metadata or broker bytes")
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

func TestIdentifyVMMProcessRequiresLowercaseHexParticipant(t *testing.T) {
	tests := []string{
		"",
		"1111111111111111111111111111111",
		"111111111111111111111111111111111",
		"1111111111111111111111111111111A",
		"1111111111111111111111111111111g",
	}
	for _, participant := range tests {
		t.Run(fmt.Sprintf("%q", participant), func(t *testing.T) {
			endpoint := serveVMMProcess(t, "", participant, nil)
			_, err := identifyVMMProcess(VMMProcess{
				SocketPath: endpoint.SocketPath,
			})
			if err == nil {
				t.Fatalf("identify participant %q was accepted", participant)
			}
		})
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

func TestVMMDurableMulticastMetadataBindsDigestWithoutRuntimeIdentity(t *testing.T) {
	ledger := vmmLedger{
		Version:    vmmLedgerVersion,
		Generation: "00112233445566778899aabbccddeeff",
		Resources: []vmmResource{
			{
				ID:   1,
				Kind: "allocation",
			},
			{
				ID:   2,
				Kind: "multicast",
				Owner: vmmMapping{
					Participant:         "11111111111111111111111111111111",
					RequestedHandleType: vmmHandleFabric,
					Multicast: &vmmMulticastMapping{
						BackingResourceID: 1,
						BackingRole:       vmmOwner,
						Size:              4096,
						BindAPI:           vmmMulticastBindMem,
						backingUUID:       [16]byte{0xaa},
					},
					retainRestoreHandle: true,
				},
				Multicast: &vmmMulticastProperties{
					HandleTypes: vmmHandleFabric,
					NumDevices:  2,
					Size:        4096,
				},
				captureUUID: [16]byte{0xbb},
			},
		},
	}
	encoded, err := json.Marshal(ledger)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, want := range []string{
		`"kind":"multicast"`,
		`"backingResourceID":1`,
		`"bindAPI":1`,
		`"numDevices":2`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("durable multicast JSON %s does not contain %s", text, want)
		}
	}
	for _, forbidden := range []string{
		"captureUUID",
		"backingUUID",
		"retainRestoreHandle",
		"fabricToken",
		"rawHandle",
		"socket",
		"context",
		"pid",
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("durable multicast JSON contains runtime field %q: %s", forbidden, text)
		}
	}
	digest := func(metadata []byte) string {
		hash := sha256.New()
		writeVMMDigestPart(hash, metadata)
		writeVMMDigestObject(hash, 1, []byte("allocation bytes"))
		return hex.EncodeToString(hash.Sum(nil))
	}
	baseline := digest(encoded)
	ledger.Resources[1].Owner.Multicast.BindAPI = vmmMulticastBindMemV2
	changed, err := json.Marshal(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if baseline == digest(changed) {
		t.Fatal("digest did not bind durable multicast metadata")
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
		nil,
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
		otherUUID    = "ffeeddcc-bbaa-9988-7766-554433221100"
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
		Version:    vmmLedgerVersion,
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
		"content:1", "content:2",
		"placement:1:owner,2:importer",
		"placement:2:owner,1:importer",
	}

	tests := []struct {
		name                  string
		corruptSecond         bool
		rejectSecondPlacement bool
		unlockError           error
		failOwner             uint64
		wantError             string
		replayEvents          []string
		noUnlock              bool
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
			name:                  "second placement rejection does not unlock",
			rejectSecondPlacement: true,
			wantError:             "placement rejected",
			noUnlock:              true,
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
					t, participantA, sourceUUID, contents, 1, test.failOwner,
					false, events,
				),
				serveVMMRestoreProcess(
					t, participantB, sourceUUID, contents, 2, test.failOwner,
					test.rejectSecondPlacement, events,
				),
			}
			err := RestoreVMM(
				context.Background(),
				processes,
				generation,
				expectedDigest,
				t.TempDir(),
				[]types.VMMPlacement{
					{
						SourceGPUUUID: sourceUUID,
						TargetGPUUUID: sourceUUID,
						TargetOrdinal: 1,
					},
					{
						SourceGPUUUID: otherUUID,
						TargetGPUUUID: otherUUID,
						TargetOrdinal: 0,
					},
				},
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
			if test.corruptSecond {
				wantEvents = wantEvents[:2]
			}
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

func TestRestoreVMMMulticastReplayOrder(t *testing.T) {
	const (
		generation   = "00112233445566778899aabbccddeeff"
		participantA = "11111111111111111111111111111111"
		participantB = "22222222222222222222222222222222"
		gpuA         = "00112233-4455-6677-8899-aabbccddeeff"
		gpuB         = "ffeeddcc-bbaa-9988-7766-554433221100"
	)
	mapping := func(participant, gpu string, address uint64) vmmMapping {
		return vmmMapping{
			Participant:         participant,
			Address:             address,
			Size:                4,
			GPUUUID:             gpu,
			RequestedHandleType: vmmHandlePOSIX,
			Access:              []byte{1, 2, 3, 4},
			AccessCount:         1,
			AccessSize:          4,
		}
	}
	allocationOwner := mapping(participantA, gpuA, 0x1000)
	allocationImporter := mapping(participantB, gpuB, 0x2000)
	multicastOwner := mapping(participantA, gpuA, 0x3000)
	multicastOwner.Multicast = &vmmMulticastMapping{
		BackingResourceID: 1,
		BackingRole:       vmmOwner,
		Size:              4,
		BindAPI:           vmmMulticastBindMem,
	}
	multicastImporter := mapping(participantB, gpuB, 0x4000)
	multicastImporter.Multicast = &vmmMulticastMapping{
		BackingResourceID: 1,
		BackingRole:       vmmImporter,
		Size:              4,
		BindAPI:           vmmMulticastBindMemV2,
	}
	ledger := vmmLedger{
		Version:    vmmLedgerVersion,
		Generation: generation,
		Participants: []vmmParticipant{
			{
				ID: participantA,
				Placement: vmmPlacement{
					Node: "node-a", GPUUUIDs: []string{gpuA},
				},
			},
			{
				ID: participantB,
				Placement: vmmPlacement{
					Node: "node-a", GPUUUIDs: []string{gpuB},
				},
			},
		},
		Resources: []vmmResource{
			{
				ID:        1,
				Kind:      "allocation",
				Owner:     allocationOwner,
				Importers: []vmmMapping{allocationImporter},
			},
			{
				ID:        2,
				Kind:      "multicast",
				Owner:     multicastOwner,
				Importers: []vmmMapping{multicastImporter},
				Multicast: &vmmMulticastProperties{
					HandleTypes: vmmHandlePOSIX,
					NumDevices:  2,
					Size:        4,
				},
			},
		},
	}
	if err := validateVMMLedger(ledger); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(ledger)
	if err != nil {
		t.Fatal(err)
	}
	contents := []byte{9, 8, 7, 6}
	digest := sha256.New()
	writeVMMDigestPart(digest, encoded)
	writeVMMDigestObject(digest, 1, contents)
	expectedDigest := hex.EncodeToString(digest.Sum(nil))
	events := make(chan string, 32)
	t.Setenv(VMMRedisAddressEnv, serveVMMRestoreRedis(t, map[string][]byte{
		vmmRedisKey(generation, "state"):      []byte("detached"),
		vmmRedisKey(generation, "ledger"):     encoded,
		vmmRedisKey(generation, "digest"):     []byte(expectedDigest),
		vmmRedisKey(generation, "resource:1"): contents,
	}, events))
	t.Setenv(VMMRedisRDBPathEnv, "")
	t.Setenv(VMMRedisRestoreCmdEnv, "/bin/true")
	t.Setenv("NODE_NAME", "node-a")

	serve := func(participant, gpu string, ordinal int32) VMMProcess {
		return serveVMMProcess(
			t, "", participant,
			func(request vmmResponse) (vmmResponse, error) {
				response := vmmResponse{
					header: vmmHeader{Operation: request.header.Operation},
					fd:     -1,
				}
				switch request.header.Operation {
				case vmmSetPlacement:
					if err := validateVMMPlacementRequest(
						request, gpu, gpu, ordinal,
					); err != nil {
						return response, err
					}
					events <- "placement:" + participant
				case vmmRestoreOwner:
					if participant != participantA ||
						(request.header.ObjectID != 1 &&
							request.header.ObjectID != 2) {
						return response, errors.New("unexpected restore owner")
					}
					if request.header.ObjectKind == vmmAllocation {
						if request.header.ObjectID != 1 ||
							request.header.HandleType != vmmHandlePOSIX ||
							binary.LittleEndian.Uint32(request.payload[60:64])&
								vmmRetainRestoreHandle == 0 {
							return response, errors.New("invalid retained allocation owner")
						}
						events <- "allocation-owner"
					} else if request.header.ObjectKind == vmmMulticast {
						if err := validateMulticastRestoreRequest(
							request, 2, vmmOwner, 1,
						); err != nil {
							return response, err
						}
						events <- "multicast-owner-add"
					} else {
						return response, errors.New("unexpected owner object kind")
					}
					response.header.HandleType = vmmHandlePOSIX
					response.header.ObjectKind = request.header.ObjectKind
					var err error
					response.fd, err = unix.MemfdCreate(
						"vmm-multicast-restore", unix.MFD_CLOEXEC,
					)
					if err != nil {
						return response, err
					}
				case vmmRestoreImporter:
					if participant != participantB || request.fd < 0 {
						return response, errors.New("unexpected restore importer")
					}
					if request.header.ObjectKind == vmmAllocation {
						if request.header.ObjectID != 1 ||
							binary.LittleEndian.Uint32(request.payload[60:64])&
								vmmRetainRestoreHandle == 0 {
							return response, errors.New("invalid retained allocation importer")
						}
						events <- "allocation-importer"
					} else if request.header.ObjectKind == vmmMulticast {
						if err := validateMulticastRestoreRequest(
							request, 2, vmmImporter, 1,
						); err != nil {
							return response, err
						}
						events <- "multicast-importer-add"
					} else {
						return response, errors.New("unexpected importer object kind")
					}
				case vmmRestoreMulticast:
					if err := validateMulticastRestoreRequest(
						request, 2,
						map[bool]uint32{true: vmmOwner, false: vmmImporter}[participant == participantA],
						1,
					); err != nil {
						return response, err
					}
					events <- "multicast-bind:" + participant
				case vmmFinalizeRestore:
					events <- "finalize:" + participant
				case vmmIdentify:
					events <- "health:" + participant
				default:
					return response, fmt.Errorf(
						"unexpected VMM operation %d", request.header.Operation,
					)
				}
				return response, nil
			},
		)
	}
	processes := []VMMProcess{
		serve(participantA, gpuA, 0),
		serve(participantB, gpuB, 1),
	}
	if err := RestoreVMM(
		context.Background(),
		processes,
		generation,
		expectedDigest,
		t.TempDir(),
		[]types.VMMPlacement{
			{SourceGPUUUID: gpuA, TargetGPUUUID: gpuA, TargetOrdinal: 0},
			{SourceGPUUUID: gpuB, TargetGPUUUID: gpuB, TargetOrdinal: 1},
		},
		func() error {
			events <- "unlock"
			return nil
		},
		logr.Discard(),
	); err != nil {
		t.Fatal(err)
	}
	var got []string
	for range len(events) {
		got = append(got, <-events)
	}
	want := []string{
		"content:1",
		"placement:" + participantA,
		"placement:" + participantB,
		"unlock",
		"allocation-owner",
		"allocation-importer",
		"multicast-owner-add",
		"multicast-importer-add",
		"multicast-bind:" + participantA,
		"multicast-bind:" + participantB,
		"finalize:" + participantA,
		"finalize:" + participantB,
		"health:" + participantA,
		"health:" + participantB,
		"redis-restored",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("multicast restore events = %v, want %v", got, want)
	}
}

func TestRestoreVMMMulticastFailureAbortsParticipants(t *testing.T) {
	const (
		generation   = "00112233445566778899aabbccddeeff"
		participantA = "11111111111111111111111111111111"
		participantB = "22222222222222222222222222222222"
		gpuA         = "00112233-4455-6677-8899-aabbccddeeff"
		gpuB         = "ffeeddcc-bbaa-9988-7766-554433221100"
	)
	mapping := func(participant, gpu string, address uint64) vmmMapping {
		return vmmMapping{
			Participant:         participant,
			Address:             address,
			Size:                4,
			GPUUUID:             gpu,
			RequestedHandleType: vmmHandlePOSIX,
			Access:              []byte{1, 2, 3, 4},
			AccessCount:         1,
			AccessSize:          4,
		}
	}
	allocationOwner := mapping(participantA, gpuA, 0x1000)
	allocationImporter := mapping(participantB, gpuB, 0x2000)
	multicastOwner := mapping(participantA, gpuA, 0x3000)
	multicastOwner.Multicast = &vmmMulticastMapping{
		BackingResourceID: 1,
		BackingRole:       vmmOwner,
		Size:              4,
		BindAPI:           vmmMulticastBindMem,
	}
	multicastImporter := mapping(participantB, gpuB, 0x4000)
	multicastImporter.Multicast = &vmmMulticastMapping{
		BackingResourceID: 1,
		BackingRole:       vmmImporter,
		Size:              4,
		BindAPI:           vmmMulticastBindMemV2,
	}
	ledger := vmmLedger{
		Version:    vmmLedgerVersion,
		Generation: generation,
		Participants: []vmmParticipant{
			{
				ID: participantA,
				Placement: vmmPlacement{
					Node: "node-a", GPUUUIDs: []string{gpuA},
				},
			},
			{
				ID: participantB,
				Placement: vmmPlacement{
					Node: "node-a", GPUUUIDs: []string{gpuB},
				},
			},
		},
		Resources: []vmmResource{
			{
				ID:        1,
				Kind:      "allocation",
				Owner:     allocationOwner,
				Importers: []vmmMapping{allocationImporter},
			},
			{
				ID:        2,
				Kind:      "multicast",
				Owner:     multicastOwner,
				Importers: []vmmMapping{multicastImporter},
				Multicast: &vmmMulticastProperties{
					HandleTypes: vmmHandlePOSIX,
					NumDevices:  2,
					Size:        4,
				},
			},
		},
	}
	encoded, err := json.Marshal(ledger)
	if err != nil {
		t.Fatal(err)
	}
	contents := []byte{9, 8, 7, 6}
	digest := sha256.New()
	writeVMMDigestPart(digest, encoded)
	writeVMMDigestObject(digest, 1, contents)
	expectedDigest := hex.EncodeToString(digest.Sum(nil))
	events := make(chan string, 32)
	t.Setenv(VMMRedisAddressEnv, serveVMMRestoreRedis(t, map[string][]byte{
		vmmRedisKey(generation, "state"):      []byte("detached"),
		vmmRedisKey(generation, "ledger"):     encoded,
		vmmRedisKey(generation, "digest"):     []byte(expectedDigest),
		vmmRedisKey(generation, "resource:1"): contents,
	}, events))
	t.Setenv(VMMRedisRDBPathEnv, "")
	t.Setenv(VMMRedisRestoreCmdEnv, "/bin/true")
	t.Setenv("NODE_NAME", "node-a")

	serve := func(participant, gpu string, ordinal int32) VMMProcess {
		return serveVMMProcess(
			t, "", participant,
			func(request vmmResponse) (vmmResponse, error) {
				response := vmmResponse{
					header: vmmHeader{Operation: request.header.Operation},
					fd:     -1,
				}
				switch request.header.Operation {
				case vmmSetPlacement:
					if err := validateVMMPlacementRequest(
						request, gpu, gpu, ordinal,
					); err != nil {
						return response, err
					}
				case vmmRestoreOwner:
					if participant != participantA {
						return response, errors.New("unexpected restore owner")
					}
					response.header.HandleType = vmmHandlePOSIX
					response.header.ObjectKind = request.header.ObjectKind
					var err error
					response.fd, err = unix.MemfdCreate(
						"vmm-multicast-abort", unix.MFD_CLOEXEC,
					)
					return response, err
				case vmmRestoreImporter:
					if participant != participantB || request.fd < 0 {
						return response, errors.New("unexpected restore importer")
					}
				case vmmRestoreMulticast:
					events <- "bind:" + participant
					if participant == participantB {
						response.header.Status = 1
						response.header.Message = "participant B bind failed"
					}
				case vmmAbortRestore:
					events <- "abort:" + participant
					if participant == participantB {
						response.header.Status = 1
						response.header.Message = "participant B abort failed"
					}
				case vmmFinalizeRestore:
					events <- "finalize:" + participant
				case vmmIdentify:
					events <- "health:" + participant
				default:
					return response, fmt.Errorf(
						"unexpected VMM operation %d", request.header.Operation,
					)
				}
				return response, nil
			},
		)
	}
	processes := []VMMProcess{
		serve(participantA, gpuA, 0),
		serve(participantB, gpuB, 1),
	}
	err = RestoreVMM(
		context.Background(),
		processes,
		generation,
		expectedDigest,
		t.TempDir(),
		[]types.VMMPlacement{
			{SourceGPUUUID: gpuA, TargetGPUUUID: gpuA, TargetOrdinal: 0},
			{SourceGPUUUID: gpuB, TargetGPUUUID: gpuB, TargetOrdinal: 1},
		},
		func() error {
			events <- "unlock"
			return nil
		},
		logr.Discard(),
	)
	if err == nil ||
		!strings.Contains(err.Error(), "participant B bind failed") ||
		!strings.Contains(err.Error(), "participant B abort failed") {
		t.Fatalf("RestoreVMM error = %v, want bind and abort failures", err)
	}
	var got []string
	for range len(events) {
		got = append(got, <-events)
	}
	want := []string{
		"content:1",
		"unlock",
		"bind:" + participantA,
		"bind:" + participantB,
		"abort:" + participantA,
		"abort:" + participantB,
	}
	if !slices.Equal(got, want) {
		t.Fatalf("multicast abort events = %v, want %v", got, want)
	}
}

func validateMulticastRestoreRequest(
	request vmmResponse,
	resourceID uint64,
	role uint32,
	backingResourceID uint64,
) error {
	if request.header.ObjectID != resourceID ||
		request.header.ObjectKind != vmmMulticast ||
		request.header.HandleType != vmmHandlePOSIX ||
		len(request.payload) < vmmRecordSize+vmmMulticastSize {
		return errors.New("invalid multicast restore header")
	}
	if binary.LittleEndian.Uint64(request.payload[16:24]) != resourceID ||
		binary.LittleEndian.Uint32(request.payload[48:52]) != role ||
		binary.LittleEndian.Uint32(request.payload[52:56]) != vmmMulticast {
		return errors.New("invalid multicast restore record")
	}
	extension := vmmRecordSize +
		int(binary.LittleEndian.Uint32(request.payload[68:72])) +
		int(binary.LittleEndian.Uint32(request.payload[72:76]))*
			int(binary.LittleEndian.Uint32(request.payload[76:80]))
	if extension+vmmMulticastSize != len(request.payload) ||
		binary.LittleEndian.Uint64(
			request.payload[extension+16:extension+24],
		) != backingResourceID {
		return errors.New("invalid multicast restore bind")
	}
	return nil
}

func TestVMMDetachPlanOrdersImportersBeforeOwners(t *testing.T) {
	plan := vmmDetachPlan(vmmLedger{Resources: []vmmResource{
		{
			ID:        1,
			Kind:      "allocation",
			Owner:     vmmMapping{Participant: "owner-a"},
			Importers: []vmmMapping{{Participant: "importer-a"}, {Participant: "importer-b"}},
		},
		{
			ID:        2,
			Kind:      "multicast",
			Owner:     vmmMapping{Participant: "owner-b"},
			Importers: []vmmMapping{{Participant: "importer-c"}},
		},
	}})
	want := []vmmDetach{
		{resourceID: 2, participant: "importer-c", role: vmmImporter},
		{resourceID: 2, participant: "owner-b", role: vmmOwner},
		{resourceID: 1, participant: "importer-a", role: vmmImporter},
		{resourceID: 1, participant: "importer-b", role: vmmImporter},
		{resourceID: 1, participant: "owner-a", role: vmmOwner},
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

func TestPrepareVMMRejectsOversizedAllocationBeforeOwnerRead(t *testing.T) {
	const (
		owner      = "11111111111111111111111111111111"
		importer   = "22222222222222222222222222222222"
		sourceUUID = "00112233-4455-6677-8899-aabbccddeeff"
	)
	allocationUUID := [16]byte{1}
	operations := make(chan uint16, 3)
	process := func(participant string, record vmmCaptureRecord) VMMProcess {
		return serveVMMProcess(
			t, "", participant,
			func(request vmmResponse) (vmmResponse, error) {
				operations <- request.header.Operation
				if request.header.Operation != vmmInspect {
					return vmmResponse{}, fmt.Errorf(
						"unexpected VMM operation %d",
						request.header.Operation,
					)
				}
				payload := encodeVMMRecordsForTest(
					[]vmmCaptureRecord{record},
				)
				return vmmResponse{
					header: vmmHeader{
						Operation: vmmInspect,
						Count:     1,
					},
					payload: payload,
					fd:      -1,
				}, nil
			},
		)
	}
	mapping := func(role uint32, address uint64) vmmCaptureRecord {
		record := vmmCaptureRecord{
			allocationUUID: allocationUUID,
			address:        address,
			size:           vmmMaximumRedisBulk + 1,
			role:           role,
			gpuUUID:        sourceUUID,
			access:         []byte{1},
			accessCount:    1,
			accessSize:     1,
		}
		if role == vmmOwner {
			record.properties = []byte{1}
		}
		return record
	}
	t.Setenv(VMMRedisAddressEnv, "127.0.0.1:1")
	t.Setenv(VMMRedisRDBPathEnv, filepath.Join(t.TempDir(), "dump.rdb"))
	t.Setenv("NODE_NAME", "node-a")
	_, err := PrepareVMM(
		context.Background(),
		[]VMMProcess{
			process(owner, mapping(vmmOwner, 0x1000)),
			process(importer, mapping(vmmImporter, 0x2000)),
		},
		"generation",
		t.TempDir(),
		logr.Discard(),
	)
	if err == nil || !strings.Contains(err.Error(), "allocation limit") {
		t.Fatalf("oversized checkpoint error = %v, want allocation limit", err)
	}
	for index := 0; index < 2; index++ {
		if operation := <-operations; operation != vmmInspect {
			t.Fatalf(
				"checkpoint operation %d = %d, want INSPECT",
				index, operation,
			)
		}
	}
	select {
	case operation := <-operations:
		t.Fatalf(
			"oversized checkpoint reached operation %d after inspection",
			operation,
		)
	default:
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
	owner := serveVMMInspectAs(t, ownerID, []vmmCaptureRecord{{
		allocationUUID: [16]byte{9, 8},
		address:        0x2000,
		size:           4096,
		role:           vmmOwner,
		properties:     []byte{1, 2},
		access:         []byte{3, 4},
		accessCount:    1,
		accessSize:     2,
	}})
	importer := serveVMMInspectAs(t, importerID, []vmmCaptureRecord{{
		allocationUUID: [16]byte{9, 8},
		address:        0x4000,
		size:           4096,
		role:           vmmImporter,
		access:         []byte{6, 7},
		accessCount:    1,
		accessSize:     2,
	}})
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

func TestInspectVMMRejectsMulticastPropertyDisagreement(t *testing.T) {
	const (
		owner       = "11111111111111111111111111111111"
		importer    = "22222222222222222222222222222222"
		ownerGPU    = "00112233-4455-6677-8899-aabbccddeeff"
		importerGPU = "ffeeddcc-bbaa-9988-7766-554433221100"
	)
	allocationUUID := [16]byte{1}
	multicastUUID := [16]byte{2}
	record := func(
		role uint32,
		kind uint32,
		address uint64,
		gpu string,
	) vmmCaptureRecord {
		result := vmmCaptureRecord{
			allocationUUID: allocationUUID,
			address:        address,
			size:           4096,
			role:           role,
			kind:           kind,
			handleType:     vmmHandleFabric,
			gpuUUID:        gpu,
			access:         []byte{1},
			accessCount:    1,
			accessSize:     1,
		}
		if role == vmmOwner {
			result.properties = []byte{1}
		}
		if kind == vmmMulticast {
			result.allocationUUID = multicastUUID
			result.multicast = &vmmCaptureMulticast{
				backingUUID:       allocationUUID,
				bindSize:          4096,
				objectHandleTypes: vmmHandleFabric,
				objectSize:        4096,
				numDevices:        2,
				backingRole:       role,
				bindAPI:           vmmMulticastBindMem,
			}
		}
		return result
	}
	ownerRecords := []vmmCaptureRecord{
		record(vmmOwner, vmmAllocation, 0x1000, ownerGPU),
		record(vmmOwner, vmmMulticast, 0x3000, ownerGPU),
	}
	importerRecords := []vmmCaptureRecord{
		record(vmmImporter, vmmAllocation, 0x2000, importerGPU),
		record(vmmImporter, vmmMulticast, 0x4000, importerGPU),
	}
	importerRecords[1].multicast.numDevices = 3
	_, err := inspectVMM(
		context.Background(),
		[]VMMProcess{
			serveVMMInspectAs(t, owner, ownerRecords),
			serveVMMInspectAs(t, importer, importerRecords),
		},
		"generation",
		"node-a",
	)
	if err == nil || !strings.Contains(err.Error(), "inconsistent object properties") {
		t.Fatalf("multicast property disagreement error = %v", err)
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
			process := serveVMMInspectAs(
				t,
				specification.participant,
				[]vmmCaptureRecord{specification.record},
			)
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
		return serveVMMInspectAs(t, participant, []vmmCaptureRecord{record})
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

func TestValidateRestoredVMMProcessesUsesAuthoritativePlacement(t *testing.T) {
	const participant = "11111111111111111111111111111111"
	const sourceUUID = "00112233-4455-6677-8899-aabbccddeeff"
	const targetUUID = "ffeeddcc-bbaa-9988-7766-554433221100"
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
		placement   []types.VMMPlacement
		wantError   string
	}{
		{
			name:        "authoritative remap",
			currentNode: "node-a",
			placement: []types.VMMPlacement{{
				SourceGPUUUID: sourceUUID,
				TargetGPUUUID: targetUUID,
				TargetOrdinal: 0,
			}},
		},
		{
			name:        "missing source UUID",
			currentNode: "node-a",
			placement: []types.VMMPlacement{{
				SourceGPUUUID: targetUUID,
				TargetGPUUUID: sourceUUID,
				TargetOrdinal: 0,
			}},
			wantError: "absent from the authoritative placement",
		},
		{
			name:        "different current node",
			currentNode: "node-b",
			placement: []types.VMMPlacement{{
				SourceGPUUUID: sourceUUID,
				TargetGPUUUID: targetUUID,
				TargetOrdinal: 0,
			}},
			wantError: "target node",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			setups, placement, err := validateRestoredVMMProcesses(
				[]VMMProcess{{Participant: participant}},
				ledger,
				test.currentNode,
				test.placement,
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
			if len(setups) != 1 {
				t.Fatalf("placement setup count = %d, want 1", len(setups))
			}
			if err := validateVMMPlacementPayload(
				setups[0].count,
				setups[0].payload,
				sourceUUID,
				targetUUID,
				0,
			); err != nil {
				t.Fatal(err)
			}
			if got := placement[participant][sourceUUID]; got != 0 {
				t.Fatalf("validated ordinal = %d, want 0", got)
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
	valid := VMMProcess{Participant: participant}
	empty := VMMProcess{Participant: extra}
	placementPlan := []types.VMMPlacement{{
		SourceGPUUUID: sourceUUID,
		TargetGPUUUID: sourceUUID,
		TargetOrdinal: 0,
	}}
	tests := []struct {
		name          string
		processes     []VMMProcess
		wantError     string
		wantPlacement bool
	}{
		{
			name:          "extra empty placement is ignored",
			processes:     []VMMProcess{valid, empty},
			wantPlacement: true,
		},
		{
			name:      "missing expected endpoint is rejected",
			processes: []VMMProcess{empty},
			wantError: "has no restored shim endpoint",
		},
		{
			name: "duplicate participant is rejected",
			processes: []VMMProcess{
				{Participant: participant},
				{Participant: participant},
			},
			wantError: "multiple restored shims claim participant",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			setups, placement, err := validateRestoredVMMProcesses(
				test.processes,
				ledger,
				"node-a",
				placementPlan,
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
			if len(setups) != 2 || setups[1].count != 0 ||
				len(setups[1].payload) != 0 {
				t.Fatalf("extra endpoint setup = %#v, want empty payload", setups)
			}
			if _, ok := placement[extra]; ok {
				t.Fatal("extra empty participant was included in execution placement")
			}
			if got := placement[participant][sourceUUID]; got != 0 {
				t.Fatalf("validated ordinal = %d, want 0", got)
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
		vmmResponseUntyped,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(response.payload, want) {
		t.Fatalf("payload = %q, want %q", response.payload, want)
	}
}

func TestExchangeVMMBindsEstablishedParticipant(t *testing.T) {
	const participant = "11111111111111111111111111111111"
	process := serveVMMProcess(
		t,
		"",
		participant,
		func(request vmmResponse) (vmmResponse, error) {
			if request.header.Participant != participant {
				return vmmResponse{}, fmt.Errorf(
					"request participant = %q, want %q",
					request.header.Participant,
					participant,
				)
			}
			return vmmResponse{
				header: vmmHeader{Operation: request.header.Operation},
				fd:     -1,
			}, nil
		},
	)
	if _, err := exchangeVMM(
		context.Background(),
		process,
		vmmHeader{Operation: vmmInspect},
		nil,
		-1,
		vmmResponseUntyped,
	); err != nil {
		t.Fatal(err)
	}
}

func TestExchangeVMMRejectsWrongResponseParticipant(t *testing.T) {
	const (
		participant = "11111111111111111111111111111111"
		wrong       = "22222222222222222222222222222222"
	)
	process := serveVMMProcess(
		t,
		"",
		participant,
		func(request vmmResponse) (vmmResponse, error) {
			return vmmResponse{
				header: vmmHeader{
					Operation:   request.header.Operation,
					Participant: wrong,
				},
				fd: -1,
			}, nil
		},
	)
	if _, err := exchangeVMM(
		context.Background(),
		process,
		vmmHeader{Operation: vmmInspect},
		nil,
		-1,
		vmmResponseUntyped,
	); err == nil || !strings.Contains(err.Error(), "response participant") {
		t.Fatalf("wrong response participant error = %v", err)
	}
}

func TestExchangeVMMRejectsTypedBrokerShapeBeforeReadingPayload(t *testing.T) {
	const participant = "11111111111111111111111111111111"
	tests := []struct {
		name        string
		expectation vmmResponseExpectation
		handleType  uint32
		payloadSize uint64
	}{
		{
			name:        "oversized fabric declaration",
			expectation: vmmResponseFabricBroker,
			handleType:  vmmHandleFabric,
			payloadSize: vmmFabricHandleSize + 1,
		},
		{
			name:        "short fabric declaration",
			expectation: vmmResponseFabricBroker,
			handleType:  vmmHandleFabric,
			payloadSize: vmmFabricHandleSize - 1,
		},
		{
			name:        "POSIX declaration with payload",
			expectation: vmmResponsePOSIXBroker,
			handleType:  vmmHandlePOSIX,
			payloadSize: 1,
		},
		{
			name:        "wrong typed handle",
			expectation: vmmResponseFabricBroker,
			handleType:  vmmHandlePOSIX,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "vmm.sock")
			listener, err := net.Listen("unix", path)
			if err != nil {
				t.Fatal(err)
			}
			defer listener.Close()
			done := make(chan struct{})
			go func() {
				defer close(done)
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
					Operation:   vmmRestoreOwner,
					HandleType:  test.handleType,
					PayloadSize: test.payloadSize,
					Participant: participant,
				}))
			}()

			_, err = exchangeVMM(
				context.Background(),
				VMMProcess{
					SocketPath:  path,
					Participant: participant,
				},
				vmmHeader{Operation: vmmRestoreOwner},
				nil,
				-1,
				test.expectation,
			)
			<-done
			if err == nil {
				t.Fatal("invalid typed broker response was accepted")
			}
		})
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
		vmmResponseUntyped,
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
	const participant = "11111111111111111111111111111111"
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
			Operation:   vmmRestoreOwner,
			HandleType:  vmmHandlePOSIX,
			Participant: participant,
		}))
	}()

	if _, err := exchangeVMM(
		context.Background(),
		VMMProcess{SocketPath: path, Participant: participant},
		vmmHeader{Operation: vmmRestoreOwner},
		nil,
		-1,
		vmmResponsePOSIXBroker,
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

func validateVMMPlacementPayload(
	count uint32,
	payload []byte,
	sourceUUID string,
	targetUUID string,
	ordinal int32,
) error {
	if count != 1 || len(payload) != vmmPlacementSize {
		return fmt.Errorf(
			"placement shape = count %d, payload %d bytes",
			count,
			len(payload),
		)
	}
	if binary.LittleEndian.Uint32(payload[0:4]) != uint32(ordinal) ||
		binary.LittleEndian.Uint32(payload[4:8]) != 0 ||
		!bytes.Equal(payload[8:24], parseGPUUUID(sourceUUID)) ||
		!bytes.Equal(payload[24:40], parseGPUUUID(targetUUID)) {
		return errors.New("placement payload does not match authoritative plan")
	}
	return nil
}

func validateVMMPlacementRequest(
	request vmmResponse,
	sourceUUID string,
	targetUUID string,
	ordinal int32,
) error {
	if request.header.Operation != vmmSetPlacement || request.fd >= 0 {
		return fmt.Errorf(
			"unexpected placement operation %d or FD %d",
			request.header.Operation,
			request.fd,
		)
	}
	return validateVMMPlacementPayload(
		request.header.Count,
		request.payload,
		sourceUUID,
		targetUUID,
		ordinal,
	)
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
	return serveVMMInspectAs(
		t,
		"11111111111111111111111111111111",
		records,
	)
}

func serveVMMInspectAs(
	t *testing.T,
	participant string,
	records []vmmCaptureRecord,
) VMMProcess {
	t.Helper()
	return serveVMMProcess(
		t,
		"",
		participant,
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
		extensionSize := 0
		if record.kind == vmmMulticast {
			extensionSize = vmmMulticastSize
		}
		encoded := make(
			[]byte,
			vmmRecordSize+len(record.properties)+len(record.access)+extensionSize,
		)
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
		if record.multicast != nil {
			offset := vmmRecordSize + len(record.properties) + len(record.access)
			extension := encoded[offset : offset+vmmMulticastSize]
			copy(extension[0:16], record.multicast.backingUUID[:])
			binary.LittleEndian.PutUint64(
				extension[16:24], record.multicast.backingObjectID,
			)
			binary.LittleEndian.PutUint64(
				extension[24:32], record.multicast.multicastOffset,
			)
			binary.LittleEndian.PutUint64(
				extension[32:40], record.multicast.memoryOffset,
			)
			binary.LittleEndian.PutUint64(
				extension[40:48], record.multicast.bindSize,
			)
			binary.LittleEndian.PutUint64(
				extension[48:56], record.multicast.bindFlags,
			)
			binary.LittleEndian.PutUint64(
				extension[56:64], record.multicast.objectFlags,
			)
			binary.LittleEndian.PutUint64(
				extension[64:72], record.multicast.objectHandleTypes,
			)
			binary.LittleEndian.PutUint64(
				extension[72:80], record.multicast.objectSize,
			)
			binary.LittleEndian.PutUint32(
				extension[80:84], record.multicast.numDevices,
			)
			binary.LittleEndian.PutUint32(
				extension[84:88], record.multicast.backingRole,
			)
			binary.LittleEndian.PutUint32(
				extension[88:92], record.multicast.bindAPI,
			)
		}
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
	rejectPlacement bool,
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
		case vmmSetPlacement:
			if err := validateVMMPlacementRequest(
				request, sourceUUID, sourceUUID, 1,
			); err != nil {
				return response, err
			}
			events <- "placement:" + roleSummary
			if rejectPlacement {
				response.header.Status = 1
				response.header.Message = "placement rejected"
			}
			return response, nil
		case vmmRestoreOwner:
			if err := validateVMMRestoreRequest(
				request, ownerResource, vmmOwner, 1,
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
			response.header.HandleType = request.header.HandleType
			response.header.ObjectKind = vmmAllocation
			response.fd, err = unix.MemfdCreate("vmm-restore-owner", unix.MFD_CLOEXEC)
			return response, err
		case vmmRestoreImporter:
			importerResource := uint64(3) - ownerResource
			if err := validateVMMRestoreRequest(
				request, importerResource, vmmImporter, 1, nil,
			); err != nil {
				return response, err
			}
			events <- fmt.Sprintf("importer:%d", importerResource)
			return response, nil
		case vmmFinalizeRestore:
			events <- "finalize:" + roleSummary
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
	wantOrdinal int32,
	wantContents []byte,
) error {
	if request.header.ObjectID != resourceID || len(request.payload) < vmmRecordSize {
		return fmt.Errorf("invalid resource %d replay header", resourceID)
	}
	if binary.LittleEndian.Uint64(request.payload[16:24]) != resourceID ||
		binary.LittleEndian.Uint32(request.payload[48:52]) != wantRole ||
		int32(binary.LittleEndian.Uint32(request.payload[64:68])) !=
			wantOrdinal {
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
	wantFD := wantRole == vmmImporter &&
		request.header.HandleType == vmmHandlePOSIX
	if (request.fd >= 0) != wantFD {
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
				if request.header.Participant != "" &&
					request.header.Participant != participant {
					err = fmt.Errorf(
						"request participant %q does not match endpoint %q",
						request.header.Participant,
						participant,
					)
				} else if request.header.Operation != vmmIdentify &&
					request.header.Participant == "" {
					err = errors.New("established request omitted participant")
				}
				response := vmmResponse{
					header: vmmHeader{Operation: vmmIdentify, Participant: participant},
					fd:     -1,
				}
				if err == nil && handler != nil {
					response, err = handler(request)
				}
				if err == nil {
					if response.header.Participant == "" {
						response.header.Participant = participant
					}
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
