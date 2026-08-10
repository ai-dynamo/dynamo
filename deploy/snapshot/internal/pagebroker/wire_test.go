package pagebroker

import (
	"bytes"
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func wireCommand(t *testing.T, name string, args ...string) *exec.Cmd {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	t.Cleanup(cancel)
	return exec.CommandContext(ctx, name, args...)
}

func TestCppWireCompatibility(t *testing.T) {
	root := filepath.Join("..", "..", "pagebroker")
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		if out, err := exec.CommandContext(ctx, "make", "-C", root, "clean").CombinedOutput(); err != nil {
			t.Errorf("clean wire fixture: %v\n%s", err, out)
		}
	})
	if out, err := wireCommand(t, "make", "-C", root, "wire-fixture").CombinedOutput(); err != nil {
		t.Fatal(string(out))
	}
	out, err := wireCommand(t, filepath.Join(root, "wire-fixture")).Output()
	if err != nil {
		t.Fatal(err)
	}
	var got Response
	if err := proto.Unmarshal(out, &got); err != nil || !got.GetOk() || got.GetTransactionId() != "txn" || got.GetDirectoryPath() != "/checkpoint" {
		t.Fatalf("C++ response: %#v %v", &got, err)
	}
	sourcePath := "/checkpoint"
	mode := RestoreRequest_MODE_STAGED
	request, err := proto.Marshal(&Request{Command: &Request_Restore{Restore: &RestoreRequest{SourcePath: &sourcePath, Mode: &mode}}})
	if err != nil {
		t.Fatal(err)
	}
	cmd := wireCommand(t, filepath.Join(root, "wire-fixture"), "decode")
	cmd.Stdin = bytes.NewReader(request)
	out, err = cmd.Output()
	if err != nil || strings.TrimSpace(string(out)) != "/checkpoint MODE_STAGED" {
		t.Fatalf("C++ decode: %s %v", out, err)
	}
}

func TestWireFieldNumbers(t *testing.T) {
	fields := []struct {
		message protoreflect.Name
		field   protoreflect.Name
		number  protoreflect.FieldNumber
	}{
		{"RestoreRequest", "source_path", 1},
		{"RestoreRequest", "mode", 2},
		{"PrepareCheckpointRequest", "destination_path", 1},
		{"CommitRequest", "transaction_id", 1},
		{"AbortRequest", "transaction_id", 1},
		{"Request", "restore", 1},
		{"Request", "prepare_checkpoint", 2},
		{"Request", "commit", 3},
		{"Request", "abort", 4},
		{"Response", "ok", 1},
		{"Response", "transaction_id", 2},
		{"Response", "directory_path", 3},
		{"Response", "error", 4},
	}
	for _, want := range fields {
		message := File_pagebroker_proto.Messages().ByName(want.message)
		if message == nil {
			t.Errorf("missing message %s", want.message)
			continue
		}
		field := message.Fields().ByName(want.field)
		if field == nil {
			t.Errorf("missing field %s.%s", want.message, want.field)
			continue
		}
		if field.Number() != want.number {
			t.Errorf("%s.%s = %d, want %d", want.message, want.field, field.Number(), want.number)
		}
	}
	assertModeNumbers(t, File_pagebroker_proto.Messages().ByName("RestoreRequest"))
}

func assertModeNumbers(t *testing.T, restore protoreflect.MessageDescriptor) {
	t.Helper()
	if restore == nil {
		t.Error("missing message RestoreRequest")
		return
	}
	mode := restore.Enums().ByName("Mode")
	if mode == nil {
		t.Error("missing enum RestoreRequest.Mode")
		return
	}
	for _, want := range []struct {
		name   protoreflect.Name
		number protoreflect.EnumNumber
	}{
		{"MODE_UNSPECIFIED", 0},
		{"MODE_DIRECT", 1},
		{"MODE_STAGED", 2},
	} {
		value := mode.Values().ByName(want.name)
		if value == nil {
			t.Errorf("missing enum value RestoreRequest.Mode.%s", want.name)
			continue
		}
		if value.Number() != want.number {
			t.Errorf("RestoreRequest.Mode.%s = %d, want %d", want.name, value.Number(), want.number)
		}
	}
}
