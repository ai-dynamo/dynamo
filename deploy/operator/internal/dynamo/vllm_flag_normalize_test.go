/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func vllmContainer(args ...string) *corev1.Container {
	return &corev1.Container{
		Name:    "main",
		Command: []string{"python3"},
		Args:    append([]string{"-m", "dynamo.vllm", "--model", "m"}, args...),
	}
}

func TestNormalizeVLLMFlags_EverySpellingReadsTheSame(t *testing.T) {
	for _, tc := range []struct {
		flag  string
		short string
	}{
		{tensorParallelSizeFlag, "-tp"},
		{pipelineParallelSizeFlag, "-pp"},
		{dataParallelSizeFlag, "-dp"},
	} {
		spellings := map[string][]string{
			"long equals":     {tc.flag + "=4"},
			"short separated": {tc.short, "4"},
			"short equals":    {tc.short + "=4"},
		}
		for name, args := range spellings {
			t.Run(fmt.Sprintf("%s/%s", tc.flag, name), func(t *testing.T) {
				got, err := getFlagValue(getExpandedCommandLine(vllmContainer(args...)), tc.flag)
				if err != nil {
					t.Fatalf("getFlagValue(%q) unexpected error: %v", args, err)
				}
				if got != 4 {
					t.Errorf("%s spelled %q read as %d, want 4 -- vLLM accepts all of these forms, "+
						"so every reader in this package must too", tc.flag, args, got)
				}
			})
		}
	}
}

// TestNormalizeVLLMFlags_QualifyingLaunchAlsoSizes covers a launch that qualifies as
// elastic EP also sizing correctly: qualification and width are read by different helpers.
// A one-rank reading renders no extra pods, no local-rank pin and no width gate while vLLM
// still places N ranks, which is the abort those three exist to prevent.
func TestNormalizeVLLMFlags_QualifyingLaunchAlsoSizes(t *testing.T) {
	for name, args := range map[string][]string{
		"equals flags": {"--enable-elastic-ep", dataParallelBackendFlag + "=ray", dataParallelSizeFlag + "=4"},
		"short flags":  {"--enable-elastic-ep", "-dpb", "ray", "-dp", "4"},
	} {
		t.Run(name, func(t *testing.T) {
			container := vllmContainer(args...)
			if !IsElasticEPRayLaunch(container) {
				t.Fatalf("spelling %q should qualify as an elastic-EP Ray launch", args)
			}
			got, err := getFlagValue(getExpandedCommandLine(container), dataParallelSizeFlag)
			if err != nil {
				t.Fatalf("getFlagValue(%q) unexpected error: %v", args, err)
			}
			if got != 4 {
				t.Errorf("qualified as elastic EP but its declared width read as %d, want 4. A shape "+
					"that qualifies must also size correctly, or the leader renders with no width "+
					"gate and vLLM aborts placing 4 ranks on one pod", got)
			}
		})
	}
}

// vllmLaunchArgs.WorldSize multiplies tensor by pipeline size, and it decides
// data-parallel-size-local and whether multinode coordination is injected at all -- so a
// size silently read as 1 is not a cosmetic miss.
func TestNormalizeVLLMFlags_WorldSizeReadsShortAndEqualsForms(t *testing.T) {
	for name, args := range map[string][]string{
		"equals": {tensorParallelSizeFlag + "=4", pipelineParallelSizeFlag + "=2"},
		"short":  {"-tp", "4", "-pp", "2"},
	} {
		t.Run(name, func(t *testing.T) {
			got := parseVLLMLaunchArgs(getExpandedCommandLine(vllmContainer(args...))).WorldSize()
			if got != 8 {
				t.Errorf("WorldSize() = %d, want 8 (tp 4 x pp 2) for spelling %q", got, args)
			}
		})
	}
}

func TestNormalizeVLLMFlags_LeavesEverythingElseAlone(t *testing.T) {
	in := []string{"python3", "-m", "dynamo.vllm", "--model", "deepseek-ai/DeepSeek-V2-Lite",
		"--some-future-flag=value", "--trust-remote-code", "-dp", "4"}
	got := normalizeVLLMFlags(in)

	want := []string{"python3", "-m", "dynamo.vllm", "--model", "deepseek-ai/DeepSeek-V2-Lite",
		"--some-future-flag=value", "--trust-remote-code", dataParallelSizeFlag, "4"}
	if len(got) != len(want) {
		t.Fatalf("normalizeVLLMFlags(%q) = %q, want %q", in, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("token %d = %q, want %q (full: %q)", i, got[i], want[i], got)
		}
	}
}

// TestNormalizeVLLMFlags_DoesNotSpoofFlagsFromUnrelatedValues guards the injection the
// vllmValueFlags restriction prevents.
func TestNormalizeVLLMFlags_DoesNotSpoofFlagsFromUnrelatedValues(t *testing.T) {
	container := vllmContainer("--served-model-name=--enable-elastic-ep",
		dataParallelBackendFlag, "ray")
	if IsElasticEPRayLaunch(container) {
		t.Fatal("--served-model-name's value must not be read as --enable-elastic-ep")
	}
}

// TestNormalizeVLLMFlags_UnderscoreSpellingReadsTheSame covers the underscore spelling of a
// sizing flag, which getFlagValue otherwise silently falls back to 1 on.
func TestNormalizeVLLMFlags_UnderscoreSpellingReadsTheSame(t *testing.T) {
	for name, args := range map[string][]string{
		"separated": {"--tensor_parallel_size", "4"},
		"equals":    {"--tensor_parallel_size=4"},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := getFlagValue(getExpandedCommandLine(vllmContainer(args...)), tensorParallelSizeFlag)
			if err != nil {
				t.Fatalf("getFlagValue(%q) unexpected error: %v", args, err)
			}
			if got != 4 {
				t.Errorf("getFlagValue(%q) = %d, want 4 -- vLLM treats \"_\" and \"-\" as "+
					"interchangeable in long option names", args, got)
			}
		})
	}
}

func TestNormalizeVLLMFlags_UnderscoreSpellingQualifiesElasticEP(t *testing.T) {
	container := vllmContainer("--enable_elastic_ep", "--data_parallel_backend=ray")
	if !IsElasticEPRayLaunch(container) {
		t.Fatal("underscore-spelled --enable_elastic_ep/--data_parallel_backend should qualify as an elastic-EP Ray launch")
	}
}

// TestValidateParallelismSizes_RejectsNonPositive pins the operator to refusing a
// parallelism size vLLM itself would refuse. Accepting it means building a topology for
// a launch that cannot start, and the malformed flag then reaches the engine unchanged
// with no operator-side error.
func TestValidateParallelismSizes_RejectsNonPositive(t *testing.T) {
	for name, rawArgs := range map[string][]string{
		"zero tensor-parallel-size":     {tensorParallelSizeFlag, "0"},
		"negative tensor-parallel-size": {tensorParallelSizeFlag, "-1"},
		"negative pipeline-parallel-size combined": {
			tensorParallelSizeFlag, "2", pipelineParallelSizeFlag, "-1",
		},
		"zero data-parallel-size":     {dataParallelSizeFlag, "0"},
		"negative data-parallel-size": {dataParallelSizeFlag, "-1"},
		"unparseable value":           {tensorParallelSizeFlag, "not-a-number"},
		"valid then unparseable":      {tensorParallelSizeFlag, "4", tensorParallelSizeFlag, "nope"},
		// A trailing occurrence with no value at all: vLLM rejects the command
		// line, so falling back to the earlier 4 -- or to the default of 1 --
		// would size a topology the engine never uses.
		"valid then no value": {tensorParallelSizeFlag, "4", tensorParallelSizeFlag},
		"lone flag no value":  {tensorParallelSizeFlag},
	} {
		t.Run(name, func(t *testing.T) {
			args := parseVLLMLaunchArgs(getExpandedCommandLine(vllmContainer(rawArgs...)))
			if err := args.ValidateParallelismSizes(); err == nil {
				t.Errorf("ValidateParallelismSizes() = nil, want an error for %q", rawArgs)
			}
		})
	}
}

// TestValidateParallelismSizes_AcceptsAbsentOrPositive confirms an absent flag
// (getFlagValue's default of 1) and explicit positive values are not errors -- only an
// explicitly bad value is rejected.
func TestValidateParallelismSizes_AcceptsAbsentOrPositive(t *testing.T) {
	for name, rawArgs := range map[string][]string{
		"nothing set":              {},
		"positive tensor-parallel": {tensorParallelSizeFlag, "4"},
		"positive data-parallel":   {dataParallelSizeFlag, "8"},
		"all three set and positive": {
			tensorParallelSizeFlag, "2", pipelineParallelSizeFlag, "2", dataParallelSizeFlag, "4",
		},
	} {
		t.Run(name, func(t *testing.T) {
			args := parseVLLMLaunchArgs(getExpandedCommandLine(vllmContainer(rawArgs...)))
			if err := args.ValidateParallelismSizes(); err != nil {
				t.Errorf("ValidateParallelismSizes() = %v, want nil for %q", err, rawArgs)
			}
		})
	}
}

// TestNormalizeVLLMFlags_ShortAliasIsNotSubstringMatched guards rewriting by prefix or
// substring rather than by whole token.
func TestNormalizeVLLMFlags_ShortAliasIsNotSubstringMatched(t *testing.T) {
	got := normalizeVLLMFlags([]string{"-dpb", "ray", "-dpx", "1", "--dp", "2"})
	want := []string{dataParallelBackendFlag, "ray", "-dpx", "1", "--dp", "2"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d = %q, want %q (full: %q)", i, got[i], want[i], got)
		}
	}
}

// TestGetFlagValue_RepeatedFlagUsesLastOccurrence pins the numeric readers to vLLM's
// FlexibleArgumentParser precedence: a repeated flag resolves to its final occurrence.
// Reading the first one makes the operator size a topology the engine will not use --
// "--tensor-parallel-size 1 --tensor-parallel-size 4" launches 4 ranks but would be read as 1.
func TestGetFlagValue_RepeatedFlagUsesLastOccurrence(t *testing.T) {
	for name, args := range map[string][]string{
		"long then long":   {tensorParallelSizeFlag, "1", tensorParallelSizeFlag, "4"},
		"long then short":  {tensorParallelSizeFlag, "1", "-tp", "4"},
		"short then long":  {"-tp", "1", tensorParallelSizeFlag, "4"},
		"short then short": {"-tp", "1", "-tp", "4"},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := getFlagValue(getExpandedCommandLine(vllmContainer(args...)), tensorParallelSizeFlag)
			if err != nil {
				t.Fatalf("getFlagValue(%q) unexpected error: %v", args, err)
			}
			if got != 4 {
				t.Errorf("getFlagValue(%q) = %d, want 4 (last occurrence) -- vLLM's argparse "+
					"applies the last value when a flag is repeated", args, got)
			}
		})
	}
}

// TestParseVLLMLaunchArgs_EnumFlagsUseLastOccurrence extends the same precedence to the
// enum-valued flags. hasArg reports true when ANY occurrence matches the sought value, so
// "--data-parallel-backend ray --data-parallel-backend mp" was read as ray even though vLLM
// resolves it to mp -- which would front the engine with a Ray head it never asked for.
func TestParseVLLMLaunchArgs_EnumFlagsUseLastOccurrence(t *testing.T) {
	t.Run("data-parallel-backend resolves to the final occurrence", func(t *testing.T) {
		container := vllmContainer("--enable-elastic-ep",
			dataParallelBackendFlag, "ray", dataParallelBackendFlag, "mp")
		if IsElasticEPRayLaunch(container) {
			t.Fatal("effective --data-parallel-backend is mp (the last occurrence); " +
				"must not qualify as an elastic-EP Ray launch")
		}
	})
	t.Run("distributed-executor-backend resolves to the final occurrence", func(t *testing.T) {
		args := parseVLLMLaunchArgs(getExpandedCommandLine(
			vllmContainer(distributedExecutorFlag, "mp", distributedExecutorFlag, "ray")))
		if args.IsMpDistributedExecutorBackend {
			t.Fatal("effective --distributed-executor-backend is ray (the last occurrence); " +
				"IsMpDistributedExecutorBackend must be false")
		}
	})
}

// vllmShellContainer mirrors how the operator ships a worker command: one /bin/sh -c
// string, which is where a shell control operator ends up glued to the last flag's value.
func vllmShellContainer(flags string) *corev1.Container {
	return &corev1.Container{
		Name:    "main",
		Command: []string{"/bin/sh", "-c"},
		Args:    []string{"exec python3 -m dynamo.vllm --model m " + flags},
	}
}

// TestNormalizeVLLMFlags_ShellTerminatedBackendQualifiesElasticEP covers a shell control
// operator glued to the backend value. The shell ends the word there, so vLLM receives
// "--data-parallel-backend=ray" and the pod needs a Ray head; without one it fails at
// runtime with vLLM's opaque "DP master node is missing or dead".
func TestNormalizeVLLMFlags_ShellTerminatedBackendQualifiesElasticEP(t *testing.T) {
	for name, flags := range map[string]string{
		"semicolon":      enableElasticEPFlag + " " + dataParallelBackendFlag + "=ray;",
		"and-and":        enableElasticEPFlag + " " + dataParallelBackendFlag + "=ray&&echo done",
		"pipe":           enableElasticEPFlag + " " + dataParallelBackendFlag + "=ray|tee log",
		"redirect":       enableElasticEPFlag + " " + dataParallelBackendFlag + "=ray>log",
		"subshell close": enableElasticEPFlag + " " + dataParallelBackendFlag + "=ray)",
		"short alias":    enableElasticEPFlag + " " + dataParallelBackendShortFlag + "=ray;",
		"underscore":     "--enable_elastic_ep --data_parallel_backend=ray;",
	} {
		t.Run(name, func(t *testing.T) {
			if !IsElasticEPRayLaunch(vllmShellContainer(flags)) {
				t.Errorf("%q should qualify as an elastic-EP Ray launch -- the shell ends the "+
					"word at the control operator, so vLLM only ever sees %q",
					flags, dataParallelBackendFlag+"=ray")
			}
		})
	}
}

// TestNormalizeVLLMFlags_ShellTerminatedExecutorBackendReadsAsMp covers the same thing for
// the reader that decides whether a worker gets the wait-for-leader init container. The
// separated spelling is the case the look-behind in normalizeVLLMFlags exists for.
func TestNormalizeVLLMFlags_ShellTerminatedExecutorBackendReadsAsMp(t *testing.T) {
	for name, flags := range map[string]string{
		"separated": distributedExecutorFlag + " mp;",
		"equals":    distributedExecutorFlag + "=mp;",
		"and-and":   distributedExecutorFlag + " mp&&echo done",
	} {
		t.Run(name, func(t *testing.T) {
			args := parseVLLMLaunchArgs(getExpandedCommandLine(vllmShellContainer(flags)))
			if !args.IsMpDistributedExecutorBackend {
				t.Errorf("%q should read as the mp distributed-executor backend; without it the "+
					"worker starts before the leader's master port is open", flags)
			}
		})
	}
}

// TestNormalizeVLLMFlags_NonShellPunctuationIsNotTrimmed pins the character set. Only a
// shell control operator may be dropped -- anything else really does reach argparse, and a
// value may legitimately be a dotted import path.
func TestNormalizeVLLMFlags_NonShellPunctuationIsNotTrimmed(t *testing.T) {
	if IsElasticEPRayLaunch(vllmShellContainer(enableElasticEPFlag + " " + dataParallelBackendFlag + "=ray,")) {
		t.Error("\",\" is not a shell control operator, so vLLM really does receive \"ray,\" and rejects it")
	}
	for _, value := range []string{"my_pkg.MyExecutor", "external_launcher"} {
		got := normalizeVLLMFlags([]string{distributedExecutorFlag, value})
		if got[1] != value {
			t.Errorf("value %q was rewritten to %q; only shell control operators may be trimmed", value, got[1])
		}
	}
}

// TestNormalizeVLLMFlags_ValueSlotTrimStopsAtTheNextFlag pins the two rules that keep the
// separated-form trim from reaching a token that is not a value.
func TestNormalizeVLLMFlags_ValueSlotTrimStopsAtTheNextFlag(t *testing.T) {
	// --enable-elastic-ep takes no argument (argparse.BooleanOptionalAction), so the token
	// after it belongs to another flag and must survive untouched.
	got := normalizeVLLMFlags([]string{enableElasticEPFlag, "--data-parallel-address", "$(HOST)"})
	if got[2] != "$(HOST)" {
		t.Errorf("token after %s = %q, want %q", enableElasticEPFlag, got[2], "$(HOST)")
	}
	// A "-"-prefixed token is the next flag, not a value, so it is still canonicalized.
	got = normalizeVLLMFlags([]string{distributedExecutorFlag, "-dp", "2"})
	if got[1] != dataParallelSizeFlag {
		t.Errorf("token after %s = %q, want the canonicalized %q", distributedExecutorFlag, got[1], dataParallelSizeFlag)
	}
}

// TestNormalizeVLLMFlags_EqualsValueSpellingAFlagStaysOneToken extends the spoofing
// guarantee to a value carried by a flag that is itself normalized.
func TestNormalizeVLLMFlags_EqualsValueSpellingAFlagStaysOneToken(t *testing.T) {
	container := vllmContainer(dataParallelBackendFlag+"="+enableElasticEPFlag, dataParallelBackendFlag, "ray")
	if IsElasticEPRayLaunch(container) {
		t.Fatalf("a listed flag's value must not be split into a standalone %s token", enableElasticEPFlag)
	}
}

// TestNormalizeVLLMFlags_BooleanFlagInEqualsFormIsNotRequested covers a boolean flag in
// equals form. vLLM registers --enable-elastic-ep with argparse.BooleanOptionalAction, so
// it takes no argument and this spelling is an argparse error, not a request.
func TestNormalizeVLLMFlags_BooleanFlagInEqualsFormIsNotRequested(t *testing.T) {
	container := vllmContainer(enableElasticEPFlag+"=false", dataParallelBackendFlag, "ray")
	if IsElasticEPRayLaunch(container) {
		t.Fatalf("%s=false must not read as elastic EP requested", enableElasticEPFlag)
	}
}

// TestNormalizeVLLMFlags_SizingFlagValuesAreNotTrimmed pins the deliberate limit on
// trimShellTerminator: a shell terminator is dropped only from the two flags whose values
// are compared as strings, never from a sizing flag. On this branch that limit is what
// makes a terminated sizing value reach getFlagValue unparsed, so it is rejected outright
// rather than silently resolving to a size vLLM will not use.
func TestNormalizeVLLMFlags_SizingFlagValuesAreNotTrimmed(t *testing.T) {
	for name, flags := range map[string]string{
		"equals":    tensorParallelSizeFlag + "=4;",
		"separated": tensorParallelSizeFlag + " 4;",
	} {
		t.Run(name, func(t *testing.T) {
			_, err := getFlagValue(getExpandedCommandLine(vllmShellContainer(flags)), tensorParallelSizeFlag)
			if err == nil {
				t.Errorf("getFlagValue(%q) = nil error, want a rejection -- a sizing value is "+
					"never shell-trimmed, so %q does not parse", flags, "4;")
			}
		})
	}
}
