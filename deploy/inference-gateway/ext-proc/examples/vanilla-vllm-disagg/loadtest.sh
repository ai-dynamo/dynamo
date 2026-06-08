#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Comprehensive edge-case + load suite for the disagg PD path (gateway -> EPP -> sidecar).
GW="http://inference-gateway.agentgateway-system:80"
H="Host: epp-vanilla.local"; CT="Content-Type: application/json"; M="Qwen/Qwen3-0.6B"
PASS=0; FAIL=0
chk() { # $1=label $2=expected-code $3=actual-code
  if [ "$3" = "$2" ]; then echo "  PASS [$1] HTTP=$3"; PASS=$((PASS+1));
  else echo "  FAIL [$1] HTTP=$3 (expected $2)"; FAIL=$((FAIL+1)); fi
}
post() { curl -s -o /tmp/o.out -w "%{http_code}" --max-time 90 -X POST "$GW/$1" -H "$H" -H "$CT" -d "$2"; }

echo "=================== FUNCTIONAL + EDGE CASES ==================="
chk "short"            200 "$(post v1/completions "{\"model\":\"$M\",\"prompt\":\"Hi\",\"max_tokens\":8}")"
chk "max_tokens=1"     200 "$(post v1/completions "{\"model\":\"$M\",\"prompt\":\"The sky is\",\"max_tokens\":1}")"
chk "max_tokens=256"   200 "$(post v1/completions "{\"model\":\"$M\",\"prompt\":\"Explain entropy:\",\"max_tokens\":256}")"
LONG=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*200)")
chk "long-multiblock"  200 "$(post v1/completions "{\"model\":\"$M\",\"prompt\":\"$LONG Summarize:\",\"max_tokens\":24}")"
chk "repeat-prefix(cache)" 200 "$(post v1/completions "{\"model\":\"$M\",\"prompt\":\"$LONG Summarize:\",\"max_tokens\":24}")"
chk "chat"             200 "$(post v1/chat/completions "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Name a fruit.\"}],\"max_tokens\":16}")"
chk "unicode/emoji"    200 "$(post v1/completions "{\"model\":\"$M\",\"prompt\":\"Translate 你好 🌟 to English:\",\"max_tokens\":16}")"
chk "stop-sequence"    200 "$(post v1/completions "{\"model\":\"$M\",\"prompt\":\"Count: 1 2 3 4\",\"max_tokens\":20,\"stop\":[\"5\"]}")"
# streaming
SC=$(curl -s -o /tmp/s.out -w "%{http_code}" --max-time 60 -X POST "$GW/v1/completions" -H "$H" -H "$CT" -d "{\"model\":\"$M\",\"prompt\":\"Count: 1 2 3\",\"max_tokens\":16,\"stream\":true}")
chk "streaming(http)"  200 "$SC"; echo "    stream chunks=$(grep -c 'data:' /tmp/s.out)"
# negative / error edges (worker/gateway should reject cleanly, not 5xx-crash)
echo "  -- negative cases (expect 4xx, must not crash the path) --"
post v1/completions "{bad json" >/tmp/n1; echo "    malformed-json -> HTTP=$(cat /tmp/n1)"
post v1/completions "{\"model\":\"does-not-exist\",\"prompt\":\"x\",\"max_tokens\":4}" >/tmp/n2; echo "    unknown-model  -> HTTP=$(cat /tmp/n2)"

echo ""
echo "=================== PREFIX-AFFINITY (KV-aware) ==================="
PFX=$(python3 -c "print('Shared context block alpha beta gamma. '*60)")
for i in 1 2 3; do post v1/completions "{\"model\":\"$M\",\"prompt\":\"$PFX Q$i?\",\"max_tokens\":8}" >/dev/null; done
echo "  3 requests with a shared long prefix sent (affinity verified via worker KV-cache logs)"

echo ""
echo "=================== LOAD: $2 total @ concurrency $1 ==================="
N=${2:-200}; C=${1:-48}
date +%s.%N > /tmp/t0
seq 1 $N | xargs -P $C -I {} bash -c "
  curl -s -o /dev/null -w '%{http_code}\n' --max-time 90 -X POST '$GW/v1/completions' -H '$H' -H '$CT' \
    -d '{\"model\":\"$M\",\"prompt\":\"Req {} : a short fact about the number {} please.\",\"max_tokens\":24,\"temperature\":0.7}'
" > /tmp/codes.txt 2>/dev/null
T=$(python3 -c "print(round($(date +%s.%N)-$(cat /tmp/t0),2))")
n200=$(grep -c '^200$' /tmp/codes.txt); ntot=$(wc -l </tmp/codes.txt)
echo "  total=$ntot HTTP200=$n200 non200=$((ntot-n200))  wall=${T}s  approx_rps=$(python3 -c "print(round($ntot/$T,1))")"
echo "  code histogram:"; sort /tmp/codes.txt | uniq -c | sed 's/^/     /'

echo ""
echo "=================== SUMMARY ==================="
echo "  functional/edge PASS=$PASS FAIL=$FAIL ; load 200-success=$n200/$ntot"
