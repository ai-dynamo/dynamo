#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# Dynamo Hero-Page Demo — cluster-free replay of the "one YAML → production"
# deploy flow. Prints pre-baked (real) output from a live 17-min DGDR run,
# fast-forwarding the long deploy waits into ~90 seconds.
# ============================================================================
#
# NOTHING here touches a cluster: no kubectl, no GPUs, no curl. Every
# "❯ kubectl ..." / "❯ curl ..." line is echo'd as a prompt string, followed
# by canned output captured verbatim from qwen-235-dgdr-demo-og.cast.
#
# Deterministic: no randomness, fixed strings -> identical every recording.
#
# Record with:
#   asciinema rec --cols 120 --rows 32 --idle-time-limit 1.5 \
#     --command ./hero-demo.sh hero-demo.cast
#
set -euo pipefail

# =============================================================================
# Timing knobs — tune total length here (target ~90s)
# =============================================================================
AGE_TICK="${AGE_TICK:-0.09}"       # delay between smooth per-second AGE frames
MIN_TICK="${MIN_TICK:-0.5}"        # delay between per-minute frames (deploy)
PAUSE_SHORT="${PAUSE_SHORT:-1.0}"
PAUSE_MED="${PAUSE_MED:-1.0}"
PAUSE_LONG="${PAUSE_LONG:-1.0}"    # static card holds (beat 0 / beat 7)
STREAM_SPEED="${STREAM_SPEED:-0.09}" # per-word delay while streaming chat reply
CHAR_SPEED="${CHAR_SPEED:-0.012}"    # per-char delay while "typing" commands / YAML
NAR_PAUSE="${NAR_PAUSE:-0.9}"        # pause between narration lines (they appear one at a time)
CMD_PAUSE="${CMD_PAUSE:-1.0}"        # linger after a command finishes "typing"
PROMPT_PAUSE="${PROMPT_PAUSE:-1.0}"  # linger on an empty prompt (cursor blinking) before typing

# =============================================================================
# Palette (lifted from demo-ecommerce-deployment-narrated.sh)
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
WHITE='\033[0;37m'
BOLD='\033[1m'
DIM='\033[0;35m'   # "dim" text repurposed as purple
NC='\033[0m'
NAR='\033[0;34m'   # narration = blue

# Table column formats — one format per table so header + rows always align,
# regardless of the (now shorter) qwen3-235b name.
DGDR_FMT='%-13s %-26s %-9s %-11s %-16s %-15s %s'
DGD_FMT='%-15s %-7s %-9s %s'
POD_FMT='%-42s %-7s %-9s %-10s %s'

# =============================================================================
# Helpers (narrate / show_command / step_header / type_heredoc / pause)
# =============================================================================
clear_screen() { printf '\033[H\033[2J\033[3J'; }

pause()        { sleep "${1:-2}"; }

# Narration. Pass one or more lines; each appears on its own after a beat, so the
# first line is separated from whatever preceded it (e.g. the step header) and
# consecutive lines reveal one at a time. Pause is built in — callers never pause.
narrate() {
    local line
    for line in "$@"; do
        pause "$NAR_PAUSE"
        echo -e "${NAR}# ${line}${NC}"
    done
}

# Cursor visibility. We keep the cursor VISIBLE by default so it blinks/advances
# while commands and YAML are "typed", and only hide it during the in-place
# redraw loops (beats 3 & 5) where a bouncing cursor would look like flicker.
cursor_show() { printf '\033[?25h'; }
cursor_hide() { printf '\033[?25l'; }

# Type a string char-by-char (keyboard feel), then newline. Typed text is shown
# in bright WHITE so it reads as "the user is typing". Cursor stays visible while
# typing so it visibly advances, then is hidden once the line lands.
# $2 = optional color (defaults to WHITE); pass "$NC" for normal-colored text.
type_out() {
    cursor_show
    local s="$1" color="${2:-$WHITE}" i ch
    printf '%b' "$color"
    for (( i=0; i<${#s}; i++ )); do
        ch="${s:i:1}"
        printf '%s' "$ch"
        sleep "$CHAR_SPEED"
    done
    printf '%b\n' "$NC"
    cursor_hide
}

# A shell command line: draw the prompt, linger a beat so the cursor blinks on an
# empty prompt (as if the user is about to type), THEN type the command, then a
# brief linger (as if the user just hit Enter).
show_command() {
    cursor_show
    printf '%b ' "${GREEN}❯${NC}"
    pause "$PROMPT_PAUSE"
    type_out "$1"
    pause "$CMD_PAUSE"
}

# Step header with two leading blank lines to separate it from the prior beat.
step_header() {
    local emoji="$1" title="$2"
    echo ""
    echo ""
    step_header_bars "$emoji" "$title"
}

# Header with no leading blank lines — for watch beats where, after a hard cut,
# the header must pin to the very top row of the screen.
step_header_bars() {
    local emoji="$1" title="$2"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${RED}${BOLD}${emoji} ${title}${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# "Create a file": type `vim <file>` at the prompt, then type the file body
# line-by-line (as if editing in the editor).
#   ❯ vim <file>
#   <line 1>
#   <line 2>
# $1 = filename, $2 = body (newline-separated).
type_heredoc() {
    local file="$1" body="$2" line
    printf '%b ' "${GREEN}❯${NC}"
    pause "$PROMPT_PAUSE"
    type_out "vim ${file}"
    pause "$CMD_PAUSE"
    while IFS= read -r line; do
        type_out "$line" "$NC"
    done <<< "$body"
    pause "$CMD_PAUSE"
}

# k8s-style AGE (matches kubectl HumanDuration): seconds up to 119s, then
# whole minutes. This is what lets the counter tick 38s,39s,... smoothly and
# flip to "2m" exactly at the 120s completion moment.
fmt_age() {
    local s="$1"
    if (( s < 120 )); then
        printf '%ds' "$s"
    else
        printf '%dm' "$(( s / 60 ))"
    fi
}

# In-place redraw: after printing a fixed-height frame, move the cursor back
# up N lines so the next render overwrites it — no clear-screen, no flash/jump.
# Each printed line ends with \033[K to erase any trailing chars from the prior
# frame. Track the height in the caller and pass it to move_up before redraw.
move_up() { printf '\033[%dA' "$1"; }
el() { printf '\033[K%b\n' "$1"; }   # erase-line + print (interprets colors)

# --- kubectl table renderers (shared so columns always line up) ---
dgdr_header() {
    printf "${DGDR_FMT}\n" "NAME" "MODEL" "BACKEND" "PHASE" "PROFILING" "DGD" "AGE"
}
dgdr_row() {  # phase, profiling, dgd, age
    printf "${DGDR_FMT}\n" "qwen3-235b" "Qwen/Qwen3-235B-A22B-FP8" "trtllm" "$1" "$2" "$3" "$4"
}
dgd_header() { printf "${DGD_FMT}\n" "NAME" "READY" "BACKEND" "AGE"; }
dgd_row()    { printf "${DGD_FMT}\n" "trtllm-disagg" "$1" "" "$2"; }  # ready, age
pod_header() { printf "${POD_FMT}\n" "NAME" "READY" "STATUS" "RESTARTS" "AGE"; }


# =============================================================================
# Canned data blocks (verbatim from the recorded run)
# =============================================================================
DGDR_YAML="apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-235b
spec:
  model: Qwen/Qwen3-235B-A22B-FP8"

# Abbreviated pod names: real component kinds, ReplicaSet hashes elided so
# each row fits ~120 cols and stays legible when the player is scaled down.
POD_NAMES=(
  "trtllm-disagg-frontend-dmtvf"
  "trtllm-disagg-trtllmdecodeworker-22g9r"
  "trtllm-disagg-trtllmdecodeworker-7x9z6"
  "trtllm-disagg-trtllmprefillworker-29rsc"
  "trtllm-disagg-trtllmprefillworker-vglb7"
  "trtllm-disagg-trtllmprefillworker-w56kh"
  "trtllm-disagg-trtllmprefillworker-z5m9v"
)

CHAT_RESPONSE="Hooray! I'm now live on Kubernetes with NVIDIA Dynamo! 🚀"

# =============================================================================
# Beat 0 — Intro (loop-safe start frame)
# =============================================================================
beat0_intro() {
    clear_screen
    # pause "$PAUSE_SHORT"          # start on a blank screen...
    step_header "🚀" "Let's deploy Qwen3-235B with Dynamo!"   # ...then print the title in
    # Reveal the one-liner segment-by-segment: narration-blue text prefixed with
    # "# " like the other narration, with the arrows drawn in bold white.
    local arrow="${WHITE}${BOLD} → ${NC}"
    pause "$NAR_PAUSE"
    printf '%b# It'\''s one step: kubectl apply%b' "$NAR" "$NC"; sleep 0.5
    printf '%b' "$arrow"; sleep 0.5
    printf '%bgo make coffee ☕%b' "$NAR" "$NC"; sleep 0.5
    printf '%b' "$arrow"; sleep 0.5
    printf '%bit'\''s live.%b\n' "$NAR" "$NC"
    pause "$PAUSE_LONG"
}

# =============================================================================
# Beat 1 — The DGDR: your only input
# =============================================================================
beat1_dgdr_yaml() {
    step_header "📝" "Step 1: Specify Your Model"
    narrate "Name a model. That's the only required field."
    echo ""
    pause "$PAUSE_SHORT"
    type_heredoc "qwen3-235b.yaml" "$DGDR_YAML"
    pause "$PAUSE_SHORT"
}

# =============================================================================
# Beat 2 — Apply: the only kubectl you run
# =============================================================================
beat2_apply() {
    step_header "🚀" "Step 2: Apply — The Only kubectl You'll Run"
    narrate "One command. Then you're done."
    echo ""
    pause "$PAUSE_SHORT"
    show_command "kubectl apply -f qwen3-235b.yaml"
    echo "dynamographdeploymentrequest.nvidia.com/qwen3-235b created"
    pause "$PAUSE_MED"
    echo ""
    show_command "kubectl get dgdr qwen3-235b -n dynamo-system"
    dgdr_header
    dgdr_row "Profiling" "Initializing" "" "4s"
    echo ""
    narrate "DGDR accepted. The operator takes it from here."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 3 — Profiling: Dynamo finds the optimal config
# =============================================================================
# The watch command is typed ONCE above this region; here we only redraw the
# live table + status + a fast-forward indicator BELOW the output. Height must
# match the number of lines emitted (see PF_H).
PF_H=6
pf_render() {  # phase prof dgd age status
    el "$(dgdr_header)"
    el "$(dgdr_row "$1" "$2" "$3" "$4")"
    el ""
    el "$5"
    el ""
    el "  ${DIM}⏩ fast-forwarding · ${4} elapsed${NC}"
}

beat3_profiling() {
    # Header + narration + watch command type in, continuing from the prior beat
    # (no clear yet). Once the command is typed and lingers, we hard-cut and
    # reprint this same block frozen at the top, then the live table renders below.
    local cmd="watch kubectl get dgdr qwen3-235b -n dynamo-system"
    local n1="First, Dynamo discovers your hardware: ${YELLOW}${BOLD}32× H100${NC}${NAR} on the cluster."
    local n2="It sets sensible defaults — ${YELLOW}${BOLD}SLA: TTFT ≤ 500ms · ITL ≤ 30ms${NC}"
    local n3="then generates the optimal config that meets these requirements — no manual tuning."

    step_header "🔬" "Step 3: Profiling — Finding the Optimal Config"
    narrate "$n1" "$n2" "$n3"
    echo ""
    show_command "$cmd"

    # Hard cut: pin the (now frozen) header + narration + command to the top.
    clear_screen
    step_header_bars "🔬" "Step 3: Profiling — Finding the Optimal Config"
    echo -e "${NAR}# ${n1}${NC}"
    echo -e "${NAR}# ${n2}${NC}"
    echo -e "${NAR}# ${n3}${NC}"
    echo ""
    echo -e "${GREEN}❯${NC} ${WHITE}${cmd}${NC}"
    echo ""
    cursor_hide                        # no bouncing cursor during the in-place redraw

    local first=1 s phase prof dgd status
    for (( s=4; s<=43; s++ )); do
        if   (( s <= 12 )); then
            phase="Profiling"; prof="Initializing"; dgd=""
            status="   ${BLUE}▸ Initializing${NC} — Detecting hardware, resolving model architecture..."
        elif (( s <= 20 )); then
            phase="Profiling"; prof="SweepingPrefill"; dgd=""
            status="   ${BLUE}▸ Sweeping Prefill${NC} — Testing TP/PP combinations for prefill latency..."
        elif (( s <= 28 )); then
            phase="Profiling"; prof="SweepingDecode"; dgd=""
            status="   ${BLUE}▸ Sweeping Decode${NC} — Testing parallelization for decode throughput..."
        elif (( s <= 35 )); then
            phase="Profiling"; prof="SelectingConfig"; dgd=""
            status="   ${BLUE}▸ Selecting Config${NC} — Filtering candidates against SLA, picking cheapest..."
        elif (( s <= 42 )); then
            phase="Profiling"; prof="GeneratingDGD"; dgd=""
            status="   ${BLUE}▸ Generating DGD${NC} — Rendering the DynamoGraphDeployment manifest..."
        else
            phase="Deploying"; prof="Done"; dgd="trtllm-disagg"
            status="   ${GREEN}✅ Profiling complete — phase is now: Deploying${NC}"
        fi
        (( first )) || move_up "$PF_H"
        first=0
        pf_render "$phase" "$prof" "$dgd" "$(fmt_age "$s")" "$status"
        sleep "$AGE_TICK"
    done
    cursor_show                        # restore cursor for the next (typed) beat
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 4 — The generated Deployment: what Dynamo built
# =============================================================================
beat4_generated_dgd() {
    step_header "📋" "Step 4: The Generated Deployment"
    narrate "Dynamo generated and applied a complete DynamoGraphDeployment." "You didn't write any of this."
    echo ""
    pause "$PAUSE_SHORT"
    show_command "kubectl get dgdr qwen3-235b -o jsonpath='{.status.profilingResults.selectedConfig}'"
    echo ""
    echo "   apiVersion:  nvidia.com/v1alpha1"
    echo "   kind:        DynamoGraphDeployment"
    echo "   name:        trtllm-disagg"
    echo ""
    echo "   services:"
    echo "   Service                            Replicas   GPUs         Type"
    echo "   -------------------------------- ---------- ------ ------------"
    echo "   Frontend                                  1      -     frontend"
    echo "   TRTLLMDecodeWorker  (decode)              2      8       worker"
    echo "   TRTLLMPrefillWorker (prefill)             4      4       worker"
    echo ""
    echo -e "   ${BOLD}Total GPU allocation: 32 GPUs${NC}"
    pause "$PAUSE_MED"
    echo ""
    echo -e "   ${BLUE}Frontend${NC} ──▶ ${BLUE}Prefill ×4${NC} ──▶ ${BLUE}Decode ×2${NC}      ${DIM}[ 32 GPUs · disagg ]${NC}"
    echo -e "   ${DIM}(router)     (4 GPU ea)     (8 GPU ea)${NC}"
    echo -e "   ${BLUE}SLA Planner ▸${NC} watches TTFT/ITL, auto-scales prefill & decode"
    pause "$PAUSE_LONG"
}

# =============================================================================
# Beat 5 — Deployment: the fast-forward mechanic (centerpiece)
# =============================================================================
# Fixed-height (DEPLOY_H) block redrawn in place via cursor-up. AGE counts up
# smoothly: every second from 38s→119s, then every minute 2m→15m (matching
# kubectl's own s→m rollover), so it reads as a sped-up clock, never a jump.
# The watch command is typed ONCE above this region; here we redraw only the
# pod table + status + a fast-forward indicator BELOW the output.
# DEPLOY_H MUST equal the number of lines deploy_render() emits (2 pod-header +
# 7 pod rows + blank + 3 status + blank + 1 fast-forward = 14). Keep in sync if
# you add/remove rows, or move_up will over/under-shoot and leave residue.
DEPLOY_H=14
deploy_render() {  # age_str ready pct
    local age="$1" ready="$2" pct="$3"
    local phase="Deploying"; (( ready >= 7 )) && phase="Deployed"

    el "$(pod_header)"
    local i rstr
    for i in "${!POD_NAMES[@]}"; do
        rstr="0/1"; (( i < ready )) && rstr="1/1"
        el "$(printf "${POD_FMT}" "${POD_NAMES[$i]}" "$rstr" "Running" "0" "$age")"
    done
    el ""
    if (( ready >= 7 )); then
        el "   ${BLUE}Pods: ${BOLD}7/7 ready${NC}"
        el "   ${GREEN}▸ All workers ready${NC}"
        el "   ${GREEN}✅ Deployment complete — all workers are healthy!${NC}"
    else
        local filled=$(( pct * 30 / 100 )) empty shards bar="" barempty="" b
        empty=$(( 30 - filled )); shards=$(( pct * 2452 / 100 ))
        for (( b=0; b<filled; b++ )); do bar+="█"; done
        for (( b=0; b<empty;  b++ )); do barempty+="░"; done
        el "   ${BLUE}Pods: ${BOLD}${ready}/7 ready${NC}"
        el "   ${BLUE}▸ Workers running${NC} — Loading model weights..."
        el "   Loading model: [${GREEN}${bar}${NC}${barempty}] ${BOLD}${pct}%${NC} | ${shards}/2452 shards"
    fi
    el ""
    el "  ${DIM}⏩ fast-forwarding · ${age} elapsed${NC}"
}

beat5_deploy_ffwd() {
    # Header + narration + watch command type in, continuing from the prior beat
    # (no clear yet). Once the command is typed and lingers, hard-cut and reprint
    # the frozen block at the top, then the live pod table renders below.
    local cmd="watch kubectl get pods -n dynamo-system -l 'nvidia.com/dynamo-graph-deployment-name=trtllm-disagg'"
    local n1="Fast-forwarding through image pulls and weight loading..."

    step_header "🚀" "Step 5: Deploy — Workers Come Online"
    narrate "$n1"
    echo ""
    show_command "$cmd"

    # Hard cut: pin the (now frozen) header + narration + command to the top.
    clear_screen
    step_header_bars "🚀" "Step 5: Deploy — Workers Come Online"
    echo -e "${NAR}# ${n1}${NC}"
    echo ""
    echo -e "${GREEN}❯${NC} ${WHITE}${cmd}${NC}"
    echo ""
    cursor_hide                        # no bouncing cursor during the in-place redraw

    local first=1 s m ready pct
    # --- seconds phase: 38s → 119s, one frame per second (frontend ready) ---
    for (( s=38; s<=119; s++ )); do
        ready=1
        pct=$(( 5 + (s - 38) * 50 / 81 ))   # ~5% → ~55%
        (( first )) || move_up "$DEPLOY_H"; first=0
        deploy_render "$(fmt_age "$s")" "$ready" "$pct"
        sleep "$AGE_TICK"
    done
    # --- minutes phase: 2m → 15m, one frame per minute (readiness milestones) ---
    for (( m=2; m<=15; m++ )); do
        case "$m" in
            2)  ready=1; pct=56 ;;  3)  ready=1; pct=60 ;;  4)  ready=1; pct=64 ;;
            5)  ready=1; pct=67 ;;  6)  ready=1; pct=70 ;;  7)  ready=1; pct=73 ;;
            8)  ready=1; pct=76 ;;  9)  ready=1; pct=79 ;;  10) ready=1; pct=82 ;;
            11) ready=2; pct=88 ;;  12) ready=3; pct=92 ;;  13) ready=5; pct=96 ;;
            14) ready=5; pct=99 ;;  15) ready=7; pct=100 ;;
        esac
        move_up "$DEPLOY_H"
        deploy_render "${m}m" "$ready" "$pct"
        sleep "$MIN_TICK"
    done
    cursor_show                        # restore cursor for the next (typed) beat
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 6 — Verify + live chat (proof it's real)
# =============================================================================
beat6_live_and_chat() {
    step_header "✅" "Step 6: Verify & Serve — Proof It's Real"
    narrate "Every service is healthy. Let's send it a real request."
    echo ""
    echo -e "   ${BOLD}Frontend:${NC}            ${GREEN}1/1 ready${NC}  ${DIM}(Deployment)${NC}"
    echo -e "   ${BOLD}TRTLLMDecodeWorker:${NC}  ${GREEN}2/2 ready${NC}  ${DIM}(Deployment)${NC}"
    echo -e "   ${BOLD}TRTLLMPrefillWorker:${NC} ${GREEN}4/4 ready${NC}  ${DIM}(Deployment)${NC}"
    echo ""
    echo -e "   ${GREEN}${BOLD}🟢 MODEL IS LIVE${NC}"
    pause "$PAUSE_MED"
    echo ""
    show_command "kubectl port-forward svc/trtllm-disagg-frontend 8000:8000 -n dynamo-system &"
    echo -e "   ${DIM}Port-forward active on localhost:8000${NC}"
    echo ""
    show_command "curl http://localhost:8000/v1/chat/completions -d '{\"model\":\"Qwen/Qwen3-235B-A22B-FP8\", \"stream\":true, ...}'"
    echo ""
    # Stream the response word-by-word for the "it's alive" payoff
    echo -e -n "   ${PURPLE}${BOLD}Model response:${NC} "
    local word
    for word in $CHAT_RESPONSE; do
        printf '%s ' "$word"
        sleep "$STREAM_SPEED"
    done
    echo ""
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 7 — Closing card (loop-safe end frame)
# =============================================================================
beat7_closing() {
    step_header "🎉" "Done. That Was It."
    echo -e "   ${PURPLE}${BOLD}One YAML. One command. Go make coffee. ☕${NC}"
    echo ""
    pause "$PAUSE_LONG"
}

# =============================================================================
# Main
# =============================================================================
main() {
    cursor_show                        # cursor visible by default (blinks while typing)
    trap 'cursor_show' EXIT            # ...always restore it on exit (normal or Ctrl-C)
    beat0_intro
    beat1_dgdr_yaml
    beat2_apply
    beat3_profiling
    beat4_generated_dgd
    beat5_deploy_ffwd
    beat6_live_and_chat
    beat7_closing
}

main "$@"
