#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# verl Integration Demo — cluster-free replay of the verl integration pages:
# verl trains Qwen2.5-7B-Instruct with GRPO on GSM8K in a KubeRay RayJob, and
# verl-recipe's Dynamo backend starts a Dynamo frontend and four vLLM engines
# inside the same job to serve its rollouts. The waits are fast-forwarded, so
# the run plays in about two minutes.
# ============================================================================
#
# NOTHING here touches a cluster: no kubectl, no GPUs. Every "❯ kubectl ..."
# line is echoed as a prompt string, followed by canned output.
#
# The steps, resource names, and checks follow installation.mdx and train.mdx
# in docs/fern/pages/use-cases/reinforcement-learning/verl/, with their inline
# manifests referred to by file name. The job ID, rewards, gradient norms,
# k3_kl values, weight-update times, test accuracies, and the times of the
# training log's milestones come from a 15-step run with the same RayJob
# overrides on Dynamo 1.5.1 (vLLM 0.28.0), verl 6cbca9ce, and verl-recipe
# 242c73fc. The setup Job's duration follows installation.mdx. The training
# log is condensed to its integration milestones, and the frames in between
# the recorded states of the fast-forwarded waits are approximate.
#
# Keep every line under 100 columns: the overview plays a 100-column
# recording, as the Slime overview does.
#
# Deterministic: no randomness, fixed strings -> identical every recording.
# Runs on the macOS system bash (3.2).
#
# Record with:
#   ./record-verl-demo.sh
#
set -euo pipefail

# =============================================================================
# Timing knobs — tune total length here (target ~2 min)
# =============================================================================
AGE_TICK="${AGE_TICK:-0.045}"        # per-frame delay while the setup Job or the Ray head starts
FAST_TICK="${FAST_TICK:-0.03}"       # per-second frame while the training log has nothing new
STEP_TICK="${STEP_TICK:-0.05}"       # per-second frame during the 15 training steps
PAUSE_SHORT="${PAUSE_SHORT:-1.0}"
PAUSE_MED="${PAUSE_MED:-1.0}"
PAUSE_LONG="${PAUSE_LONG:-1.5}"      # static card hold (intro)
CLOSING_HOLD="${CLOSING_HOLD:-4}"    # whole seconds on the closing card before the loop restarts
CHAR_SPEED="${CHAR_SPEED:-0.012}"    # delay per typed character
NAR_PAUSE="${NAR_PAUSE:-0.9}"        # pause before each narration line
CMD_PAUSE="${CMD_PAUSE:-0.8}"        # linger after a command finishes "typing"
PROMPT_PAUSE="${PROMPT_PAUSE:-0.8}"  # linger on an empty prompt before typing

# =============================================================================
# Palette: each escape picks a syntax role, not a hue
# =============================================================================
# The verl Integration overview plays this recording with TerminalDemo's
# "fern-code" theme, which colors each ANSI slot the way Fern's code blocks
# color that role, in the site's light and dark mode. The slots match
# slime-demo.sh, so ../slime-demo/apply-fern-code-theme.py embeds the right
# dark colors for other players. This demo types no Python, so slots 1, 11,
# and 12 go unused.
RULE='\033[0;30m'     # 0  step header rules
STR='\033[0;32m'      # 2  quoted strings; success
CMD='\033[0;33m'      # 3  command names
FUNC='\033[0;34m'     # 4  shell builtins
DECL='\033[0;35m'     # 5  declaration keywords: export
OP='\033[0;36m'       # 6  operators and control keywords: | =
NAR='\033[0;37m'      # 7  narration and comments
DIM='\033[0;90m'      # 8  progress and fast-forward notes
NUM='\033[0;91m'      # 9  numbers called out in narration; in-progress status
ARG='\033[0;92m'      # 10 unquoted arguments and flags
QUOTE='\033[0;95m'    # 13 quote marks
PUNCT='\033[0;96m'    # 14 punctuation
PROMPT='\033[0;97m'   # 15 the prompt
BOLD='\033[1m'
NC='\033[0m'
OK="$STR"             # success marks share the string green
STATUS="$NUM"         # in-progress status shares the number color

# Table column format, padded the way kubectl pads it (widest value + 3).
CRD_FMT='%-21s%s'

# =============================================================================
# Helpers (same behavior as slime-demo.sh)
# =============================================================================
clear_screen() { printf '\033[H\033[2J\033[3J'; }

pause() { sleep "${1:-2}"; }

# Hold the screen for N seconds. The recording and the player cap idle gaps,
# so emit an invisible event (show cursor) every second to keep a longer hold.
hold() {
    local i
    for (( i=0; i<$1; i++ )); do
        sleep 1
        printf '\033[?25h'
    done
}

# Narration. Each line appears on its own after a beat.
narrate() {
    local line
    for line in "$@"; do
        pause "$NAR_PAUSE"
        echo -e "${NAR}# ${line}${NC}"
    done
}

# The cursor stays visible while text is typed and hides during in-place redraws.
cursor_show() { printf '\033[?25h'; }
cursor_hide() { printf '\033[?25l'; }

# -----------------------------------------------------------------------------
# Syntax highlighting. hl_bash splits one line into segments the way Fern's
# code blocks color bash, appending each segment's text to SEG_TXT and its
# role's escape to SEG_COL.
# -----------------------------------------------------------------------------
SEG_TXT=()
SEG_COL=()
seg_reset() { SEG_TXT=(); SEG_COL=(); }
seg() { SEG_COL+=("$1"); SEG_TXT+=("$2"); }

# Bash state that carries across the lines of one command: inside "...",
# inside $( ) within it, whether the next word is a command name, and a role
# that the next word takes after export.
HB_DQ=0
HB_SUB=0
HB_CMD=1
HB_NEXT=""
hl_bash_reset() { HB_DQ=0; HB_SUB=0; HB_CMD=1; HB_NEXT=""; }

# Color one word of a command.
hl_bash_word() {
    local w="$1"
    if [[ "$HB_NEXT" == "assign" && "$w" == *=* ]]; then
        HB_NEXT=""
        seg "$NC" "${w%%=*}"; seg "$OP" "="; seg "$NC" "${w#*=}"
        return 0
    fi
    case "$w" in
        export) seg "$DECL" "$w"; HB_CMD=0; HB_NEXT=assign; return 0 ;;
    esac
    if (( HB_CMD )); then
        HB_CMD=0
        case "$w" in
            echo|cd|test) seg "$FUNC" "$w" ;;
            *) seg "$CMD" "$w" ;;
        esac
    else
        seg "$ARG" "$w"
    fi
}

hl_bash() {
    local s="$1" n=${#1} i=0 j c
    while (( i < n )); do
        c=${s:i:1}
        if [[ "$c" == " " ]]; then
            j=$i
            while [[ "${s:j:1}" == " " ]]; do j=$((j + 1)); done
            seg "$NC" "${s:i:j-i}"; i=$j
            continue
        fi
        if (( HB_DQ && !HB_SUB )); then
            # Inside "...": variables, $( to open a command, or string text.
            if [[ "$c" == '"' ]]; then
                seg "$QUOTE" '"'; HB_DQ=0; i=$((i + 1))
            elif [[ "$c" == '$' && "${s:i+1:1}" == "(" ]]; then
                seg "$QUOTE" '$('; HB_SUB=1; HB_CMD=1; i=$((i + 2))
            elif [[ "$c" == '$' ]]; then
                j=$((i + 1))
                while [[ "${s:j:1}" =~ [A-Za-z0-9_] ]]; do j=$((j + 1)); done
                seg "$NC" "${s:i:j-i}"; i=$j
            else
                j=$i
                while (( j < n )) && [[ "${s:j:1}" != '"' && "${s:j:1}" != '$' ]]; do j=$((j + 1)); done
                seg "$STR" "${s:i:j-i}"; i=$j
            fi
            continue
        fi
        case "$c" in
            '"') seg "$QUOTE" '"'; HB_DQ=1; HB_CMD=0; i=$((i + 1)); continue ;;
            "'")
                j=$((i + 1))
                while (( j < n )) && [[ "${s:j:1}" != "'" ]]; do j=$((j + 1)); done
                seg "$QUOTE" "'"; seg "$STR" "${s:i+1:j-i-1}"; seg "$QUOTE" "'"
                HB_CMD=0; i=$((j + 1))
                continue ;;
            '|') seg "$OP" '|'; HB_CMD=1; i=$((i + 1)); continue ;;
            ';') seg "$PUNCT" ';'; HB_CMD=1; i=$((i + 1)); continue ;;
            ')') seg "$QUOTE" ')'; HB_SUB=0; i=$((i + 1)); continue ;;
            '\') seg "$NC" '\'; i=$((i + 1)); continue ;;
            '$')
                j=$((i + 1))
                while [[ "${s:j:1}" =~ [A-Za-z0-9_] ]]; do j=$((j + 1)); done
                seg "$NC" "${s:i:j-i}"; HB_CMD=0; i=$j
                continue ;;
        esac
        # A word runs to the next space, quote, operator, or variable.
        j=$i
        while (( j < n )); do
            case "${s:j:1}" in ' '|'"'|"'"|'|'|';'|')'|'$') break ;; esac
            j=$((j + 1))
        done
        hl_bash_word "${s:i:j-i}"
        i=$j
    done
}

# Type the segments one character at a time.
type_segments() {
    local k i t
    cursor_show
    for (( k=0; k<${#SEG_TXT[@]}; k++ )); do
        t=${SEG_TXT[$k]}
        printf '%b' "${SEG_COL[$k]}"
        for (( i=0; i<${#t}; i++ )); do
            printf '%s' "${t:i:1}"
            sleep "$CHAR_SPEED"
        done
    done
    printf '%b' "$NC"
}

# Print the segments at once.
print_segments() {
    local k
    for (( k=0; k<${#SEG_TXT[@]}; k++ )); do
        printf '%b%s' "${SEG_COL[$k]}" "${SEG_TXT[$k]}"
    done
    printf '%b' "$NC"
}

# A shell command: draw the prompt, linger, type the command, linger.
# $2 = optional comment printed after the command.
show_command() {
    cursor_show
    printf '%b ' "${PROMPT}❯${NC}"
    pause "$PROMPT_PAUSE"
    seg_reset; hl_bash_reset; hl_bash "$1"
    type_segments
    if [[ -n "${2:-}" ]]; then
        printf '%b' "${NAR}${2}${NC}"
    fi
    printf '\n'
    cursor_hide
    pause "$CMD_PAUSE"
}

# A command that spans several lines: the first line follows the prompt and
# each continuation line is typed below it.
show_command_lines() {
    cursor_show
    printf '%b ' "${PROMPT}❯${NC}"
    pause "$PROMPT_PAUSE"
    hl_bash_reset
    local line
    for line in "$@"; do
        seg_reset; hl_bash "$line"
        type_segments
        printf '\n'
        cursor_hide
    done
    pause "$CMD_PAUSE"
}

# The same prompt line, printed at once. Used to re-pin a command after a hard cut.
print_command() {
    printf '%b ' "${PROMPT}❯${NC}"
    seg_reset; hl_bash_reset; hl_bash "$1"
    print_segments
    printf '%b\n' "${2:+${NAR}${2}${NC}}"
}

# Step header with two leading blank lines to separate it from the prior beat.
step_header() {
    echo ""
    echo ""
    step_header_bars "$1" "$2"
}

# Header with no leading blank lines, for watch beats that pin it to the top row.
step_header_bars() {
    local emoji="$1" title="$2"
    echo -e "${RULE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${BOLD}${emoji} ${title}${NC}"
    echo -e "${RULE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Hard cut: clear the screen and pin a beat's header, narration, and command
# to the top, so the fast-forwarded block below it never scrolls.
pin_beat() {  # emoji title narration command note
    clear_screen
    step_header_bars "$1" "$2"
    echo -e "${NAR}# ${3}${NC}"
    echo ""
    print_command "$4" "$5"
    echo ""
}

# kubectl's AGE format (HumanDuration): seconds below 2 minutes, then XmYs.
# Sets AGE_STR instead of printing, so redraw loops don't spawn subshells.
fmt_age() {
    local s="$1"
    if (( s < 120 )); then
        AGE_STR="${s}s"
    elif (( s % 60 == 0 )); then
        AGE_STR="$(( s / 60 ))m"
    else
        AGE_STR="$(( s / 60 ))m$(( s % 60 ))s"
    fi
}

# In-place redraw: move the cursor up N lines so the next frame overwrites the
# last one. Every line is printed with el, which erases the old line first.
move_up() { printf '\033[%dA' "$1"; }
el() { printf '\033[K%b\n' "$1"; }

# Pad plain text into a named variable (padding counts bytes, so keep it ASCII).
pad() { printf -v "$1" "%-${2}s" "$3"; }

# =============================================================================
# Canned data (from the recorded run)
# =============================================================================
JOB_ID="verl-grpo-nrcsf"

# Driver-log milestones, in seconds after Ray started the job (02:24:45Z).
RAY_T=13           # verl connects to the Ray cluster that KubeRay started
FSDP_T=196         # actor and reference model loaded into the FSDP workers
DYNAMO_T=213       # etcd and NATS up; the frontend and 4 engines starting
READY_T=336        # 4/4 engines registered, refit self-test passed, probe OK
VAL0_T=366         # step 0: test accuracy before training
DONE_T=740         # Job 'verl-grpo-nrcsf' succeeded

# For each training step: rollouts done and the engines paused for training
# (abort_all_requests), then the new weights loaded and generation resumed
# (resume_generation). FSDP finishes about one weight update before resuming.
ROLLOUT_T=(378 407 431 449 473 503 527 545 569 587 617 641 658 682 700)
SYNC_T=(406 424 448 466 490 520 544 563 586 604 634 658 676 699 718)
REWARD=(0.840 0.779 0.865 0.891 0.936 0.914 0.928 0.924 0.922 0.895 0.916 0.955 0.885 0.934 0.936)
GRAD_NORM=(0.57 0.71 0.38 0.47 0.33 0.38 0.57 0.53 0.38 0.44 0.51 0.18 0.31 0.52 0.29)
K3_KL=(0.0003 0.0010 0.0003 0.0018 0.0003 0.0005 0.0003 0.0035 0.0003 0.0008 0.0024 0.0003
       0.0008 0.0003 0.0007)
SYNC_SECS=(5.5 5.1 5.5 5.3 5.3 5.4 5.3 5.3 5.3 5.5 5.4 5.5 5.3 5.2 5.5)
VAL0="0.780"
# Test accuracy after steps 5, 10, and 15; the test takes about 12 seconds.
VAL_STEP=(5 10 15)
VAL_ACC=(0.897 0.904 0.914)
VAL_SECS=12

# =============================================================================
# Beat 0 — Intro (loop-safe start frame)
# =============================================================================
beat0_intro() {
    clear_screen
    step_header "🚀" "Let's train Qwen2.5-7B-Instruct with verl and Dynamo!"
    # Reveal the loop segment by segment, with the arrows in bold white.
    local arrow="${NC}${BOLD} → ${NC}"
    pause "$NAR_PAUSE"
    printf '%b# Each step: Dynamo serves the rollouts%b' "$NAR" "$NC"; sleep 0.5
    printf '%b' "$arrow"; sleep 0.5
    printf '%bverl trains on them%b' "$NAR" "$NC"; sleep 0.5
    printf '%b' "$arrow"; sleep 0.5
    printf '%bthe new weights sync back.%b\n' "$NAR" "$NC"
    narrate "GRPO on GSM8K, 15 steps, ${NUM}${BOLD}8 GPUs${NC}${NAR}: the FSDP trainer and 4 vLLM engines in Dynamo share them."
    pause "$PAUSE_LONG"
}

# =============================================================================
# Beat 1 — Build verl and stage the model and data on a shared volume
# =============================================================================
SETUP_H=2
SETUP_SECS=780     # about 13 minutes, 9 of them for the flash-attn compile
setup_render() {  # seconds since the Job started
    local t="$1" status
    fmt_age "$t"
    if (( t < 150 )); then
        status="Cloning verl and verl-recipe, then installing verl, Ray, and TransferQueue..."
    elif (( t < 690 )); then
        status="Compiling flash-attn 2.8.3.post1 for H100 and H200 GPUs..."
    elif (( t < 720 )); then
        status="Saving every package the Job added as a wheel in /data/wheelhouse..."
    else
        status="Downloading Qwen2.5-7B-Instruct and converting GSM8K to Parquet..."
    fi
    el "   ${STATUS}▸ ${status}${NC}"
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

beat1_build_and_stage() {
    step_header "📦" "Step 1: Build verl and Stage the Model"
    narrate "Prerequisite: the KubeRay operator. verl needs no Dynamo platform on the cluster:" \
            "verl-recipe starts Dynamo from the Python packages in Dynamo's vLLM runtime image."
    echo ""
    show_command "kubectl get crd rayjobs.ray.io rayclusters.ray.io"
    printf "${CRD_FMT}\n" "NAME" "CREATED AT"
    printf "${CRD_FMT}\n" "rayjobs.ray.io" "2026-07-09T18:47:36Z"
    printf "${CRD_FMT}\n" "rayclusters.ray.io" "2026-07-09T18:47:32Z"
    echo ""
    narrate "The setup Job and the Ray head share one volume, mounted at /data."
    echo ""
    show_command 'export NAMESPACE=verl-dynamo'
    show_command 'kubectl create namespace $NAMESPACE'
    echo "namespace/verl-dynamo created"
    echo ""
    narrate "A one-time Job in that runtime image builds verl and flash-attn into wheels on /data," \
            "then downloads Qwen2.5-7B-Instruct and GSM8K."
    echo ""
    show_command 'kubectl apply -n $NAMESPACE -f verl-data.yaml -f verl-setup.yaml'
    echo "persistentvolumeclaim/verl-data created"
    echo "job.batch/verl-setup created"
    pause "$PAUSE_SHORT"
    show_command 'kubectl wait -n $NAMESPACE --for=condition=complete job/verl-setup --timeout=60m'

    cursor_hide
    local t first=1
    for (( t=0; t<=SETUP_SECS; t+=5 )); do
        if (( first )); then first=0; else move_up "$SETUP_H"; fi
        setup_render "$t"
        sleep "$AGE_TICK"
    done
    move_up "$SETUP_H"
    el "job.batch/verl-setup condition met"
    printf '\033[K'
    cursor_show
    pause "$PAUSE_SHORT"

    show_command 'kubectl logs -n $NAMESPACE job/verl-setup --tail=6'
    echo "/data/gsm8k:"
    echo "test.parquet"
    echo "train.parquet"
    echo ""
    echo "/data/models:"
    echo "Qwen2.5-7B-Instruct"
    echo ""
    narrate "The runtime image has no Ray or verl, so the Ray head installs the wheels when it starts."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 2 — One RayJob starts the trainer and, inside it, Dynamo
# =============================================================================
HEAD_H=2
HEAD_SECS=105
head_render() {  # seconds since the RayJob was applied
    local t="$1" status
    fmt_age "$t"
    if (( t < 75 )); then
        status="Ray head installing the wheels from /data/wheelhouse..."
    else
        status="Ray head starting Ray 2.60.0, then the submitter runs ray job submit..."
    fi
    el "   ${STATUS}▸ ${status}${NC}"
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

beat2_rayjob() {
    step_header "🧠" "Step 2: Train with verl on KubeRay"
    narrate "One RayJob runs verl's trainer on a Ray head Pod that holds all 8 GPUs." \
            "verl-recipe's Dynamo backend starts Dynamo inside the same job, so there's no DGD." \
            "These overrides send verl's rollouts through Dynamo:"
    echo ""
    show_command "grep -E 'rollout.name|rollout.mode|tensor_model|router_mode|free_engine' verl-grpo.yaml"
    echo "    actor_rollout_ref.rollout.name=dynamo"
    echo "    actor_rollout_ref.rollout.mode=async"
    echo "    actor_rollout_ref.rollout.tensor_model_parallel_size=2"
    echo "    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv"
    echo "    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true"
    echo ""
    narrate "Dynamo rollouts · 4 vLLM engines of 2 GPUs · KV-aware routing · engines sleep in training"
    echo ""
    show_command 'kubectl apply -n $NAMESPACE -f verl-grpo.yaml'
    echo "rayjob.ray.io/verl-grpo created"
    echo ""
    narrate "KubeRay starts the Ray head, then a submitter Pod runs ray job submit with the entrypoint."
    echo ""
    show_command_lines "kubectl wait -n \$NAMESPACE --for=jsonpath='{.status.jobStatus}'=RUNNING \\" \
                       "    rayjob/verl-grpo --timeout=30m"

    cursor_hide
    local t first=1
    for (( t=0; t<=HEAD_SECS; t++ )); do
        if (( first )); then first=0; else move_up "$HEAD_H"; fi
        head_render "$t"
        sleep "$AGE_TICK"
    done
    move_up "$HEAD_H"
    el "rayjob.ray.io/verl-grpo condition met"
    printf '\033[K'
    cursor_show
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 3 — Dynamo starts inside the training job
# =============================================================================
# START_H must equal the number of lines start_render() emits.
START_H=9

# One milestone line: done at or after $2, in progress from $3, blank before.
milestone() {  # t done_t from_t done_text progress_text
    local t="$1"
    if (( t >= $2 )); then
        el "  ${OK}✓${NC} $4"
    elif (( t >= $3 )) && [[ -n "$5" ]]; then
        el "  ${DIM}▸ $5${NC}"
    else
        el ""
    fi
}

start_render() {  # seconds since Ray started the job
    local t="$1"
    milestone "$t" "$RAY_T" 0 \
        "verl joined the Ray cluster that KubeRay started" \
        "Starting verl's driver on the Ray head..."
    milestone "$t" "$FSDP_T" "$RAY_T" \
        "8 FSDP workers loaded Qwen2.5-7B-Instruct: the policy and the reference model" \
        "Loading Qwen2.5-7B-Instruct into 8 FSDP workers..."
    milestone "$t" "$DYNAMO_T" "$FSDP_T" \
        "verl-recipe started etcd, NATS, and the Dynamo frontend with the KV router" \
        "Starting etcd and NATS..."
    milestone "$t" "$DYNAMO_T" "$FSDP_T" \
        "4 dynamo.vllm engines starting, 2 GPUs each: 0-1, 2-3, 4-5, 6-7" \
        ""
    milestone "$t" "$READY_T" "$DYNAMO_T" \
        "Frontend healthy: all 4 engines registered" \
        "Engines loading the model and registering with the frontend..."
    milestone "$t" "$READY_T" "$READY_T" \
        "Weight-update self-test passed · token IDs and logprobs come back through Dynamo" \
        ""
    milestone "$t" "$VAL0_T" "$READY_T" \
        "Test accuracy before training: ${BOLD}${VAL0}${NC} on the 1,319 GSM8K test questions" \
        "Scoring the GSM8K test split before training..."
    el ""
    start_status "$t"
}

start_status() {  # t
    fmt_age "$1"
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

# The seconds at which start_render() changes, so the other frames redraw
# only the status line.
START_EVENTS=("$RAY_T" "$FSDP_T" "$DYNAMO_T" "$READY_T" "$VAL0_T")
EVENTS=0
count_events() {  # t milestone...
    local t="$1" m n=0
    shift
    for m in "$@"; do
        if (( m <= t )); then n=$((n + 1)); fi
    done
    EVENTS=$n
}

LOG_CMD='kubectl logs -n $NAMESPACE -f job/verl-grpo'
LOG_NOTE="   # condensed to the integration milestones"

beat3_dynamo_starts() {
    local title="Step 3: Dynamo Starts Inside the Job"
    local n1="verl loads the model, then verl-recipe starts Dynamo on the same GPUs."

    step_header "🔌" "$title"
    narrate "$n1"
    echo ""
    show_command "$LOG_CMD" "$LOG_NOTE"
    pin_beat "🔌" "$title" "$n1" "$LOG_CMD" "$LOG_NOTE"
    cursor_hide

    local t first=1 last=-1
    for (( t=0; t<=VAL0_T; t++ )); do
        count_events "$t" "${START_EVENTS[@]}"
        if (( first )); then
            first=0
            start_render "$t"
        elif (( EVENTS != last )); then
            move_up "$START_H"
            start_render "$t"
        else
            move_up 1
            start_status "$t"
        fi
        last=$EVENTS
        sleep "$FAST_TICK"
    done
    # Drop the fast-forward note: the narration takes its line.
    move_up 1
    printf '\033[K'
    cursor_show
    pause "$PAUSE_MED"
    narrate "Dynamo runs inside the job: KubeRay removes it with the Ray cluster when the job ends."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 4 — The RL loop: rollout → train → sync, 15 times (centerpiece)
# =============================================================================
# Fixed-height block redrawn in place. LOOP_H MUST equal the number of lines
# loop_render() emits: header + 15 steps + status = 17. With the 7 pinned
# lines above it, that fills 24 of 25 rows.
LOOP_H=17

loop_row() {  # step t
    local k="$1" t="$2" start train_t c0 c1 c2 c3 v i
    local r_txt="" r_col="$NC" t_txt="" t_col="$NC" s_txt="" s_col="$NC" v_txt="" v_col="$NC"
    if (( k == 0 )); then start=$VAL0_T; else start=${SYNC_T[$((k - 1))]}; fi
    if (( t < start )); then
        el ""
        return 0
    fi
    train_t=$(( ${SYNC_T[$k]} - 5 ))
    if (( t >= ${ROLLOUT_T[$k]} )); then
        r_txt="512 answers, reward ${REWARD[$k]}"
    else
        r_txt="generating..."; r_col="$DIM"
    fi
    if (( t >= train_t )); then
        t_txt="grad norm ${GRAD_NORM[$k]}, k3_kl ${K3_KL[$k]}"
    elif (( t >= ${ROLLOUT_T[$k]} )); then
        t_txt="training..."; t_col="$DIM"
    fi
    if (( t >= ${SYNC_T[$k]} )); then
        s_txt="4 engines in ${SYNC_SECS[$k]}s"; s_col="$OK"
    elif (( t >= train_t )); then
        s_txt="syncing..."; s_col="$DIM"
    fi
    for i in 0 1 2; do
        if (( k + 1 == ${VAL_STEP[$i]} )); then
            if (( t >= ${SYNC_T[$k]} + VAL_SECS )); then
                v_txt="${VAL_ACC[$i]}"; v_col="$BOLD"
            elif (( t >= ${SYNC_T[$k]} )); then
                v_txt="testing..."; v_col="$DIM"
            fi
        fi
    done
    printf -v c0 '  %-6s' "$((k + 1))"
    pad c1 27 "$r_txt"
    pad c2 30 "$t_txt"
    pad c3 24 "$s_txt"
    el "${c0}${r_col}${c1}${NC}${t_col}${c2}${NC}${s_col}${c3}${NC}${v_col}${v_txt}${NC}"
}

loop_status() {  # t
    local t="$1"
    fmt_age "$t"
    if (( t >= DONE_T )); then
        el "  ${OK}✅ Job '${JOB_ID}' succeeded · ${AGE_STR} after Ray started it${NC}"
    else
        el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
    fi
}

loop_render() {  # t
    local t="$1" k header
    printf -v header '  %-6s%-27s%-30s%-24s%s' "STEP" "ROLLOUTS (Dynamo)" "TRAINING (FSDP)" \
        "WEIGHT SYNC (CUDA IPC)" "TEST ACC"
    el "${BOLD}${header}${NC}"
    for (( k=0; k<15; k++ )); do
        loop_row "$k" "$t"
    done
    loop_status "$t"
}

# Every second at which loop_render() changes.
LOOP_EVENTS=()
loop_events_init() {
    local k
    for (( k=0; k<15; k++ )); do
        LOOP_EVENTS+=("${ROLLOUT_T[$k]}" "$(( ${SYNC_T[$k]} - 5 ))" "${SYNC_T[$k]}")
    done
    for k in 4 9 14; do
        LOOP_EVENTS+=("$(( ${SYNC_T[$k]} + VAL_SECS ))")
    done
    LOOP_EVENTS+=("$DONE_T")
}

beat4_rl_loop() {
    local title="Step 4: Rollout → Train → Sync, 15 Times"
    local n1="Each step: 512 rollouts through Dynamo → FSDP trains → new weights to all 4 engines."

    loop_events_init
    clear_screen
    step_header_bars "🔁" "$title"
    pause "$NAR_PAUSE"
    echo -e "${NAR}# ${n1}${NC}"
    print_command "$LOG_CMD" "$LOG_NOTE"
    echo ""
    cursor_hide

    local t first=1 last=-1
    for (( t=VAL0_T; t<=DONE_T; t++ )); do
        count_events "$t" "${LOOP_EVENTS[@]}"
        if (( first )); then
            first=0
            loop_render "$t"
        elif (( EVENTS != last )); then
            move_up "$LOOP_H"
            loop_render "$t"
        else
            move_up 1
            loop_status "$t"
        fi
        last=$EVENTS
        sleep "$STEP_TICK"
    done
    cursor_show
    pause "$PAUSE_LONG"
    echo ""
    narrate "Test accuracy rose from ${NUM}${BOLD}0.780${NC}${NAR} to ${NUM}${BOLD}0.914${NC}${NAR} in 15 steps." \
            "k3_kl stays near 0.0003 when the engines generate a step's batch with the newest weights." \
            "Steps that verl started a step early, on the older weights, read higher by design."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 5 — Verify: the engines took every weight update. The loop above
# already ends on the job's success, so the RayJob status table is left out.
# =============================================================================
beat5_verify() {
    step_header "✅" "Step 5: Verify What Dynamo Did"
    narrate "The engines pause while FSDP trains and resume once the new weights arrive."
    echo ""
    show_command_lines "kubectl logs -n \$NAMESPACE job/verl-grpo \\" \
                       "    | grep -c 'resume_generation: gate open'"
    echo "15"
    echo ""
    narrate "One resume after each of the 15 steps: the engines got new weights every step."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 6 — Closing card (loop-safe end frame)
# =============================================================================
beat6_closing() {
    step_header "🎉" "Done: 15 GRPO Steps, Every Rollout Served by Dynamo"
    echo -e "   ${PROMPT}${BOLD}verl trains. Dynamo serves. New weights every step. 🔁${NC}"
    echo ""
    hold "$CLOSING_HOLD"
}

# =============================================================================
# Main
# =============================================================================
main() {
    cursor_show
    trap 'cursor_show' EXIT
    beat0_intro
    beat1_build_and_stage
    beat2_rayjob
    beat3_dynamo_starts
    beat4_rl_loop
    beat5_verify
    beat6_closing
}

main "$@"
