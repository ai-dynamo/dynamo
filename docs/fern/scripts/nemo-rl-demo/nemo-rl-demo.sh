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
# NeMo RL Integration Demo — cluster-free replay of the NeMo RL integration
# pages: NeMo RL trains Qwen2.5-1.5B with GRPO on OpenMathInstruct-2 in a
# KubeRay RayJob, and with policy.generation.dynamo: true its driver starts a
# Dynamo frontend inside the same job for NeMo RL's two vLLM engines. The
# waits are fast-forwarded, so the run plays in about two minutes.
# ============================================================================
#
# NOTHING here touches a cluster or a registry: no kubectl, no docker, no
# GPUs. Every "❯ kubectl ..." line is echoed as a prompt string, followed by
# canned output.
#
# The steps, resource names, and checks follow installation.mdx and train.mdx
# in docs/fern/pages/use-cases/reinforcement-learning/nemo-rl/, with their
# inline manifests referred to by file name. The rewards, gen_kl_error values,
# requests per engine, weight-sync times, step end times, and the times of the
# training log's milestones come from a 25-step run of the same NeMo RL
# command on NeMo RL d51b2842 (ai-dynamo 1.5.1, vLLM 0.29.0), with the
# resources renamed to the pages' names. The image build's and the setup Job's
# durations follow installation.mdx. The training log is condensed to its
# integration milestones, and the frames in between the recorded states of
# the fast-forwarded waits are approximate.
#
# Keep every line under 100 columns: the overview plays a 100-column
# recording, as the Slime and verl overviews do.
#
# Deterministic: no randomness, fixed strings -> identical every recording.
# Runs on the macOS system bash (3.2).
#
# Record with:
#   ./record-nemo-rl-demo.sh
#
set -euo pipefail

# =============================================================================
# Timing knobs — tune total length here (target ~2 min)
# =============================================================================
AGE_TICK="${AGE_TICK:-0.045}"        # per-frame delay while the build, setup Job, or Ray head runs
FAST_TICK="${FAST_TICK:-0.03}"       # per-second frame while the training log has nothing new
STEP_TICK="${STEP_TICK:-0.04}"       # per-second frame during the 25 training steps
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
# The NeMo RL Integration overview plays this recording with TerminalDemo's
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

# Image build durations: hours and minutes from an hour on.
fmt_hm() {
    local s="$1"
    if (( s >= 3600 )); then
        AGE_STR="$(( s / 3600 ))h$(( (s % 3600) / 60 ))m"
    else
        fmt_age "$s"
    fi
}

# =============================================================================
# Canned data (from the recorded run)
# =============================================================================
JOB_ID="nemo-rl-grpo-28bgq"
IMAGE_REF="registry.example.com/nemo-rl:d51b2842-dynamo"

# Driver-log milestones, in seconds after the driver started (00:12:38Z).
RAY_T=17           # NeMo RL joins the Ray cluster that KubeRay started
DATA_T=34          # OpenMathInstruct-2 loaded from the volume's cache
DYNAMO_T=53        # etcd, NATS, and the frontend with the KV router up
ENGINES_T=73       # both vLLM 0.29.0 engines starting
REG_T=93           # both engines registered: the frontend routes to 2 engines
TRAINER_T=106      # Megatron Core model and optimizer ready
LOOP_T=144         # first weight sync to both engines; the rollouts start
ROLLOUT1_T=151     # step 1's batch done (later batches finish before their step starts)
DONE_T=937         # Job 'nemo-rl-grpo-28bgq' succeeded

# For each training step: the second at which its weight update to both
# engines finished (each engine resets its prefix cache then), its mean reward
# and gen_kl_error, how many of its batch's 512 requests Dynamo's KV router
# sent to each engine, and the weight update's duration.
STEPS=25
STEP_END=(191 222 252 281 311 341 372 403 433 464 493 524 555 586 618 648 678 708 737 768
          800 831 861 892 922)
REWARD=(0.072 0.088 0.080 0.081 0.066 0.049 0.083 0.051 0.119 0.145 0.260 0.157 0.146 0.176
        0.152 0.242 0.328 0.285 0.343 0.182 0.164 0.238 0.338 0.191 0.305)
GEN_KL=(0.00062 0.00060 0.00064 0.00068 0.00070 0.00075 0.00073 0.00134 0.00102 0.00102 0.00209
        0.00142 0.00093 0.00078 0.00090 0.00086 0.00081 0.00083 0.00097 0.00093 0.00096 0.00071
        0.00076 0.00074 0.00098)
E1=(261 230 294 269 269 265 240 257 240 258 268 242 277 270 277 269 256 235 267 288 256 249 257
    275 250)
E2=(251 282 218 243 243 247 272 255 272 254 244 270 235 242 235 243 256 277 245 224 256 263 255
    237 262)
SYNC_SECS=(0.11 0.12 0.12 0.12 0.11 0.12 0.11 0.12 0.12 0.12 0.11 0.11 0.10 0.11 0.12 0.11 0.10
           0.10 0.11 0.11 0.11 0.11 0.11 0.11 0.10)

# =============================================================================
# Beat 0 — Intro (loop-safe start frame)
# =============================================================================
beat0_intro() {
    clear_screen
    step_header "🚀" "Let's train Qwen2.5-1.5B with NeMo RL and Dynamo!"
    # Reveal the loop segment by segment, with the arrows in bold white.
    local arrow="${NC}${BOLD} → ${NC}"
    pause "$NAR_PAUSE"
    printf '%b# Each step: Dynamo serves the rollouts%b' "$NAR" "$NC"; sleep 0.5
    printf '%b' "$arrow"; sleep 0.5
    printf '%bNeMo RL trains on them%b' "$NAR" "$NC"; sleep 0.5
    printf '%b' "$arrow"; sleep 0.5
    printf '%bthe new weights sync back.%b\n' "$NAR" "$NC"
    narrate "GRPO on OpenMathInstruct-2, 25 steps, ${NUM}${BOLD}3 GPUs${NC}${NAR}: a Megatron Core trainer and 2 vLLM engines."
    pause "$PAUSE_LONG"
}

# =============================================================================
# Beat 1 — Build the image, then stage the model and data on a shared volume
# =============================================================================
BUILD_H=2
BUILD_SECS=5880    # 1 h 38 min: about 75 minutes of building, then 23 to export and push
build_render() {  # seconds since the build started
    local t="$1" status
    fmt_hm "$t"
    if (( t < 4500 )); then
        status="Building NeMo RL's environments and compiling CUDA extensions from source..."
    else
        status="Exporting and pushing the image, 17 GB compressed..."
    fi
    el "   ${STATUS}▸ ${status}${NC}"
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

SETUP_H=2
SETUP_SECS=1380    # 23 minutes: a 12-minute image pull, then the model and the dataset
setup_render() {  # seconds since the Job started
    local t="$1" status
    fmt_age "$t"
    if (( t < 720 )); then
        status="Pulling the NeMo RL image, 17 GB compressed, onto a GPU node..."
    elif (( t < 780 )); then
        status="Downloading Qwen2.5-1.5B into the Hugging Face cache on /data..."
    else
        status="Preparing all four splits of OpenMathInstruct-2, about 38 GB..."
    fi
    el "   ${STATUS}▸ ${status}${NC}"
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

beat1_build_and_stage() {
    step_header "📦" "Step 1: Build the Image and Stage the Data"
    narrate "Prerequisite: the KubeRay operator. NeMo RL needs no Dynamo platform on the cluster:" \
            "it starts Dynamo from the ai-dynamo package in its own image."
    echo ""
    show_command "kubectl get crd rayjobs.ray.io rayclusters.ray.io"
    printf "${CRD_FMT}\n" "NAME" "CREATED AT"
    printf "${CRD_FMT}\n" "rayjobs.ray.io" "2026-07-09T18:47:36Z"
    printf "${CRD_FMT}\n" "rayclusters.ray.io" "2026-07-09T18:47:32Z"
    echo ""
    narrate "No NeMo RL release has the Dynamo setting yet, so build an image from NeMo RL d51b2842:"
    echo ""
    show_command "export IMAGE=${IMAGE_REF}"
    show_command_lines \
        "docker buildx build --platform linux/amd64 --file docker/Dockerfile --target release \\" \
        "    --build-context nemo-rl=. --build-arg BUILD_DYNAMO=1 --build-arg SKIP_SGLANG_BUILD=1 \\" \
        "    --build-arg SKIP_TRTLLM_BUILD=1 --build-arg STRIP_UV_CACHE=1 --build-arg NVTE_CUDA_ARCHS=90 \\" \
        "    --build-arg MAX_JOBS=4 --build-arg CUSTOM_SETUP_FNAME= --tag \$IMAGE --push ."

    cursor_hide
    local t first=1
    for (( t=0; t<=BUILD_SECS; t+=60 )); do
        if (( first )); then first=0; else move_up "$BUILD_H"; fi
        build_render "$t"
        sleep "$AGE_TICK"
    done
    move_up "$BUILD_H"
    el "  ${OK}✓${NC} pushed ${IMAGE_REF} after 1h38m"
    printf '\033[K'
    cursor_show
    pause "$PAUSE_SHORT"

    narrate "The setup Job and the Ray head share one volume, mounted at /data."
    echo ""
    show_command 'export NAMESPACE=nemo-rl-dynamo'
    show_command 'kubectl create namespace $NAMESPACE'
    echo "namespace/nemo-rl-dynamo created"
    echo ""
    narrate "A one-time Job in the image caches Qwen2.5-1.5B and OpenMathInstruct-2 on /data."
    echo ""
    show_command 'kubectl apply -n $NAMESPACE -f nemo-rl-data.yaml -f nemo-rl-setup.yaml'
    echo "persistentvolumeclaim/nemo-rl-data created"
    echo "job.batch/nemo-rl-setup created"
    pause "$PAUSE_SHORT"
    show_command 'kubectl wait -n $NAMESPACE --for=condition=complete job/nemo-rl-setup --timeout=90m'

    cursor_hide
    first=1
    for (( t=0; t<=SETUP_SECS; t+=15 )); do
        if (( first )); then first=0; else move_up "$SETUP_H"; fi
        setup_render "$t"
        sleep "$AGE_TICK"
    done
    move_up "$SETUP_H"
    el "job.batch/nemo-rl-setup condition met"
    printf '\033[K'
    cursor_show
    pause "$PAUSE_SHORT"

    show_command 'kubectl logs -n $NAMESPACE job/nemo-rl-setup --tail=1'
    echo "OpenMathInstruct-2: 950,000 training and 50,000 held-out problems"
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 2 — One RayJob starts the trainer and, inside it, Dynamo
# =============================================================================
HEAD_H=2
HEAD_SECS=52
head_render() {  # seconds since the RayJob was applied
    local t="$1" status
    fmt_age "$t"
    if (( t < 25 )); then
        status="Ray head starting: KubeRay runs ray start with NeMo RL's Ray 2.58.0..."
    else
        status="Submitter pulling rayproject/ray:2.58.0-py313, then ray job submit..."
    fi
    el "   ${STATUS}▸ ${status}${NC}"
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

beat2_rayjob() {
    step_header "🧠" "Step 2: Train with NeMo RL on KubeRay"
    narrate "One RayJob runs NeMo RL's GRPO driver on a Ray head Pod that holds all 3 GPUs." \
            "Its config sets policy.generation.dynamo: true, so the driver starts Dynamo in the job."
    echo ""
    show_command "grep -A6 'entrypoint:' nemo-rl-grpo.yaml"
    echo "  entrypoint: >-  # KubeRay pastes this into a shell script unquoted, so it can't hold comments"
    echo "    /opt/nemo_rl_venv/bin/python examples/run_grpo_single_controller.py"
    echo "    --config examples/configs/grpo_math_1B_dynamo_single_controller.yaml"
    echo "    cluster.gpus_per_node=3"
    echo "    policy.generation.colocated.resources.gpus_per_node=2"
    echo "    grpo.max_num_steps=25"
    echo "    logger.monitor_gpus=false"
    echo ""
    narrate "Dynamo rollouts · KV-aware routing · 2 engine GPUs and 1 trainer GPU · no DGD"
    echo ""
    show_command 'kubectl apply -n $NAMESPACE -f nemo-rl-grpo.yaml'
    echo "rayjob.ray.io/nemo-rl-grpo created"
    echo ""
    narrate "KubeRay starts the Ray head, then a submitter Pod runs ray job submit with the entrypoint."
    echo ""
    show_command_lines "kubectl wait -n \$NAMESPACE --for=jsonpath='{.status.jobStatus}'=RUNNING \\" \
                       "    rayjob/nemo-rl-grpo --timeout=30m"

    cursor_hide
    local t first=1
    for (( t=0; t<=HEAD_SECS; t++ )); do
        if (( first )); then first=0; else move_up "$HEAD_H"; fi
        head_render "$t"
        sleep "$AGE_TICK"
    done
    move_up "$HEAD_H"
    el "rayjob.ray.io/nemo-rl-grpo condition met"
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

start_render() {  # seconds since the driver started
    local t="$1"
    milestone "$t" "$RAY_T" 0 \
        "NeMo RL joined the Ray cluster that KubeRay started" \
        "Starting NeMo RL's driver on the Ray head..."
    milestone "$t" "$DATA_T" "$RAY_T" \
        "Loaded OpenMathInstruct-2 from the volume: 950,000 training problems" \
        "Loading the dataset that the setup Job prepared..."
    milestone "$t" "$DYNAMO_T" "$DATA_T" \
        "The driver started etcd, NATS, and the Dynamo frontend with the KV router" \
        "Starting etcd and NATS..."
    milestone "$t" "$ENGINES_T" "$DYNAMO_T" \
        "2 vLLM engines starting in NeMo RL's own workers, 1 GPU each" \
        ""
    milestone "$t" "$REG_T" "$ENGINES_T" \
        "Both engines registered with the frontend and publish KV-cache events" \
        "Engines loading Qwen2.5-1.5B and registering with the frontend..."
    milestone "$t" "$TRAINER_T" "$REG_T" \
        "Megatron Core trainer ready on the third GPU" \
        "Megatron Core loading the policy and the reference model..."
    milestone "$t" "$LOOP_T" "$TRAINER_T" \
        "First weight sync to both engines over NCCL: the training loop starts" \
        "Starting NeMo RL's training loop..."
    el ""
    start_status "$t"
}

start_status() {  # t
    fmt_age "$1"
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

# The seconds at which start_render() changes, so the other frames redraw
# only the status line.
START_EVENTS=("$RAY_T" "$DATA_T" "$DYNAMO_T" "$ENGINES_T" "$REG_T" "$TRAINER_T" "$LOOP_T")
EVENTS=0
count_events() {  # t milestone...
    local t="$1" m n=0
    shift
    for m in "$@"; do
        if (( m <= t )); then n=$((n + 1)); fi
    done
    EVENTS=$n
}

LOG_CMD="kubectl logs -n \$NAMESPACE -f job/nemo-rl-grpo | grep -v 'request_id='"
LOG_NOTE="   # milestones only"

beat3_dynamo_starts() {
    local title="Step 3: Dynamo Starts Inside the Job"
    local n1="NeMo RL joins KubeRay's Ray cluster, then its driver starts Dynamo for its 2 engines."

    step_header "🔌" "$title"
    narrate "$n1"
    echo ""
    show_command "$LOG_CMD" "$LOG_NOTE"
    pin_beat "🔌" "$title" "$n1" "$LOG_CMD" "$LOG_NOTE"
    cursor_hide

    local t first=1 last=-1
    for (( t=0; t<=LOOP_T; t++ )); do
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
# Beat 4 — The RL loop: rollout → train → sync, 25 times (centerpiece)
# =============================================================================
# Fixed-height block redrawn in place. LOOP_H MUST equal the number of lines
# loop_render() emits: header + a 15-step window + status = 17. With the 7
# pinned lines above it, that fills 24 of 25 rows. Once step 16 starts, the
# window scrolls to keep the newest step on its last row.
LOOP_H=17
WINDOW=15

# The second at which a step (1-based) starts: when the step before it synced.
STEP_START=0
step_start() {  # step
    if (( $1 == 1 )); then STEP_START=$LOOP_T; else STEP_START=${STEP_END[$(( $1 - 2 ))]}; fi
}

loop_row() {  # step t
    local s="$1" t="$2" k=$(( $1 - 1 )) rollout_t train_t end_t c0 c1 c2 c3
    local r_txt="" r_col="$NC" e_txt="" t_txt="" t_col="$NC" s_txt="" s_col="$NC"
    step_start "$s"
    if (( t < STEP_START )); then
        el ""
        return 0
    fi
    # The engines generate a step's batch during the step before it, so only
    # step 1 waits for its rollouts.
    if (( s == 1 )); then rollout_t=$ROLLOUT1_T; else rollout_t=$STEP_START; fi
    end_t=${STEP_END[$k]}
    train_t=$(( end_t - 1 ))
    if (( t >= rollout_t )); then
        r_txt="512 answers, reward ${REWARD[$k]}"
        e_txt="${E1[$k]} / ${E2[$k]}"
    else
        r_txt="generating..."; r_col="$DIM"
    fi
    if (( t >= train_t )); then
        t_txt="gen_kl_error ${GEN_KL[$k]}"
    elif (( t >= rollout_t )); then
        t_txt="training..."; t_col="$DIM"
    fi
    if (( t >= end_t )); then
        s_txt="2 engines in ${SYNC_SECS[$k]}s"; s_col="$OK"
    elif (( t >= train_t )); then
        s_txt="syncing..."; s_col="$DIM"
    fi
    printf -v c0 '  %-6s' "$s"
    pad c1 27 "$r_txt"
    pad c2 15 "$e_txt"
    pad c3 23 "$t_txt"
    el "${c0}${r_col}${c1}${NC}${c2}${t_col}${c3}${NC}${s_col}${s_txt}${NC}"
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

# The newest step that has started.
CUR_STEP=1
current_step() {  # t
    local s=1
    while (( s < STEPS )) && (( $1 >= ${STEP_END[$(( s - 1 ))]} )); do s=$((s + 1)); done
    CUR_STEP=$s
}

loop_render() {  # t
    local t="$1" s lo header
    printf -v header '  %-6s%-27s%-15s%-23s%s' "STEP" "ROLLOUTS (Dynamo)" "ENGINES 1 / 2" \
        "TRAINING (Megatron)" "WEIGHT SYNC (NCCL)"
    el "${BOLD}${header}${NC}"
    current_step "$t"
    lo=$(( CUR_STEP - WINDOW + 1 ))
    if (( lo < 1 )); then lo=1; fi
    for (( s=lo; s<lo+WINDOW; s++ )); do
        loop_row "$s" "$t"
    done
    loop_status "$t"
}

# Every second at which loop_render() changes. A step starts when the step
# before it syncs, so the sync times also cover the window's scrolling.
LOOP_EVENTS=()
loop_events_init() {
    local k
    LOOP_EVENTS+=("$ROLLOUT1_T")
    for (( k=0; k<STEPS; k++ )); do
        LOOP_EVENTS+=("$(( ${STEP_END[$k]} - 1 ))" "${STEP_END[$k]}")
    done
    LOOP_EVENTS+=("$DONE_T")
}

beat4_rl_loop() {
    local title="Step 4: Rollout → Train → Sync, 25 Times"
    local n1="Each step: train on 512 rollouts served by Dynamo → send the new weights to both engines."

    loop_events_init
    clear_screen
    step_header_bars "🔁" "$title"
    pause "$NAR_PAUSE"
    echo -e "${NAR}# ${n1}${NC}"
    print_command "$LOG_CMD" "$LOG_NOTE"
    echo ""
    cursor_hide

    local t first=1 last=-1
    for (( t=LOOP_T; t<=DONE_T; t++ )); do
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
    narrate "The mean reward rose from ${NUM}${BOLD}0.08${NC}${NAR} in the first 3 steps to ${NUM}${BOLD}0.28${NC}${NAR} in the last 3." \
            "gen_kl_error stayed near 0.001: the engines generated with the weights from each sync." \
            "The engines generate a step's batch during the step before, so training never waits."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 5 — Verify: Dynamo routed every rollout. The loop above already ends
# on the job's success, so the RayJob status table is left out.
# =============================================================================
beat5_verify() {
    step_header "✅" "Step 5: Verify What Dynamo Did"
    narrate "The frontend logs one 'Selected worker' line for each request that it routes."
    echo ""
    show_command 'kubectl logs -n $NAMESPACE job/nemo-rl-grpo > nemo-rl-grpo.log'
    show_command "grep 'Selected worker' nemo-rl-grpo.log | grep -o 'worker_id=[0-9]*' | sort | uniq -c"
    echo "   6549 worker_id=7220855501956877835"
    echo "   6763 worker_id=7220855501956877837"
    echo ""
    narrate "Every rollout went through Dynamo's KV router: 13,312 requests, split between 2 engines."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 6 — Closing card (loop-safe end frame)
# =============================================================================
beat6_closing() {
    step_header "🎉" "Done: 25 GRPO Steps, Every Rollout Served by Dynamo"
    echo -e "   ${PROMPT}${BOLD}NeMo RL trains. Dynamo serves. New weights every step. 🔁${NC}"
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
