#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# Slime Integration Demo — cluster-free replay of the Slime tutorial: Slime
# trains Qwen3-0.6B with GRPO on GSM8K in a KubeRay RayJob while a Dynamo
# DynamoGraphDeployment serves its rollouts. The waits are fast-forwarded, so
# the run plays in about two minutes.
# ============================================================================
#
# NOTHING here touches a cluster: no kubectl, no GPUs. Every "❯ kubectl ..."
# line is echoed as a prompt string, followed by canned output.
#
# The steps, resource names, and checks follow the tutorial at
# docs/fern/pages/use-cases/reinforcement-learning/slime/tutorial.mdx, with its
# inline manifests referred to by file name. Pod names, rewards, gradient
# norms, log-prob gaps, sync times, timestamps, and the request and reload
# counts come from a 10-step run of that setup with Dynamo 1.6.1 and
# slimerl/slime:nightly-dev-20261027a-cu130. The training log is condensed to
# its integration milestones. The frames in between the recorded states of the
# fast-forwarded waits are approximate.
#
# Deterministic: no randomness, fixed strings -> identical every recording.
# Runs on the macOS system bash (3.2).
#
# Record with:
#   ./record-slime-demo.sh
#
set -euo pipefail

# =============================================================================
# Timing knobs — tune total length here (target ~2 min)
# =============================================================================
AGE_TICK="${AGE_TICK:-0.045}"        # per-second frame while the setup Job or the engines start
FAST_TICK="${FAST_TICK:-0.04}"       # per-second frame while the training log has nothing new
STEP_TICK="${STEP_TICK:-0.16}"       # per-second frame during the 10 training steps
PAUSE_SHORT="${PAUSE_SHORT:-1.0}"
PAUSE_MED="${PAUSE_MED:-1.0}"
PAUSE_LONG="${PAUSE_LONG:-1.5}"      # static card hold (intro)
CLOSING_HOLD="${CLOSING_HOLD:-4}"    # whole seconds on the closing card before the loop restarts
CHAR_SPEED="${CHAR_SPEED:-0.012}"    # delay per typed chunk
FILE_CHUNK="${FILE_CHUNK:-3}"        # chars per chunk while typing a file body (commands type 1 at a time)
NAR_PAUSE="${NAR_PAUSE:-0.9}"        # pause before each narration line
CMD_PAUSE="${CMD_PAUSE:-0.8}"        # linger after a command finishes "typing"
PROMPT_PAUSE="${PROMPT_PAUSE:-0.8}"  # linger on an empty prompt before typing

# =============================================================================
# Palette: each escape picks a syntax role, not a hue
# =============================================================================
# The Slime Integration overview plays this recording with TerminalDemo's
# "fern-code" theme, which colors each ANSI slot the way Fern's code blocks
# color that role, in the site's light and dark mode. The two modes give some
# roles different hues: command names are purple in light mode and yellow in
# dark mode. apply-fern-code-theme.py embeds the dark colors for other players.
RULE='\033[0;30m'     # 0  step header rules
PROP='\033[0;31m'     # 1  Python attribute names
STR='\033[0;32m'      # 2  quoted strings; success
CMD='\033[0;33m'      # 3  command names
FUNC='\033[0;34m'     # 4  shell builtins and Python function names
DECL='\033[0;35m'     # 5  declaration keywords: export, async, def
OP='\033[0;36m'       # 6  operators and control keywords: | = for do import return
NAR='\033[0;37m'      # 7  narration and comments
DIM='\033[0;90m'      # 8  progress and fast-forward notes
NUM='\033[0;91m'      # 9  numbers called out in narration; in-progress status
ARG='\033[0;92m'      # 10 unquoted arguments and flags
PARAM='\033[0;93m'    # 11 Python parameters
CALLARG='\033[0;94m'  # 12 Python call arguments
QUOTE='\033[0;95m'    # 13 quote marks
PUNCT='\033[0;96m'    # 14 punctuation
PROMPT='\033[0;97m'   # 15 the prompt
BOLD='\033[1m'
NC='\033[0m'
OK="$STR"             # success marks share the string green
STATUS="$NUM"         # in-progress status shares the number color

# Table column formats, padded the way kubectl pads them (widest value + 3).
CRD_FMT='%-36s%s'
POD_FMT='%-32s%-8s%-10s%-11s%s'
DGD_FMT='%-15s%-8s%-10s%s'

# =============================================================================
# Helpers (same behavior as hero-demo.sh)
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
# Syntax highlighting. hl_bash and hl_python split one line into segments the
# way Fern's code blocks color bash and Python, appending each segment's text
# to SEG_TXT and its role's escape to SEG_COL.
# -----------------------------------------------------------------------------
SEG_TXT=()
SEG_COL=()
seg_reset() { SEG_TXT=(); SEG_COL=(); }
seg() { SEG_COL+=("$1"); SEG_TXT+=("$2"); }

# Bash state that carries across the lines of one command: inside "...",
# inside $( ) within it, whether the next word is a command name, and a role
# that the next word takes after for or export.
HB_DQ=0
HB_SUB=0
HB_CMD=1
HB_NEXT=""
hl_bash_reset() { HB_DQ=0; HB_SUB=0; HB_CMD=1; HB_NEXT=""; }

# Color one word of a command.
hl_bash_word() {
    local w="$1"
    if [[ "$HB_NEXT" == "var" ]]; then
        HB_NEXT=""
        seg "$NC" "$w"
        return 0
    fi
    if [[ "$HB_NEXT" == "assign" && "$w" == *=* ]]; then
        HB_NEXT=""
        seg "$NC" "${w%%=*}"; seg "$OP" "="; seg "$NC" "${w#*=}"
        return 0
    fi
    case "$w" in
        for) seg "$OP" "$w"; HB_CMD=0; HB_NEXT=var; return 0 ;;
        in|done) seg "$OP" "$w"; return 0 ;;
        do) seg "$OP" "$w"; HB_CMD=1; return 0 ;;
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

# Color one line of Python. Each line of the demo's file is self-contained.
hl_python() {
    local s="$1" n=${#1} i=0 j c w prev next role
    local import_line=0 after_def=0 params=0 depth=0
    case "$s" in "import "*|"from "*) import_line=1 ;; esac
    while (( i < n )); do
        c=${s:i:1}
        if [[ "$c" == " " ]]; then
            j=$i
            while [[ "${s:j:1}" == " " ]]; do j=$((j + 1)); done
            seg "$NC" "${s:i:j-i}"; i=$j
            continue
        fi
        if [[ "$c" == '"' || "$c" == "'" ]]; then
            j=$((i + 1))
            while (( j < n )) && [[ "${s:j:1}" != "$c" ]]; do j=$((j + 1)); done
            seg "$QUOTE" "$c"; seg "$STR" "${s:i+1:j-i-1}"; seg "$QUOTE" "$c"
            i=$((j + 1))
            continue
        fi
        if [[ "$c" =~ [A-Za-z_] ]]; then
            j=$i
            while [[ "${s:j:1}" =~ [A-Za-z0-9_] ]]; do j=$((j + 1)); done
            w=${s:i:j-i}
            next=${s:j:1}
            prev=""
            (( i > 0 )) && prev=${s:i-1:1}
            case "$w" in
                import|from|as|return|await|if|else|for|in|with) role=$OP ;;
                async|def|class) role=$DECL; [[ "$w" == def ]] && after_def=1 ;;
                *)
                    if (( after_def )); then
                        role=$FUNC; after_def=0; params=1
                    elif (( params == 2 )); then
                        role=$PARAM
                    elif [[ "$prev" == "." ]] && (( !import_line )); then
                        if [[ "$next" == "(" ]]; then role=$FUNC; else role=$PROP; fi
                    elif [[ "$next" == "(" ]]; then
                        role=$FUNC
                    elif (( depth > 0 )); then
                        role=$CALLARG
                    else
                        role=$NC
                    fi ;;
            esac
            seg "$role" "$w"; i=$j
            continue
        fi
        case "$c" in
            '(')
                if (( params == 1 )); then params=2; else depth=$((depth + 1)); fi
                seg "$PUNCT" "$c" ;;
            ')')
                if (( params == 2 )); then params=0; else depth=$((depth - 1)); fi
                seg "$PUNCT" "$c" ;;
            '=') seg "$OP" "$c" ;;
            *) seg "$PUNCT" "$c" ;;
        esac
        i=$((i + 1))
    done
}

# Type the segments, sleeping after every $1 characters (default 1).
type_segments() {
    local chunk="${1:-1}" k i t count=0
    cursor_show
    for (( k=0; k<${#SEG_TXT[@]}; k++ )); do
        t=${SEG_TXT[$k]}
        printf '%b' "${SEG_COL[$k]}"
        for (( i=0; i<${#t}; i++ )); do
            printf '%s' "${t:i:1}"
            count=$((count + 1))
            if (( count % chunk == 0 )); then sleep "$CHAR_SPEED"; fi
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

# "Create a file": type `vim <file>` at the prompt, then type the file body.
type_file() {
    local file="$1" body="$2" line
    cursor_show
    printf '%b ' "${PROMPT}❯${NC}"
    pause "$PROMPT_PAUSE"
    seg_reset; hl_bash_reset; hl_bash "vim ${file}"
    type_segments
    printf '\n'
    cursor_hide
    pause "$CMD_PAUSE"
    while IFS= read -r line; do
        seg_reset; hl_python "$line"
        type_segments "$FILE_CHUNK"
        printf '\n'
        cursor_hide
    done <<< "$body"
    pause "$CMD_PAUSE"
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

# =============================================================================
# Canned data (from the recorded run)
# =============================================================================
DYNAMO_GENERATE_PY='import copy
import os
from urllib.parse import urlsplit

from slime.rollout.sglang_streaming_rollout import generate_streaming as slime_generate_streaming

async def generate_streaming(args, sample, sampling_params):
    frontend = urlsplit(os.environ["DYNAMO_ROLLOUT_URL"])
    frontend_args = copy.copy(args)
    frontend_args.sglang_router_ip = frontend.hostname
    frontend_args.sglang_router_port = frontend.port
    return await slime_generate_streaming(frontend_args, sample, sampling_params)

generate_streaming.abort_mode = "request"'

# DGD Pods and the second after the apply at which each one turns ready.
POD_NAMES=(slime-sglang-0-frontend-bg7cd slime-sglang-0-worker-0-gsmhv slime-sglang-0-worker-1-smkj7)
POD_READY_T=(9 66 68)

# Driver-log milestones, in seconds after the RayJob started (00:57:54Z), and
# the values that each training step reported.
ENGINES_T=49       # Slime reads /server_info from both engines
FIRST_SYNC_T=84    # weight_v000001 written and loaded by both engines
DONE_T=153         # Job 'slime-grpo-d5tcx' succeeded
ROLLOUT_T=(89 100 105 110 116 121 127 132 138 143)
TRAIN_T=(95 100 106 111 116 121 127 133 138 143)
SYNC_T=(97 102 108 113 118 123 130 135 140 145)
REWARD=(0.641 0.516 0.578 0.656 0.562 0.578 0.516 0.703 0.547 0.734)
GRAD_NORM=(2.38 1.74 1.82 2.19 1.99 2.33 2.54 2.18 1.73 2.15)
LOGPROB_GAP=(0.0101 0.0105 0.0108 0.0113 0.0109 0.0121 0.0120 0.0111 0.0114 0.0099)
SYNC_SECS=(1.6 1.6 1.6 1.5 1.4 1.5 1.5 1.5 1.4 1.5)
MILESTONES=("$ENGINES_T" "$FIRST_SYNC_T" "${ROLLOUT_T[@]}" "${TRAIN_T[@]}" "${SYNC_T[@]}" "$DONE_T")

# =============================================================================
# Beat 0 — Intro (loop-safe start frame)
# =============================================================================
beat0_intro() {
    clear_screen
    step_header "🚀" "Let's train Qwen3-0.6B with Slime and Dynamo!"
    # Reveal the loop segment by segment, with the arrows in bold white.
    local arrow="${NC}${BOLD} → ${NC}"
    pause "$NAR_PAUSE"
    printf '%b# Each step: Dynamo serves the rollouts%b' "$NAR" "$NC"; sleep 0.5
    printf '%b' "$arrow"; sleep 0.5
    printf '%bSlime trains on them%b' "$NAR" "$NC"; sleep 0.5
    printf '%b' "$arrow"; sleep 0.5
    printf '%bthe new weights sync back.%b\n' "$NAR" "$NC"
    narrate "GRPO on GSM8K, 10 steps, ${NUM}${BOLD}3 GPUs${NC}${NAR}: two SGLang engines in Dynamo, one Megatron-LM trainer on KubeRay."
    pause "$PAUSE_LONG"
}

# =============================================================================
# Beat 1 — Stage the model and data on a shared volume
# =============================================================================
SETUP_H=2
setup_render() {  # seconds since the Job started
    local t="$1" status
    fmt_age "$t"
    if (( t < 25 )); then
        status="Downloading Qwen/Qwen3-0.6B and zhuzilin/gsm8k from Hugging Face..."
    elif (( t < 85 )); then
        status="Converting the checkpoint to torch_dist on 1 GPU..."
    else
        status="Creating /data/weight-updates for the weight syncs..."
    fi
    el "   ${STATUS}▸ ${status}${NC}"
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

beat1_stage_data() {
    step_header "📦" "Step 1: Stage the Model and Data"
    narrate "Prerequisites: the Dynamo platform and the KubeRay operator, both installed with Helm."
    echo ""
    show_command "kubectl get crd dynamographdeployments.nvidia.com rayjobs.ray.io"
    printf "${CRD_FMT}\n" "NAME" "CREATED AT"
    printf "${CRD_FMT}\n" "dynamographdeployments.nvidia.com" "2026-07-01T15:17:29Z"
    printf "${CRD_FMT}\n" "rayjobs.ray.io" "2026-07-09T18:47:32Z"
    echo ""
    narrate "The trainer and both engines share one ReadWriteMany volume, mounted at /data."
    echo ""
    show_command 'export NAMESPACE=slime-dynamo'
    show_command 'kubectl create namespace $NAMESPACE'
    echo "namespace/slime-dynamo created"
    echo ""
    narrate "A one-time Job downloads Qwen3-0.6B and GSM8K, then converts the checkpoint to torch_dist:" \
            "Megatron-LM loads its own checkpoint formats, not Hugging Face ones."
    echo ""
    show_command 'kubectl apply -n $NAMESPACE -f slime-data.yaml -f slime-setup.yaml'
    echo "persistentvolumeclaim/slime-data created"
    echo "job.batch/slime-setup created"
    pause "$PAUSE_SHORT"
    show_command 'kubectl wait -n $NAMESPACE --for=condition=complete job/slime-setup --timeout=30m'

    cursor_hide
    local t first=1
    for (( t=0; t<=90; t++ )); do
        if (( first )); then first=0; else move_up "$SETUP_H"; fi
        setup_render "$t"
        sleep "$AGE_TICK"
    done
    move_up "$SETUP_H"
    el "job.batch/slime-setup condition met"
    printf '\033[K'
    cursor_show
    pause "$PAUSE_SHORT"

    show_command 'kubectl logs -n $NAMESPACE job/slime-setup --tail=6'
    echo "/data/datasets:"
    echo "gsm8k"
    echo ""
    echo "/data/models:"
    echo "Qwen3-0.6B"
    echo "Qwen3-0.6B_torch_dist"
    echo ""
    narrate "SGLang serves the Hugging Face checkpoint. The trainer starts from the torch_dist copy."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 2 — The custom functions that point Slime at Dynamo
# =============================================================================
beat2_custom_functions() {
    step_header "🧩" "Step 2: Send Slime's Rollouts to Dynamo"
    narrate "Slime loads custom Python functions. This one sends every rollout request to the Dynamo frontend:"
    echo ""
    pause "$PAUSE_SHORT"
    type_file "dynamo_generate.py" "$DYNAMO_GENERATE_PY"
    echo ""
    narrate "A second one, weight_permissions.py, makes each weight update readable by the non-root engines."
    echo ""
    show_command_lines 'kubectl create configmap slime-functions -n $NAMESPACE \' \
                       '    --from-file=dynamo_generate.py --from-file=weight_permissions.py'
    echo "configmap/slime-functions created"
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 3 — Dynamo serves the rollouts: a DGD with two SGLang engines
# =============================================================================
# WATCH_H must equal the number of lines pods_render() emits.
WATCH_H=8
pods_render() {  # seconds since the DGD was applied
    local t="$1" i ready line
    fmt_age "$t"
    printf -v line "$POD_FMT" "NAME" "READY" "STATUS" "RESTARTS" "AGE"
    el "$line"
    for i in 0 1 2; do
        ready="0/1"
        if (( t >= ${POD_READY_T[$i]} )); then ready="1/1"; fi
        printf -v line "$POD_FMT" "${POD_NAMES[$i]}" "$ready" "Running" "0" "$AGE_STR"
        el "$line"
    done
    el ""
    if (( t >= ${POD_READY_T[2]} )); then
        el "   ${OK}✅ All 3 Pods ready: both engines can generate${NC}"
    else
        el "   ${STATUS}▸ Engines loading Qwen3-0.6B from /data${NC} — ready once /health_generate succeeds"
    fi
    el ""
    el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
}

beat3_serve_rollouts() {
    step_header "🚀" "Step 3: Serve Rollouts with Dynamo"
    narrate "A DynamoGraphDeployment (DGD) runs a Dynamo frontend and two SGLang engines, one GPU each." \
            "Each engine is stock sglang.launch_server. Dynamo's sidecar registers it with the frontend." \
            "One Service per engine exposes SGLang's own API on port 30000 for Slime's weight updates."
    echo ""
    show_command 'kubectl apply -n $NAMESPACE -f slime-sglang.yaml'
    echo "service/slime-sglang-engine-0 created"
    echo "service/slime-sglang-engine-1 created"
    echo "dynamographdeployment.nvidia.com/slime-sglang created"
    pause "$PAUSE_SHORT"

    local title="Step 3: Serve Rollouts with Dynamo"
    local cmd='watch kubectl get pods -n $NAMESPACE -l dynamo.nvidia.com/slime-role'
    local n1="Fast-forwarding while both engines load the model..."
    echo ""
    narrate "$n1"
    echo ""
    show_command "$cmd"

    # Hard cut: pin the header, narration, and command to the top.
    clear_screen
    step_header_bars "🚀" "$title"
    echo -e "${NAR}# ${n1}${NC}"
    echo ""
    print_command "$cmd"
    echo ""
    cursor_hide

    local t first=1
    for (( t=4; t<=70; t++ )); do
        if (( first )); then first=0; else move_up "$WATCH_H"; fi
        pods_render "$t"
        sleep "$AGE_TICK"
    done
    cursor_show
    pause "$PAUSE_SHORT"

    show_command 'kubectl get dgd -n $NAMESPACE slime-sglang'
    printf "${DGD_FMT}\n" "NAME" "READY" "BACKEND" "AGE"
    printf "${DGD_FMT}\n" "slime-sglang" "True" "sglang" "72s"
    echo ""
    narrate "The worker probes call /health_generate, so Ready means both engines can generate."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 4 — Slime trains in a KubeRay RayJob
# =============================================================================
beat4_rayjob() {
    step_header "🧠" "Step 4: Train with Megatron-LM on KubeRay"
    narrate "A KubeRay RayJob runs Slime's train.py, with one GPU on the Ray head for the Megatron-LM trainer." \
            "These train.py flags swap the engines Slime would launch for the two Dynamo engines:"
    echo ""
    show_command "grep -E 'external-engine|generate-function|update-weight' slime-grpo.yaml"
    echo "    --rollout-external-engine-addrs slime-sglang-engine-0:30000 slime-sglang-engine-1:30000"
    echo "    --custom-generate-function-path dynamo_generate.generate_streaming"
    echo "    --update-weight-transport disk"
    echo "    --update-weight-disk-dir /data/weight-updates"
    echo "    --custom-update-weight-post-write-path weight_permissions.make_weights_readable"
    echo ""
    narrate "Use the DGD's engines · send rollouts to the Dynamo frontend · sync weights as files on /data"
    echo ""
    show_command 'kubectl apply -n $NAMESPACE -f slime-grpo.yaml'
    echo "rayjob.ray.io/slime-grpo created"
    echo ""
    narrate "KubeRay starts the Ray head, then a submitter Pod runs ray job submit with the entrypoint."
    echo ""
    show_command "kubectl get pods -n \$NAMESPACE | grep -E 'NAME|slime-grpo'"
    echo "NAME                            READY   STATUS      RESTARTS   AGE"
    echo "slime-grpo-2cn5s                1/1     Running     0          5s"
    echo "slime-grpo-6gsc7-head-dk4s9     1/1     Running     0          19s"
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 5 — The RL loop: rollout → train → sync, 10 times (centerpiece)
# =============================================================================
# Fixed-height block redrawn in place. LOOP_H MUST equal the number of lines
# loop_render() emits: 2 milestone lines + blank + header + 10 steps + blank +
# status = 16. With the 8 pinned lines above it, that fills 24 of 25 rows.
LOOP_H=16

# Pad plain text into a named variable (padding counts bytes, so keep it ASCII).
pad() { printf -v "$1" "%-${2}s" "$3"; }

loop_milestones() {  # t
    local t="$1"
    if (( t >= ENGINES_T )); then
        el "  ${OK}✓${NC} Slime attached to the Dynamo engines: slime-sglang-engine-0:30000, slime-sglang-engine-1:30000"
    else
        el "  ${DIM}▸ Ray starts Slime's driver and the Megatron-LM trainer on the head Pod...${NC}"
    fi
    if (( t >= FIRST_SYNC_T )); then
        el "  ${OK}✓${NC} Initial weight sync: v1 → both engines in 2.9s, so rollouts start from the trainer's weights"
    elif (( t >= ENGINES_T )); then
        el "  ${DIM}▸ Loading the torch_dist checkpoint, then syncing it to both engines...${NC}"
    else
        el ""
    fi
}

loop_row() {  # step t
    local k="$1" t="$2" start c0 c1 c2 sync
    local r_txt="" r_col="$NC" t_txt="" t_col="$NC" s_txt="" s_col="$NC"
    if (( k == 0 )); then start=$FIRST_SYNC_T; else start=${SYNC_T[$((k - 1))]}; fi
    if (( t < start )); then
        el ""
        return 0
    fi
    if (( t >= ${ROLLOUT_T[$k]} )); then
        r_txt="64 samples, reward ${REWARD[$k]}"
    else
        r_txt="generating..."; r_col="$DIM"
    fi
    if (( t >= ${TRAIN_T[$k]} )); then
        t_txt="grad norm ${GRAD_NORM[$k]}, logprob gap ${LOGPROB_GAP[$k]}"
    elif (( t >= ${ROLLOUT_T[$k]} )); then
        t_txt="training..."; t_col="$DIM"
    fi
    sync="v$((k + 2))"
    if (( t >= ${SYNC_T[$k]} )); then
        s_txt="${sync} → both engines in ${SYNC_SECS[$k]}s"; s_col="$OK"
    elif (( t >= ${TRAIN_T[$k]} )); then
        s_txt="writing ${sync}..."; s_col="$DIM"
    fi
    printf -v c0 '  %-6s' "$k"
    pad c1 28 "$r_txt"
    pad c2 40 "$t_txt"
    el "${c0}${r_col}${c1}${NC}${t_col}${c2}${NC}${s_col}${s_txt}${NC}"
}

loop_status() {  # t
    local t="$1"
    fmt_age "$t"
    if (( t >= DONE_T )); then
        el "  ${OK}✅ Job 'slime-grpo-d5tcx' succeeded · ${AGE_STR} after the RayJob started${NC}"
    else
        el "  ${DIM}⏩ fast-forwarding · ${AGE_STR} elapsed${NC}"
    fi
}

loop_render() {  # t
    local t="$1" k header
    loop_milestones "$t"
    el ""
    printf -v header '  %-6s%-28s%-40s%s' "STEP" "ROLLOUTS (Dynamo)" "TRAINING (Megatron-LM)" "WEIGHT SYNC (disk)"
    el "${BOLD}${header}${NC}"
    for (( k=0; k<10; k++ )); do
        loop_row "$k" "$t"
    done
    el ""
    loop_status "$t"
}

# Count the milestones reached by second t, so a frame with no new milestone
# redraws only the status line.
EVENTS=0
count_events() {  # t
    local t="$1" m n=0
    for m in "${MILESTONES[@]}"; do
        if (( m <= t )); then n=$((n + 1)); fi
    done
    EVENTS=$n
}

beat5_rl_loop() {
    local title="Step 5: Rollout → Train → Sync, 10 Times"
    local cmd='kubectl logs -n $NAMESPACE -f job/slime-grpo'
    local note="   # condensed to the integration milestones"
    local n1="Each step: 64 rollouts through Dynamo → one training step → new weights loaded by both engines."

    step_header "🔁" "$title"
    narrate "$n1"
    echo ""
    show_command "$cmd" "$note"

    # Hard cut: pin the header, narration, and command to the top.
    clear_screen
    step_header_bars "🔁" "$title"
    echo -e "${NAR}# ${n1}${NC}"
    echo ""
    print_command "$cmd" "$note"
    echo ""
    cursor_hide

    local t first=1 last=-1
    for (( t=20; t<=DONE_T; t++ )); do
        count_events "$t"
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
        if (( t >= FIRST_SYNC_T && t <= ${SYNC_T[9]} )); then
            sleep "$STEP_TICK"
        else
            sleep "$FAST_TICK"
        fi
    done
    cursor_show
    pause "$PAUSE_MED"
    echo ""
    narrate "Logprob gap ${NUM}${BOLD}≈ 0.01${NC}${NAR} at every step: the engines generated with the weights Slime had just synced." \
            "The reward moves with the 8 prompts in each step. Ten steps test the pipeline, not learning."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 6 — Verify: the job, Dynamo's routing, and every engine reload
# =============================================================================
beat6_verify() {
    step_header "✅" "Step 6: Verify What Dynamo Did"
    show_command 'kubectl get rayjob -n $NAMESPACE slime-grpo'
    echo "NAME         JOB STATUS   DEPLOYMENT STATUS   RAY CLUSTER NAME   START TIME             END TIME               AGE"
    echo "slime-grpo   SUCCEEDED    Complete            slime-grpo-6gsc7   2026-11-05T00:57:54Z   2026-11-05T01:00:30Z   2m40s"
    echo ""
    narrate "The frontend logs one routing decision per rollout request: 10 steps × 64 samples."
    echo ""
    show_command "kubectl logs -n \$NAMESPACE -l dynamo.nvidia.com/slime-role=frontend --tail=-1 | grep -c 'Selected worker'"
    echo "640"
    echo ""
    narrate "Round-robin sent 320 requests to each engine. Each engine also loaded every weight version:"
    echo ""
    show_command_lines 'for worker in worker-0 worker-1; do' \
                       '    echo "$worker: $(kubectl logs -n $NAMESPACE -l dynamo.nvidia.com/slime-role=$worker --tail=-1 \' \
                       "      | grep -c 'POST /update_weights_from_disk HTTP/1.1\" 200')\"" \
                       '  done'
    echo "worker-0: 11"
    echo "worker-1: 11"
    echo ""
    narrate "One sync before the first rollout, plus one after each of the 10 training steps."
    pause "$PAUSE_MED"
}

# =============================================================================
# Beat 7 — Closing card (loop-safe end frame)
# =============================================================================
beat7_closing() {
    step_header "🎉" "Done: 10 GRPO Steps, Every Rollout Served by Dynamo"
    echo -e "   ${PROMPT}${BOLD}Slime trains. Dynamo serves. New weights every step. 🔁${NC}"
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
    beat1_stage_data
    beat2_custom_functions
    beat3_serve_rollouts
    beat4_rayjob
    beat5_rl_loop
    beat6_verify
    beat7_closing
}

main "$@"
