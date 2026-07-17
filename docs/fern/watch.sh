#!/usr/bin/env bash
# fern/watch.sh — non-intrusive hot reload for the canonical docs/ layout.
#
# WHY THIS EXISTS
#   `fern docs dev` only watches content that lives *under the fern root* (the
#   directory with fern.config.json). This repo keeps docs in ../docs, which Fern
#   builds but never watches — so editing a page triggers no reload at all.
#
#   This script keeps documentation exactly where it is (docs/) and works around
#   that limitation with a tiny zero-dependency Node watcher: it monitors docs/
#   and `touch`es fern/main.css whenever a content file changes. Fern *does*
#   watch main.css, so the touch forces a reload that re-reads docs/.
#
# WHY NOT just move docs/ under fern/ (see fern/dev.sh)?
#   Moving the tree gives faster (~300ms incremental) reloads, but it turns docs/
#   into a symlink and shows every page as "deleted" in `git status`, destroying
#   per-file git status/diff while you work. This script trades reload speed for a
#   pristine git tree.
#
# TRADEOFF
#   Each reload is a FULL rebuild (~5s), not incremental — the changed file is
#   outside the fern root, so Fern can't do a targeted update. In exchange:
#     - docs/ never moves; per-file `git status` / `git diff` stay intact.
#     - `touch` only changes mtime, and git is content-based, so the touch creates
#       ZERO git churn. fern/main.css never shows as modified.
#
# USAGE
#   ./fern/watch.sh                          # frontend 3001 / backend 3002 (fern's
#                                            # defaults); auto-bumps if either is taken
#   ./fern/watch.sh --port 3939              # force the frontend (preview) port
#   ./fern/watch.sh --backend-port 3940      # force the backend port
#   ./fern/watch.sh --ext md,mdx,yml,yaml,svg,png   # override watched extensions
#   Ctrl-C stops both the dev server and the watcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FERN_DIR="$SCRIPT_DIR"
REPO_ROOT="$(dirname "$FERN_DIR")"
DOCS="$REPO_ROOT/docs"
MAIN_CSS="$FERN_DIR/main.css"

c_grn=$'\033[0;32m'; c_yel=$'\033[0;33m'; c_red=$'\033[0;31m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
die()  { echo "${c_red}error:${c_off} $*" >&2; exit 1; }
info() { echo "${c_dim}$*${c_off}"; }

# --- args ---------------------------------------------------------------------
PORT=""
BACKEND_PORT=""
EXTS="md,mdx,yml,yaml"
FERN_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port)         PORT="${2:-}"; shift 2 ;;
    --backend-port) BACKEND_PORT="${2:-}"; shift 2 ;;
    --ext)          EXTS="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) FERN_ARGS+=("$1"); shift ;;
  esac
done

# --- guards -------------------------------------------------------------------
[ -f "$FERN_DIR/fern.config.json" ] || die "not a fern root: $FERN_DIR"
[ -f "$MAIN_CSS" ] || die "missing $MAIN_CSS (the reload trigger file)"
command -v node >/dev/null 2>&1 || die "node not found (required for the watcher)"
if [ -L "$DOCS" ]; then
  die "docs/ is a symlink — you're in fern/dev.sh DEV MODE.
       Run './fern/dev.sh disable' first to restore the canonical docs/ layout,
       then re-run this script. (This watcher is for the docs/-in-place layout.)"
fi
[ -d "$DOCS" ] || die "$DOCS is not a directory"

# --- pick ports (fern defaults to 3001 frontend / 3002 backend) ---------------
port_taken() { ss -ltn 2>/dev/null | grep -q ":$1 "; }
next_free()  { local p="$1"; while port_taken "$p" || [ "$p" = "${2:-}" ]; do p=$((p+1)); done; echo "$p"; }

# Frontend (preview) port: default 3001, bump upward if busy.
if [ -z "$PORT" ]; then
  PORT="$(next_free 3001)"
elif port_taken "$PORT"; then
  die "requested --port $PORT is already in use"
fi

# Backend port: default 3002, bump upward if busy or equal to the frontend port.
if [ -z "$BACKEND_PORT" ]; then
  BACKEND_PORT="$(next_free 3002 "$PORT")"
elif port_taken "$BACKEND_PORT"; then
  die "requested --backend-port $BACKEND_PORT is already in use"
fi

# --- background watcher -------------------------------------------------------
WATCHER_JS='
const fs = require("fs");
const path = require("path");
const DIR  = process.env.WATCH_DIR;
const FILE = process.env.TOUCH_FILE;
const exts = new Set(process.env.WATCH_EXTS.split(",").map(e => "." + e.trim().toLowerCase()));
let timer = null;
const pending = new Set();
function flush() {
  timer = null;
  const n = pending.size; const sample = [...pending].slice(0, 3).join(", ");
  pending.clear();
  try {
    const now = new Date();
    fs.utimesSync(FILE, now, now);            // touch: mtime only, no content change
    console.log(`\x1b[2m[watch] ${n} change(s) [${sample}${n>3?", …":""}] → touched main.css → reload\x1b[0m`);
  } catch (e) { console.error("[watch] touch failed:", e.message); }
}
try {
  fs.watch(DIR, { recursive: true }, (_ev, file) => {
    if (!file) return;
    if (!exts.has(path.extname(file).toLowerCase())) return;   // content files only
    pending.add(file);
    if (timer) clearTimeout(timer);
    timer = setTimeout(flush, 200);                            // debounce editor multi-writes
  });
  console.log(`\x1b[2m[watch] watching docs/ (${[...exts].join(", ")}) → touch fern/main.css\x1b[0m`);
} catch (e) {
  console.error("[watch] fs.watch failed:", e.message);
  process.exit(1);
}
'

WATCH_DIR="$DOCS" TOUCH_FILE="$MAIN_CSS" WATCH_EXTS="$EXTS" \
  node -e "$WATCHER_JS" &
WATCHER_PID=$!

# --- cleanup ------------------------------------------------------------------
cleanup() {
  trap - EXIT INT TERM
  echo
  info "Shutting down watcher + dev server..."
  kill "$WATCHER_PID" 2>/dev/null || true
  # kill the fern dev server and its next-server child, if still up
  if [ -n "${FERN_PID:-}" ]; then
    for child in $(pgrep -P "$FERN_PID" 2>/dev/null); do kill "$child" 2>/dev/null || true; done
    kill "$FERN_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# --- foreground dev server ----------------------------------------------------
echo "${c_grn}Watch mode${c_off} — docs/ stays in place, git status stays clean."
echo "  preview port : $PORT  (open http://localhost:$PORT)"
echo "  backend port : $BACKEND_PORT"
echo "  trigger      : touch fern/main.css on docs/ changes (${EXTS})"
echo "${c_yel}  note: reloads are full rebuilds (~5s); this is the no-churn tradeoff.${c_off}"
echo
( cd "$FERN_DIR" && exec fern docs dev --port "$PORT" --backend-port "$BACKEND_PORT" "${FERN_ARGS[@]}" ) &
FERN_PID=$!
wait "$FERN_PID"
