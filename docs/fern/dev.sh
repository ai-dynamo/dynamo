#!/usr/bin/env bash
# fern/dev.sh — toggle a fast-hot-reload "fern-rooted" docs layout for local dev.
#
# WHY THIS EXISTS
#   `fern docs dev` only hot-reloads content that lives *under the fern root*
#   (the directory containing fern.config.json). This repo keeps documentation in
#   ../docs, which Fern builds but never watches — so every save triggers a full
#   ~5s rebuild instead of a ~300ms incremental reload.
#
#   This script temporarily relocates the content to a real fern/docs/ directory
#   and points the repo-root `docs` at it via a symlink. With the content under
#   the fern root, edits hot-reload incrementally (~42ms). `disable` reverts to
#   the canonical layout so your commit / PR stays clean.
#
# WHAT WAS RULED OUT (don't "simplify" this back into a plain symlink)
#   A `fern/docs -> ../docs` symlink does NOT work: Fern resolves the symlink,
#   sees the target is outside the fern root, and skips watching it (5 inotify
#   watches, zero reload). Only real content inodes under fern/ get watched
#   (measured: 65 watches, incremental reload filesEdited:1). That is why this
#   script *moves* the directory rather than just linking to it.
#
# USAGE
#   ./fern/dev.sh enable      # switch to fast-reload layout (docs/ -> fern/docs/)
#   ./fern/dev.sh dev [args]  # enable (if needed) then run `fern docs dev`
#   ./fern/dev.sh disable     # revert to canonical docs/ layout (run before commit)
#   ./fern/dev.sh status      # show current mode
#
# SAFETY
#   - The directory move is a same-filesystem rename: instant, content-preserving,
#     and it keeps file inodes stable (open editors / handles stay valid).
#   - docs.yml path rewrites are exact inverses (../docs/ <-> ./docs/), so any
#     *other* edits you make to docs.yml while in dev mode are preserved on revert.
#   - `disable` restores the canonical layout exactly; `git status` returns to the
#     same state it had before `enable`.
#   - DO NOT commit while DEV MODE is ON. Run `disable` first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FERN_DIR="$SCRIPT_DIR"
REPO_ROOT="$(dirname "$FERN_DIR")"
DOCS="$REPO_ROOT/docs"
FERN_DOCS="$FERN_DIR/docs"
DOCS_YML="$FERN_DIR/docs.yml"

c_red=$'\033[0;31m'; c_grn=$'\033[0;32m'; c_yel=$'\033[0;33m'; c_dim=$'\033[2m'; c_off=$'\033[0m'

die()  { echo "${c_red}error:${c_off} $*" >&2; exit 1; }
info() { echo "${c_dim}$*${c_off}"; }

# DEV MODE is ON when: repo-root docs is a symlink AND fern/docs is a real dir.
is_enabled() { [ -L "$DOCS" ] && [ -d "$FERN_DOCS" ] && [ ! -L "$FERN_DOCS" ]; }

git_exclude_file() {
  git -C "$REPO_ROOT" rev-parse --git-path info/exclude 2>/dev/null || true
}

# Hide the relocated content tree from `git status` so the 200+ moved files don't
# flood it and can't be staged by accident. (Tracked docs/** deletions still show
# — git always reports those — but this keeps the new fern/docs/ tree quiet.)
add_git_exclude() {
  local f; f="$(git_exclude_file)"; [ -n "$f" ] && [ -f "$f" ] || return 0
  grep -q '# >>> fern-dev' "$f" 2>/dev/null && return 0
  {
    echo '# >>> fern-dev (added by fern/dev.sh; removed on disable)'
    echo '/fern/docs/'
    echo '# <<< fern-dev'
  } >> "$f"
}

remove_git_exclude() {
  local f; f="$(git_exclude_file)"; [ -n "$f" ] && [ -f "$f" ] || return 0
  grep -q '# >>> fern-dev' "$f" 2>/dev/null || return 0
  sed -i '/# >>> fern-dev/,/# <<< fern-dev/d' "$f"
}

# The relocation moves docs/ -> fern/docs/, so every TRACKED docs/** path now
# reads as deleted in the worktree — 400+ ' D docs/...' lines that bury your real
# diff. `git add_git_exclude` can't hide these (git always reports tracked-file
# changes). skip-worktree tells git to stop comparing those index entries against
# the (now-absent) worktree files, so `git status` shows only your actual edits.
#   set   on enable  (after the move)
#   clear on disable (BEFORE the move-back, else the restore would race the flag)
# The set is scoped to the exact paths tracked at enable time; clear is scoped the
# same way, so no unrelated skip-worktree flags are touched.
DEV_SKIP_LIST="$FERN_DIR/.git-skip-worktree-list" # records what we flagged

skip_worktree_docs() {
  git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 || return 0
  # Tracked paths under docs/ (index side), NUL-safe for odd filenames.
  git -C "$REPO_ROOT" ls-files -z -- docs > "$DEV_SKIP_LIST.z" 2>/dev/null || return 0
  [ -s "$DEV_SKIP_LIST.z" ] || { rm -f "$DEV_SKIP_LIST.z"; return 0; }
  xargs -0 -a "$DEV_SKIP_LIST.z" git -C "$REPO_ROOT" update-index --skip-worktree 2>/dev/null || true
  mv "$DEV_SKIP_LIST.z" "$DEV_SKIP_LIST"
}

unskip_worktree_docs() {
  [ -f "$DEV_SKIP_LIST" ] || return 0
  xargs -0 -a "$DEV_SKIP_LIST" git -C "$REPO_ROOT" update-index --no-skip-worktree 2>/dev/null || true
  rm -f "$DEV_SKIP_LIST"
}

cmd_enable() {
  if is_enabled; then info "Already in DEV MODE (fast reload)."; cmd_status; return 0; fi

  [ -f "$FERN_DIR/fern.config.json" ] || die "not a fern root: $FERN_DIR (missing fern.config.json)"
  [ -d "$DOCS" ] && [ ! -L "$DOCS" ] || die "$DOCS is not a real directory; refusing to enable."
  [ -e "$FERN_DOCS" ] && die "$FERN_DOCS already exists; clean up a botched state before enabling."
  [ -f "$DOCS_YML" ] || die "missing $DOCS_YML"

  info "Relocating docs/ -> fern/docs/ (same-filesystem rename)..."
  mv "$DOCS" "$FERN_DOCS"
  ln -s "fern/docs" "$DOCS"                       # repo-root docs -> fern/docs

  info "Rewriting docs.yml paths ../docs/ -> ./docs/ ..."
  sed -i 's|\.\./docs/|./docs/|g' "$DOCS_YML"

  add_git_exclude
  skip_worktree_docs

  echo "${c_grn}DEV MODE ON${c_off} — content is now under the fern root and hot-reloads incrementally."
  echo "${c_yel}Do NOT commit in this state.${c_off} Run '${0##*/} disable' before committing/merging."
  echo
  info "Start the preview with:   ./fern/dev.sh dev      (or: cd fern && fern docs dev)"
}

cmd_disable() {
  if ! is_enabled; then info "Already in canonical layout (DEV MODE off)."; cmd_status; return 0; fi

  [ -L "$DOCS" ] || die "$DOCS is not a symlink; unexpected state, not reverting automatically."
  [ -d "$FERN_DOCS" ] && [ ! -L "$FERN_DOCS" ] || die "$FERN_DOCS missing or not a real dir; cannot revert."

  # Clear skip-worktree FIRST, while the docs/ worktree files are still absent —
  # this is a pure index op. The mv below then restores the files so index and
  # worktree agree and git status is clean.
  unskip_worktree_docs

  info "Restoring fern/docs/ -> docs/ ..."
  rm "$DOCS"                                       # remove the symlink
  mv "$FERN_DOCS" "$DOCS"

  info "Rewriting docs.yml paths ./docs/ -> ../docs/ ..."
  sed -i 's|\./docs/|../docs/|g' "$DOCS_YML"

  remove_git_exclude

  echo "${c_grn}DEV MODE OFF${c_off} — canonical docs/ layout restored. Safe to commit."
}

cmd_dev() {
  is_enabled || cmd_enable
  echo
  info "Launching 'fern docs dev' from $FERN_DIR (Ctrl-C to stop)."
  ( cd "$FERN_DIR" && fern docs dev "$@" ) || true
  echo
  echo "${c_yel}fern docs dev stopped — still in DEV MODE.${c_off}"
  echo "Run '${0##*/} disable' before you commit or merge."
}

cmd_status() {
  if is_enabled; then
    echo "mode: ${c_grn}DEV (fast hot reload)${c_off}"
    echo "  docs       -> $(readlink "$DOCS")"
    echo "  content dir : fern/docs/ (real, watched by fern)"
    echo "  docs.yml    : paths use ./docs/"
    echo "${c_yel}  reminder: run '${0##*/} disable' before committing.${c_off}"
  else
    echo "mode: ${c_grn}CANONICAL (commit-ready)${c_off}"
    echo "  content dir : docs/ (real)"
    echo "  docs.yml    : paths use ../docs/"
  fi
}

case "${1:-}" in
  enable|up|on)        cmd_enable ;;
  disable|down|off|restore) cmd_disable ;;
  dev|preview|serve)   shift; cmd_dev "$@" ;;
  status|st)           cmd_status ;;
  ""|-h|--help|help)
    sed -n '2,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
  *) die "unknown command: $1  (try: enable | dev | disable | status)" ;;
esac
