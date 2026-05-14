#!/usr/bin/env bash

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install gh
  else
    echo "ERROR: install GitHub CLI first: https://cli.github.com/" >&2
    exit 1
  fi
fi

if ! gh auth status >/dev/null 2>&1; then
  gh auth login --hostname github.com --git-protocol ssh
fi

user="$(gh api user --jq .login)"
email="$(gh api user --jq '.email // empty')"
if [ -z "$email" ]; then
  id="$(gh api user --jq .id)"
  email="${id}+${user}@users.noreply.github.com"
fi

if [ -z "$(git config --global user.name 2>/dev/null || true)" ]; then
  git config --global user.name "$user"
fi
if [ -z "$(git config --global user.email 2>/dev/null || true)" ]; then
  git config --global user.email "$email"
fi

gh auth status
ssh -T git@github.com 2>&1 | head -1 || true
