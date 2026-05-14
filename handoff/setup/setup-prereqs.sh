#!/usr/bin/env bash

set -euo pipefail

install_macos() {
  if ! xcode-select -p >/dev/null 2>&1; then
    xcode-select --install || true
    until xcode-select -p >/dev/null 2>&1; do
      echo "Waiting for Xcode Command Line Tools installation..."
      sleep 10
    done
  fi

  if ! command -v brew >/dev/null 2>&1; then
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -x /opt/homebrew/bin/brew ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
  fi

  brew install git curl python@3.12 kubectl helm gh || true
}

install_linux() {
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y git curl python3 python3-pip gcc gcc-c++ make openssl-devel pkgconfig
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git curl python3 python3-venv python3-pip build-essential pkg-config libssl-dev
  else
    echo "WARNING: no supported package manager found; install git, curl, python3, and build tools manually"
  fi
}

case "$(uname -s)" in
  Darwin) install_macos ;;
  Linux) install_linux ;;
  *) echo "WARNING: unsupported OS $(uname -s); continuing with command checks" ;;
esac

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  . "$HOME/.cargo/env"
fi

rustup show active-toolchain >/dev/null
rustup component add rustfmt >/dev/null

if ! command -v kubectl >/dev/null 2>&1; then
  bin="$HOME/.local/bin/kubectl"
  mkdir -p "$(dirname "$bin")"
  version="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$arch" in
    arm64|aarch64) karch=arm64 ;;
    x86_64|amd64) karch=amd64 ;;
    *) echo "ERROR: unsupported kubectl arch: $arch" >&2; exit 1 ;;
  esac
  curl -L -s -o "$bin" "https://dl.k8s.io/release/${version}/bin/${os}/${karch}/kubectl"
  chmod +x "$bin"
  export PATH="$HOME/.local/bin:$PATH"
fi

python3 --version
git --version
cargo --version
kubectl version --client=true
