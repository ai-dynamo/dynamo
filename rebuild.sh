#!/bin/bash

# Function to rebuild the Dynamo project
# Usage: From the repo root, run: source rebuild.sh && rebuild [--with-vllm]
# Or add this function to your ~/.zshrc or ~/.bashrc
# Options:
#   --with-vllm    Install with vllm optional dependencies

rebuild() {
    # Parse command line arguments
    local install_extras=""
    for arg in "$@"; do
        case $arg in
            --with-vllm)
                install_extras="[vllm]"
                shift
                ;;
            *)
                # Unknown option
                ;;
        esac
    done
    # Save current directory
    local original_dir=$(pwd)
    
    # Find the repo root (looks for the deploy directory)
    local repo_root="$original_dir"
    while [[ ! -d "$repo_root/deploy" ]] && [[ "$repo_root" != "/" ]]; do
        repo_root=$(dirname "$repo_root")
    done
    
    if [[ ! -d "$repo_root/deploy" ]]; then
        echo "Error: Could not find dynamo-7 repo root"
        return 1
    fi
    
    echo "🔨 Starting Dynamo rebuild..."
    
    # Activate virtual environment
    if [[ -f "$repo_root/deploy/dynamo/bin/activate" ]]; then
        source "$repo_root/deploy/dynamo/bin/activate"
        echo "✓ Virtual environment activated"
    else
        echo "Error: Virtual environment not found at $repo_root/deploy/dynamo/bin/activate"
        return 1
    fi
    
    # Build Python bindings with maturin
    echo "🍹 Building Python bindings..."
    cd "$repo_root/lib/bindings/python" || return 1
    maturin develop --uv || {
        echo "Error: maturin build failed"
        cd "$original_dir"
        return 1
    }
    
    # Install package in editable mode
    echo "📦 Installing package${install_extras:+ with extras: $install_extras}..."
    cd "$repo_root" || return 1
    uv pip install -e ".${install_extras}" || {
        echo "Error: pip install failed"
        cd "$original_dir"
        return 1
    }
    
    echo "✅ Rebuild complete!"
    
    # Return to original directory
    cd "$original_dir"
}

# If the script is sourced, the function will be available
# If executed directly, run the function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    rebuild "$@"
fi

