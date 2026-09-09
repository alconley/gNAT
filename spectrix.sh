#!/usr/bin/env sh
set -eu

# Bootstrap Spectrix on Linux and macOS without requiring Python up front.
# uv installs the Python version pinned by .python-version.

SCRIPT_DIR=${0%/*}
if [ "$SCRIPT_DIR" = "$0" ]; then
    SCRIPT_DIR=.
fi
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR" && pwd)
DRY_RUN=0
CHECK_ONLY=0
PLATFORM_OVERRIDE=

usage() {
    cat <<'EOF'
Usage: sh ./spectrix.sh [bootstrap options] [-- launcher options]

Bootstrap options:
  --check                 Check prerequisites without installing or launching.
  --dry-run               Print the bootstrap plan without changing the machine.
  --platform linux|macos  Override the platform in dry-run mode (proof testing only).
  -h, --help              Show this help.

Launcher options:
  --info                  Show info-level Rust logs in the terminal.
  --debug                 Show debug-level Rust logs and use a debug build.
  --debug-build           Use Cargo's debug build without changing RUST_LOG.
  --no-sync               Skip the Python environment sync performed at launch.
  --reset-state           Back up persisted state and start with a clean session.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)
            CHECK_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --platform)
            if [ "$#" -lt 2 ]; then
                echo "spectrix: --platform requires linux or macos" >&2
                exit 2
            fi
            PLATFORM_OVERRIDE=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [ -n "$PLATFORM_OVERRIDE" ] && [ "$DRY_RUN" -ne 1 ]; then
    echo "spectrix: --platform is only available with --dry-run" >&2
    exit 2
fi

if [ -n "$PLATFORM_OVERRIDE" ]; then
    PLATFORM=$PLATFORM_OVERRIDE
else
    case "$(uname -s)" in
        Linux) PLATFORM=linux ;;
        Darwin) PLATFORM=macos ;;
        *)
            echo "spectrix: use spectrix.ps1 on Windows; this script supports Linux and macOS" >&2
            exit 1
            ;;
    esac
fi

case "$PLATFORM" in
    linux|macos) ;;
    *)
        echo "spectrix: unsupported platform '$PLATFORM'" >&2
        exit 2
        ;;
esac

proof_plan() {
    echo "Spectrix bootstrap proof: $PLATFORM"
    echo "  check/install: native compiler and GUI build dependencies"
    echo "  check/install: uv"
    echo "  check/install: rustup and Cargo"
    echo "  provision:     Python from .python-version via uv"
    echo "  synchronize:   Python packages from uv.lock"
    echo "  launch:        spectrix.py $*"
}

if [ "$DRY_RUN" -eq 1 ]; then
    proof_plan "$@"
    exit 0
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

status=0
check_command() {
    name=$1
    if command -v "$name" >/dev/null 2>&1; then
        echo "[ok] $name"
    else
        echo "[missing] $name"
        status=1
    fi
}

linux_native_ready() {
    command -v pkg-config >/dev/null 2>&1 &&
        pkg-config --exists gtk+-3.0 xkbcommon openssl xcb-render xcb-shape xcb-xfixes
}

macos_native_ready() {
    xcode-select -p >/dev/null 2>&1
}

check_native_dependencies() {
    if [ "$PLATFORM" = linux ]; then
        if linux_native_ready; then
            echo "[ok] Linux compiler and GUI build dependencies"
        else
            echo "[missing] Linux compiler and/or GUI build dependencies"
            status=1
        fi
    elif macos_native_ready; then
        echo "[ok] Xcode Command Line Tools"
    else
        echo "[missing] Xcode Command Line Tools"
        status=1
    fi
}

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "Spectrix prerequisite check: $PLATFORM"
    check_native_dependencies
    check_command uv
    check_command rustup
    check_command cargo
    if command -v uv >/dev/null 2>&1 && uv python find 3.13 >/dev/null 2>&1; then
        echo "[ok] Python 3.13 available to uv"
    else
        echo "[missing] Python 3.13 (uv installs it during normal bootstrap)"
        status=1
    fi
    exit "$status"
fi

require_curl() {
    if ! command -v curl >/dev/null 2>&1; then
        echo "spectrix: curl is required to bootstrap uv and Rust" >&2
        exit 1
    fi
}

install_native_dependencies() {
    if [ "$PLATFORM" = macos ]; then
        if ! macos_native_ready; then
            echo "Installing Xcode Command Line Tools..."
            xcode-select --install
            echo "Complete the Apple installer, then run sh ./spectrix.sh again." >&2
            exit 1
        fi
        return
    fi

    if linux_native_ready; then
        return
    fi

    if command -v apt-get >/dev/null 2>&1; then
        if [ "$(id -u)" -eq 0 ]; then
            SUDO=
        elif command -v sudo >/dev/null 2>&1; then
            SUDO=sudo
        else
            echo "spectrix: sudo is required to install Linux build dependencies" >&2
            exit 1
        fi
        echo "Installing Linux compiler and GUI build dependencies..."
        $SUDO apt-get update
        $SUDO apt-get install -y \
            build-essential pkg-config libxcb-render0-dev libxcb-shape0-dev \
            libxcb-xfixes0-dev libxkbcommon-dev libssl-dev libgtk-3-dev
    else
        echo "spectrix: automatic native dependency installation currently supports apt-based Linux distributions" >&2
        echo "Install a C compiler, pkg-config, GTK 3, XKB, XCB render/shape/xfixes, and OpenSSL development packages." >&2
        exit 1
    fi
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    require_curl
    echo "Installing uv from Astral's official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || {
        echo "spectrix: uv installation completed but uv was not found on PATH" >&2
        exit 1
    }
}

install_rust() {
    if command -v rustup >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
        return
    fi
    require_curl
    echo "Installing Rust from the official rustup installer..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
    command -v cargo >/dev/null 2>&1 || {
        echo "spectrix: Rust installation completed but Cargo was not found on PATH" >&2
        exit 1
    }
}

install_native_dependencies
install_uv
install_rust

cd "$PROJECT_ROOT"
echo "Synchronizing the locked Python 3.13 environment..."
uv sync --locked
echo "Launching Spectrix..."
exec uv run --locked --no-sync python "$PROJECT_ROOT/spectrix.py" --no-sync "$@"
