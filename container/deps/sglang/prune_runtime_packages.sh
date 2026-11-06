#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# The public SGLang image includes Debian packaging/debug tools, desktop
# dependencies and kernel-driver installers. They are not serving dependencies.
# Keep libgdrapi, the host compiler, Python headers, CUDA and RDMA libraries.
# Explicit removals avoid apt autoremove treating those runtime JIT dependencies
# as orphans. Names absent from an architecture's base image are skipped.
packages=(
    at-spi2-common autoconf automake autopoint autotools-dev bear bsdextrautils check cloc
    dbus-user-session dconf-gsettings-backend dconf-service debhelper debugedit devscripts
    dh-autoreconf dh-strip-nondeterminism dkms dwz fakeroot gdb gdrcopy gdrdrv-dkms gettext
    gettext-base 'gfortran-*' googletest groff-base
    gtk-update-icon-cache hicolor-icon-theme htop humanity-icon-theme intltool-debian kmod
    libatk-bridge2.0-0t64 libatk1.0-0t64 libatspi2.0-0t64 libavahi-client3
    libavahi-common-data libavahi-common3 libbabeltrace1 libc-ares-dev libcolord2 libdconf1
    libdebhelper-perl libdebuginfod-common libdebuginfod1t64 libdw-dev libear libegl1
    libelf-dev libevdev2 libevent-dev libfakeroot libfile-stripnondeterminism-perl libgd3
    'libgfortran-*-dev' libgrpc++-dev libgrpc-dev libgtest-dev libgtk-3-0t64 libgtk-3-common
    libgudev-1.0-0 libheif-plugin-aomdec libheif1 libhwloc-dev libinput-bin libinput10
    libjs-underscore libjsoncpp-dev libltdl-dev libltdl7 liblzma-dev libnl-genl-3-200
    libnotify4 libopenmpi-dev libpipeline1 libpmix-dev libpopt0 libqt5core5t64 libqt5dbus5t64
    libqt5gui5t64 libqt5network5t64 libqt5printsupport5t64 libqt5widgets5t64
    libsource-highlight-common libsource-highlight4t64 libsystemd-dev libtirpc-common
    libtirpc3t64 libtool libuchardet0 libunwind-dev libutempter0 libwayland-server0
    libwxbase3.2-1t64 libwxgtk3.2-1t64 libxcb-icccm4 libxcb-image0 libxcb-keysyms1
    libxcb-render-util0 libxcb-util1 libxcb-xinerama0 libxcb-xinput0 libxcb-xkb1
    libxcomposite1 libxdamage1 libxinerama1 libxkbcommon-x11-0 libxpm4 libxtst6 lsof m4
    man-db mpi-default-bin mpi-default-dev openmpi-bin openmpi-common pahole patchutils
    po-debconf protobuf-compiler-grpc python3-dev rsync silversearcher-ag tree ubuntu-mono
    uuid-dev wdiff zsh zsh-common
)

# Match the codec families removed by the vLLM runtime image as well. The
# separately built VP9-only ffmpeg is copied into the image after this cleanup.
codec_pattern='^(ffmpeg|libav[a-z]|libsw[a-z]|libpostproc|libx264|libx265|libmp3lame|libaom|libdav1d|libvpx|libtheora|libvorbis|libopus|libsoxr|libcaca|libcdio|libzvbi|libgme|libvidstab|libdc1394|libraw1394|libiec61883|libtwolame|libshine|libsrt[0-9]|libudfread|libsvtav1|libbs2b|librubberband|libchromaprint|libcodec2|libgsm|libass[0-9]|libbluray|libxvidcore|libflite)'
mapfile -t codec_packages < <(
    dpkg-query -W -f='${Package}\n' | grep -E "$codec_pattern" || true
)
mapfile -t purge < <(
    dpkg-query -W -f='${Package}\t${db:Status-Status}\n' \
        "${packages[@]}" "${codec_packages[@]}" 2>/dev/null \
        | awk '$2 != "not-installed" {print $1}' || true
)
if [[ ${#purge[@]} -eq 0 ]]; then
    exit 0
fi

# Refuse an indirect dependency cascade into the inference/JIT toolchain.
plan="$(apt-get -o APT::Get::AutomaticRemove=false --simulate purge "${purge[@]}")"
printf '%s\n' "$plan"
while read -r action package _; do
    [[ "$action" == Purg || "$action" == Remv ]] || continue
    case "${package%%:*}" in
        gcc*|g++*|cpp*|make|cmake|ninja-build|libc6*|libgcc*|libstdc++*|libgomp*|\
        python3|python3-minimal|python3.[0-9]*|libpython3.[0-9]*|cuda-*|libcublas*|libcusparse*|\
        libcusolver*|libcurand*|libcudnn*|libnccl*|nvidia-*|libibverbs*|\
        ibverbs-providers|librdmacm*|libnuma*|libgdrapi*|libfabric*)
            echo "Refusing to remove SGLang runtime dependency: $package" >&2
            exit 1
            ;;
    esac
done <<< "$plan"

apt-get -o APT::Get::AutomaticRemove=false purge -y "${purge[@]}"
for tool in cc gcc g++ make; do
    command -v "$tool" > /dev/null
done
python3 -c 'import pathlib, sysconfig; assert (pathlib.Path(sysconfig.get_path("include")) / "Python.h").is_file(), "Missing Python headers required by runtime JIT"'
