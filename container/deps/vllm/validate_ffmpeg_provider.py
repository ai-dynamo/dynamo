# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that vLLM uses only Dynamo's allowlisted FFmpeg installation."""

import importlib
import importlib.metadata
import os
import re
import subprocess
from pathlib import Path

FFMPEG = Path("/usr/local/bin/ffmpeg")
ALLOWED_LIBRARY_ROOT = Path("/usr/local/lib")
SEARCH_ROOTS = (Path("/usr"), Path("/opt"), Path("/workspace"))
ALLOWED_DECODERS = {"rawvideo", "vp8", "vp9"}
ALLOWED_ENCODERS = {"h264_nvenc", "libvpx_vp9"}
REQUIRED_BUILD_CONFIGURATION = {
    "--disable-bsfs",
    "--disable-decoders",
    "--disable-demuxers",
    "--disable-encoders",
    "--disable-parsers",
    "--disable-protocols",
    "--enable-decoder=vp8,vp9,rawvideo",
    "--enable-demuxer=mov,matroska,rawvideo",
    "--enable-encoder=h264_nvenc,libvpx_vp9",
    "--enable-parser=vp8,vp9",
}
FORBIDDEN_DISTRIBUTIONS = {
    "decord",
    "decord2",
    "opencv-python-headless",
    "pynvvideocodec",
}
MEDIA_LIBRARY_RE = re.compile(
    r"^(?:libav(?:codec|device|filter|format|util)|libpostproc|"
    r"libsw(?:resample|scale)|libx26[45]|libopenh264|libfdk-aac|libfaac|"
    r"libvo-aacenc|libaacplus).*\.so(?:\..*)?$"
)
MEDIA_BINARY_RE = re.compile(r"^ff(?:mpeg|probe)(?:[-_].*)?$")
COMPONENT_RE = re.compile(r"^[A-Z.]{6}\s+(\S+)")


def run(*command: str) -> str:
    """Run a validation command and return stdout."""

    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_distributions() -> dict[str, importlib.metadata.Distribution]:
    return {
        normalize_distribution_name(distribution.metadata["Name"]): distribution
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }


def ffmpeg_components(kind: str) -> set[str]:
    output = run(str(FFMPEG), "-hide_banner", f"-{kind}")
    components = set()
    for line in output.splitlines():
        match = COMPONENT_RE.match(line.strip())
        if match is not None:
            components.add(match.group(1))
    return components


def unexpected_media_artifacts() -> list[Path]:
    artifacts = []
    for root in SEARCH_ROOTS:
        for directory, _, filenames in os.walk(root):
            directory_path = Path(directory)
            if directory_path.is_relative_to(Path("/usr/local/src/ffmpeg")):
                continue
            for filename in filenames:
                path = directory_path / filename
                if MEDIA_LIBRARY_RE.match(filename):
                    if not path.resolve().is_relative_to(ALLOWED_LIBRARY_ROOT):
                        artifacts.append(path)
                elif MEDIA_BINARY_RE.match(filename) and os.access(path, os.X_OK):
                    if path.resolve() != FFMPEG:
                        artifacts.append(path)
    return artifacts


def extension_libraries(module_name: str, pattern: str = "*.so") -> list[Path]:
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    package_root = module_file if module_file.suffix == ".so" else module_file.parent
    if package_root.is_file():
        return [package_root]
    return sorted(package_root.rglob(pattern))


def assert_links_to_dynamo_ffmpeg(module_name: str, pattern: str = "*.so") -> None:
    extensions = extension_libraries(module_name, pattern)
    if not extensions:
        raise RuntimeError(f"{module_name} contains no extension libraries to inspect")

    resolved_media_libraries = set()
    for extension in extensions:
        for line in run("ldd", str(extension)).splitlines():
            if not re.search(r"lib(?:av|sw)", line):
                continue
            if "not found" in line:
                raise RuntimeError(
                    f"unresolved FFmpeg dependency for {extension}: {line}"
                )
            match = re.search(r"=>\s+(/\S+)", line)
            if match is None:
                continue
            resolved = Path(match.group(1)).resolve()
            resolved_media_libraries.add(resolved)
            if not resolved.is_relative_to(ALLOWED_LIBRARY_ROOT):
                raise RuntimeError(
                    f"{extension} loads FFmpeg outside {ALLOWED_LIBRARY_ROOT}: {resolved}"
                )

    if not resolved_media_libraries:
        raise RuntimeError(f"{module_name} does not link to Dynamo's FFmpeg libraries")


def main() -> None:
    if not FFMPEG.is_file():
        raise RuntimeError(f"missing Dynamo FFmpeg executable: {FFMPEG}")

    expected_version = os.environ["EXPECTED_FFMPEG_VERSION"]
    first_version_line = run(str(FFMPEG), "-version").splitlines()[0]
    if not first_version_line.startswith(f"ffmpeg version {expected_version}"):
        raise RuntimeError(
            f"expected FFmpeg {expected_version}, found: {first_version_line}"
        )

    build_configuration = run(str(FFMPEG), "-buildconf")
    missing_configuration = REQUIRED_BUILD_CONFIGURATION - set(
        build_configuration.split()
    )
    if missing_configuration:
        raise RuntimeError(
            "FFmpeg is missing required allowlist configuration: "
            + ", ".join(sorted(missing_configuration))
        )

    decoders = ffmpeg_components("decoders")
    if decoders != ALLOWED_DECODERS:
        raise RuntimeError(
            f"unexpected FFmpeg decoder set: expected {sorted(ALLOWED_DECODERS)}, "
            f"found {sorted(decoders)}"
        )

    encoders = ffmpeg_components("encoders")
    if encoders != ALLOWED_ENCODERS:
        raise RuntimeError(
            f"unexpected FFmpeg encoder set: expected {sorted(ALLOWED_ENCODERS)}, "
            f"found {sorted(encoders)}"
        )

    distributions = installed_distributions()
    unexpected_distributions = FORBIDDEN_DISTRIBUTIONS & distributions.keys()
    if unexpected_distributions:
        raise RuntimeError(
            "codec-bearing upstream distributions remain installed: "
            + ", ".join(sorted(unexpected_distributions))
        )
    for required_distribution in ("av", "opencv-python", "torchcodec"):
        if required_distribution not in distributions:
            raise RuntimeError(
                f"required distribution is missing: {required_distribution}"
            )

    unexpected_artifacts = unexpected_media_artifacts()
    if unexpected_artifacts:
        raise RuntimeError(
            "FFmpeg or prohibited codec artifacts remain outside Dynamo's "
            "installation:\n"
            + "\n".join(str(path) for path in sorted(unexpected_artifacts))
        )

    for module_name in ("av", "cv2"):
        assert_links_to_dynamo_ffmpeg(module_name)
    ffmpeg_major = expected_version.partition(".")[0]
    assert_links_to_dynamo_ffmpeg("torchcodec", f"libtorchcodec_core{ffmpeg_major}.so")

    imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    selected_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if selected_ffmpeg != FFMPEG:
        raise RuntimeError(
            f"imageio-ffmpeg selected {selected_ffmpeg}, expected {FFMPEG}"
        )

    print(
        f"validated FFmpeg {expected_version}: one provider, "
        f"decoders={sorted(decoders)}, encoders={sorted(encoders)}"
    )


if __name__ == "__main__":
    main()
