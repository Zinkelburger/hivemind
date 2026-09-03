#!/usr/bin/env python3
"""Stage a redistributable ONNX Runtime engine bundle.

The shipped layout is deliberately flat -- the engine and the inference runtime
side by side in one directory -- because that is what the RPATH baked in by
`engine/CMakeLists.txt` (`$ORIGIN` / `@loader_path`) resolves against, and what
Windows' "look next to the .exe" DLL search finds. A consumer therefore needs
no environment variables to launch the engine.

    python3 tools/package_engine.py --build engine/build-ort --target linux-x64

Writes dist/hivemind-<target>.{tar.gz,zip} plus a .sha256 beside it.
"""
import argparse
import hashlib
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# target -> (engine name in the bundle, runtime glob in third_party/onnxruntime/lib)
TARGETS = {
    "linux-x64":     ("hivemind",     "libonnxruntime.so.*.*"),
    "windows-x64":   ("hivemind.exe", "onnxruntime.dll"),
    "macos-arm64":   ("hivemind",     "libonnxruntime.*.dylib"),
    "macos-x86_64":  ("hivemind",     "libonnxruntime.*.dylib"),
}

# The name the loader actually asks for, which is the SONAME/install-name --
# not the versioned filename the archive happens to ship.
RUNTIME_INSTALL_NAME = {
    "linux-x64":    "libonnxruntime.so.1",
    "windows-x64":  "onnxruntime.dll",
    "macos-arm64":  "libonnxruntime.dylib",
    "macos-x86_64": "libonnxruntime.dylib",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_target() -> str:
    system, machine = platform.system(), platform.machine().lower()
    if system == "Windows":
        return "windows-x64"
    if system == "Darwin":
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x86_64"
    return "linux-x64"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="engine/build-ort", type=Path,
                    help="CMake build directory holding hivemind.bin")
    ap.add_argument("--onnxruntime", default="third_party/onnxruntime", type=Path,
                    help="vendored ONNX Runtime root")
    ap.add_argument("--target", default=None, choices=sorted(TARGETS),
                    help="platform slug (default: this machine's)")
    ap.add_argument("--out", default="dist", type=Path)
    a = ap.parse_args()

    target = a.target or default_target()
    engine_name, runtime_glob = TARGETS[target]

    build = (ROOT / a.build) if not a.build.is_absolute() else a.build
    ort = (ROOT / a.onnxruntime) if not a.onnxruntime.is_absolute() else a.onnxruntime

    # CMake names the executable hivemind.bin on POSIX and hivemind.bin.exe
    # under mingw; take whichever exists.
    engine = next((p for p in (build / "hivemind.bin",
                               build / "hivemind.bin.exe",
                               build / "hivemind.exe") if p.is_file()), None)
    if engine is None:
        print(f"No engine binary in {build}", file=sys.stderr)
        return 1

    runtime = next(iter(sorted((ort / "lib").glob(runtime_glob))), None)
    if runtime is None:
        print(f"No runtime matching {runtime_glob} in {ort / 'lib'}", file=sys.stderr)
        return 1

    out = (ROOT / a.out) if not a.out.is_absolute() else a.out
    stage = out / f"hivemind-{target}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    shutil.copy2(engine, stage / engine_name)
    (stage / engine_name).chmod(0o755)
    shutil.copy2(runtime, stage / RUNTIME_INSTALL_NAME[target])
    shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")

    for name, path in [("engine", stage / engine_name),
                       ("runtime", stage / RUNTIME_INSTALL_NAME[target])]:
        print(f"  {name:8} {path.name:24} {path.stat().st_size / 1e6:7.1f} MB")

    if target == "windows-x64":
        archive = out / f"hivemind-{target}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(stage.iterdir()):
                zf.write(path, path.name)
    else:
        archive = out / f"hivemind-{target}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=9) as tf:
            for path in sorted(stage.iterdir()):
                info = tf.gettarinfo(path, arcname=path.name)
                # Reproducible-ish: the mtime and owner of a CI checkout are noise.
                info.mtime, info.uid, info.gid = 0, 0, 0
                info.uname = info.gname = ""
                with path.open("rb") as fh:
                    tf.addfile(info, fh)

    digest = sha256(archive)
    (out / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"\n{archive.relative_to(ROOT) if out.is_relative_to(ROOT) else archive}"
          f"  {archive.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
