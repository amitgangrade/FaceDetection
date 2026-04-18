"""Best-effort downloader for auxiliary ONNX models.

InsightFace (buffalo_l) auto-downloads on first use into ~/.insightface/models/,
so this script only needs to fetch the passive anti-spoof model (MiniFASNet
converted to ONNX).

If the download fails, the app still runs but with passive anti-spoof disabled —
the blink challenge and quality gates remain in place. To install the model
manually, place an ONNX file at `models/antispoof/antispoof.onnx`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

# Community-converted ONNX of the Silent-Face-Anti-Spoofing MiniFASNetV2.
# Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing (original .pth weights).
# This URL points to one of several community ONNX conversions — if it changes,
# override by dropping your own ONNX file at `models/antispoof/antispoof.onnx`.
CANDIDATE_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/2.7_80x80_MiniFASNetV2.onnx",
    "https://huggingface.co/tensorworks-onnx/minifasnet-antispoof/resolve/main/2.7_80x80_MiniFASNetV2.onnx",
]


def download(url: str, dest: Path) -> bool:
    try:
        print(f"Trying {url}...")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
        print(f"Downloaded to {dest} ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  failed: {e}")
        return False


def main() -> int:
    project_root = Path.cwd()
    dest = project_root / "models" / "antispoof" / "antispoof.onnx"
    if dest.exists():
        print(f"Already present: {dest}")
        return 0
    for url in CANDIDATE_URLS:
        if download(url, dest):
            return 0
    print()
    print("All download URLs failed.")
    print(f"To install manually, place a MiniFASNet (80x80) ONNX file at: {dest}")
    print("The app will run without it; passive anti-spoof will be disabled.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
