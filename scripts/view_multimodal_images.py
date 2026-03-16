"""
View images from PersonaMem multimodal chat histories.

Multimodal = chat history messages can contain both text and images.
Images are stored as base64 in messages where content is a list of parts
with type "image_url" and data:image/...;base64,...

Usage:
  python view_multimodal_images.py                    # print where images are
  python view_multimodal_images.py --show 1          # show first image (saves to file or opens)
"""

import argparse
import base64
import json
from pathlib import Path

# Default cache path for PersonaMem-v2 (adjust if your HF cache is elsewhere)
HF_CACHE = Path.home() / ".cache/huggingface/hub/datasets--bowen-upenn--PersonaMem-v2/snapshots"
MULTIMODAL_DIR = "data/chat_history_multimodal_32k"


def find_snapshot_dir() -> Path | None:
    if not HF_CACHE.exists():
        return None
    for p in HF_CACHE.iterdir():
        if p.is_dir():
            chat_dir = p / MULTIMODAL_DIR
            if chat_dir.exists():
                return p
    return None


def iter_images_in_chat(chat_path: Path):
    """Yield (message_index, part_index, b64_data_url) from a chat history JSON."""
    with open(chat_path) as f:
        data = json.load(f)
    for i, msg in enumerate(data.get("chat_history", [])):
        c = msg.get("content")
        if not isinstance(c, list):
            continue
        for j, part in enumerate(c):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url", "")
                if isinstance(url, str) and url.startswith("data:image"):
                    yield i, j, url


def b64_to_image_path(data_url: str, out_path: Path) -> Path:
    """Decode data:image/...;base64,... and save to out_path. Returns out_path."""
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    header, b64 = data_url.split(",", 1)
    ext = "png"
    if "image/jpeg" in header or "image/jpg" in header:
        ext = "jpg"
    elif "image/png" in header:
        ext = "png"
    if not out_path.suffix:
        out_path = out_path.with_suffix(f".{ext}")
    raw = base64.b64decode(b64)
    out_path.write_bytes(raw)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="View images from PersonaMem multimodal chat histories")
    ap.add_argument("--show", type=int, default=None, metavar="N", help="Decode and save the Nth image (1-based) from the first chat file that has images")
    ap.add_argument("--out", type=Path, default=Path("multimodal_sample_image.jpg"), help="Output path for --show")
    args = ap.parse_args()

    snap = find_snapshot_dir()
    if not snap:
        print("PersonaMem-v2 cache not found at", HF_CACHE)
        return
    chat_dir = snap / MULTIMODAL_DIR
    print("Multimodal chat histories:", chat_dir)

    if args.show is not None:
        n = args.show
        for chat_file in sorted(chat_dir.glob("*.json")):
            for mi, pj, data_url in iter_images_in_chat(chat_file):
                n -= 1
                if n == 0:
                    out = Path(args.out)
                    b64_to_image_path(data_url, out)
                    print("Saved image to", out.absolute())
                    return
        print("Not enough images; try a smaller --show N")
        return

    # List which files have images
    total = 0
    for chat_file in sorted(chat_dir.glob("*.json"))[:20]:
        count = sum(1 for _ in iter_images_in_chat(chat_file))
        if count:
            print(chat_file.name, "->", count, "image(s)")
            total += count
    print("... (showing first 20 files with images)")
    print("\nTo view an image: python view_multimodal_images.py --show 1 --out first_image.jpg")


if __name__ == "__main__":
    main()
