"""Run the same image twice in one process to check timing consistency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import Service
from core.config import JobConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the same image twice to inspect timing fields")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cpu", help='Device string, default "cpu"')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")

    svc = Service()
    for run_idx in (1, 2):
        result = svc.run_job(image=image, config=JobConfig(), device=args.device)
        summary_path = Path(result["summary_json"])
        with summary_path.open("r", encoding="utf-8") as f:
            content = f.read()
        print(f"\nRun#{run_idx} summary path: {summary_path}")
        print(content)


if __name__ == "__main__":
    main()
