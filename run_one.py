"""CLI to run a single image through the PromptGuard pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from app.service import Service
from core.config import JobConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one image through PromptGuard service")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cpu", help='Device string, default "cpu"')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    service = Service()
    result = service.run_job(image=image, config=JobConfig(), device=args.device)

    output_dir = Path(result["output_dir"])
    print(f"Output directory: {output_dir}")
    if output_dir.exists():
        for p in sorted(output_dir.iterdir()):
            print(p.name)


if __name__ == "__main__":
    main()
