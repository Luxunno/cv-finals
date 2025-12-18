from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore


TARGET_ORDER = ["person", "car", "bicycle", "motorcycle", "bus", "traffic light"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview YOLO GT boxes on a random subset of images."
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "custom" / "images",
        help="Directory containing images.",
    )
    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "custom" / "labels",
        help="Directory containing YOLO label files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "custom" / "preview",
        help="Directory to save preview images.",
    )
    parser.add_argument("--num", type=int, default=10, help="Number of images to preview.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def require_pillow() -> None:
    if Image is None or ImageDraw is None:
        print("Pillow is required for visualization. Please install via `pip install pillow`.", file=sys.stderr)
        sys.exit(1)


def load_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def denorm_bbox(
    bbox: Tuple[int, float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int, str]:
    cls, cx, cy, w, h = bbox
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return int(x1), int(y1), int(x2), int(y2), TARGET_ORDER[cls] if 0 <= cls < len(TARGET_ORDER) else str(cls)


def preview_image(img_path: Path, label_path: Path, output_path: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes = load_yolo_label(label_path)
    width, height = img.size
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
    font = None
    if ImageFont:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    for idx, bbox in enumerate(boxes):
        x1, y1, x2, y2, name = denorm_bbox(bbox, width, height)
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{name}"
        if font:
            try:
                # PIL >=8 provides textbbox, which is safer than textsize
                bbox_text = draw.textbbox((x1, y1), text, font=font)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            except Exception:
                tw, th = font.getsize(text)
            draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
            draw.text((x1, y1 - th), text, fill="black", font=font)
        else:
            draw.text((x1, y1), text, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    args = parse_args()
    require_pillow()
    images = sorted([p for p in args.images_dir.glob("*.jpg")])
    if not images:
        print(f"No images found in {args.images_dir}", file=sys.stderr)
        sys.exit(1)
    rng = random.Random(args.seed)
    rng.shuffle(images)
    selected = images[: args.num]
    print(f"Previewing {len(selected)} images...")
    for img_path in selected:
        label_path = args.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"Label missing for {img_path.name}, skip.")
            continue
        out_path = args.output_dir / img_path.name
        preview_image(img_path, label_path, out_path)
        print(f"Saved preview: {out_path}")


if __name__ == "__main__":
    main()
