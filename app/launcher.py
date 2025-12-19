"""PromptGuard EXE launcher.

Responsibilities:
- Prepare runtime environment for Gradio/Ultralytics in packaged EXE.
- Provide --selfcheck mode for build verification.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _setup_env() -> None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        local_app_data = str(Path.home() / "AppData" / "Local")

    base = Path(local_app_data) / "PromptGuard"
    tmp_dir = base / "tmp"
    settings_dir = base / "ultralytics"
    _ensure_dir(tmp_dir)
    _ensure_dir(settings_dir)

    os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
    os.environ["GRADIO_TEMP_DIR"] = str(tmp_dir)
    os.environ["ULTRALYTICS_SETTINGS_DIR"] = str(settings_dir)
    os.environ.setdefault("PROMPTGUARD_OUTPUT_DIR", str(Path.cwd() / "outputs"))


def main() -> int:
    parser = argparse.ArgumentParser(description="PromptGuard launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    parser.add_argument("--selfcheck", action="store_true", help="Start UI and exit after a short check")
    args = parser.parse_args()

    _setup_env()

    from app import ui

    demo = ui.build_interface()
    if args.selfcheck:
        demo.launch(server_name=args.host, server_port=args.port, inbrowser=False, prevent_thread_lock=True)
        print(f"SELFHECK_OK http://{args.host}:{args.port}")
        time.sleep(2.5)
        try:
            demo.close()
        except Exception:
            pass
        return 0

    demo.launch(server_name=args.host, server_port=args.port, inbrowser=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
