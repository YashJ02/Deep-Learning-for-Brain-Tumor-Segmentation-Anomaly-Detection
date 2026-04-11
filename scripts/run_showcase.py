"""One-command showcase launcher for the web app."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def check_health(base_url: str, timeout_seconds: float = 2.0) -> bool:
    health_url = f"{base_url}/api/health"
    try:
        with urlopen(health_url, timeout=timeout_seconds) as response:
            return int(response.status) == 200
    except (HTTPError, URLError, TimeoutError):
        return False


def wait_for_health(base_url: str, wait_seconds: float) -> bool:
    deadline = time.time() + float(wait_seconds)
    while time.time() < deadline:
        if check_health(base_url):
            return True
        time.sleep(0.5)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run web showcase with one command.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Uvicorn with auto-reload (default: true).",
    )
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the browser automatically (default: true).",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=35.0,
        help="How long to wait for server readiness.",
    )
    parser.add_argument(
        "--force-new-server",
        action="store_true",
        help="Start a new server even if one is already running on this port.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_url = f"http://localhost:{args.port}"

    if not args.force_new_server and check_health(local_url):
        log(f"A showcase server is already running at {local_url}")
        if args.open_browser:
            webbrowser.open(local_url)
            log("Opened browser to existing server")
        else:
            log("Browser auto-open disabled")
        return 0

    uvicorn_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        uvicorn_command.append("--reload")

    log("Starting showcase server")
    log("Command: " + " ".join(uvicorn_command))

    process = subprocess.Popen(uvicorn_command, cwd=str(PROJECT_ROOT))

    try:
        if not wait_for_health(local_url, wait_seconds=args.wait_seconds):
            process.terminate()
            log("Server did not become healthy within timeout")
            return 1

        log(f"Showcase is live at {local_url}")
        if args.open_browser:
            webbrowser.open(local_url)
            log("Opened browser")
        else:
            log("Browser auto-open disabled")

        log("Press Ctrl+C to stop the showcase server")
        return_code = process.wait()
        return int(return_code)
    except KeyboardInterrupt:
        log("Stopping showcase server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log("Showcase server stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
