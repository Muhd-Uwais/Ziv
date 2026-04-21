"""Local feedback server for the Ziv CLI."""

from __future__ import annotations

import json
import mimetypes
import socket
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


console = Console()

TEMPLATES_DIR = files("ziv.cli.templates")
STATIC_DIR = files("ziv.static")

FEEDBACK_DIR = Path.home() / ".ziv" / "feedback"
REMOTE_ENDPOINT = "https://ziv-feedback-api.vercel.app/api/feedback"
DEFAULT_TIMEOUT_SEC = 300
BROWSER_OPEN_DELAY_SEC = 0.3
SHUTDOWN_DELAY_SEC = 0.6


def _find_free_port() -> int:
    """Return an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _save_feedback(data: dict[str, Any]) -> Path:
    """Persist submitted feedback to the local feedback directory."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = FEEDBACK_DIR / f"feedback_{timestamp}.json"
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


class _FeedbackServer(HTTPServer):
    """HTTP server carrying feedback session state."""

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        version: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.received = False
        self.saved_path: Path | None = None
        self.version = version


class _FeedbackHandler(BaseHTTPRequestHandler):
    """Serve the feedback page, static assets, and local save endpoint."""

    server: _FeedbackServer

    def log_message(self, format: str, *args: Any) -> None:
        """Silence default HTTP request logging."""
        return

    def do_GET(self) -> None:
        """Serve the feedback page or bundled static assets."""
        path = urlparse(self.path).path

        if path.startswith("/static/"):
            self._serve_static(path.removeprefix("/static/"))
            return

        self._serve_feedback_page()

    def do_POST(self) -> None:
        """Handle feedback submission and trigger shutdown."""
        if self.path != "/feedback":
            self.send_error(404, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        saved_path = _save_feedback(payload)
        self._send_bytes(
            status_code=200,
            content_type="application/json; charset=utf-8",
            body=json.dumps({"ok": True}).encode("utf-8"),
        )

        self.server.received = True
        self.server.saved_path = saved_path

        threading.Thread(target=self._shutdown_later, daemon=True).start()

    def _serve_feedback_page(self) -> None:
        """Render the packaged feedback HTML template."""
        try:
            html = (TEMPLATES_DIR / "feedback.html").read_text(encoding="utf-8")
        except FileNotFoundError:
            self.send_error(404, "feedback.html missing from package")
            return

        html = html.replace("__ZIV_VERSION__", self.server.version)
        html = html.replace("__REMOTE_ENDPOINT__", REMOTE_ENDPOINT)

        self._send_bytes(
            status_code=200,
            content_type="text/html; charset=utf-8",
            body=html.encode("utf-8"),
        )

    def _serve_static(self, relative_path: str) -> None:
        """Serve a bundled static asset."""
        try:
            asset = STATIC_DIR.joinpath(relative_path)
            body = asset.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "Static file not found")
            return

        content_type = mimetypes.guess_type(
            relative_path)[0] or "application/octet-stream"
        self._send_bytes(status_code=200, content_type=content_type, body=body)

    def _shutdown_later(self) -> None:
        """Shut down the server shortly after a successful submission."""
        time.sleep(SHUTDOWN_DELAY_SEC)
        self.server.shutdown()

    def _send_bytes(self, *, status_code: int, content_type: str, body: bytes) -> None:
        """Send a byte response with headers."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def launch_feedback(version: str = "0.3.0", timeout: int = DEFAULT_TIMEOUT_SEC) -> None:
    """Launch the local browser-based feedback flow."""
    port = _find_free_port()
    server = _FeedbackServer(
        ("127.0.0.1", port), _FeedbackHandler, version=version)
    url = f"http://127.0.0.1:{port}/?v={version}&endpoint={REMOTE_ENDPOINT}"

    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("Opening feedback form in your browser\n\n", "bold white"),
                ("URL  ", "dim"),
                (f"http://127.0.0.1:{port}\n", "cyan"),
                ("Timeout  ", "dim"),
                (f"{timeout}s  ", "white"),
                ("·  ", "dim"),
                ("Ctrl+C to cancel", "dim"),
            ),
            title="[bold cyan]ziv[/bold cyan] [dim]·[/dim] [bold white]feedback[/bold white]",
            border_style="blue",
            expand=False,
        )
    )

    threading.Timer(BROWSER_OPEN_DELAY_SEC,
                    webbrowser.open, args=[url]).start()

    shutdown_timer = threading.Timer(timeout, server.shutdown)
    shutdown_timer.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Feedback cancelled.[/dim]")
        server.shutdown()
    finally:
        shutdown_timer.cancel()
        server.server_close()

    if server.received:
        console.print(
            Panel(
                Text.assemble(
                    ("✔  Feedback received\n\n", "bold green"),
                    ("Saved to  ", "dim"),
                    (str(server.saved_path), "cyan"),
                ),
                title="[bold cyan]ziv[/bold cyan]",
                border_style="green",
                expand=False,
            )
        )
    else:
        console.print("[dim]Session ended without submission.[/dim]")
