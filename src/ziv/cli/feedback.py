import json
import socket
import threading
import time
import webbrowser
import mimetypes
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse
from importlib.resources import files

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


console = Console()


# ─── PACKAGE RESOURCE PATHS (SAFE FOR PIP INSTALL) ─────────────


TEMPLATES_DIR = files("ziv.cli.templates")
STATIC_DIR = files("ziv.static")

_FEEDBACK_DIR = Path.home() / ".ziv" / "feedback"
_REMOTE_ENDPOINT = "https://ziv-feedback-api.vercel.app/api/feedback"
_VERSION = "0.3.0"


# ─── UTIL ─────────────────────────────────────────────────────


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _save_feedback(data: dict) -> Path:
    _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _FEEDBACK_DIR / f"feedback_{ts}.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out


# ─── HANDLER ──────────────────────────────────────────────────


class _FeedbackHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # ─── STATIC FILES ───────────────────────────────

        if path.startswith("/static/"):
            self._serve_static(path.replace("/static/", ""))
            return

        # ─── HTML ──────────────────────────────────────
        try:
            html = (TEMPLATES_DIR / "feedback.html").read_text(encoding="utf-8")
            html = html.replace("__ZIV_VERSION__", self.server._version)
            html = html.replace("__REMOTE_ENDPOINT__", _REMOTE_ENDPOINT)
        except FileNotFoundError:
            self.send_error(404, "feedback.html missing from package")
            return

        self._send_response(200, "text/html; charset=utf-8",
                            html.encode("utf-8"))

    def _serve_static(self, rel_path: str):
        try:
            file = STATIC_DIR.joinpath(rel_path)
            data = file.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "Static file not found")
            return

        mime, _ = mimetypes.guess_type(rel_path)
        mime = mime or "application/octet-stream"

        self._send_response(200, mime, data)

    def do_POST(self):
        if self.path != "/feedback":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        saved_path = _save_feedback(data)

        body = json.dumps({"ok": True}).encode()

        self._send_response(200, "application/json", body)

        self.server._received = True
        self.server._saved_path = saved_path

        threading.Thread(target=self._delayed_shutdown, daemon=True).start()

    def _delayed_shutdown(self):
        time.sleep(0.6)
        self.server.shutdown()

    def _send_response(self, code, content_type, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ─── SERVER ───────────────────────────────────────────────────


class _FeedbackServer(HTTPServer):
    def __init__(self, *args, version=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._received = False
        self._saved_path = None
        self._version = version


# ─── ENTRYPOINT ───────────────────────────────────────────────


def launch_feedback(version: str = "0.1.0", timeout: int = 300) -> None:
    port = _find_free_port()
    server = _FeedbackServer(
        ("127.0.0.1", port), _FeedbackHandler, version=version)

    url = f"http://127.0.0.1:{port}/?v={version}&endpoint={_REMOTE_ENDPOINT}"

    console.print()
    console.print(Panel(
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
    ))

    threading.Timer(0.3, webbrowser.open, args=[url]).start()

    kill_timer = threading.Timer(
        timeout,
        lambda: server.shutdown()
    )
    kill_timer.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Feedback cancelled.[/dim]")
        server.shutdown()
    finally:
        kill_timer.cancel()

    if server._received:
        console.print(Panel(
            Text.assemble(
                ("✔  Feedback received\n\n", "bold green"),
                ("Saved to  ", "dim"),
                (str(server._saved_path), "cyan"),
            ),
            border_style="green",
            title="[bold cyan]ziv[/bold cyan]",
            expand=False,
        ))
    else:
        console.print("[dim]Session ended without submission.[/dim]")
