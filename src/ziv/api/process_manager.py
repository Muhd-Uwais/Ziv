"""Start, stop, and inspect the embedding server process."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import requests
from rich.console import Console


logger = logging.getLogger(__name__)
console = Console()

ZIV_HOME = Path.home() / ".ziv"
INSTANCE_FILE = ZIV_HOME / "server.instance"
LOG_FILE = ZIV_HOME / "server.log"
SERVER_APP = "ziv.api.embed_server:app"
SERVER_HOST = "127.0.0.1"
READY_TIMEOUT_SEC = 30
HEALTH_TIMEOUT_SEC = 1


def _find_free_port() -> int:
    """Return an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((SERVER_HOST, 0))
        return sock.getsockname()[1]


def _is_port_in_use(port: int) -> bool:
    """Return True when the port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((SERVER_HOST, port))
            return False
        except OSError:
            return True


def _read_instance() -> tuple[int | None, int | None, str | None]:
    """Read the persisted server process metadata."""
    if not INSTANCE_FILE.exists():
        return None, None, None

    try:
        pid_s, port_s, url = INSTANCE_FILE.read_text(
            encoding="utf-8").splitlines()[:3]
        return int(pid_s), int(port_s), url
    except (ValueError, IndexError, OSError):
        return None, None, None


def _write_instance(pid: int, port: int, url: str) -> None:
    """Persist the running server metadata."""
    ZIV_HOME.mkdir(parents=True, exist_ok=True)
    INSTANCE_FILE.write_text(f"{pid}\n{port}\n{url}\n", encoding="utf-8")


def _remove_instance_file() -> None:
    """Delete the instance file if it exists."""
    try:
        INSTANCE_FILE.unlink()
    except FileNotFoundError:
        pass


def _process_is_alive(pid: int) -> bool:
    """Check whether a PID still exists."""
    return pid > 0 and psutil.pid_exists(pid)


def get_server_url() -> str | None:
    """Return the current server URL if the recorded PID is alive."""
    pid, _, url = _read_instance()
    if pid is None or url is None:
        return None
    return url if _process_is_alive(pid) else None


def get_server_status() -> tuple[bool, int | None, dict[str, Any] | None]:
    """Return server liveness and health payload if available."""
    pid, _, url = _read_instance()
    if pid is None or url is None or not _process_is_alive(pid):
        return False, None, None

    try:
        response = requests.get(f"{url}health", timeout=HEALTH_TIMEOUT_SEC)
        if response.ok:
            return True, pid, response.json()
    except requests.RequestException:
        pass

    return True, pid, None


def _build_subprocess_kwargs(log_file) -> dict[str, Any]:
    """Build platform-specific subprocess options."""
    kwargs: dict[str, Any] = {
        "stdout": log_file,
        "stderr": log_file,
        "close_fds": True,
    }

    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        kwargs.update(
            {
                "startupinfo": startupinfo,
                "creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            }
        )

    return kwargs


def start_server(port: int | None = None) -> None:
    """Start the embedding server in the background."""
    ZIV_HOME.mkdir(parents=True, exist_ok=True)

    is_alive, pid, _ = get_server_status()
    if is_alive:
        _, _, url = _read_instance()
        console.print(
            f"[yellow]ℹ️  Server already running (PID: {pid}) at {url}[/yellow]")
        return

    _remove_instance_file()

    if port is None:
        resolved_port = _find_free_port()
    else:
        if _is_port_in_use(port):
            console.print(
                f"[bold red]❌ Port {port} is already in use.[/bold red] "
                "Use a different port or omit [cyan]--port[/cyan] to auto-select."
            )
            return
        resolved_port = port

    server_url = f"http://{SERVER_HOST}:{resolved_port}/"

    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        try:
            with console.status(
                "[bold green]Starting AI Embedding Server...[/bold green]",
                spinner="arc",
            ):
                process = subprocess.Popen(
                    [
                        "uvicorn",
                        SERVER_APP,
                        "--host",
                        SERVER_HOST,
                        "--port",
                        str(resolved_port),
                    ],
                    **_build_subprocess_kwargs(log_file),
                )

                _write_instance(process.pid, resolved_port, server_url)
                console.print(f"[dim]→ Binding on {server_url}[/dim]")

                deadline = time.monotonic() + READY_TIMEOUT_SEC
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        console.print(
                            "\n[bold red]❌ Server failed to start.[/bold red]")
                        console.print(
                            f"🔍 Check logs: [cyan]cat {LOG_FILE}[/cyan]")
                        _remove_instance_file()
                        return

                    is_running, _, health = get_server_status()
                    if is_running and health and health.get("model_status") == "Ready":
                        elapsed = READY_TIMEOUT_SEC - \
                            max(0.0, deadline - time.monotonic())
                        console.print(
                            f"✅ [bold green]Server ready![/bold green] "
                            f"[white](Took {elapsed:.1f}s, port {resolved_port})[/white]"
                        )
                        logger.info("Server ready at %s", server_url)
                        return

                    time.sleep(1)

                console.print(
                    "\n[bold yellow]⚠️ Timed out waiting for server. "
                    "It may still be loading.[/bold yellow]"
                )

        except PermissionError:
            logger.exception("Permission denied while starting uvicorn")
            console.print(
                "[bold red]❌ Permission denied.[/bold red] "
                "Try running with elevated privileges."
            )
            _remove_instance_file()
        except OSError as exc:
            logger.exception("OS error while starting server")
            console.print(f"[bold red]❌ System error:[/bold red] {exc}")
            _remove_instance_file()
        except Exception as exc:
            logger.exception("Unexpected error while starting server")
            console.print(f"[bold red]❌ Unexpected error:[/bold red] {exc}")
            _remove_instance_file()


def _terminate_pid(pid: int) -> None:
    """Terminate a process gracefully, then force kill if needed."""
    if sys.platform == "win32":

        result = subprocess.run(
            ["taskkill", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + 3
        while time.time() < deadline:
            if not _process_is_alive(pid):
                return
            time.sleep(0.2)

        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.time() + 2
    while time.time() < deadline:
        if not _process_is_alive(pid):
            return
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def stop_server() -> None:
    """Stop the running embedding server."""
    pid, _, _ = _read_instance()

    if pid is None:
        console.print("[yellow]ℹ️  No server is currently running.[/yellow]")
        return

    if not _process_is_alive(pid):
        console.print(
            "[yellow]ℹ️  Recorded server PID is already dead.[/yellow]")
        _remove_instance_file()
        return

    with console.status("[bold red]Stopping server...[/bold red]", spinner="bouncingBall"):
        try:
            _terminate_pid(pid)
            _remove_instance_file()
            console.print("🛑 [bold red]Server stopped.[/bold red]")
            logger.info("Server stopped by user.")
        except Exception as exc:
            logger.exception("Error stopping server")
            console.print(
                f"[bold red]❌ Failed to stop server PID {pid}:[/bold red] {exc}")
