import subprocess
import os
import sys
import signal
import logging
import time
import requests
import socket
from rich.console import Console


logger = logging.getLogger(__name__)
console = Console()

ZIV_HOME =  os.path.join(os.path.expanduser("~"), ".ziv")
INSTANCE_FILE = os.path.join(ZIV_HOME, "server.instance")
LOG_FILE = os.path.join(ZIV_HOME, "server.log")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def _read_instance():
    if not os.path.exists(INSTANCE_FILE):
        return None, None, None
    try:
        with open(INSTANCE_FILE, "r") as f:
            lines = f.read().strip().splitlines()

        pid = int(lines[0])
        port = int(lines[1])
        url = lines[2]
        return pid, port, url
    except (ValueError, IndexError, OSError):
        return None, None, None


def _write_instance(pid: int, port: int, url: str):
    os.makedirs(".ziv", exist_ok=True)
    with open(INSTANCE_FILE, "w") as f:
        f.write(f"{pid}\n{port}\n{url}\n")


def get_server_url() -> str | None:
    pid, _, url = _read_instance()

    try:
        os.kill(pid, 0)
        return url
    except OSError:
        return None


def get_server_status():
    pid, port, url = _read_instance()

    if pid is None:
        return False, None, None

    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False, None, None

    try:
        response = requests.get(f"{url}/health", timeout=1)
        if response.status_code == 200:
            return True, pid, response.json()
    except:
        return True, pid, None

    return True, pid, None


def start_server(port: int | None = None):
    os.makedirs(".ziv", exist_ok=True)
    is_alive, pid, _ = get_server_status()

    if is_alive:
        _, _, existing_url = _read_instance()
        console.print(
            f"[yellow]ℹ️  Server is already running (PID: {pid}) at {existing_url}[/yellow]"
        )
        return

    if os.path.exists(INSTANCE_FILE):
        os.remove(INSTANCE_FILE)

    if port is not None:
        if _is_port_in_use(port):
            console.print(
                f"[bold red]❌ Port {port} is already in use.[/bold red] "
                "Use a different port or omit [cyan]--port[/cyan] to auto-select."
            )
            return
        resolved_port = port
    else:
        resolved_port = _find_free_port()

    server_url = f"http://127.0.0.1:{resolved_port}/"

    with console.status(
        "[bold green]Starting AI Embedding Server... Initialization may take a few moments.",
        spinner="arc",
    ):
        log_out = open(LOG_FILE, "a")
        kwargs = {"stdout": log_out, "stderr": log_out}

        kwargs["close_fds"] = True

        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE  # force-hide any window the child opens

            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW        # no console window, ever
                | subprocess.CREATE_NEW_PROCESS_GROUP  # isolate Ctrl+C signal group
            )
            kwargs["startupinfo"] = si

        try:
            process = subprocess.Popen(
                [
                    "uvicorn",
                    "ziv.api.embed_server:app",
                    "--host", "127.0.0.1",
                    "--port", str(resolved_port),
                ],
                **kwargs,
            )
        except PermissionError:
            logger.error(
                "Permission denied when trying to start uvicorn process.")
            console.print(
                "[bold red]❌ Permission denied.[/bold red] "
                "Try running with elevated privileges."
            )
            return
        except OSError as e:
            logger.error("OS error while starting server: %s", e)
            console.print(f"[bold red]❌ System error:[/bold red] {e}")
            return
        except Exception as e:
            logger.exception(
                "Unexpected error while starting the embedding server.")
            console.print(f"[bold red]❌ Unexpected error:[/bold red] {e}")
            return

        _write_instance(process.pid, resolved_port, server_url)
        console.print(f"[dim]  → Binding on {server_url}[/dim]")

        # Waiting loop
        start_time = time.time()
        max_time = 30
        while (time.time() - start_time) < max_time:

            if process.poll() is not None:
                console.print(
                    "\n[bold red]❌ Server failed to start.[/bold red]")
                logger.error("Server failed to start")
                console.print(
                    f"🔍 Check the logs for details: [cyan]cat {LOG_FILE}[/cyan]")
                if os.path.exists(INSTANCE_FILE):
                    os.remove(INSTANCE_FILE)
                return

            # 2. Check if API is ready
            _, _, api_data = get_server_status()
            if api_data and api_data.get("model_status") == "Ready":
                elapsed = time.time() - start_time
                console.print(
                    f"✅ [bold green]Server ready![/bold green] "
                    f"[white](Took {elapsed:.1f}s, port {resolved_port})[/white]"
                )
                logger.info("Server ready at %s", server_url)
                return

            time.sleep(1)

        console.print(
            "\n[bold yellow]⚠️  Timed out waiting for server. "
            "It might still be loading in the background.[/bold yellow]"
        )


def stop_server():
    if not os.path.exists(INSTANCE_FILE):
        console.print("[yellow]ℹ️  No server is currently running.[/yellow]")
        return

    pid, port, url = _read_instance()
    if pid is None:
        console.print(
            "[bold red]❌ Corrupted instance file. Removing it.[/bold red]")

        os.remove(INSTANCE_FILE)
        return

    with console.status("[bold red]Stopping server...", spinner="bouncingBall"):
        try:
            if sys.platform == "win32":
                # On windows: os.kill with SIGINT is unrelaible.
                # Use CTRL_C_EVENT
                try:
                    os.kill(pid, signal.CTRL_C_EVENT)
                except (OSError, KeyboardInterrupt):
                    pass

                time.sleep(1)
                try:
                    os.kill(pid, 0)
                    # Still running - force terminate
                    subprocess.call(
                        ["taskkill", "/F", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except OSError:
                    pass
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

            if os.path.exists(INSTANCE_FILE):
                os.remove(INSTANCE_FILE)

            console.print("🛑 [bold red]Server stopped.[/bold red]")
            logger.info("Server stopped by user.")
        except FileNotFoundError:
            console.print(
                "[bold red]❌ PID file missing or invalid.[/bold red]")
        except ValueError:
            console.print("[bold red]❌ Corrupted PID file.[/bold red]")
            if os.path.exists(INSTANCE_FILE):
                os.remove(INSTANCE_FILE)
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            console.print(
                f"[bold red]❌ Failed to stop server PID {pid}.[/bold red]")

    logger.info("Server stopped")
