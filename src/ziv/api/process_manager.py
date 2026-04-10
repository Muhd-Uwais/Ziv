import subprocess
import os
import sys
import signal
import logging
import time
import requests
from rich.console import Console


logger = logging.getLogger(__name__)
console = Console()

PID_FILE = ".ziv/server.pid"
LOG_FILE = ".ziv/server.log"
SERVER_URL = "http://127.0.0.1:8000"


def get_server_status():
    if not os.path.exists(PID_FILE):
        return False, None, None

    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False, None, None

    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=1)
        if response.status_code == 200:
            return True, pid, response.json()
    except:
        return True, pid, None

    return True, pid, None


def start_server():
    os.makedirs(".ziv", exist_ok=True)
    is_alive, pid, _ = get_server_status()

    if is_alive:
        console.print(
            f"[yellow]ℹ️  Server is already running (PID: {pid}).[/yellow]")
        return

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

    with console.status("[bold green]Starting AI Embedding Server... Initialization may take few moments.", spinner="arc") as status:
        log_out = open(LOG_FILE, "a")
        kwargs = {"stdout": log_out, "stderr": log_out}

        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS |
                subprocess.CREATE_NEW_PROCESS_GROUP |
                subprocess.CREATE_NO_WINDOW
            )
        else:
            kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(
                ["uvicorn", "ziv.api.embed_server:app", "--port", "8000"],
                **kwargs
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

        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))

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
                if os.path.exists(PID_FILE):
                    os.remove(PID_FILE)
                return

            # 2. Check if API is ready
            alive_now, _, api_data = get_server_status()
            if api_data and api_data.get("model_status") == "Ready":
                console.print(
                    f"✅ [bold green]Server ready![/bold green] [white](Took {time.time()-start_time:.1f}s)[/white]")
                logger.info("Server ready!")
                return

            time.sleep(1)

        console.print(
            "\n[bold yellow]⚠️  Timed out waiting for server. It might still be loading in the background.[/bold yellow]")


def stop_server():
    if not os.path.exists(PID_FILE):
        console.print("[yellow]ℹ️  No server is currently running.[/yellow]")
        return

    with console.status("[bold red]Stopping server...", spinner="bouncingBall"):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read())

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

            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)

            console.print("🛑 [bold red]Server stopped.[/bold red]")
            logger.info("Server stopped by user.")
        except FileNotFoundError:
            console.print(
                "[bold red]❌ PID file missing or invalid.[/bold red]")
        except ValueError:
            console.print("[bold red]❌ Corrupted PID file.[/bold red]")
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            console.print(
                f"[bold red]❌ Failed to stop server PID {pid}.[/bold red]")

    logger.info("Server stopped")
