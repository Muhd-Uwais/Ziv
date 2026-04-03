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

PID_FILE = ".lfit/server.pid"
LOG_FILE = ".lfit/server.log"
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
    os.makedirs(".lfit", exist_ok=True)
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
            # Detach process = 0x00000008
            # Create new process = 0x00000200
            kwargs["creationflags"] = 0x00000008 | 0x00000200
        else:
            kwargs["preexec_fn"] = os.setsid

        process = subprocess.Popen(
            ["uvicorn", "lfit.api.embed_server:app", "--port", "8000"],
            **kwargs
        )

        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))

        # Waiting loop
        start_time = time.time()
        max_time = 200
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
            if api_data and api_data.get("model_status") == "ready":
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

            os.kill(pid, signal.SIGINT)

            time.sleep(1)
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)

            console.print("🛑 [bold red]Server stopped.[/bold red]")
            logger.info("Server stopped by user.")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            console.print(
                f"[bold red]❌ Failed to stop server PID {pid}.[/bold red]")

    logger.info("Server stopped")
