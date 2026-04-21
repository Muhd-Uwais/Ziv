"""Ziv CLI: semantic code search for your local codebase."""

import typer
import logging
import time

from importlib.metadata import version
from ziv.pipelines.index_builder import BuildIndex
from ziv.pipelines.retriever import Retriever, ServerUnavailable
from ziv.api.process_manager import start_server, stop_server, get_server_status
from ziv.core.downloader import download_model, _is_model_installed
from ziv.cli.feedback import launch_feedback

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


app = typer.Typer(
    context_settings={"help_option_names": ["--help", "-h"]},
    add_completion=False,
    help=(
        "Local File Intelligence Tool (Semantic Code Search)\n\n"
        "ziv lets you build a semantic index of your code and then\n"
        "search it with natural language queries. You index once, then\n"
        "search many times."
    ),
)
console = Console()

try:
    VERSION = version("ziv")
except Exception:
    VERSION = "0.1.0-dev"


def setup_logging(verbose: bool) -> None:
    """Configure logging level and formatter."""
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True,
                              console=console, show_path=False)],
    )


def version_callback(value: bool) -> None:
    if value:
        console.print(
            f"[bold cyan]ziv[/bold cyan] version: [green]{VERSION}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show Ziv version and exit.",
    ),
):
    """
    Main entry point for ziv.

    Examples:
      ziv init
      ziv start
      ziv build-index .
      ziv search "find config loading"
      ziv status
    """
    pass


@app.command(
    help=(
        "Download the embedding model to local storage.\n\n"
        "The embedding model is required to run ziv. This command\n"
        "downloads it once into .ziv/models/ so you can start the server.\n\n"
        "Example usage:\n"
        "  ziv init\n"
        "  ziv init -v          # with verbose logs"
    )
)
def init(
    model: str = typer.Option(
        "fast",
        "--model",
        "-m",
        help=(
            "Model variant to download. Current only model: fast.\n"
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed logs during download.",
    ),
):
    setup_logging(verbose)
    if model == "fast":
        download_model()
    else:
        console.print(
            f"[bold red]✖ Unknown model:[/bold red] '{model}'. Only model: fast."
        )
        raise typer.Exit(code=1)


@app.command(
    help=(
        "Start the background embedding server.\n\n"
        "Before running ziv search or build‑index, the server must\n"
        "be running to handle embedding requests. Use ziv status to\n"
        "check if the server is already alive.\n\n"
        "Example usage:\n"
        "  ziv start\n"
        "  ziv start -p 44291   # bind on specific port"
    )
)
def start(
    port: int = typer.Option(
        None,
        "--port",
        "-p",
        help=(
            "Port to bind the server on. If not specified, a free port\n"
            "is auto‑selected. Use this if you need a fixed port number."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed logs.",
    ),
):
    setup_logging(verbose)

    if not _is_model_installed():
        console.print(
            Panel(
                Text.assemble(
                    ("✖ Model not found\n\n", "bold red"),
                    ("Run ", "white"),
                    ("ziv init", "bold cyan"),
                    (" to download the model before starting the server.", "white"),
                ),
                border_style="red",
                title="[bold red]ziv[/bold red]",
                expand=False,
            )
        )
        raise typer.Exit(code=1)

    start_server(port)


@app.command(
    help=(
        "Build a semantic index over your codebase.\n\n"
        "ziv scans all files, splits them into chunks, and computes\n"
        "embeddings for each chunk. The index is stored in .ziv/.\n\n"
        "Typical usage:\n"
        "  ziv build-index .\n"
        "  ziv build-index src --batch-size 64"
    )
)
def build_index(
    path: str = typer.Argument(
        ".",
        help="Root path to index (default: current directory).",
    ),
    batch_size: int = typer.Option(
        128,
        "--batch-size",
        "-b",
        help=(
            "Batch size for embedding requests [32, 64, 128, 512].\n"
            "Higher values are faster but use more memory."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show internal logs.",
    ),
):
    setup_logging(verbose)

    if batch_size not in (32, 64, 128, 512):
        raise typer.BadParameter(
            "batch-size must be one of [32, 64, 128, 512]", param_hint="--batch-size"
        )

    indexer = BuildIndex()
    indexer.build_index(root_path=path, batch_size=batch_size)


@app.command(
    help=(
        "Search your indexed codebase with a natural language query.\n\n"
        "ziv embeds your query, finds the closest code chunks, and\n"
        "displays them ranked by relevance.\n\n"
        "Example usage:\n"
        "  ziv search \"where is request context handled?\"\n"
        "  ziv search -l 5 \"session management\""
    )
)
def search(
    query: str = typer.Argument(
        ...,
        help="Natural language query (no quotes needed).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show internal logs.",
    ),
    limit: int = typer.Option(
        3,
        "--limit",
        "-l",
        help="Maximum number of results to show (default: 3).",
    ),
):
    setup_logging(verbose)
    retriever = Retriever()
    try:
        results = retriever.search(query, limit)

    except ServerUnavailable:
        console.print("[red]Start server with ziv start[/red]")
        return

    console.print()

    if not results:
        console.print(
            Panel(
                (
                    "[yellow]No results found.[/yellow]\n\n"
                    "[dim]Try rephrasing your query or run [cyan]ziv build-index[/cyan] "
                    "to refresh the index.[/dim]"
                ),
                title="[bold]ziv[/bold]",
                border_style="yellow",
                padding=(1, 2),
                expand=False,
            )
        )
        return

    console.print(
        f"[bold]🔍 Query:[/bold] [italic cyan]\"{query}\"[/italic cyan]"
        f" [dim]— {len(results)} result(s) found[/dim]"
    )
    console.print()

    table = Table(
        show_header=True,
        header_style="bold dim",
        box=box.ROUNDED,
        style="honeydew2",
        padding=(0, 2),
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Score", style="yellow", justify="right", width=7)

    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        file_path = r["file_path"]
        table.add_row(str(i), file_path, f"{score:.3f}")

    console.print(table)


@app.command(
    help=(
        "Stop the background embedding server.\n\n"
        "If the server is running, this stops it cleanly. If no server\n"
        "is running, this prints that no server is found.\n\n"
        "Usage:\n"
        "  ziv stop"
    )
)
def stop(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed logs.",
    ),
):
    setup_logging(verbose)
    stop_server()


@app.command(
    help=(
        "Show the status of the background embedding server.\n\n"
        "This prints whether the server process is running, its PID,\n"
        "and the model’s readiness state. Use this to check if your\n"
        "server is ready before running ziv search or build‑index.\n\n"
        "Usage:\n"
        "  ziv status"
    )
)
@app.command()
def status():
    is_alive, pid, api_data = get_server_status()

    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    status_text = (
        "[bold green]● running[/bold green]"
        if is_alive
        else (
            "[bold red]○ offline[/bold red]\n"
            "[dim]Run [bold cyan]ziv start[/bold cyan] to launch the server.[/dim]"
        )
    )
    table.add_row("Process", status_text)

    if is_alive:
        table.add_row("PID", f"[white]{pid}[/white]")

        if api_data:
            model_status_text = (
                "[bold green]✔ ready[/bold green]"
                if api_data.get("model_status") == "Ready"
                else "[yellow]⠿ loading[/yellow]"
            )
            table.add_row("Model Status", model_status_text)
            table.add_row(
                "Model", f"[dim]{api_data.get('model_name', '')}[/dim]")
        else:
            table.add_row("API Health", "[yellow]⠿ initializing[/yellow]")

    panel = Panel(
        table,
        title="[bold cyan]ziv[/bold cyan] [dim]·[/dim] [bold white]server status[/bold white]",
        border_style="blue" if is_alive else "red",
        expand=False,
    )
    console.print(panel)


@app.command(
    help=(
        "Open a feedback form in your browser.\n\n"
        "ziv opens a local web server and your default browser so you\n"
        "can submit feedback or suggestions.\n\n"
        "Usage:\n"
        "  ziv feedback"
    )
)
def feedback():
    launch_feedback(version=VERSION)


def run_cli():
    start_time = time.time()
    try:
        app()
    finally:
        end_time = time.time()
        duration = end_time - start_time
        console.print(
            f"\n[dim]✨ Finished in [white]{duration:.2f}s[/white][/dim]")


if __name__ == "__main__":
    run_cli()
