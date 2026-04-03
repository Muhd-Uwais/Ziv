import typer
import logging
import time
from importlib.metadata import version
from lfit.pipelines.index_builder import BuildIndex
from lfit.core.retriever import Retriever
from lfit.api.process_manager import start_server, stop_server, get_server_status
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel


app = typer.Typer(
    context_settings={"help_option_names": ["--help", "-h"]},
    add_completion=False,
    help="🚀 lfit: Local File Intelligence Tool (Semantic Code Search)",
)
console = Console()

try:
    VERSION = version("lfit")
except Exception:
    VERSION = "0.1.0-dev"


def setup_logging(verbose: bool):
    """Configures logging to show technical details only if requested."""
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True,
                              console=console, show_path=False)]
    )


def version_callback(value: bool):
    if value:
        console.print(
            f"[bold cyan]lfit[/bold cyan] version: [green]{VERSION}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show Version",
    ),
):
    """
    Main entry point for lfit. 
    Use --help to see available commands.
    """
    pass


@app.command()
def build_index(
    path: str = typer.Argument(".", help="The root directory to index"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show internal logs")
):
    """Build the semantic index of your codebase"""
    setup_logging(verbose)
    indexer = BuildIndex()

    indexer.build_index(root_path=path)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show internal logs"),
    limit: int = typer.Option(
        3, "--limit", "-l", help="Number of results to show")
):
    """Search your codebase using semantic meaning."""
    setup_logging(verbose)
    s = Retriever()

    results = s.search(query, limit)

    if not results:
        return

    console.print(
        f"[bold]Found {len(results[:3])} matches for:[/bold] [italic]'{query}'[/italic]")

    # Diplay results in the pretty table
    table = Table(title="Semantic Search Results",
                  show_header=True, header_style="bold magenta")
    table.add_column("File Path", style="cyan", no_wrap=False)
    table.add_column("Lines", style="green", justify="center")
    table.add_column("Score", style="yellow", justify="right")

    for r in results:
        table.add_row(
            r['file_path'],
            f"{r['start_line']}-{r['end_line']}",
            f"{r.get('score', 0.0):.2f}"
        )

    console.print(table)


@app.command()
def start(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Start the background embeddings server."""
    setup_logging(verbose)
    start_server()


@app.command()
def stop(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Stop the background embedding server."""
    setup_logging(verbose)
    stop_server()


@app.command()
def status():
    """View the current status of the background AI server."""
    is_alive, pid, api_data = get_server_status()

    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    status_text = "[bold green]● Active[/bold green]" if is_alive else "[bold red]○ Offline[/bold red]"
    table.add_row("Background Process", status_text)

    if is_alive:
        table.add_row("Process ID", f"[white]{pid}[/white]")

        if api_data:
            m_status = "[green]Ready[/green]" if api_data['model_status'] == "ready" else "[yellow]Loading...[/yellow]"
            table.add_row("Model Status", m_status)
            table.add_row("Model Name", f"[dim]{api_data['model_name']}[/dim]")
        else:
            table.add_row("API Health", "[yellow]Initializing API...[/yellow]")

    panel = Panel(
        table,
        title="[bold blue]LFIT System Status[/bold blue]",
        border_style="blue" if is_alive else "red",
        expand=False
    )
    console.print(panel)


def run_cli():
    start_time = time.time()
    try:
        app()
    finally:
        end_time = time.time()
        duration = end_time - start_time

        console.print(
            f"\n[dim][white]✨ Finished in {duration:.2f}s[/white][/dim]")


if __name__ == "__main__":
    run_cli()
