from pathlib import Path
from importlib.metadata import version
from lfit.pipelines.index_builder import BuildIndex
from lfit.core.retriever import Retriever
from lfit.api.process_manage import start_server, stop_server
import typer
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["--help", "-h"]},
    help="Simple CLI interface, DEV stage Local File Intelligence Tool",
)

VERSION = version("lfit")

API = "http://localhost:8000/"


def version_callback(value: bool):
    if value:
        typer.echo(f"lfit version: {VERSION}")
        raise typer.Exit()
    

@app.callback()
def version(
    version: bool = typer.Option(
        None,
        "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show Version",
    )
):
    pass

@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show Version",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose", "-v",
        count=True,
        help="Increase verbosity (-v, -vv)"
    )
):
    # setup_logging(verbose)
    if verbose:
        typer.echo(f"Verbosity level: {verbose}")


@app.command()
def build_index(path: str = "."):
    """
    To build index
    """
    indexer = BuildIndex()

    indexer.build_index(root_path=path)


@app.command()
def search(query: str):
    """
    To search give query
    """
    s = Retriever()

    results = s.search(query)

    print(f"Found in {results[0]['file_path']}, between {results[0]['start_line'], results[0]['end_line']} lines")


@app.command()
def start():
    """
    To start server
    """
    logger.info("Server is starting. This process may take a few moments.")
    start_server()


@app.command()
def stop():
    """
    To stop server
    """
    logger.info("Shuting down.")
    stop_server()

if __name__ == "__main__":
    app()   