from typing import Optional

import typer

import joernti.llm

app = typer.Typer()


@app.command(name="llm", help="Load an arbitrary Large Language Machine Learning model for type inference")
def llm(
        checkpoint: Optional[str] = typer.Argument(None, help="The path to the checkpoint to use."),
        input_slice: Optional[str] = typer.Option(None, help="The input usage slice to infer types from."),
        language: Optional[str] = typer.Option(None, help="The programming language that object to inference."),
        run_as_server: Optional[bool] = typer.Option(False, help="Run the model as a server."),
        port: Optional[int] = typer.Option(1337, help="The port to run the server on."),
):
    if input_slice is None and not run_as_server:
        print("[!] No input specified and also not running in server mode! Bailing out...")
        exit(1)

    joernti.llm.main(input_slice, checkpoint=checkpoint, language=language, server_mode=run_as_server,
                     port=port)


@app.command(name="codetidal5",
             help="CodeT5-based LLM model for JavaScript type inference. Will download `codetidal5` model checkpoint if no local model is specified.")
def codeTidal5(
        checkpoint: Optional[str] = typer.Argument("joernio/codetidal5",
                                                   help="The path to the checkpoint to use."),
        input_slice: Optional[str] = typer.Option(None, help="The input usage slice to infer types from."),
        run_as_server: Optional[bool] = typer.Option(False, help="Run the model as a server."),
        port: Optional[int] = typer.Option(1337, help="The port to run the server on."),
):
    if input_slice is None and not run_as_server:
        print("[!] No input specified and also not running in server mode! Bailing out...")
        exit(1)

    joernti.llm.main(input_slice, checkpoint=checkpoint, language="Javascript", server_mode=run_as_server,
                     port=port)


@app.command(help="Print joernti version")
def version():
    from joernti import __version__
    print("joernti \"{}\"".format(__version__))


def main():
    """CLI entrypoint for joernti"""
    app()
