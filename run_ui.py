"""Launcher for the Chainlit Desk on Python 3.14.

Why this exists: chainlit 2.11.0 calls ``nest_asyncio.apply()`` unconditionally
at CLI startup. On Python 3.14 that patch breaks anyio's event-loop detection,
and every starlette ``FileResponse`` route (the JS bundles, the custom CSS)
starts answering 500. Pre-empting the patch restores all routes.

Run the Desk with::

    .venv/Scripts/python run_ui.py            # http://localhost:8000

``chainlit run app.py`` still starts the server (the chat itself works), but
its static assets break under this Python/chainlit combination — always prefer
this launcher.
"""

from __future__ import annotations

import sys


def main() -> None:
    import nest_asyncio

    # Neutralize the patch BEFORE chainlit's CLI applies it.
    nest_asyncio.apply = lambda *args, **kwargs: None  # type: ignore[method-assign]

    sys.argv = [
        "chainlit",
        "run",
        "app.py",
        "--port",
        "8000",
        *sys.argv[1:],
    ]
    from chainlit.cli import cli

    cli()


if __name__ == "__main__":
    main()
