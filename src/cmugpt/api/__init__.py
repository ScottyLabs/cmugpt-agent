"""HTTP layer.

`app` is the ASGI application uvicorn serves and `main` is the
console-script entry point that runs it under uvicorn. Route handlers live
in routes/ and the checks they share live in deps.py.
"""

from .server import app, main

__all__ = ["app", "main"]
