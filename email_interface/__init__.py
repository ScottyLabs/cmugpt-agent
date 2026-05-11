"""Email interface for CMUGPT.

Provides a background worker that polls Gmail for incoming emails
and responds using the same agent pipeline as the HTTP API.
"""

from email_interface.handler import EmailHandler
from email_interface.worker import EmailWorker

__all__ = ["EmailHandler", "EmailWorker"]
