"""Email formatting utilities.

Handles HTML-to-text extraction, email quoting, signature generation,
and MIME message construction for replies.
"""

import base64
import html
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown as md

EMAIL_SIGNATURE_HTML = """\
<br><br>
<div style="font-size: 13px; color: #666; border-top: 1px solid #ddd; \
padding-top: 10px; margin-top: 20px;">
    <b>CMUGPT</b> | Carnegie Mellon University Assistant<br>
    Powered by <a href="https://scottylabs.org" style="color: #c41230;">\
ScottyLabs</a>
</div>"""

EMAIL_SIGNATURE_PLAIN = """
--
CMUGPT | Carnegie Mellon University Assistant
Powered by ScottyLabs — scottylabs.org"""


def extract_text_from_html(html_content: str) -> str:
    """Extract readable plain text from an HTML email body."""
    if not html_content:
        return ""
    text = html_content
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<hr\s*/?>", "\n---\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_body_text(payload: dict) -> str:
    """Recursively extract plain text from a Gmail message payload.

    Prefers text/plain; falls back to text/html with tag stripping.
    """
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if mime_type == "text/plain" and body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
    if mime_type == "text/html" and body_data:
        raw_html = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
        return extract_text_from_html(raw_html)

    for part in payload.get("parts", []):
        text = extract_body_text(part)
        if text:
            return text
    return ""


def extract_html_body(payload: dict) -> str:
    """Recursively extract HTML body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if mime_type == "text/html" and body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        result = extract_html_body(part)
        if result:
            return result
    return ""


def get_header(headers: list[dict], name: str) -> str:
    """Get a header value by name (case-insensitive)."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def parse_email_address(raw: str) -> str:
    """Extract bare email address from 'Name <email>' string."""
    match = re.search(r"<([^>]+)>", raw)
    return match.group(1) if match else raw.strip()


def parse_sender_name(raw: str) -> str:
    """Extract display name from 'Name <email>' string."""
    match = re.match(r"^(.+?)\s*<", raw)
    if match:
        return match.group(1).strip().strip('"')
    return raw.strip()


def markdown_to_html(text: str) -> str:
    """Convert Markdown text to HTML.

    Used as a fallback when the agent outputs Markdown despite being
    instructed to use HTML for email replies.
    """
    return md.markdown(text, extensions=["tables", "fenced_code"])


def quote_original_message(sender: str, date: str, body_text: str) -> str:
    """Format the original message as a quoted block for plain text replies."""
    quoted_lines = "\n".join(f"> {line}" for line in body_text.splitlines())
    return f"\n\nOn {date}, {sender} wrote:\n{quoted_lines}"


def quote_original_message_html(
    sender: str,
    date: str,
    body_html: str | None,
    body_text: str,
) -> str:
    """Format the original message as a quoted block for HTML replies."""
    if body_html:
        original_content = body_html
    else:
        escaped = html.escape(body_text)
        original_content = escaped.replace("\n", "<br>")

    return (
        '<div style="border-left: 2px solid #ccc; padding-left: 10px; '
        'margin-top: 20px; color: #555;">'
        f"<p>On {html.escape(date)}, {html.escape(sender)} wrote:</p>"
        f"{original_content}"
        "</div>"
    )


def build_reply_message(
    to: str,
    subject: str,
    body_html: str,
    body_plain: str,
    original_message_id: str,
    references: str = "",
    thread_id: str | None = None,
    from_addr: str | None = None,
) -> dict:
    """Build a Gmail API-ready reply message with proper threading headers.

    Returns a dict with ``raw`` (base64url-encoded) and optionally ``threadId``.
    """
    msg = MIMEMultipart("alternative")
    if from_addr:
        msg["From"] = f"CMUGPT <{from_addr}>"
    msg["To"] = to
    msg["Subject"] = subject
    if original_message_id:
        msg["In-Reply-To"] = original_message_id
    msg["References"] = (
        f"{references} {original_message_id}".strip()
        if references
        else (original_message_id or "")
    )

    part_plain = MIMEText(body_plain + EMAIL_SIGNATURE_PLAIN, "plain")
    part_html = MIMEText(body_html + EMAIL_SIGNATURE_HTML, "html")
    msg.attach(part_plain)
    msg.attach(part_html)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result: dict = {"raw": raw}
    if thread_id:
        result["threadId"] = thread_id
    return result
