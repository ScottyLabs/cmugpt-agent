"""Email processing handler for CMUGPT.

Fetches emails labelled ``cmugpt-unread`` sent to the configured address,
generates AI responses using the same agent pipeline as the HTTP API, and
sends replies while maintaining thread context.

Uses custom Gmail labels (``cmugpt-unread`` / ``cmugpt-read``) to track
processing state independently of Gmail's native read/unread status.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from email_interface.config import EMAIL_MODEL
from email_interface.utils import (
    build_reply_message,
    extract_body_text,
    extract_html_body,
    extract_text_from_html,
    get_header,
    markdown_to_html,
    parse_email_address,
    parse_sender_name,
    quote_original_message,
    quote_original_message_html,
)

logger = logging.getLogger(__name__)

# Serialises synchronous Gmail API calls — the Google client library is
# not thread-safe, so concurrent asyncio.to_thread calls can corrupt memory.
_gmail_lock = threading.Lock()

_CMUGPT_UNREAD_LABEL = "cmugpt-unread"
_CMUGPT_READ_LABEL = "cmugpt-read"

EMAIL_SYSTEM_ADDENDUM = """\
You are communicating via email. Adopt a professional yet approachable tone.

## Email formatting (MANDATORY)
Your response_text MUST be valid HTML. NEVER use Markdown syntax in emails.
Email clients do not render Markdown — recipients see raw **, *, -, ##, []()
characters.

Use these HTML elements instead of their Markdown equivalents:
- Bold: <b>text</b> or <strong>text</strong>  (NEVER **text**)
- Italic: <i>text</i> or <em>text</em>  (NEVER *text* or _text_)
- Links: <a href="URL">text</a>  (NEVER [text](URL))
- Headings: <h3>heading</h3>  (NEVER ## heading)
- Bullet lists: <ul><li>item</li></ul>  (NEVER - item)
- Numbered lists: <ol><li>item</li></ol>  (NEVER 1. item)
- Paragraphs: <p>text</p>
- Line breaks: <br>

## Email behaviour
- Open with an appropriate greeting (e.g. "Hi <name>," or "Hello,").
- Close with a brief sign-off (e.g. "Best," or "Thanks,") — your
  signature block is added automatically, do NOT add one yourself.
- Be thorough — email readers expect more complete answers than chat users.
- Address the sender's questions or concerns directly.
- Each incoming email is prefixed with "[From: ...]" and "[Subject: ...]"
  so you know the context. Address the sender by name when appropriate.
- Before finishing, mentally review: if you see **, *, -, ##, or []()
  used for formatting, STOP and replace with equivalent HTML tags.
"""


@dataclass
class InboundEmail:
    """Parsed representation of an inbound Gmail message."""

    message_id: str
    thread_id: str
    sender_raw: str
    sender_email: str
    sender_name: str
    to: str
    subject: str
    date: str
    body_text: str
    body_html: str
    gmail_message_id_header: str
    references_header: str


@dataclass
class EmailHandler:
    """Processes inbound emails and generates AI-powered replies.

    Maintains per-thread message history so follow-up emails in the same
    Gmail thread retain conversational context.

    Accepts Google ``Credentials`` externally so the caller controls where
    tokens come from (Keycloak broker, direct OAuth, etc.).
    """

    _thread_histories: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    _processed_ids: set[str] = field(default_factory=set)
    _own_email: str | None = None
    _label_ids: dict[str, str] = field(default_factory=dict)
    _gmail_service: Any = field(default=None, repr=False)

    async def initialise(self, credentials: Credentials) -> None:
        """Build the Gmail service from *credentials*, resolve own email, and ensure labels.

        Raises ``RuntimeError`` if the authenticated user's email address
        cannot be determined — emails will only be sent from that address.
        """
        self._gmail_service = build("gmail", "v1", credentials=credentials)

        self._own_email = await self._resolve_own_email()
        if not self._own_email:
            raise RuntimeError(
                "Could not resolve the authenticated user's email address. "
                "Emails can only be sent from the authenticated user's own "
                "email. Check your Google credentials."
            )
        logger.info("Resolved own email address: %s", self._own_email)

        try:
            await self._ensure_labels()
        except Exception:
            logger.exception("Failed to ensure custom Gmail labels")

    # ------------------------------------------------------------------
    # Gmail API helpers
    # ------------------------------------------------------------------

    def _get_gmail_service(self) -> Any:
        if self._gmail_service is None:
            raise RuntimeError("EmailHandler not initialised — call initialise() first")
        return self._gmail_service

    async def _resolve_own_email(self) -> str:
        svc = self._get_gmail_service()

        def _get_profile() -> str:
            with _gmail_lock:
                profile = svc.users().getProfile(userId="me").execute()
                return profile.get("emailAddress", "")

        return await asyncio.to_thread(_get_profile)

    async def _ensure_labels(self) -> None:
        svc = self._get_gmail_service()

        def _list_labels() -> list[dict]:
            with _gmail_lock:
                resp = svc.users().labels().list(userId="me").execute()
                return resp.get("labels", [])

        def _create_label(name: str) -> dict:
            with _gmail_lock:
                body = {
                    "name": name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                }
                return svc.users().labels().create(userId="me", body=body).execute()

        existing = await asyncio.to_thread(_list_labels)
        existing_by_name = {lbl["name"]: lbl["id"] for lbl in existing}

        for label_name in (_CMUGPT_UNREAD_LABEL, _CMUGPT_READ_LABEL):
            if label_name in existing_by_name:
                self._label_ids[label_name] = existing_by_name[label_name]
                lid = self._label_ids[label_name]
                logger.debug("Found label %r → %s", label_name, lid)
            else:
                created = await asyncio.to_thread(
                    _create_label, label_name
                )
                self._label_ids[label_name] = created["id"]
                logger.info(
                    "Created Gmail label %r → %s",
                    label_name,
                    created["id"],
                )

    def _get_label_id(self, label_name: str) -> str:
        try:
            return self._label_ids[label_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Label {label_name!r} not found — did initialise() run?"
            ) from exc

    def _is_self_sent(self, email: InboundEmail) -> bool:
        if not self._own_email:
            return False
        return email.sender_email.lower() == self._own_email.lower()

    # ------------------------------------------------------------------
    # Fetch & parse
    # ------------------------------------------------------------------

    async def fetch_unread_emails(self, target_address: str) -> list[InboundEmail]:
        """Fetch unread emails with the cmugpt-unread label."""
        svc = self._get_gmail_service()
        query = f"to:{target_address} label:{_CMUGPT_UNREAD_LABEL} -from:me"

        def _search() -> list[dict]:
            with _gmail_lock:
                results = (
                    svc.users()
                    .messages()
                    .list(userId="me", q=query, maxResults=10)
                    .execute()
                )
                return results.get("messages", [])

        try:
            stubs = await asyncio.to_thread(_search)
        except Exception:
            logger.exception("Gmail API search failed for query: %s", query)
            return []

        if not stubs:
            return []

        emails: list[InboundEmail] = []
        for stub in stubs:
            msg_id = stub["id"]
            if msg_id in self._processed_ids:
                continue
            try:
                email = await self._fetch_and_parse(svc, msg_id)
                if email is None:
                    continue
                if self._is_self_sent(email):
                    self._processed_ids.add(msg_id)
                    continue
                emails.append(email)
            except Exception:
                logger.exception("Failed to fetch/parse email %s", msg_id)
        return emails

    async def _fetch_and_parse(self, svc: Any, msg_id: str) -> InboundEmail | None:
        def _get() -> dict:
            with _gmail_lock:
                return (
                    svc.users()
                    .messages()
                    .get(userId="me", id=msg_id, format="full")
                    .execute()
                )

        msg = await asyncio.to_thread(_get)
        headers = msg.get("payload", {}).get("headers", [])
        payload = msg.get("payload", {})

        sender_raw = get_header(headers, "From")
        subject = get_header(headers, "Subject")
        date = get_header(headers, "Date")
        to = get_header(headers, "To")
        message_id_header = get_header(headers, "Message-ID") or get_header(
            headers, "Message-Id"
        )
        references_header = get_header(headers, "References")

        body_text = extract_body_text(payload)
        body_html = extract_html_body(payload)

        if not body_text and not body_html:
            logger.warning("Email %s has no extractable body — skipping", msg_id)
            return None

        return InboundEmail(
            message_id=msg_id,
            thread_id=msg.get("threadId", msg_id),
            sender_raw=sender_raw,
            sender_email=parse_email_address(sender_raw),
            sender_name=parse_sender_name(sender_raw),
            to=to,
            subject=subject,
            date=date,
            body_text=body_text or "",
            body_html=body_html,
            gmail_message_id_header=message_id_header,
            references_header=references_header,
        )

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    async def process_email(self, email: InboundEmail) -> bool:
        """Generate an AI response and send it as a threaded reply.

        Returns True if the reply was sent, False otherwise.
        """
        from agent import UserInput, run_agent

        # Retrieve or start thread history
        history = self._thread_histories.get(email.thread_id, [])

        user_message = (
            f"[From: {email.sender_name} <{email.sender_email}>]\n"
            f"[Subject: {email.subject}]\n\n"
            f"{email.body_text}"
        )

        user_input = UserInput(query=user_message)

        try:
            agent_response = await run_agent(
                user_input=user_input,
                model=EMAIL_MODEL,
                message_history=history if history else None,
                system_prompt_addendum=EMAIL_SYSTEM_ADDENDUM,
            )
        except Exception:
            logger.exception("Agent error while processing email %s", email.message_id)
            return False

        response_text = agent_response.response_text or ""

        if not response_text or response_text.strip() == "__NO_REPLY__":
            logger.info("Agent chose not to reply to email %s", email.message_id)
            self._processed_ids.add(email.message_id)
            return False

        # Update thread history for future context
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})
        self._thread_histories[email.thread_id] = history

        # The agent should output HTML, but fall back to Markdown conversion
        if response_text.strip().startswith("<") or "<p>" in response_text:
            response_html = response_text
        else:
            response_html = markdown_to_html(response_text)

        response_plain = extract_text_from_html(response_html)
        if not response_plain:
            response_plain = response_text

        quoted_plain = quote_original_message(
            email.sender_raw, email.date, email.body_text
        )
        quoted_html = quote_original_message_html(
            email.sender_raw, email.date, email.body_html or None, email.body_text
        )

        reply_subject = email.subject
        if not reply_subject.lower().startswith("re:"):
            reply_subject = f"Re: {reply_subject}"

        if not self._own_email:
            logger.error(
                "Cannot send reply — authenticated user's email is not resolved"
            )
            return False

        reply_body = build_reply_message(
            from_addr=self._own_email,
            to=email.sender_email,
            subject=reply_subject,
            body_html=response_html + quoted_html,
            body_plain=response_plain + quoted_plain,
            original_message_id=email.gmail_message_id_header,
            references=email.references_header,
            thread_id=email.thread_id,
        )

        try:
            success = await self._send_reply(reply_body)
            if success:
                self._processed_ids.add(email.message_id)
                return True
            return False
        except Exception:
            logger.exception("Failed to send reply for email %s", email.message_id)
            return False

    async def _send_reply(self, message_body: dict) -> bool:
        svc = self._get_gmail_service()

        def _send() -> dict:
            with _gmail_lock:
                return (
                    svc.users()
                    .messages()
                    .send(userId="me", body=message_body)
                    .execute()
                )

        result = await asyncio.to_thread(_send)
        logger.info("Reply sent — Gmail message ID: %s", result.get("id", "?"))
        return True

    async def mark_as_read(self, message_id: str) -> None:
        """Swap cmugpt-unread → cmugpt-read."""
        svc = self._get_gmail_service()
        unread_id = self._get_label_id(_CMUGPT_UNREAD_LABEL)
        read_id = self._get_label_id(_CMUGPT_READ_LABEL)

        def _modify() -> None:
            with _gmail_lock:
                svc.users().messages().modify(
                    userId="me",
                    id=message_id,
                    body={
                        "addLabelIds": [read_id],
                        "removeLabelIds": [unread_id],
                    },
                ).execute()

        await asyncio.to_thread(_modify)
        logger.info("Marked email %s as cmugpt-read", message_id)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cleanup_old_conversations(self, max_threads: int = 200) -> None:
        """Evict oldest thread histories to prevent unbounded memory growth."""
        if len(self._thread_histories) > max_threads:
            keep = max_threads // 2
            keys = list(self._thread_histories.keys())
            for key in keys[:-keep]:
                del self._thread_histories[key]

        if len(self._processed_ids) > 5000:
            self._processed_ids = set(list(self._processed_ids)[-2500:])
