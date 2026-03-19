"""Authentication helpers for Intelli-Credit.

Provides a lightweight OTP-based login system (email or SMS) and stores
state securely using SQLite.

This module stores OTP hashes in the local database and validates them
using an HMAC secret key, so the raw OTP never persists.

Configuration (via environment variables / .env):
- OTP_SECRET_KEY: secret used to hash OTPs (required)
- OTP_EXPIRY_SECONDS: OTP lifetime in seconds (defaults to 300)

Email (optional):
- SMTP_HOST
- SMTP_PORT
- SMTP_USER
- SMTP_PASS
- OTP_FROM_EMAIL

SMS (optional via Twilio):
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_FROM_NUMBER
"""

from __future__ import annotations

from dotenv import load_dotenv
import os
import re
import sqlite3
import time
import hmac
import hashlib
import random
import logging
import threading
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional, Tuple

# Ensure environment variables from .env are loaded when importing this module.
load_dotenv()

try:
    from twilio.rest import Client as TwilioClient
    _HAS_TWILIO = True
except ImportError:  # pragma: no cover
    _HAS_TWILIO = False

from pathlib import Path
from config import SQLITE_PATH

logger = logging.getLogger(__name__)

# Minimum OTP length
OTP_LENGTH = 6

# Basic regex validators
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[0-9]{7,15}$")

# Global lock to serialize SQLite writes across Streamlit threads/process callbacks.
_DB_LOCK = threading.RLock()


def _get_otp_secret() -> str:
    secret = os.getenv("OTP_SECRET_KEY", "")
    if not secret:
        raise ValueError("OTP_SECRET_KEY must be set in environment for secure operation.")
    return secret


def _hmac_hash(value: str, key: str) -> str:
    return hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def _now_ts() -> int:
    return int(time.time())


@dataclass
class AuthUser:
    contact: str
    contact_type: str  # 'email' or 'phone'
    created_at: int
    last_login_at: Optional[int] = None


AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", str(Path(SQLITE_PATH).with_name("auth.db")))


class AuthStore:
    """SQLite-backed auth store."""

    def __init__(self, db_path: str | os.PathLike = AUTH_DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        # Use WAL + higher timeout to reduce "database is locked" errors under concurrent Streamlit threads.
        conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _with_retry(self, fn):
        """Run a DB operation with simple retry on SQLITE_BUSY/locked and hold a process-wide lock."""
        backoff = 0.15
        for attempt in range(6):
            try:
                with _DB_LOCK:
                    return fn()
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "locked" in msg or "busy" in msg:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.7, 2.0)
                    continue
                raise
        # final attempt without catching to surface error
        with _DB_LOCK:
            return fn()

    def _init_db(self):
        def _create():
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contact TEXT NOT NULL UNIQUE,
                    contact_type TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_login_at INTEGER
                )"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS otps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contact TEXT NOT NULL,
                    contact_type TEXT NOT NULL,
                    otp_hash TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    UNIQUE(contact, contact_type, used)
                )"""
            )
            conn.commit()
            conn.close()

        self._with_retry(_create)

    def find_user(self, contact: str, contact_type: str) -> Optional[AuthUser]:
        def _select():
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM users WHERE contact = ? AND contact_type = ?", (contact, contact_type)
            )
            row = cur.fetchone()
            conn.close()
            return row

        row = self._with_retry(_select)
        if not row:
            return None
        return AuthUser(
            contact=row["contact"],
            contact_type=row["contact_type"],
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )

    def upsert_user(self, contact: str, contact_type: str) -> AuthUser:
        now = _now_ts()

        def _upsert():
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO users (contact, contact_type, created_at) VALUES (?, ?, ?)",
                (contact, contact_type, now),
            )
            cur.execute(
                "UPDATE users SET last_login_at = ? WHERE contact = ? AND contact_type = ?",
                (now, contact, contact_type),
            )
            conn.commit()
            conn.close()

        self._with_retry(_upsert)
        return AuthUser(contact=contact, contact_type=contact_type, created_at=now, last_login_at=now)

    def _store_otp(self, contact: str, contact_type: str, otp_hash: str, expires_at: int):
        now = _now_ts()

        def _insert():
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO otps (contact, contact_type, otp_hash, expires_at, used, created_at) VALUES (?, ?, ?, ?, 0, ?)",
                (contact, contact_type, otp_hash, expires_at, now),
            )
            conn.commit()
            conn.close()

        self._with_retry(_insert)

    def verify_otp(self, contact: str, contact_type: str, otp: str) -> bool:
        """Check an OTP, mark it used if valid."""
        secret = _get_otp_secret()
        otp_hash = _hmac_hash(otp, secret)
        now = _now_ts()
        def _verify():
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT id, otp_hash, expires_at, used FROM otps WHERE contact = ? AND contact_type = ? AND used = 0 ORDER BY created_at DESC LIMIT 1",
                (contact, contact_type),
            )
            row = cur.fetchone()
            if not row:
                conn.close()
                return False

            if row["used"] or now > row["expires_at"]:
                conn.close()
                return False

            valid = hmac.compare_digest(row["otp_hash"], otp_hash)
            if valid:
                cur.execute("UPDATE otps SET used = 1 WHERE id = ?", (row["id"],))
                conn.commit()
            conn.close()
            return valid

        return self._with_retry(_verify)


def _is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email.strip()))


def _is_valid_phone(phone: str) -> bool:
    return bool(_PHONE_RE.match(phone.strip()))


def _format_phone(phone: str) -> str:
    # Normalize common phone formats (remove spaces, dashes, parentheses)
    return re.sub(r"[^0-9+]", "", phone).strip()


def _get_expiry_seconds() -> int:
    try:
        return int(os.getenv("OTP_EXPIRY_SECONDS", "300"))
    except ValueError:
        return 300


def _generate_otp(length: int = OTP_LENGTH) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(length))


def _send_email(recipient: str, subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "0") or 0)
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")
    from_email = os.getenv("OTP_FROM_EMAIL", user)

    if not host or not port or not user or not password or not from_email:
        raise RuntimeError("SMTP is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, OTP_FROM_EMAIL.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = recipient
    msg.set_content(body)

    import smtplib

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)


def _send_sms(phone: str, message: str) -> None:
    if not _HAS_TWILIO:
        raise RuntimeError("Twilio package not installed. Install 'twilio' to send SMS messages.")

    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER", "")
    if not sid or not token or not from_number:
        raise RuntimeError("Twilio is not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER.")

    client = TwilioClient(sid, token)
    client.messages.create(body=message, from_=from_number, to=phone)


def send_otp(contact: str, contact_type: str) -> Tuple[bool, str]:
    """Generate and send an OTP. Returns (sent, message).

    If sending via email/SMS is not configured, the OTP is returned in the message for
    local/demo use (NOT for production).
    """
    contact_type = contact_type.lower()
    contact = contact.strip()

    if contact_type not in {"email", "phone"}:
        raise ValueError("contact_type must be 'email' or 'phone'.")

    if contact_type == "email":
        if not _is_valid_email(contact):
            raise ValueError("Invalid email address")
    else:
        contact = _format_phone(contact)
        if not _is_valid_phone(contact):
            raise ValueError("Invalid phone number")

    otp = _generate_otp()
    secret = _get_otp_secret()
    expires_at = _now_ts() + _get_expiry_seconds()
    otp_hash = _hmac_hash(otp, secret)

    store = AuthStore()
    store._store_otp(contact, contact_type, otp_hash, expires_at)

    subject = "Your Intelli-Credit Login Code"
    body = f"Your login code is: {otp}\n\nThis code will expire in {int(_get_expiry_seconds()/60)} minutes."

    # Attempt to send via configured channel; on failure, return code in message.
    if contact_type == "email":
        try:
            _send_email(contact, subject, body)
            return True, "OTP sent to your email address."
        except Exception as e:
            logger.warning("Failed to send email OTP: %s", e)
            return False, f"OTP (fallback): {otp}"

    if contact_type == "phone":
        try:
            _send_sms(contact, body)
            return True, "OTP sent via SMS."
        except Exception as e:
            logger.warning("Failed to send SMS OTP: %s", e)
            return False, f"OTP (fallback): {otp}"

    return False, "Unable to send OTP."


def is_authenticated() -> bool:
    """Utility for UI: check if Streamlit has a logged-in user."""
    try:
        import streamlit as st
    except ImportError:  # pragma: no cover
        return False
    return bool(st.session_state.get("user"))


def logout():
    """Remove user session state."""
    try:
        import streamlit as st
    except ImportError:  # pragma: no cover
        return

    for key in list(st.session_state.keys()):
        if key in {"user", "otp_sent_contact", "otp_sent_type", "otp_step"}:
            del st.session_state[key]


def get_current_user() -> Optional[AuthUser]:
    try:
        import streamlit as st
    except ImportError:  # pragma: no cover
        return None
    u = st.session_state.get("user")
    if not u:
        return None
    return AuthUser(contact=u.get("contact"), contact_type=u.get("contact_type"), created_at=u.get("created_at", 0), last_login_at=u.get("last_login_at"))
