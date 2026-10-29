"""Parsing helpers for the Enron email case study.

The stock-market case study needs no helpers beyond events_to_symbolic_sequences;
the brain (Allen Institute) preprocessing lives in
notebooks/brain_neuropixel_preprocessing.ipynb.
"""
import re

__all__ = [
    "extract_email_info",
    "extract_surname",
    "all_recipients_enron",
    "any_recipient_enron",
]


def _header(head, name):
    """Value of header `name` in an email's header block, including its wrapped
    continuation lines (lines starting with a space or a tab); None if absent."""
    m = re.search(rf"^{name}: (.*(?:\n[ \t].*)*)", head, re.M)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def extract_email_info(email_text):
    """
    Returns (timestamp, sender, recipients) from the headers of a raw email.

    recipients joins the To, Cc and Bcc addresses in one comma-separated string.
    Only the header block (up to the first empty line) is read, and long headers
    wrapped over several lines are read in full.
    """
    head = email_text.split("\n\n", 1)[0]
    timestamp = _header(head, "Date")
    sender = _header(head, "From")
    recipients = ", ".join(v for v in (_header(head, name) for name in ("To", "Cc", "Bcc")) if v) or None
    return timestamp, sender, recipients

def extract_surname(file_path):
    first_part = file_path.split('/')[0]  # Extracts 'allen-p' from 'allen-p/xyz/something'
    surname = first_part.split('-')[0]  # 'allen-p' → 'allen'
    surname_and_initial = first_part
    return surname.lower(), surname_and_initial


def all_recipients_enron(recipients):
    """True if the recipient list is not empty and every address in it is an @enron.com address."""
    if not isinstance(recipients, str):  # Handle NaN cases
        return False
    recipient_list = [email.strip() for email in recipients.split(',') if email.strip()]
    return bool(recipient_list) and all(r.endswith('@enron.com') for r in recipient_list)

def any_recipient_enron(recipients):
    """True if at least one recipient is an @enron.com address (looser alternative to all_recipients_enron)."""
    if not isinstance(recipients, str):
        return False
    recipient_list = [email.strip() for email in recipients.split(',') if email.strip()]
    return any(r.endswith('@enron.com') for r in recipient_list)
