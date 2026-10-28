"""Logging set-up: plain lines in a terminal, and real priorities in the
systemd journal, so `journalctl -u nuc-assistant -p warning` shows problems."""

import logging
import os
import sys

# syslog priorities, which journald reads from a "<N>" prefix on each line
PRIORITIES = {
    logging.DEBUG: 7,
    logging.INFO: 6,
    logging.WARNING: 4,
    logging.ERROR: 3,
    logging.CRITICAL: 2,
}


class JournalFormatter(logging.Formatter):
    def format(self, record):
        priority = PRIORITIES.get(record.levelno, 6)
        # One prefix per line, so a traceback keeps its priority too
        return "\n".join(f"<{priority}>{line}" for line in super().format(record).splitlines())


def setup(debug: bool = False):
    handler = logging.StreamHandler(sys.stdout)
    # systemd sets JOURNAL_STREAM when stdout goes to the journal
    handler.setFormatter(JournalFormatter() if os.environ.get("JOURNAL_STREAM")
                         else logging.Formatter())
    root = logging.getLogger("nuc_assistant")
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.propagate = False
