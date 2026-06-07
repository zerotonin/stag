# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — utils.csv_formatter                                      ║
# ║  « small CSV pretty-printing helpers »                           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Tiny helpers for one-off CSV exports inside scripts.            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""CSV-formatted log handler for Python logging."""

import csv
import io
import logging


class CsvFormatter(logging.Formatter):
    """``logging.Formatter`` that emits each record as one CSV row.

    Each row is ``"<LEVELNAME>,<message>"`` with both fields fully
    quoted so embedded commas in the message are preserved.  Useful
    for log files that downstream pandas / Excel readers want to
    consume directly.
    """

    def __init__(self):
        """Create the StringIO-backed CSV writer."""
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record):
        """Render one ``logging.LogRecord`` as a CSV-formatted string."""
        self.writer.writerow([record.levelname, record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()
