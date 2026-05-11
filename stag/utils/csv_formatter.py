# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — utils.csv_formatter                                      ║
# ║  « small CSV pretty-printing helpers »                           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Tiny helpers for one-off CSV exports inside scripts.            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""CSV-formatted log handler for Python logging."""

import logging
import csv
import io

class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record):
        self.writer.writerow([record.levelname, record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()
