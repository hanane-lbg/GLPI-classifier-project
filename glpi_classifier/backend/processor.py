import re
from bs4 import BeautifulSoup


class GLPITicketProcessor:

    def __init__(self):

        self.patterns = {
            "html_tags": r"[^<]+<|&[a-zA-Z0-9]+;",
            "urls": r"https?://\S|www\.\S+",
            "path": r"(?:[a-zA-Z]:\\|[~/])[\w\d\s\.\-\\]+\.\w+",
            "ip_addresses": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]{1,5})?\b",
            "dates": r"\b\d{2,4}[-/\.]\d{2}[-/\.]\d{2,4}\b",
            "times": r"\b\d{2}[h:]\d{2}(?::\d{2})?\b",
            "email_headers": r"(?:From|To|Sent|Cc|Sujet):\s*.+",
        }

        self.compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.patterns.items()
        }

    def clean_text(self, text: str) -> str:

        if not isinstance(text, str):
            return ""

        text = BeautifulSoup(
            text,
            "html.parser"
        ).get_text()

        text = self.compiled_patterns[
            "email_headers"
        ].sub("", text)

        text = self.compiled_patterns[
            "urls"
        ].sub("[URL]", text)

        text = self.compiled_patterns[
            "path"
        ].sub("[PATH]", text)

        text = self.compiled_patterns[
            "ip_addresses"
        ].sub("[IP]", text)

        text = self.compiled_patterns[
            "dates"
        ].sub("[DATE]", text)

        text = self.compiled_patterns[
            "times"
        ].sub("[TIME]", text)

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text