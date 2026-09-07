from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

KEYWORDS = re.compile(
    r"hunting|monotemporal|multitemporal|LOYO|Boyce|CBI|AUC|feature importance|"
    r"response curve|MESS|NT1|NT2|novelty|configuration|predictor",
    re.IGNORECASE,
)
CAPTION = re.compile(r"^(table|figure|supplement|supplementary|s\d+)\b", re.IGNORECASE)


def docx_paragraphs(path: Path) -> Iterable[tuple[int | None, str]]:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", ns):
        text = "".join(
            (node.text or "") for node in paragraph.findall(".//w:t", ns)
        ).strip()
        if text:
            yield None, text


def pdf_pages(path: Path) -> Iterable[tuple[int, str]]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages, start=1):
            yield page_no, page.extract_text() or ""
        return
    except Exception:
        pass

    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError("Install pypdf or pdftotext to index the supplement PDF")
    with tempfile.TemporaryDirectory() as tmp:
        text_path = Path(tmp) / "supplement.txt"
        subprocess.run([executable, "-layout", str(path), str(text_path)], check=True)
        yield 0, text_path.read_text(encoding="utf-8", errors="replace")


def index_sources(manuscript: Path | None, supplement: Path | None, output: Path) -> None:
    rows = []
    if manuscript and manuscript.exists():
        for page, text in docx_paragraphs(manuscript):
            if CAPTION.match(text) or KEYWORDS.search(text):
                rows.append(
                    {
                        "source": "manuscript",
                        "page": page,
                        "text": text,
                        "caption_like": bool(CAPTION.match(text)),
                        "likely_affected": bool(KEYWORDS.search(text)),
                    }
                )
    if supplement and supplement.exists():
        for page, page_text in pdf_pages(supplement):
            for line in (line.strip() for line in page_text.splitlines()):
                if line and (CAPTION.match(line) or KEYWORDS.search(line)):
                    rows.append(
                        {
                            "source": "supplement",
                            "page": page,
                            "text": line,
                            "caption_like": bool(CAPTION.match(line)),
                            "likely_affected": bool(KEYWORDS.search(line)),
                        }
                    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "page", "text", "caption_like", "likely_affected"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Index current manuscript/supplement references.")
    parser.add_argument("--manuscript", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    index_sources(args.manuscript, args.supplement, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
