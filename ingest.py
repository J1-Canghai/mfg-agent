"""Split the equipment manuals into chunks and write data/chunks.jsonl.

Usage: python ingest.py
"""
import json
from pathlib import Path

BASE = Path(__file__).parent
MANUAL_DIR = BASE / "data" / "manuals"
CHUNK_FILE = BASE / "data" / "chunks.jsonl"

CHUNK_SIZE = 400
OVERLAP = 80


def load_docs(folder):
    docs = []
    for path in sorted(folder.glob("*.md")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def split_sections(doc):
    """Split on markdown headings: '# ' is the doc title, '## ' starts a section."""
    title = doc["source"]
    sections = []
    section, lines = "简介", []
    for line in doc["text"].splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append({"section": section, "text": "\n".join(lines).strip()})
            section, lines = line[3:].strip(), []
        else:
            lines.append(line)
    sections.append({"section": section, "text": "\n".join(lines).strip()})

    result = []
    for s in sections:
        if s["text"]:
            result.append({"source": doc["source"], "title": title, **s})
    return result


def split_long(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Sliding window over long sections, preferring to end a piece at a full stop."""
    if len(text) <= size:
        return [text]
    pieces = []
    start = 0
    while True:
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("。", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return pieces


def build_chunks(sections):
    chunks = []
    sec_no = {}
    for s in sections:
        stem = s["source"].removesuffix(".md")
        sec_no[stem] = sec_no.get(stem, 0) + 1
        for i, piece in enumerate(split_long(s["text"]), start=1):
            chunks.append({
                "chunk_id": f"{stem}-{sec_no[stem]}-{i}",
                "source": s["source"],
                "section": s["section"],
                # the title prefix helps retrieval when the body alone is ambiguous
                "text": f"{s['title']} > {s['section']}\n{piece}",
            })
    return chunks


def main():
    sections = []
    for doc in load_docs(MANUAL_DIR):
        secs = split_sections(doc)
        print(f"{doc['source']}: {len(secs)} sections")
        sections.extend(secs)

    chunks = build_chunks(sections)
    with CHUNK_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lengths = [len(c["text"]) for c in chunks]
    print(f"{len(chunks)} chunks, length min {min(lengths)} / avg {sum(lengths) / len(lengths):.0f} / max {max(lengths)}")


if __name__ == "__main__":
    main()
