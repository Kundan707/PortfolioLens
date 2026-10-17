"""Small ASCII PDF writer used for the sample brief and tests."""

from __future__ import annotations

from pathlib import Path


def write_text_pdf(path: Path, pages: list[list[str]]) -> None:
    """Write a multi-page PDF. Each page is a list of paragraphs."""
    if not pages:
        raise ValueError("At least one page is required")

    rendered = [_layout(paragraphs) for paragraphs in pages]
    font_id = 3
    next_id = 4
    page_plan: list[tuple[int, int, bytes]] = []
    for lines in rendered:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_plan.append((page_id, content_id, _content_stream(lines)))

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = " ".join(f"{page_id} 0 R" for page_id, _, _ in page_plan)
    objects[2] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_plan)} >>".encode()
    )
    for page_id, content_id, stream in page_plan:
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id in range(1, next_id):
        offsets.append(len(out))
        out.extend(f"{obj_id} 0 obj\n".encode())
        out.extend(objects[obj_id])
        out.extend(b"\nendobj\n")

    xref_at = len(out)
    out.extend(f"xref\n0 {next_id}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, next_id):
        out.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer << /Size {next_id} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n"
        ).encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def _layout(paragraphs: list[str]) -> list[str]:
    lines: list[str] = []
    for paragraph in paragraphs:
        lines.extend(_wrap(paragraph))
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) > 46:
        raise ValueError("A page has too many lines to lay out")
    return lines


def _wrap(text: str, width: int = 85) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = [words[0]]
    for word in words[1:]:
        if len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


def _content_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", "16 TL", "54 740 Td"]
    for index, line in enumerate(lines):
        if index:
            commands.append("T*")
        commands.append(f"({_escape(line)}) Tj")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("latin-1")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
