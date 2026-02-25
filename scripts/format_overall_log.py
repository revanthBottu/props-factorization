#!/usr/bin/env python3
"""Format a CSV-style overall_log.txt into a neat table.

Usage examples:
  python scripts/format_overall_log.py \
      --input logs/walker2d_props/overall_log.txt \
      --format md \
      --out logs/walker2d_props/overall_log.md

  python scripts/format_overall_log.py --input logs/walker2d_props/overall_log.txt

Defaults to printing a Markdown table to stdout when --out is omitted.
"""
import argparse
import csv
from pathlib import Path
from typing import List


def read_csv_rows(path: Path) -> (List[str], List[List[str]]):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [ [cell.strip() for cell in row] for row in reader if row and any(c.strip() for c in row) ]
    if not rows:
        raise SystemExit(f"Empty or missing file: {path}")
    header = rows[0]
    body = rows[1:]
    return header, body


def fmt_number_like(s: str) -> str:
    # Try to format numeric values for consistent column widths
    try:
        f = float(s)
    except Exception:
        return s
    # large integers should be printed without decimal
    if abs(f - int(f)) < 1e-9:
        return str(int(f))
    # otherwise use 3 decimal places but strip trailing zeros
    out = f"{f:.3f}".rstrip("0").rstrip(".")
    return out


def make_markdown_table(header: List[str], rows: List[List[str]]) -> str:
    formatted = [[fmt_number_like(cell) for cell in row] for row in rows]
    # compute column widths
    cols = len(header)
    widths = [len(h) for h in header]
    for r in formatted:
        for i in range(cols):
            if i < len(r):
                widths[i] = max(widths[i], len(r[i]))

    def pad(cell, i):
        return cell + " " * (widths[i] - len(cell))

    lines = []
    lines.append("| " + " | ".join(pad(header[i], i) for i in range(cols)) + " |")
    lines.append("| " + " | ".join("-" * widths[i] for i in range(cols)) + " |")
    for r in formatted:
        # safe-get cell or empty
        line_cells = [pad(r[i] if i < len(r) else "", i) for i in range(cols)]
        lines.append("| " + " | ".join(line_cells) + " |")

    return "\n".join(lines)


def make_ascii_table(header: List[str], rows: List[List[str]]) -> str:
    formatted = [[fmt_number_like(cell) for cell in row] for row in rows]
    cols = len(header)
    widths = [len(h) for h in header]
    for r in formatted:
        for i in range(cols):
            if i < len(r):
                widths[i] = max(widths[i], len(r[i]))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def rowline(cells):
        cells = [(cells[i] if i < len(cells) else "") for i in range(cols)]
        return "| " + " | ".join(cells[i] + " " * (widths[i] - len(cells[i])) for i in range(cols)) + " |"

    lines = [sep, rowline(header), sep]
    for r in formatted:
        lines.append(rowline(r))
    lines.append(sep)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="logs/walker2d_props/overall_log.txt")
    p.add_argument("--format", "-f", choices=("md", "ascii"), default="md")
    p.add_argument("--out", "-o", help="Write output to file (optional)")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    header, rows = read_csv_rows(path)

    if args.format == "md":
        out = make_markdown_table(header, rows)
    else:
        out = make_ascii_table(header, rows)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        print(f"Wrote formatted table to: {out_path}")
    else:
        print(out)


if __name__ == "__main__":
    main()
