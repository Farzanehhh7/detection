
import argparse
import io
import tokenize
from pathlib import Path

EXCLUDE_DIRS = {".venv", "venv", "env", ".git", "__pycache__", "node_modules", ".idea"}


def strip_comments_from_source(source: str, strip_docstrings: bool = False) -> str:
    lines = source.splitlines(keepends=True)
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))

    edits = []
    prev_toktype = tokenize.NEWLINE

    for tok in tokens:
        tok_type, tok_string, start, end, _line = tok

        if tok_type == tokenize.COMMENT:
            edits.append((start[0], start[1], end[1]))

        elif strip_docstrings and tok_type == tokenize.STRING:
            is_standalone = prev_toktype in (
                tokenize.NEWLINE, tokenize.INDENT, tokenize.NL, tokenize.ENCODING
            )
            if is_standalone:
                if start[0] != end[0]:
                    for row in range(start[0], end[0] + 1):
                        col_start = start[1] if row == start[0] else 0
                        col_end = end[1] if row == end[0] else len(lines[row - 1])
                        edits.append((row, col_start, col_end))
                else:
                    edits.append((start[0], start[1], end[1]))

        if tok_type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev_toktype = tok_type

    edits_by_line: dict[int, list[tuple[int, int]]] = {}
    for row, col_start, col_end in edits:
        edits_by_line.setdefault(row, []).append((col_start, col_end))

    new_lines = list(lines)
    for row, spans in edits_by_line.items():
        line = new_lines[row - 1]
        for col_start, col_end in sorted(spans, reverse=True):
            line = line[:col_start] + line[col_end:]
        new_lines[row - 1] = line

    result_lines = []
    for line in new_lines:
        if line.strip() == "":
            result_lines.append("\n" if line.endswith("\n") else "")
        else:
            result_lines.append(line)

    return "".join(result_lines)


def process_file(path: Path, strip_docstrings: bool, dry_run: bool) -> None:
    source = path.read_text(encoding="utf-8")
    try:
        cleaned = strip_comments_from_source(source, strip_docstrings=strip_docstrings)
    except Exception as e:
        print(f"رد شد (خطای تجزیه): {path} -> {e}")
        return

    if cleaned == source:
        print(f"بدون تغییر: {path}")
        return

    if dry_run:
        print(f"تغییر می‌کنه (dry-run، چیزی نوشته نشد): {path}")
    else:
        path.write_text(cleaned, encoding="utf-8")
        print(f"پاک‌سازی شد: {path}")


def find_python_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    files = []
    for f in target.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in f.parts):
            continue
        files.append(f)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="فایل یا پوشه‌ی پایتون")
    parser.add_argument("--strip-docstrings", action="store_true",
                         help="docstring ها رو هم پاک کن (پیش‌فرض: نگه‌داشته میشن)")
    parser.add_argument("--dry-run", action="store_true",
                         help="فقط نشون بده چی تغییر می‌کنه، چیزی روی دیسک ننویس")
    args = parser.parse_args()

    files = find_python_files(Path(args.target))

    if not files:
        print("هیچ فایل .py پیدا نشد.")
        return

    print(f"{len(files)} فایل پیدا شد.\n")
    for f in files:
        process_file(f, args.strip_docstrings, args.dry_run)


if __name__ == "__main__":
    main()