#!/usr/bin/env python3
r"""Copy figures and optional tables referenced by a Paper1 TeX entry point.

The script parses \includegraphics targets after recursively expanding simple
\input{...} files. It is intentionally narrow: it is a source-packaging helper,
not a general LaTeX parser.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

GRAPHIC_EXTS = (".pdf", ".png", ".jpg", ".jpeg")
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHICSPATH_RE = re.compile(r"\\graphicspath\{((?:\{[^{}]*\})+)\}")
GRAPHICSPATH_ENTRY_RE = re.compile(r"\{([^{}]+)\}")


def strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        out = []
        escaped = False
        for ch in line:
            if ch == "%" and not escaped:
                break
            out.append(ch)
            escaped = (ch == "\\" and not escaped)
            if ch != "\\":
                escaped = False
        lines.append("".join(out))
    return "\n".join(lines)


def resolve_input(name: str, current: Path, base_dir: Path) -> Path:
    candidates = []
    raw = Path(name)
    names = [raw]
    if raw.suffix == "":
        names.append(raw.with_suffix(".tex"))
    for item in names:
        candidates.append((current.parent / item).resolve())
        candidates.append((base_dir / item).resolve())
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing TeX input {name!r}; searched:\n  {searched}")


def collect_tex(tex_path: Path, base_dir: Path, seen: set[Path]) -> tuple[str, list[str]]:
    tex_path = tex_path.resolve()
    if tex_path in seen:
        return "", []
    seen.add(tex_path)
    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    graphic_paths: list[str] = []
    for match in GRAPHICSPATH_RE.finditer(text):
        graphic_paths.extend(GRAPHICSPATH_ENTRY_RE.findall(match.group(1)))
    pieces = [text]
    for name in INPUT_RE.findall(text):
        child = resolve_input(name, tex_path, base_dir)
        child_text, child_paths = collect_tex(child, base_dir, seen)
        pieces.append(child_text)
        graphic_paths.extend(child_paths)
    return "\n".join(pieces), graphic_paths


def candidate_paths(target: str, base_dir: Path, graphic_paths: list[str]) -> list[Path]:
    raw = Path(target)
    names = [raw]
    if raw.suffix == "":
        names = [raw.with_suffix(ext) for ext in GRAPHIC_EXTS]

    roots: list[Path] = [base_dir]
    roots.extend((base_dir / p) for p in graphic_paths)
    roots.extend([
        base_dir / "figures",
        base_dir.parent / "assets" / "paper1_figs",
        base_dir / "assets" / "paper1_figs",
    ])

    candidates: list[Path] = []
    for name in names:
        if name.is_absolute():
            candidates.append(name)
            continue
        for root in roots:
            candidates.append((root / name).resolve())
    return candidates


def find_figure(target: str, base_dir: Path, graphic_paths: list[str]) -> Path:
    for candidate in candidate_paths(target, base_dir, graphic_paths):
        if candidate.exists() and candidate.is_file():
            return candidate
    searched = "\n  ".join(str(p) for p in candidate_paths(target, base_dir, graphic_paths))
    raise FileNotFoundError(f"missing figure for {target!r}; searched:\n  {searched}")


def output_relative_path(target: str, source: Path) -> Path:
    """Return the safe path for a collected figure inside the output directory.

    Paper1 currently uses basename-only include targets. Reject directory-bearing
    targets instead of silently flattening them into a bundle that TeX cannot
    compile. Supporting nested target paths later should be an explicit contract
    change with matching bundle-layout tests.
    """

    raw = Path(target)
    if raw.is_absolute() or raw.parent != Path("."):
        raise ValueError(
            f"figure target {target!r} contains a directory; "
            "collect_tex_figures currently requires basename-only targets"
        )
    return raw if raw.suffix else raw.with_suffix(source.suffix)


def copy_referenced_tables(text: str, base_dir: Path, out_dir: Path) -> list[Path]:
    """Copy only direct ``tables/...`` inputs found in expanded TeX text.

    Blind bundles must not include every historical table in the repository.
    Restricting targets to one basename below ``tables/`` also prevents an
    input from escaping the intended bundle directory.
    """

    targets: list[str] = []
    for target in INPUT_RE.findall(text):
        raw = Path(target)
        if raw.parent != Path("tables") or raw.name in {"", ".", ".."}:
            continue
        if target not in targets:
            targets.append(target)

    copied: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        raw = Path(target)
        if raw.suffix not in {"", ".tex"}:
            raise ValueError(f"table input {target!r} must be a TeX file")
        source = (base_dir / raw).with_suffix(".tex").resolve()
        table_root = (base_dir / "tables").resolve()
        if source.parent != table_root or not source.is_file():
            raise FileNotFoundError(f"missing referenced table input: {target}")
        destination = out_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
        print(f"copied {source} -> {destination}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", required=True, help="TeX entry point to parse")
    parser.add_argument("--out-dir", required=True, help="Directory to receive copied figures")
    parser.add_argument(
        "--table-out-dir",
        default=None,
        help="Optional directory to receive only referenced tables/... inputs",
    )
    parser.add_argument("--base-dir", default=None, help="Directory for TeX-relative figure lookup; defaults to the TeX parent")
    parser.add_argument("--dry-run", action="store_true", help="List figures without copying")
    args = parser.parse_args()

    tex_path = Path(args.tex).resolve()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else tex_path.parent.resolve()
    out_dir = Path(args.out_dir).resolve()

    text, graphic_paths = collect_tex(tex_path, base_dir, set())
    targets = []
    for target in INCLUDE_RE.findall(text):
        if target not in targets:
            targets.append(target)

    if not targets:
        raise SystemExit(f"no includegraphics targets found in {tex_path}")

    resolved: list[tuple[str, Path, Path]] = []
    used_paths: dict[Path, Path] = {}
    for target in targets:
        try:
            source = find_figure(target, base_dir, graphic_paths)
            relative_path = output_relative_path(target, source)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        if relative_path in used_paths and used_paths[relative_path] != source:
            raise SystemExit(
                f"duplicate output path {relative_path}: "
                f"{used_paths[relative_path]} and {source}"
            )
        used_paths[relative_path] = source
        resolved.append((target, source, relative_path))

    if args.dry_run:
        for target, source, _ in resolved:
            print(f"{target}\t{source}")
        if args.table_out_dir:
            for target in INPUT_RE.findall(text):
                if Path(target).parent == Path("tables"):
                    print(f"table\t{target}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for _, source, relative_path in resolved:
        destination = out_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"copied {source} -> {destination}")
    if args.table_out_dir:
        copy_referenced_tables(text, base_dir, Path(args.table_out_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
